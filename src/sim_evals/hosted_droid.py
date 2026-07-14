"""Hosted Cybernetic Physics DROID rollout orchestration."""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, Protocol, cast

import numpy as np
from PIL import Image

from .inference.cybernetics_dreamzero import (
    DroidObservationSamplingAPI,
    _action_chunk,
)
from .inference.droid_observation import DroidObservation

_TRANSIENT_MCP_FAILURE_MARKERS = (
    "BRIDGE_OFFLINE",
    "ISAAC_UNREACHABLE",
    "Isaac Sim MCP extension is not ready yet",
    "No bridge connected",
)
_IDEMPOTENT_TRANSPORT_RETRY_TOOLS = frozenset({"isaac.set_joint_positions"})
_TRANSIENT_TRANSPORT_FAILURE_MARKERS = ("HTTP 502",)
_TRANSIENT_MCP_RETRIES = 12
_DROID_EXTERNAL_CAMERA_ROOT_PREFIX = "/World/droid_eval_"
_DROID_WRIST_CAMERA_PREFIX = "droid_eval_wrist_cam_"
_PI0_DROID_POLICY_PROFILE: dict[str, Any] = {
    "base_model": "pi0-droid",
    "openpi_config": "pi0_droid_jointpos_polaris",
    "checkpoint_uri": "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris",
    "openpi_source_commit": "714ec9aa5e4e9b73b98c6bf3a328f377268e26f9",
    "action_space": "droid_joint_position",
    "action_horizon": 10,
    "action_dim": 8,
}


class HostedDroidError(RuntimeError):
    """A hosted DROID rollout could not satisfy its runtime contract."""


class MCPClient(Protocol):
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


class SimulationLaunch(Protocol):
    session_id: str


class SimulationClientAPI(Protocol):
    def launch(self, environment_uri: str, **kwargs: Any) -> SimulationLaunch: ...

    def wait_for_session(
        self,
        session_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> Mapping[str, Any]: ...

    def mcp_session(self, session_id: str) -> AbstractContextManager[MCPClient]: ...

    def stop_session(self, session_id: str) -> None: ...


@dataclass(frozen=True)
class CameraSpec:
    prim_path: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    focal_length: float
    clipping_range: tuple[float, float] = (0.05, 100.0)
    focus_distance: float = 28.0
    horizontal_aperture: float = 5.376
    vertical_aperture: float = 3.024


@dataclass(frozen=True)
class HostedDroidConfig:
    environment_uri: str
    session_id: str | None = None
    base_model: str = "dreamzero-droid"
    instruction: str = "put the cube in the bowl"
    robot_prim_path: str = "/World/robot"
    robot_usd_path: str = "/data/workspace/franka_robotiq_2f_85_flattened.usd"
    cameras: tuple[CameraSpec, ...] = field(default_factory=lambda: _default_cameras())
    image_width: int = 640
    image_height: int = 360
    max_action_steps: int = 450
    open_loop_horizon: int = 8
    physics_steps_per_action: int | None = None
    target_control_hz: float = 15.0
    runtime_provider: str | None = None
    policy_mode: str = "native"
    include_predicted_video: bool = False
    request_timeout_seconds: float = 2400.0
    launch_timeout_seconds: float = 1200.0
    readiness_timeout_seconds: float = 600.0
    readiness_poll_seconds: float = 5.0
    keep_session: bool = True
    record_video: bool = False
    video_fps: int = 15
    results_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.environment_uri.strip():
            raise ValueError("environment_uri must not be empty")
        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.base_model.strip():
            raise ValueError("base_model must not be empty")
        if self.policy_mode not in {"native", "sde"}:
            raise ValueError("policy_mode must be native or sde")
        if len(self.cameras) != 3:
            raise ValueError("DROID requires exactly three RGB cameras")
        for name, value in (
            ("image_width", self.image_width),
            ("image_height", self.image_height),
            ("max_action_steps", self.max_action_steps),
            ("open_loop_horizon", self.open_loop_horizon),
            ("video_fps", self.video_fps),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if (
            self.physics_steps_per_action is not None
            and self.physics_steps_per_action < 1
        ):
            raise ValueError("physics_steps_per_action must be at least 1")
        if not math.isfinite(self.target_control_hz) or self.target_control_hz <= 0:
            raise ValueError("target_control_hz must be positive and finite")


@dataclass(frozen=True)
class HostedDroidRunResult:
    session_id: str
    samples: int
    action_steps: int
    repaired_robot: bool
    created_cameras: tuple[str, ...]
    session_retained: bool
    physics_dt: float
    physics_steps_per_action: int
    control_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "samples": self.samples,
            "action_steps": self.action_steps,
            "repaired_robot": self.repaired_robot,
            "created_cameras": list(self.created_cameras),
            "session_retained": self.session_retained,
            "physics_dt": self.physics_dt,
            "physics_steps_per_action": self.physics_steps_per_action,
            "control_hz": self.control_hz,
        }


_EVIDENCE_SCHEMA_VERSION = 5
_EVIDENCE_CAMERA_NAMES = ("exterior-1", "exterior-2", "wrist")


def _mediapy_module() -> Any | None:
    try:
        return importlib.import_module("mediapy")
    except ModuleNotFoundError as exc:
        if exc.name != "mediapy":
            raise
        return None


def _require_video_backend() -> None:
    if _mediapy_module() is not None:
        return
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    raise HostedDroidError(
        "video recording requires mediapy or both ffmpeg and ffprobe; "
        "install a video backend before launching the hosted rollout"
    )


def _run_video_command(
    command: list[str], operation: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "no diagnostic output").strip()
        raise HostedDroidError(f"{operation} failed: {detail}") from exc


def _write_video_with_ffmpeg(
    temporary_path: Path,
    frame_paths: list[Path],
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise HostedDroidError("ffmpeg fallback requires both ffmpeg and ffprobe")

    indexes = [int(path.stem.removeprefix("action-")) for path in frame_paths]
    expected_indexes = list(range(indexes[0], indexes[0] + len(indexes)))
    if indexes != expected_indexes:
        raise HostedDroidError("video source frame indexes are not contiguous")

    _run_video_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            str(indexes[0]),
            "-i",
            str(frame_paths[0].parent / "action-%05d.png"),
            "-frames:v",
            str(len(frame_paths)),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ],
        "ffmpeg encode",
    )
    _run_video_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-f",
            "null",
            "-",
        ],
        "ffmpeg decode validation",
    )
    probe = _run_video_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames",
            "-of",
            "json",
            str(temporary_path),
        ],
        "ffprobe validation",
    )
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        shape = (int(stream["height"]), int(stream["width"]))
        frame_count = int(stream["nb_read_frames"])
        codec = stream["codec_name"]
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HostedDroidError("ffprobe returned incomplete video metadata") from exc
    if shape != (height, width) or frame_count != len(frame_paths) or codec != "h264":
        raise HostedDroidError(
            "encoded rollout video failed probe validation: "
            f"shape={shape}, frames={frame_count}, codec={codec}"
        )


def finalize_hosted_video_evidence(
    results_dir: Path,
    *,
    fps: int,
    source_camera: str,
) -> dict[str, Any] | None:
    """Encode and validate already-persisted rollout frames without rerunning Isaac."""

    video_frames_dir = results_dir / "video-frames"
    frame_paths = sorted(video_frames_dir.glob("action-*.png"))
    if not frame_paths:
        return None
    _require_video_backend()

    frames: list[np.ndarray] = []
    frame_manifest: list[dict[str, Any]] = []
    for path in frame_paths:
        raw = path.read_bytes()
        frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        frames.append(frame)
        frame_manifest.append(
            {
                "path": str(path.relative_to(results_dir)),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    height, width = frames[0].shape[:2]
    if any(frame.shape != (height, width, 3) for frame in frames):
        raise HostedDroidError("video source frames do not share one RGB shape")

    manifest_path = video_frames_dir / "manifest.json"
    _atomic_write(
        manifest_path,
        (
            json.dumps(
                {
                    "schema_version": _EVIDENCE_SCHEMA_VERSION,
                    "source_camera": source_camera,
                    "frames": frame_manifest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    video_path = results_dir / "rollout.mp4"
    temporary_path = video_path.with_suffix(".tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    mediapy = _mediapy_module()
    if mediapy is not None:
        mediapy.write_video(temporary_path, frames, fps=fps, codec="h264")
        decoded = np.asarray(mediapy.read_video(temporary_path))
        if decoded.shape != (len(frames), height, width, 3):
            raise HostedDroidError(
                "encoded rollout video failed decode validation: "
                f"expected {[len(frames), height, width, 3]}, got {list(decoded.shape)}"
            )
    else:
        _write_video_with_ffmpeg(
            temporary_path,
            frame_paths,
            fps=fps,
            width=width,
            height=height,
        )
    os.replace(temporary_path, video_path)
    video_bytes = video_path.read_bytes()
    return {
        "path": str(video_path.relative_to(results_dir)),
        "bytes": len(video_bytes),
        "sha256": hashlib.sha256(video_bytes).hexdigest(),
        "frames": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps,
        "width": width,
        "height": height,
        "codec": "h264",
        "source_camera": source_camera,
        "source_frames_manifest": str(manifest_path.relative_to(results_dir)),
    }


def recover_hosted_video_evidence(results_dir: Path) -> dict[str, Any]:
    """Finalize video for a completed rollout whose local post-processing failed."""

    config_path = results_dir / "config.json"
    actions_path = results_dir / "actions.jsonl"
    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = config_payload["config"]
        source_camera = config["cameras"][0]["prim_path"]
        fps = int(config["video_fps"])
    except (
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise HostedDroidError(
            "hosted evidence config is incomplete or invalid"
        ) from exc
    if fps < 1:
        raise HostedDroidError("hosted evidence video_fps must be positive")

    counts: dict[str, int] = {}
    try:
        for line in actions_path.read_text(encoding="utf-8").splitlines():
            record_type = json.loads(line).get("record_type")
            if not isinstance(record_type, str):
                raise HostedDroidError("action evidence record is missing record_type")
            counts[record_type] = counts.get(record_type, 0) + 1
    except (OSError, json.JSONDecodeError) as exc:
        raise HostedDroidError(
            "hosted action evidence is incomplete or invalid"
        ) from exc

    video = finalize_hosted_video_evidence(
        results_dir,
        fps=fps,
        source_camera=source_camera,
    )
    if video is None:
        raise HostedDroidError("hosted evidence does not contain rollout video frames")

    original_status = None
    error_path = results_dir / "error.json"
    if error_path.is_file():
        try:
            original_status = json.loads(error_path.read_text(encoding="utf-8")).get(
                "status"
            )
        except (OSError, json.JSONDecodeError):
            original_status = "unreadable"
    recovery = {
        "schema_version": _EVIDENCE_SCHEMA_VERSION,
        "status": "video_recovered",
        "recovered_at": _utc_now(),
        "original_status": original_status,
        "action_records": counts,
        "video": video,
    }
    _atomic_write(
        results_dir / "video-recovery.json",
        (json.dumps(recovery, indent=2, sort_keys=True) + "\n").encode(),
    )
    return recovery


class _EvidenceRecorder:
    def __init__(self, results_dir: Path, config: HostedDroidConfig) -> None:
        self.results_dir = results_dir
        self.config = config
        self.frames_dir = results_dir / "frames"
        self.samples_dir = results_dir / "samples"
        self.actions_path = results_dir / "actions.jsonl"
        self.video_frames_dir = results_dir / "video-frames"
        self.video_path = results_dir / "rollout.mp4"
        self.video_manifest_path = self.video_frames_dir / "manifest.json"
        self.runtime_path = results_dir / "runtime.json"
        self._video_metadata: dict[str, Any] | None = None
        self._action_record_count = 0
        self.started_at = _utc_now()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.video_frames_dir.mkdir(parents=True, exist_ok=True)
        self._clear_previous_evidence()
        self._write_json(
            self.results_dir / "config.json",
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "created_at": self.started_at,
                "config": _config_dict(config),
            },
        )

    def write_frame(self, sample_index: int, camera_index: int, raw: bytes) -> None:
        name = _evidence_frame_name(sample_index, camera_index)
        _atomic_write(self.frames_dir / name, raw)

    def write_video_frame(self, action_index: int, rgb: np.ndarray) -> None:
        output = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            output, format="PNG"
        )
        _atomic_write(
            self.video_frames_dir / f"action-{action_index:05d}.png",
            output.getvalue(),
        )

    def write_runtime_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._write_json(
            self.runtime_path,
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                **metadata,
            },
        )

    def finalize_video(self, fps: int) -> None:
        self._video_metadata = finalize_hosted_video_evidence(
            self.results_dir,
            fps=fps,
            source_camera=self.config.cameras[0].prim_path,
        )

    def write_sample(
        self,
        sample_index: int,
        response: Any,
        sampled_action_chunk: np.ndarray,
        action_chunk: np.ndarray,
    ) -> None:
        record: dict[str, Any] = {
            "schema_version": _EVIDENCE_SCHEMA_VERSION,
            "record_type": "sample",
            "sample_index": sample_index,
            "sampled_action_chunk_shape": list(sampled_action_chunk.shape),
            "sampled_action_chunk": sampled_action_chunk.astype(float).tolist(),
            "action_chunk": action_chunk.astype(float).tolist(),
        }
        predicted_video = _response_field(response, "predicted_video")
        if predicted_video is None:
            predicted_video = _response_field(response, "video")
        if predicted_video is not None:
            array = _tensor_array(predicted_video, "predicted_video")
            path = self.samples_dir / f"sample-{sample_index:05d}-predicted-video.npy"
            output = io.BytesIO()
            np.save(output, array, allow_pickle=False)
            _atomic_write(path, output.getvalue())
            record["predicted_video"] = {
                "path": str(path.relative_to(self.results_dir)),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        policy_metadata = _response_field(response, "policy_metadata")
        if policy_metadata is not None:
            if not isinstance(policy_metadata, Mapping):
                raise HostedDroidError("policy_metadata must be a mapping")
            record["policy_metadata"] = dict(policy_metadata)

        trajectory = _response_field(response, "trajectory")
        if trajectory is not None:
            if not isinstance(trajectory, list):
                raise HostedDroidError("trajectory must be a list of tensor mappings")
            arrays: dict[str, np.ndarray] = {}
            steps: list[dict[str, Any]] = []
            for step_index, step in enumerate(trajectory):
                if not isinstance(step, Mapping):
                    raise HostedDroidError(
                        f"trajectory step {step_index} must be a tensor mapping"
                    )
                step_metadata: dict[str, Any] = {}
                for key, value in step.items():
                    array = _tensor_array(value, f"trajectory[{step_index}].{key}")
                    source_key = str(key)
                    encoded_key = source_key.encode("utf-8").hex()
                    archive_key = f"step_{step_index:03d}__key_{encoded_key}"
                    arrays[archive_key] = array
                    step_metadata[source_key] = {
                        "archive_key": archive_key,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                    }
                steps.append(step_metadata)
            path = self.samples_dir / f"sample-{sample_index:05d}-trajectory.npz"
            output = io.BytesIO()
            np.savez_compressed(output, **arrays)
            _atomic_write(path, output.getvalue())
            record["trajectory"] = {
                "path": str(path.relative_to(self.results_dir)),
                "steps": steps,
            }
        self._append_action_record(record)

    def write_applied_action(
        self,
        *,
        sample_index: int,
        chunk_index: int,
        action_index: int,
        policy_action: np.ndarray,
        joint_positions: list[float],
        joint_indices: list[int],
        simulation_timing: Mapping[str, Any],
    ) -> None:
        self._append_action_record(
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "record_type": "applied_action",
                "sample_index": sample_index,
                "chunk_index": chunk_index,
                "action_index": action_index,
                "policy_action": policy_action.astype(float).tolist(),
                "joint_positions": joint_positions,
                "joint_indices": joint_indices,
                "simulation_timing": dict(simulation_timing),
            }
        )

    def write_action_target(
        self,
        joint_positions: list[float],
        joint_indices: list[int],
        *,
        sample_index: int,
        chunk_index: int,
        action_index: int,
        policy_action: np.ndarray,
    ) -> None:
        self._append_action_record(
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "record_type": "action_target",
                "sample_index": sample_index,
                "chunk_index": chunk_index,
                "action_index": action_index,
                "policy_action": policy_action.astype(float).tolist(),
                "joint_positions": joint_positions,
                "joint_indices": joint_indices,
            }
        )

    def _append_action_record(self, record: Mapping[str, Any]) -> None:
        encoded = (json.dumps(record, sort_keys=True) + "\n").encode()
        descriptor = os.open(
            self.actions_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written == 0:
                    raise OSError("actions.jsonl write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._action_record_count += 1

    def write_result(self, result: HostedDroidRunResult) -> None:
        self._write_json(
            self.results_dir / "result.json",
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "status": "succeeded",
                "started_at": self.started_at,
                "finished_at": _utc_now(),
                "result": result.to_dict(),
                "evidence": self._evidence_dict(),
            },
        )

    def write_error(
        self,
        error: BaseException,
        session_id: str | None,
        *,
        evidence_errors: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": _EVIDENCE_SCHEMA_VERSION,
            "status": "failed",
            "started_at": self.started_at,
            "finished_at": _utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "evidence": self._evidence_dict(),
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if evidence_errors:
            payload["evidence_errors"] = evidence_errors
        self._write_json(self.results_dir / "error.json", payload)

    def _clear_previous_evidence(self) -> None:
        for path in (
            self.results_dir / "result.json",
            self.results_dir / "error.json",
            self.actions_path,
            self.runtime_path,
        ):
            path.unlink(missing_ok=True)
        for path in self.frames_dir.glob("sample-*.png"):
            path.unlink()
        for path in self.samples_dir.glob("sample-*.*"):
            path.unlink()
        for path in self.video_frames_dir.glob("action-*.png"):
            path.unlink()
        self.video_path.unlink(missing_ok=True)
        self.video_path.with_suffix(".tmp.mp4").unlink(missing_ok=True)
        self.video_manifest_path.unlink(missing_ok=True)

    def _evidence_dict(self) -> dict[str, Any]:
        frames = []
        for path in sorted(self.frames_dir.glob("sample-*.png")):
            stem = path.stem
            sample_index = int(stem.split("-", 2)[1])
            camera_name = stem.split("-", 2)[2]
            frames.append(
                {
                    "sample_index": sample_index,
                    "camera": camera_name,
                    "path": str(path.relative_to(self.results_dir)),
                }
            )
        sample_artifacts = [
            str(path.relative_to(self.results_dir))
            for path in sorted(self.samples_dir.glob("sample-*.*"))
        ]
        actions = None
        if self.actions_path.is_file():
            actions = {
                "path": str(self.actions_path.relative_to(self.results_dir)),
                "records": self._action_record_count,
            }
        return {
            "frames": frames,
            "actions": actions,
            "sample_artifacts": sample_artifacts,
            "runtime": (
                str(self.runtime_path.relative_to(self.results_dir))
                if self.runtime_path.is_file()
                else None
            ),
            "video": self._video_metadata if self.video_path.is_file() else None,
        }

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        _atomic_write(path, encoded)


ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
GRIPPER_JOINT_NAME = "finger_joint"
GRIPPER_CLOSED_RADIANS = math.pi / 4
_CAMERA_CAPTURE_ATTEMPTS = 10
_CAMERA_CAPTURE_RETRY_STEPS = 2
_CAMERA_CAPTURE_RETRY_SECONDS = 0.5
_CAMERA_WARMUP_STEPS = 32
_CAMERA_WARMUP_SECONDS = 1.0
_MIN_LUMINANCE_P99 = 12.0
_MIN_LUMINANCE_STDDEV = 2.0
_MIN_NON_DARK_FRACTION = 0.01
_MIN_NON_WHITE_FRACTION = 0.02
_NON_WHITE_LUMINANCE_CUTOFF = 245.0


def _default_cameras() -> tuple[CameraSpec, ...]:
    generation = uuid.uuid4().hex[:12]
    external_root = f"/World/droid_eval_{generation}"
    wrist_parent = "/World/robot/Gripper/Robotiq_2F_85/base_link"
    return (
        CameraSpec(
            f"{external_root}/external_cam",
            (0.05, 0.57, 0.66),
            (-0.393, -0.195, 0.399, 0.805),
            2.1,
        ),
        CameraSpec(
            f"{external_root}/external_cam_2",
            (0.05, -0.57, 0.66),
            (0.805, 0.399, -0.195, -0.393),
            2.1,
        ),
        CameraSpec(
            f"{wrist_parent}/droid_eval_wrist_cam_{generation}",
            (0.011, -0.031, -0.074),
            (-0.420, 0.570, 0.576, -0.409),
            2.8,
        ),
    )


class HostedDroidRunner:
    """Run a Cybernetics DROID policy against a hosted Isaac MCP session."""

    def __init__(
        self,
        simulation_client: SimulationClientAPI,
        sampling_api: DroidObservationSamplingAPI,
        config: HostedDroidConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.simulation_client = simulation_client
        self.sampling_api = sampling_api
        self.config = config
        self._monotonic = monotonic
        self._sleep = sleep

    def run(self) -> HostedDroidRunResult:
        if self.config.record_video and self.config.results_dir is not None:
            _require_video_backend()
        evidence = (
            _EvidenceRecorder(self.config.results_dir, self.config)
            if self.config.results_dir is not None
            else None
        )
        session_id = self.config.session_id
        owns_session = False
        cleanup_errors: list[str] = []
        try:
            try:
                if session_id is None:
                    launch_kwargs: dict[str, Any] = {
                        "wait": True,
                        "timeout_seconds": self.config.launch_timeout_seconds,
                        "poll_interval_seconds": self.config.readiness_poll_seconds,
                    }
                    if self.config.runtime_provider is not None:
                        launch_kwargs["runtime_provider"] = self.config.runtime_provider
                    launch = self.simulation_client.launch(
                        self.config.environment_uri,
                        **launch_kwargs,
                    )
                    session_id = launch.session_id
                    owns_session = True
                else:
                    self.simulation_client.wait_for_session(
                        session_id,
                        timeout_seconds=self.config.launch_timeout_seconds,
                        poll_interval_seconds=self.config.readiness_poll_seconds,
                    )
                mcp_session = getattr(self.simulation_client, "mcp_session", None)
                if not callable(mcp_session):
                    raise HostedDroidError(
                        "Cybernetics SimulationClient must expose "
                        "mcp_session(session_id)"
                    )
                mcp_session = cast(
                    Callable[[str], AbstractContextManager[MCPClient]], mcp_session
                )
                with mcp_session(session_id) as mcp:
                    self._wait_for_isaac(mcp)
                    repaired_robot = self._ensure_robot(mcp)
                    self._retire_previous_cameras(mcp)
                    created_cameras = self._ensure_cameras(mcp)
                    self._set_viewer_camera(mcp, created_cameras[0])
                    joint_indices = self._joint_indices(mcp)
                    physics_dt, physics_steps_per_action = self._control_cadence(mcp)
                    control_hz = 1.0 / (physics_dt * physics_steps_per_action)
                    if evidence is not None:
                        evidence.write_runtime_metadata(
                            {
                                "base_model": self.config.base_model,
                                "physics_dt": physics_dt,
                                "physics_steps_per_action": physics_steps_per_action,
                                "target_control_hz": self.config.target_control_hz,
                                "control_hz": control_hz,
                            }
                        )
                    self._step_while_playing(
                        mcp,
                        num_steps=_CAMERA_WARMUP_STEPS,
                    )
                    self._sleep(_CAMERA_WARMUP_SECONDS)
                    self.sampling_api.reset_sampling_session()
                    samples, action_steps = self._rollout(
                        mcp,
                        joint_indices,
                        physics_steps_per_action,
                        evidence,
                    )
                result = HostedDroidRunResult(
                    session_id=session_id,
                    samples=samples,
                    action_steps=action_steps,
                    repaired_robot=repaired_robot,
                    created_cameras=tuple(created_cameras),
                    session_retained=not owns_session or self.config.keep_session,
                    physics_dt=physics_dt,
                    physics_steps_per_action=physics_steps_per_action,
                    control_hz=control_hz,
                )
            except BaseException:
                cleanup_errors.extend(
                    self._cleanup_after_failure(session_id, owns_session)
                )
                raise
            else:
                self.sampling_api.close()
                if (
                    owns_session
                    and session_id is not None
                    and not self.config.keep_session
                ):
                    self.simulation_client.stop_session(session_id)
            if evidence is not None and self.config.record_video:
                evidence.finalize_video(self.config.video_fps)
        except BaseException as exc:
            if evidence is not None:
                evidence_errors = list(cleanup_errors)
                if self.config.record_video and not evidence.video_path.is_file():
                    try:
                        evidence.finalize_video(self.config.video_fps)
                    except Exception as evidence_exc:
                        evidence_errors.append(
                            f"video finalization failed: {type(evidence_exc).__name__}: "
                            f"{evidence_exc}"
                        )
                evidence.write_error(
                    exc,
                    session_id,
                    evidence_errors=evidence_errors,
                )
            raise
        if evidence is not None:
            evidence.write_result(result)
        return result

    def _cleanup_after_failure(
        self,
        session_id: str | None,
        owns_session: bool,
    ) -> list[str]:
        cleanup_errors: list[str] = []
        try:
            self.sampling_api.close()
        except Exception as exc:
            cleanup_errors.append(
                f"sampling API close failed: {type(exc).__name__}: {exc}"
            )
        if owns_session and session_id is not None and not self.config.keep_session:
            try:
                self.simulation_client.stop_session(session_id)
            except Exception as exc:
                cleanup_errors.append(
                    f"session stop failed: {type(exc).__name__}: {exc}"
                )
        return cleanup_errors

    def _wait_for_isaac(self, mcp: MCPClient) -> None:
        deadline = self._monotonic() + self.config.readiness_timeout_seconds
        last_error = "not checked"
        while True:
            try:
                self._call(mcp, "isaac.get_scene_info", {})
                return
            except Exception as exc:
                last_error = str(exc)
            if self._monotonic() >= deadline:
                raise HostedDroidError(
                    "Isaac MCP was not ready after "
                    f"{self.config.readiness_timeout_seconds}s: {last_error}"
                )
            self._sleep(self.config.readiness_poll_seconds)

    def _ensure_robot(self, mcp: MCPClient) -> bool:
        info = self._try_call(
            mcp,
            "isaac.get_robot_info",
            {"prim_path": self.config.robot_prim_path},
        )
        if info is not None and _has_droid_joints(info):
            return False

        if (
            self._try_call(
                mcp,
                "isaac.get_prim_info",
                {"prim_path": self.config.robot_prim_path},
            )
            is not None
        ):
            self._call(
                mcp,
                "isaac.delete_object",
                {"prim_path": self.config.robot_prim_path},
            )
        self._call(
            mcp,
            "isaac.load_usd",
            {
                "usd_path": self.config.robot_usd_path,
                "prim_path": self.config.robot_prim_path,
            },
        )
        self._call(mcp, "isaac.play_simulation", {})
        self._call(
            mcp,
            "isaac.step_simulation",
            {"num_steps": 1, "observe_joints": [self.config.robot_prim_path]},
        )
        repaired = self._call(
            mcp,
            "isaac.get_robot_info",
            {"prim_path": self.config.robot_prim_path},
        )
        if not _has_droid_joints(repaired):
            raise HostedDroidError(
                f"loaded robot at {self.config.robot_prim_path} is not DROID-compatible"
            )
        return True

    def _ensure_cameras(self, mcp: MCPClient) -> list[str]:
        created: list[str] = []
        for camera in self.config.cameras:
            arguments = {
                "prim_path": camera.prim_path,
                "position": list(camera.position),
                "orientation": list(camera.orientation_wxyz),
                "resolution": [self.config.image_width, self.config.image_height],
                "focal_length": camera.focal_length,
                "clipping_range": list(camera.clipping_range),
                "focus_distance": camera.focus_distance,
                "horizontal_aperture": camera.horizontal_aperture,
                "vertical_aperture": camera.vertical_aperture,
            }
            if self._try_call(mcp, "isaac.create_camera", arguments) is None:
                self._configure_legacy_camera(mcp, camera)
                self._call(
                    mcp,
                    "isaac.create_camera",
                    {
                        "prim_path": camera.prim_path,
                        "resolution": [
                            self.config.image_width,
                            self.config.image_height,
                        ],
                    },
                )
            created.append(camera.prim_path)
        return created

    def _retire_previous_cameras(self, mcp: MCPClient) -> None:
        current_paths = {camera.prim_path for camera in self.config.cameras}
        current_external_roots = {
            path.rsplit("/", 1)[0]
            for path in current_paths
            if path.startswith(_DROID_EXTERNAL_CAMERA_ROOT_PREFIX)
        }
        stale_paths: set[str] = set()

        world_prims = self._listed_prim_paths(mcp, "/World")
        stale_paths.update(
            path
            for path in world_prims
            if path.startswith(_DROID_EXTERNAL_CAMERA_ROOT_PREFIX)
            and path not in current_external_roots
        )

        wrist_parent = "/World/robot/Gripper/Robotiq_2F_85/base_link"
        wrist_prefix = f"{wrist_parent}/{_DROID_WRIST_CAMERA_PREFIX}"
        wrist_prims = self._listed_prim_paths(mcp, wrist_parent)
        stale_paths.update(
            path
            for path in wrist_prims
            if path.startswith(wrist_prefix) and path not in current_paths
        )

        for path in sorted(stale_paths, key=lambda item: (-item.count("/"), item)):
            self._call(mcp, "isaac.delete_object", {"prim_path": path})

    def _listed_prim_paths(self, mcp: MCPClient, root_path: str) -> set[str]:
        payload = self._call(mcp, "isaac.list_prims", {"root_path": root_path})
        prims = payload.get("prims")
        if not isinstance(prims, list):
            raise HostedDroidError("isaac.list_prims did not return prims")
        paths = {
            prim.get("path")
            for prim in prims
            if isinstance(prim, Mapping) and isinstance(prim.get("path"), str)
        }
        if len(paths) != len(prims):
            raise HostedDroidError("isaac.list_prims returned malformed prim records")
        return cast(set[str], paths)

    def _set_viewer_camera(self, mcp: MCPClient, prim_path: str) -> None:
        if (
            self._try_call(
                mcp,
                "isaac.set_active_camera",
                {"prim_path": prim_path},
            )
            is not None
        ):
            return
        encoded_path = json.dumps(prim_path)
        code = f"""
import omni.kit.app
from omni.kit.viewport.utility import get_active_viewport

viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("no active viewport")
viewport.camera_path = {encoded_path}
omni.kit.app.get_app().update()
print({{"status": "success", "active_camera": str(viewport.camera_path)}})
"""
        self._call(mcp, "isaac.execute_script", {"code": code})

    def _configure_legacy_camera(self, mcp: MCPClient, camera: CameraSpec) -> None:
        position = json.dumps(list(camera.position))
        orientation = json.dumps(list(camera.orientation_wxyz))
        prim_path = json.dumps(camera.prim_path)
        code = f"""
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
camera = UsdGeom.Camera.Define(stage, {prim_path})
prim = camera.GetPrim()
xformable = UsdGeom.Xformable(prim)
xformable.ClearXformOpOrder()
position = {position}
orientation = {orientation}
xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Vec3d(*position)
)
xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:]))
)
camera.GetFocalLengthAttr().Set({camera.focal_length})
camera.GetClippingRangeAttr().Set(
    Gf.Vec2f({camera.clipping_range[0]}, {camera.clipping_range[1]})
)
camera.GetFocusDistanceAttr().Set({camera.focus_distance})
camera.GetHorizontalApertureAttr().Set({camera.horizontal_aperture})
camera.GetVerticalApertureAttr().Set({camera.vertical_aperture})
print({{"status": "success", "prim_path": {prim_path}}})
"""
        self._call(mcp, "isaac.execute_script", {"code": code})

    def _joint_indices(self, mcp: MCPClient) -> tuple[list[int], int]:
        info = self._call(
            mcp,
            "isaac.get_robot_info",
            {"prim_path": self.config.robot_prim_path},
        )
        names = info.get("joint_names")
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise HostedDroidError("isaac.get_robot_info did not return joint_names")
        missing = [
            name for name in (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME) if name not in names
        ]
        if missing:
            raise HostedDroidError(
                f"DROID robot is missing joints: {', '.join(missing)}"
            )
        return [names.index(name) for name in ARM_JOINT_NAMES], names.index(
            GRIPPER_JOINT_NAME
        )

    def _rollout(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
        physics_steps_per_action: int,
        evidence: _EvidenceRecorder | None = None,
    ) -> tuple[int, int]:
        samples = 0
        action_steps = 0
        while action_steps < self.config.max_action_steps:
            observation = self._observation(mcp, joint_indices, samples, evidence)
            response = self.sampling_api.sample_droid(
                observation,
                timeout=self.config.request_timeout_seconds,
            )
            sampled_chunk = _action_chunk(response)
            _validate_policy_response(self.config.base_model, response, sampled_chunk)
            chunk = sampled_chunk[: self.config.open_loop_horizon]
            sample_index = samples
            if evidence is not None:
                evidence.write_sample(
                    sample_index,
                    response,
                    sampled_chunk,
                    chunk,
                )
            samples += 1
            for chunk_index, action in enumerate(chunk):
                if action_steps >= self.config.max_action_steps:
                    break
                on_target_accepted: Callable[[list[float], list[int]], None] | None = (
                    None
                )
                if evidence is not None:
                    on_target_accepted = partial(
                        evidence.write_action_target,
                        sample_index=sample_index,
                        chunk_index=chunk_index,
                        action_index=action_steps,
                        policy_action=action,
                    )
                (
                    joint_positions,
                    applied_joint_indices,
                    simulation_timing,
                ) = self._apply_action(
                    mcp,
                    joint_indices,
                    action,
                    physics_steps_per_action,
                    on_target_accepted=on_target_accepted,
                )
                if evidence is not None:
                    evidence.write_applied_action(
                        sample_index=sample_index,
                        chunk_index=chunk_index,
                        action_index=action_steps,
                        policy_action=action,
                        joint_positions=joint_positions,
                        joint_indices=applied_joint_indices,
                        simulation_timing=simulation_timing,
                    )
                    if self.config.record_video:
                        video_rgb = self._capture_rgb(
                            mcp,
                            self.config.cameras[0].prim_path,
                            action_steps,
                            0,
                            None,
                        )
                        evidence.write_video_frame(action_steps, video_rgb)
                action_steps += 1
        return samples, action_steps

    def _observation(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
        sample_index: int,
        evidence: _EvidenceRecorder | None = None,
    ) -> DroidObservation:
        images = [
            self._capture_rgb(
                mcp,
                camera.prim_path,
                sample_index,
                camera_index,
                evidence,
            )
            for camera_index, camera in enumerate(self.config.cameras)
        ]
        positions_payload = self._call(
            mcp,
            "isaac.get_joint_positions",
            {"prim_path": self.config.robot_prim_path},
        )
        positions = np.asarray(
            positions_payload.get("joint_positions"), dtype=np.float32
        )
        arm_indices, gripper_index = joint_indices
        if positions.ndim != 1 or positions.size <= max(*arm_indices, gripper_index):
            raise HostedDroidError(
                "isaac.get_joint_positions returned an incomplete joint vector"
            )
        gripper = np.clip(
            positions[gripper_index] / GRIPPER_CLOSED_RADIANS,
            0.0,
            1.0,
        )
        return DroidObservation(
            exterior_image_1_left=images[0],
            exterior_image_2_left=images[1],
            wrist_image_left=images[2],
            joint_position=np.ascontiguousarray(positions[arm_indices]),
            gripper_position=np.asarray([gripper], dtype=np.float32),
            instruction=self.config.instruction,
        )

    def _capture_rgb(
        self,
        mcp: MCPClient,
        camera_prim_path: str,
        sample_index: int,
        camera_index: int,
        evidence: _EvidenceRecorder | None = None,
    ) -> np.ndarray:
        output_path = (
            f"/data/workspace/media/droid-{sample_index:05d}-{camera_index}.png"
        )
        last_error: HostedDroidError | None = None
        for attempt in range(1, _CAMERA_CAPTURE_ATTEMPTS + 1):
            try:
                capture = self._call(
                    mcp,
                    "isaac.capture_camera_image",
                    {"prim_path": camera_prim_path, "output_path": output_path},
                )
                encoded = _encoded_image(capture)
                if encoded is None:
                    artifact_path = capture.get("output_path", output_path)
                    artifact = self._call(
                        mcp,
                        "isaac.download_artifact",
                        {"path": artifact_path},
                    )
                    encoded = _encoded_image(artifact)
                if encoded is None:
                    raise HostedDroidError(
                        f"Isaac did not return RGB bytes for camera {camera_prim_path}"
                    )
                raw, rgb = _decode_valid_rgb(
                    encoded,
                    camera_prim_path=camera_prim_path,
                    expected_width=self.config.image_width,
                    expected_height=self.config.image_height,
                )
                if evidence is not None:
                    evidence.write_frame(sample_index, camera_index, raw)
                return rgb
            except HostedDroidError as exc:
                last_error = exc
                if attempt == _CAMERA_CAPTURE_ATTEMPTS:
                    break
                if not _is_transport_failure(exc):
                    self._step_while_playing(
                        mcp,
                        num_steps=_CAMERA_CAPTURE_RETRY_STEPS,
                    )
                self._sleep(_CAMERA_CAPTURE_RETRY_SECONDS)
        raise HostedDroidError(
            f"camera {camera_prim_path} did not produce a valid rendered frame after "
            f"{_CAMERA_CAPTURE_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _control_cadence(self, mcp: MCPClient) -> tuple[float, int]:
        state = self._simulation_state(mcp)
        physics_dt = state["physics_dt"]
        configured_steps = self.config.physics_steps_per_action
        if configured_steps is not None:
            return physics_dt, configured_steps
        steps = max(1, round(1.0 / (self.config.target_control_hz * physics_dt)))
        return physics_dt, steps

    def _simulation_state(self, mcp: MCPClient) -> dict[str, Any]:
        state = self._call(mcp, "isaac.get_simulation_state", {})
        physics_dt = state.get("physics_dt")
        current_time = state.get("current_time")
        timeline_state = state.get("timeline_state")
        if (
            isinstance(physics_dt, bool)
            or not isinstance(physics_dt, (int, float))
            or not math.isfinite(float(physics_dt))
            or float(physics_dt) <= 0
        ):
            raise HostedDroidError(
                f"isaac.get_simulation_state returned invalid physics_dt: {physics_dt!r}"
            )
        if (
            isinstance(current_time, bool)
            or not isinstance(current_time, (int, float))
            or not math.isfinite(float(current_time))
        ):
            raise HostedDroidError(
                "isaac.get_simulation_state returned invalid current_time: "
                f"{current_time!r}"
            )
        if timeline_state not in {"playing", "paused", "stopped"}:
            raise HostedDroidError(
                "isaac.get_simulation_state returned invalid timeline_state: "
                f"{timeline_state!r}"
            )
        return {
            "physics_dt": float(physics_dt),
            "current_time": float(current_time),
            "timeline_state": timeline_state,
        }

    def _step_while_playing(
        self,
        mcp: MCPClient,
        *,
        num_steps: int,
        observe_joints: list[str] | None = None,
        observe_cap: int | None = None,
    ) -> dict[str, Any]:
        self._call(mcp, "isaac.play_simulation", {})
        try:
            arguments: dict[str, Any] = {"num_steps": num_steps}
            if observe_joints is not None:
                arguments["observe_joints"] = observe_joints
            if observe_cap is not None:
                arguments["observe_cap"] = observe_cap
            result = self._call(mcp, "isaac.step_simulation", arguments)
        except BaseException as step_exc:
            try:
                self._call(mcp, "isaac.pause_simulation", {})
            except Exception as pause_exc:
                raise HostedDroidError(
                    f"simulation step failed ({step_exc}); pause also failed: {pause_exc}"
                ) from step_exc
            raise
        else:
            self._call(mcp, "isaac.pause_simulation", {})
            return result

    def _apply_action(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
        action: np.ndarray,
        physics_steps_per_action: int,
        *,
        on_target_accepted: Callable[[list[float], list[int]], None] | None = None,
    ) -> tuple[list[float], list[int], dict[str, Any]]:
        arm_indices, gripper_index = joint_indices
        gripper = GRIPPER_CLOSED_RADIANS if float(action[7]) > 0.5 else 0.0
        applied_joint_indices = [*arm_indices, gripper_index]
        joint_positions = [*action[:7].astype(float).tolist(), gripper]
        self._call(
            mcp,
            "isaac.set_joint_positions",
            {
                "prim_path": self.config.robot_prim_path,
                "joint_positions": joint_positions,
                "joint_indices": applied_joint_indices,
            },
        )
        if on_target_accepted is not None:
            on_target_accepted(joint_positions, applied_joint_indices)
        before = self._simulation_state(mcp)
        step = self._step_while_playing(
            mcp,
            num_steps=physics_steps_per_action,
            observe_joints=[self.config.robot_prim_path],
            observe_cap=1,
        )
        after = self._simulation_state(mcp)
        stepped = step.get("stepped")
        if (
            isinstance(stepped, bool)
            or not isinstance(stepped, int)
            or stepped != physics_steps_per_action
            or step.get("timed_out") is True
        ):
            raise HostedDroidError(
                "isaac.step_simulation applied an incomplete action: "
                f"expected {physics_steps_per_action} frames, "
                f"stepped={stepped!r}, timed_out={step.get('timed_out')!r}"
            )
        observed_seconds = after["current_time"] - before["current_time"]
        if observed_seconds < 0:
            raise HostedDroidError(
                "simulation time moved backward while applying action"
            )
        expected_seconds = physics_steps_per_action * before["physics_dt"]
        return (
            joint_positions,
            applied_joint_indices,
            {
                "before": before,
                "after": after,
                "stepped": stepped,
                "expected_simulation_seconds": expected_seconds,
                "observed_simulation_seconds": observed_seconds,
                "timeline_drift_seconds": observed_seconds - expected_seconds,
            },
        )

    def _try_call(
        self, mcp: MCPClient, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        try:
            return self._call(mcp, name, arguments)
        except HostedDroidError:
            return None

    def _call(
        self, mcp: MCPClient, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        for attempt in range(_TRANSIENT_MCP_RETRIES):
            try:
                return self._call_once(mcp, name, arguments)
            except HostedDroidError as exc:
                if not _is_retryable_mcp_failure(name, exc):
                    raise
                if attempt + 1 == _TRANSIENT_MCP_RETRIES:
                    raise
                self._sleep(min(self.config.readiness_poll_seconds, 5.0))
        raise AssertionError("unreachable")

    def _call_once(
        self, mcp: MCPClient, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            raw = mcp.call_tool(name, dict(arguments))
            payload = _tool_payload(raw)
            _raise_tool_error(name, payload)
            if isinstance(payload.get("data"), Mapping):
                payload = dict(payload["data"])
                _raise_tool_error(name, payload)
            return payload
        except HostedDroidError:
            raise
        except Exception as exc:
            raise HostedDroidError(f"{name} failed: {exc}") from exc


def _tool_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, Mapping):
        return dict(structured)
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
    raise HostedDroidError(
        f"MCP tool returned unsupported result type {type(result).__name__}"
    )


def _raise_tool_error(name: str, payload: Mapping[str, Any]) -> None:
    status = str(payload.get("status", "")).lower()
    if payload.get("success") is False or status in {"error", "failed", "failure"}:
        message = payload.get("message") or payload.get("error") or "unknown error"
        raise HostedDroidError(f"{name} failed: {message}")


def _is_retryable_mcp_failure(name: str, exc: HostedDroidError) -> bool:
    message = str(exc)
    if any(marker in message for marker in _TRANSIENT_MCP_FAILURE_MARKERS):
        return True
    return name in _IDEMPOTENT_TRANSPORT_RETRY_TOOLS and any(
        marker in message for marker in _TRANSIENT_TRANSPORT_FAILURE_MARKERS
    )


def _is_transport_failure(exc: HostedDroidError) -> bool:
    return any(marker in str(exc) for marker in _TRANSIENT_TRANSPORT_FAILURE_MARKERS)


def _has_droid_joints(payload: Mapping[str, Any]) -> bool:
    names = payload.get("joint_names")
    return isinstance(names, list) and all(
        name in names for name in (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME)
    )


def _encoded_image(payload: Mapping[str, Any]) -> str | None:
    for key in ("image_base64", "base64", "data"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _decode_valid_rgb(
    encoded: str,
    *,
    camera_prim_path: str,
    expected_width: int,
    expected_height: int,
) -> tuple[bytes, np.ndarray]:
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (expected_width, expected_height):
                raise HostedDroidError(
                    f"camera {camera_prim_path} returned {image.size[0]}x{image.size[1]}, "
                    f"expected {expected_width}x{expected_height}"
                )
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except HostedDroidError:
        raise
    except Exception as exc:
        raise HostedDroidError(
            f"invalid RGB artifact for camera {camera_prim_path}: {exc}"
        ) from exc

    luminance = rgb.astype(np.float32).mean(axis=2)
    p99 = float(np.percentile(luminance, 99))
    stddev = float(luminance.std())
    non_dark_fraction = float(np.count_nonzero(luminance > 8.0) / luminance.size)
    non_white_fraction = float(
        np.count_nonzero(luminance < _NON_WHITE_LUMINANCE_CUTOFF) / luminance.size
    )
    if (
        p99 < _MIN_LUMINANCE_P99
        or stddev < _MIN_LUMINANCE_STDDEV
        or non_dark_fraction < _MIN_NON_DARK_FRACTION
        or non_white_fraction < _MIN_NON_WHITE_FRACTION
    ):
        raise HostedDroidError(
            f"camera {camera_prim_path} returned an unrendered or low-information frame "
            f"(p99={p99:.2f}, stddev={stddev:.2f}, "
            f"non_dark_fraction={non_dark_fraction:.4f}, "
            f"non_white_fraction={non_white_fraction:.4f})"
        )
    return raw, np.ascontiguousarray(rgb)


def _response_field(response: Any, name: str) -> Any:
    if isinstance(response, Mapping):
        return response.get(name)
    return getattr(response, name, None)


def _tensor_array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif isinstance(value, Mapping) and "data" in value:
        shape = value.get("shape")
        value = np.asarray(value["data"])
        if shape is not None:
            value = value.reshape(shape)
    array = np.asarray(value)
    if array.dtype.kind not in {"b", "f", "i", "u"}:
        raise HostedDroidError(f"{name} must contain numeric tensor data")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise HostedDroidError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _validate_policy_response(
    base_model: str,
    response: Any,
    action_chunk: np.ndarray,
) -> None:
    if base_model != "pi0-droid":
        return
    metadata = _response_field(response, "policy_metadata")
    if not isinstance(metadata, Mapping) or dict(metadata) != _PI0_DROID_POLICY_PROFILE:
        raise HostedDroidError(
            "pi0-droid response did not prove the pinned joint-position policy profile"
        )
    if action_chunk.shape != (10, 8):
        raise HostedDroidError(
            f"pi0-droid must return action_chunk [10,8], got {list(action_chunk.shape)}"
        )


def _config_dict(config: HostedDroidConfig) -> dict[str, Any]:
    return {
        "environment_uri": config.environment_uri,
        "session_id": config.session_id,
        "base_model": config.base_model,
        "instruction": config.instruction,
        "robot_prim_path": config.robot_prim_path,
        "robot_usd_path": config.robot_usd_path,
        "cameras": [
            {
                "prim_path": camera.prim_path,
                "position": list(camera.position),
                "orientation_wxyz": list(camera.orientation_wxyz),
                "focal_length": camera.focal_length,
                "clipping_range": list(camera.clipping_range),
                "focus_distance": camera.focus_distance,
                "horizontal_aperture": camera.horizontal_aperture,
                "vertical_aperture": camera.vertical_aperture,
            }
            for camera in config.cameras
        ],
        "image_width": config.image_width,
        "image_height": config.image_height,
        "max_action_steps": config.max_action_steps,
        "open_loop_horizon": config.open_loop_horizon,
        "physics_steps_per_action": config.physics_steps_per_action,
        "target_control_hz": config.target_control_hz,
        "runtime_provider": config.runtime_provider,
        "policy_mode": config.policy_mode,
        "include_predicted_video": config.include_predicted_video,
        "request_timeout_seconds": config.request_timeout_seconds,
        "launch_timeout_seconds": config.launch_timeout_seconds,
        "readiness_timeout_seconds": config.readiness_timeout_seconds,
        "readiness_poll_seconds": config.readiness_poll_seconds,
        "keep_session": config.keep_session,
        "record_video": config.record_video,
        "video_fps": config.video_fps,
        "results_dir": str(config.results_dir) if config.results_dir else None,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _evidence_frame_name(sample_index: int, camera_index: int) -> str:
    return f"sample-{sample_index:05d}-{_EVIDENCE_CAMERA_NAMES[camera_index]}.png"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
