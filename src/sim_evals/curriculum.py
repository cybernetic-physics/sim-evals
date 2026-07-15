"""Deterministic DROID scene-curriculum planning and sequential workflow launch.

The module owns one invariant: every generated scene starts from the exact same
immutable environment version.  Planning is pure and deterministic; launching
is an explicit, sequential side effect that persists the workflow run and exact
output version URI after every state transition.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, cast


CURRICULUM_SCHEMA_VERSION = "droid-scene-curriculum/v1"
DEFAULT_CONTROL_PLANE_URL = "https://api.cyberneticphysics.com"
MAX_VARIANTS_PER_SPLIT = 32
MAX_TOTAL_VARIANTS = 64
MAX_ROOT_SEED = (1 << 63) - 1
MAX_VARIANT_SEED = (1 << 31) - 1
SPLITS = ("train", "validation", "held_out")

CurriculumSplit = Literal["train", "validation", "held_out"]
ExecutionStatus = Literal["planned", "running", "completed", "failed"]
WorkflowStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

_ENVIRONMENT_VERSION_URI = re.compile(
    r"cybernetics://envs/(?P<environment_id>env_[A-Za-z0-9_-]+)"
    r"/versions/(?P<version_id>ver_[A-Za-z0-9_-]+)"
)
_TERMINAL_WORKFLOW_STATUSES = frozenset({"completed", "failed", "cancelled"})


class CurriculumError(RuntimeError):
    """Base error for invalid plans, manifests, or workflow results."""


class CurriculumWorkflowError(CurriculumError):
    """A workflow request or terminal workflow result was unusable."""


class CurriculumLaunchTimeout(CurriculumWorkflowError):
    """A launched workflow remained non-terminal past the caller's deadline."""


@dataclass(frozen=True)
class ImmutableEnvironmentVersion:
    """An exact Cybernetics environment version, never an environment default."""

    environment_id: str
    version_id: str

    @property
    def uri(self) -> str:
        return f"cybernetics://envs/{self.environment_id}/versions/{self.version_id}"

    @classmethod
    def parse(cls, value: str) -> "ImmutableEnvironmentVersion":
        if not isinstance(value, str):
            raise CurriculumError("base environment URI must be a string")
        match = _ENVIRONMENT_VERSION_URI.fullmatch(value)
        if match is None:
            raise CurriculumError(
                "base environment URI must name an exact immutable version: "
                "cybernetics://envs/env_.../versions/ver_..."
            )
        return cls(
            environment_id=match.group("environment_id"),
            version_id=match.group("version_id"),
        )


@dataclass(frozen=True)
class CurriculumPlanConfig:
    """Bounded inputs for a reproducible train/validation/held-out scene plan."""

    base_environment_uri: str
    root_seed: int = 20260715
    train_variants: int = 8
    validation_variants: int = 4
    held_out_variants: int = 4

    def __post_init__(self) -> None:
        ImmutableEnvironmentVersion.parse(self.base_environment_uri)
        _require_bounded_int(
            self.root_seed,
            name="root_seed",
            minimum=0,
            maximum=MAX_ROOT_SEED,
        )
        counts = self.split_counts
        for split, count in counts.items():
            _require_bounded_int(
                count,
                name=f"{split}_variants",
                minimum=1,
                maximum=MAX_VARIANTS_PER_SPLIT,
            )
        if sum(counts.values()) > MAX_TOTAL_VARIANTS:
            raise CurriculumError(
                f"curriculum may contain at most {MAX_TOTAL_VARIANTS} variants"
            )

    @property
    def base_environment(self) -> ImmutableEnvironmentVersion:
        return ImmutableEnvironmentVersion.parse(self.base_environment_uri)

    @property
    def split_counts(self) -> dict[CurriculumSplit, int]:
        return {
            "train": self.train_variants,
            "validation": self.validation_variants,
            "held_out": self.held_out_variants,
        }


@dataclass(frozen=True)
class VariantExecution:
    """Persisted launch state for exactly one independently generated variant."""

    status: ExecutionStatus = "planned"
    workflow_run_id: str | None = None
    output_environment_uri: str | None = None
    environment_version_status: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "workflowRunId": self.workflow_run_id,
            "outputEnvironmentUri": self.output_environment_uri,
            "environmentVersionStatus": self.environment_version_status,
            "error": self.error,
        }


@dataclass(frozen=True)
class CurriculumVariant:
    """A fully specified scene edit plus its independently owned execution."""

    variant_id: str
    split: CurriculumSplit
    index: int
    seed: int
    changes: dict[str, Any]
    prompt: str
    execution: VariantExecution = VariantExecution()

    def plan_dict(self) -> dict[str, Any]:
        return {
            "variantId": self.variant_id,
            "split": self.split,
            "index": self.index,
            "seed": self.seed,
            "changes": self.changes,
            "prompt": self.prompt,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.plan_dict(), "execution": self.execution.to_dict()}


@dataclass(frozen=True)
class CurriculumManifest:
    """Deterministic plan plus resumable, per-variant workflow execution state."""

    base_environment: ImmutableEnvironmentVersion
    root_seed: int
    split_counts: dict[CurriculumSplit, int]
    variants: tuple[CurriculumVariant, ...]
    plan_sha256: str
    schema_version: str = CURRICULUM_SCHEMA_VERSION

    @property
    def output_environment_uris(self) -> tuple[str, ...]:
        return tuple(
            variant.execution.output_environment_uri
            for variant in self.variants
            if variant.execution.output_environment_uri is not None
        )

    def with_execution(
        self,
        variant_index: int,
        execution: VariantExecution,
    ) -> "CurriculumManifest":
        variants = list(self.variants)
        variants[variant_index] = replace(variants[variant_index], execution=execution)
        return replace(self, variants=tuple(variants))

    def plan_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "baseEnvironmentUri": self.base_environment.uri,
            "rootSeed": self.root_seed,
            "splitCounts": self.split_counts,
            "variants": [variant.plan_dict() for variant in self.variants],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.plan_dict(),
            "planSha256": self.plan_sha256,
            "outputEnvironmentUris": list(self.output_environment_uris),
            "variants": [variant.to_dict() for variant in self.variants],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CurriculumManifest":
        schema_version = _require_string(payload, "schemaVersion")
        if schema_version != CURRICULUM_SCHEMA_VERSION:
            raise CurriculumError(
                f"manifest schemaVersion must be {CURRICULUM_SCHEMA_VERSION!r}"
            )
        base_environment = ImmutableEnvironmentVersion.parse(
            _require_string(payload, "baseEnvironmentUri")
        )
        root_seed = _require_manifest_int(payload, "rootSeed")
        raw_counts = _require_mapping(payload, "splitCounts")
        split_counts = cast(
            dict[CurriculumSplit, int],
            {split: _require_manifest_int(raw_counts, split) for split in SPLITS},
        )
        raw_variants = payload.get("variants")
        if not isinstance(raw_variants, list):
            raise CurriculumError("manifest variants must be an array")
        variants = tuple(_variant_from_dict(item) for item in raw_variants)
        manifest = cls(
            base_environment=base_environment,
            root_seed=root_seed,
            split_counts=split_counts,
            variants=variants,
            plan_sha256=_require_string(payload, "planSha256"),
            schema_version=schema_version,
        )
        config = CurriculumPlanConfig(
            base_environment_uri=base_environment.uri,
            root_seed=root_seed,
            train_variants=split_counts["train"],
            validation_variants=split_counts["validation"],
            held_out_variants=split_counts["held_out"],
        )
        if len(variants) != sum(config.split_counts.values()):
            raise CurriculumError("manifest variant count does not match splitCounts")
        expected_digest = _sha256_json(manifest.plan_dict())
        if not _constant_time_equal(expected_digest, manifest.plan_sha256):
            raise CurriculumError("manifest planSha256 does not match its plan")
        canonical = plan_curriculum(config)
        if not _constant_time_equal(canonical.plan_sha256, manifest.plan_sha256):
            raise CurriculumError(
                "manifest plan is not the deterministic plan for its inputs"
            )
        output_uris = payload.get("outputEnvironmentUris", [])
        if not isinstance(output_uris, list) or any(
            not isinstance(item, str) for item in output_uris
        ):
            raise CurriculumError(
                "manifest outputEnvironmentUris must be an array of strings"
            )
        if tuple(output_uris) != manifest.output_environment_uris:
            raise CurriculumError(
                "manifest outputEnvironmentUris does not match completed variant records"
            )
        return manifest


@dataclass(frozen=True)
class WorkflowRunSnapshot:
    """Only the control-plane workflow fields required by the curriculum owner."""

    run_id: str
    status: WorkflowStatus
    input: dict[str, Any]
    result: dict[str, Any] | None = None
    error_message: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkflowRunSnapshot":
        run_id = _require_string(payload, "runId")
        if not run_id.startswith("wfr_"):
            raise CurriculumWorkflowError("workflow run id must start with 'wfr_'")
        raw_status = _require_string(payload, "status")
        if raw_status not in {"queued", "running", "completed", "failed", "cancelled"}:
            raise CurriculumWorkflowError(
                f"workflow {run_id} returned unsupported status {raw_status!r}"
            )
        raw_input = payload.get("input", {})
        if not isinstance(raw_input, dict):
            raise CurriculumWorkflowError(f"workflow {run_id} input must be an object")
        raw_result = payload.get("result")
        if raw_result is not None and not isinstance(raw_result, dict):
            raise CurriculumWorkflowError(f"workflow {run_id} result must be an object")
        raw_error = payload.get("errorMessage")
        if raw_error is not None and not isinstance(raw_error, str):
            raise CurriculumWorkflowError(
                f"workflow {run_id} errorMessage must be a string"
            )
        return cls(
            run_id=run_id,
            status=cast(WorkflowStatus, raw_status),
            input=dict(raw_input),
            result=dict(raw_result) if raw_result is not None else None,
            error_message=raw_error,
        )


class WorkflowRunClient(Protocol):
    """Narrow control-plane workflow boundary used by the sequential launcher."""

    def create_simulation_from_prompt(
        self,
        *,
        prompt: str,
        environment: ImmutableEnvironmentVersion,
        budget_turns: int,
        budget_seconds: float,
        workspace_id: str | None,
    ) -> WorkflowRunSnapshot: ...

    def get_workflow_run(self, run_id: str) -> WorkflowRunSnapshot: ...


class CyberneticsWorkflowRunClient:
    """Small adapter for the existing control-plane workflow REST contract.

    The current Python SDK exposes simulation environment/session/MCP clients,
    but not a public workflow namespace.  This adapter therefore owns only the
    two authenticated workflow routes needed here and never reaches into the
    SDK simulation client's private request method.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
    ) -> None:
        try:
            from cybernetics.lib.credentials import (  # pyright: ignore[reportMissingImports]
                resolve_api_key,
                resolve_base_url,
            )
        except ImportError as exc:
            raise CurriculumWorkflowError(
                "launching requires the Cybernetics Python SDK"
            ) from exc

        resolved_key = resolve_api_key(api_key)
        if not resolved_key:
            raise CurriculumWorkflowError(
                "No Cybernetics API key found; authenticate or set CYBERNETICS_API_KEY"
            )
        resolved_base = resolve_base_url(base_url) or DEFAULT_CONTROL_PLANE_URL
        self.api_key = resolved_key
        self.base_url = str(resolved_base).rstrip("/")
        self._owns_client = http_client is None
        if http_client is None:
            import httpx

            http_client = httpx.Client(base_url=self.base_url, timeout=180.0)
        self._client = http_client

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "CyberneticsWorkflowRunClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def create_simulation_from_prompt(
        self,
        *,
        prompt: str,
        environment: ImmutableEnvironmentVersion,
        budget_turns: int,
        budget_seconds: float,
        workspace_id: str | None,
    ) -> WorkflowRunSnapshot:
        input_payload = {
            "prompt": prompt,
            "sessionMode": "environment",
            "environmentId": environment.environment_id,
            "environmentVersionId": environment.version_id,
        }
        body: dict[str, Any] = {
            "kind": "simulation_from_prompt",
            "input": input_payload,
            "budget": {"turns": budget_turns, "seconds": budget_seconds},
        }
        if workspace_id is not None:
            body["workspaceId"] = workspace_id
        payload = self._request("POST", "/v1/workflows/runs", json_body=body)
        return WorkflowRunSnapshot.from_dict(payload)

    def get_workflow_run(self, run_id: str) -> WorkflowRunSnapshot:
        if not isinstance(run_id, str) or not run_id.startswith("wfr_"):
            raise CurriculumWorkflowError("workflow run id must start with 'wfr_'")
        payload = self._request("GET", f"/v1/workflows/runs/{run_id}")
        return WorkflowRunSnapshot.from_dict(payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(
            method,
            path,
            json=json_body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        status_code = getattr(response, "status_code", None)
        if not isinstance(status_code, int) or status_code < 200 or status_code >= 300:
            detail = _safe_http_error_detail(response)
            suffix = f" ({detail})" if detail else ""
            raise CurriculumWorkflowError(
                f"{method} {path} failed with HTTP {status_code or 'unknown'}{suffix}"
            )
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise CurriculumWorkflowError(
                f"{method} {path} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise CurriculumWorkflowError(f"{method} {path} returned non-object JSON")
        return payload


def plan_curriculum(config: CurriculumPlanConfig) -> CurriculumManifest:
    """Build a byte-stable curriculum plan for the same bounded inputs."""

    used_seeds: set[int] = set()
    variants: list[CurriculumVariant] = []
    for split in SPLITS:
        typed_split = cast(CurriculumSplit, split)
        for index in range(config.split_counts[typed_split]):
            seed = _derive_unique_seed(config.root_seed, typed_split, index, used_seeds)
            changes = _generate_changes(typed_split, seed)
            variant_id = f"{typed_split}-{index:03d}-{seed:08x}"
            variants.append(
                CurriculumVariant(
                    variant_id=variant_id,
                    split=typed_split,
                    index=index,
                    seed=seed,
                    changes=changes,
                    prompt=_render_prompt(
                        base_environment=config.base_environment,
                        variant_id=variant_id,
                        split=typed_split,
                        seed=seed,
                        changes=changes,
                    ),
                )
            )
    partial = CurriculumManifest(
        base_environment=config.base_environment,
        root_seed=config.root_seed,
        split_counts=config.split_counts,
        variants=tuple(variants),
        plan_sha256="pending",
    )
    return replace(partial, plan_sha256=_sha256_json(partial.plan_dict()))


def write_manifest(manifest: CurriculumManifest, path: str | Path) -> Path:
    """Atomically persist a complete manifest from its single launcher owner."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=True
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_manifest(path: str | Path) -> CurriculumManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CurriculumError(f"failed to read curriculum manifest {path}") from exc
    if not isinstance(payload, dict):
        raise CurriculumError("curriculum manifest root must be an object")
    return CurriculumManifest.from_dict(payload)


def load_or_create_manifest(
    config: CurriculumPlanConfig,
    path: str | Path,
) -> CurriculumManifest:
    """Resume only when the persisted immutable plan matches the requested plan."""

    planned = plan_curriculum(config)
    destination = Path(path)
    if not destination.exists():
        write_manifest(planned, destination)
        return planned
    existing = load_manifest(destination)
    if not _constant_time_equal(existing.plan_sha256, planned.plan_sha256):
        raise CurriculumError(
            "existing manifest describes a different curriculum plan; choose a new path"
        )
    return existing


def launch_curriculum(
    manifest: CurriculumManifest,
    client: WorkflowRunClient,
    *,
    manifest_path: str | Path,
    max_launches: int = 1,
    budget_turns: int = 24,
    budget_seconds: float = 3600.0,
    workflow_timeout_seconds: float = 7200.0,
    poll_interval_seconds: float = 5.0,
    workspace_id: str | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> CurriculumManifest:
    """Launch variants one at a time and persist every exact output version URI.

    ``max_launches`` counts new workflow creations, not safe polling of a run
    already recorded as running.  A timeout preserves that running run id so a
    later invocation resumes it instead of creating a duplicate.
    """

    _require_bounded_int(
        max_launches,
        name="max_launches",
        minimum=1,
        maximum=MAX_TOTAL_VARIANTS,
    )
    _require_bounded_int(budget_turns, name="budget_turns", minimum=2, maximum=64)
    if (
        isinstance(budget_seconds, bool)
        or not isinstance(budget_seconds, (int, float))
        or not math.isfinite(float(budget_seconds))
        or budget_seconds <= 0
        or budget_seconds > 3600
    ):
        raise CurriculumError("budget_seconds must be finite and in (0, 3600]")
    if (
        isinstance(workflow_timeout_seconds, bool)
        or not isinstance(workflow_timeout_seconds, (int, float))
        or not math.isfinite(float(workflow_timeout_seconds))
        or workflow_timeout_seconds <= 0
    ):
        raise CurriculumError("workflow_timeout_seconds must be finite and positive")
    if (
        isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, (int, float))
        or not math.isfinite(float(poll_interval_seconds))
        or poll_interval_seconds < 0
    ):
        raise CurriculumError("poll_interval_seconds must be finite and non-negative")

    current = manifest
    new_launches = 0
    for variant_index, variant in enumerate(current.variants):
        execution = variant.execution
        if execution.status == "completed":
            continue
        if execution.status == "failed":
            raise CurriculumWorkflowError(
                f"variant {variant.variant_id} is failed; inspect the manifest before retrying"
            )

        if execution.status == "running":
            run_id = execution.workflow_run_id
            if run_id is None:
                raise CurriculumError(
                    f"running variant {variant.variant_id} has no workflow run id"
                )
            snapshot = client.get_workflow_run(run_id)
            try:
                _validate_source_binding(snapshot, current.base_environment)
            except Exception as exc:
                current = current.with_execution(
                    variant_index,
                    VariantExecution(
                        status="failed",
                        workflow_run_id=run_id,
                        error=_safe_error(exc),
                    ),
                )
                write_manifest(current, manifest_path)
                raise
        else:
            if new_launches >= max_launches:
                break
            try:
                snapshot = client.create_simulation_from_prompt(
                    prompt=variant.prompt,
                    environment=current.base_environment,
                    budget_turns=budget_turns,
                    budget_seconds=float(budget_seconds),
                    workspace_id=workspace_id,
                )
            except Exception as exc:
                current = current.with_execution(
                    variant_index,
                    VariantExecution(status="failed", error=_safe_error(exc)),
                )
                write_manifest(current, manifest_path)
                raise
            try:
                _validate_source_binding(snapshot, current.base_environment)
            except Exception as exc:
                current = current.with_execution(
                    variant_index,
                    VariantExecution(
                        status="failed",
                        workflow_run_id=snapshot.run_id,
                        error=_safe_error(exc),
                    ),
                )
                write_manifest(current, manifest_path)
                raise
            new_launches += 1
            current = current.with_execution(
                variant_index,
                VariantExecution(status="running", workflow_run_id=snapshot.run_id),
            )
            write_manifest(current, manifest_path)

        terminal = wait_for_workflow(
            client,
            snapshot,
            timeout_seconds=float(workflow_timeout_seconds),
            poll_interval_seconds=float(poll_interval_seconds),
            monotonic=monotonic,
            sleep=sleep,
        )
        try:
            _validate_source_binding(terminal, current.base_environment)
        except Exception as exc:
            current = current.with_execution(
                variant_index,
                VariantExecution(
                    status="failed",
                    workflow_run_id=terminal.run_id,
                    error=_safe_error(exc),
                ),
            )
            write_manifest(current, manifest_path)
            raise
        if terminal.status != "completed":
            error = terminal.error_message or f"workflow ended as {terminal.status}"
            current = current.with_execution(
                variant_index,
                VariantExecution(
                    status="failed",
                    workflow_run_id=terminal.run_id,
                    error=error[:1000],
                ),
            )
            write_manifest(current, manifest_path)
            raise CurriculumWorkflowError(
                f"variant {variant.variant_id} workflow {terminal.run_id} {error}"
            )

        try:
            output_environment, version_status = _completed_output(terminal)
            if output_environment.uri == current.base_environment.uri:
                raise CurriculumWorkflowError(
                    f"workflow {terminal.run_id} returned the immutable base version as output"
                )
        except Exception as exc:
            current = current.with_execution(
                variant_index,
                VariantExecution(
                    status="failed",
                    workflow_run_id=terminal.run_id,
                    error=_safe_error(exc),
                ),
            )
            write_manifest(current, manifest_path)
            raise
        current = current.with_execution(
            variant_index,
            VariantExecution(
                status="completed",
                workflow_run_id=terminal.run_id,
                output_environment_uri=output_environment.uri,
                environment_version_status=version_status,
            ),
        )
        write_manifest(current, manifest_path)
    return current


def wait_for_workflow(
    client: WorkflowRunClient,
    initial: WorkflowRunSnapshot,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WorkflowRunSnapshot:
    """Poll one workflow to terminal before any next workflow may be created."""

    snapshot = initial
    deadline = monotonic() + timeout_seconds
    while snapshot.status not in _TERMINAL_WORKFLOW_STATUSES:
        if monotonic() >= deadline:
            raise CurriculumLaunchTimeout(
                f"workflow {snapshot.run_id} did not finish within {timeout_seconds}s"
            )
        sleep(poll_interval_seconds)
        updated = client.get_workflow_run(snapshot.run_id)
        if updated.run_id != snapshot.run_id:
            raise CurriculumWorkflowError(
                f"workflow poll changed run id from {snapshot.run_id} to {updated.run_id}"
            )
        snapshot = updated
    return snapshot


def _generate_changes(split: CurriculumSplit, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    profiles = {
        "train": {
            "object_xy": 0.025,
            "cube_yaw": 12.0,
            "bowl_yaw": 8.0,
            "light": (0.90, 1.10),
            "camera_xyz": 0.010,
            "camera_yaw": 1.5,
            "distractors": (0, 1),
            "cube_colors": ("red", "blue", "green", "yellow"),
            "bowl_colors": ("white", "gray", "navy"),
        },
        "validation": {
            "object_xy": 0.035,
            "cube_yaw": 18.0,
            "bowl_yaw": 12.0,
            "light": (0.82, 1.18),
            "camera_xyz": 0.016,
            "camera_yaw": 2.5,
            "distractors": (1, 2),
            "cube_colors": ("orange", "cyan", "purple"),
            "bowl_colors": ("beige", "charcoal"),
        },
        "held_out": {
            "object_xy": 0.050,
            "cube_yaw": 25.0,
            "bowl_yaw": 18.0,
            "light": (0.72, 1.28),
            "camera_xyz": 0.022,
            "camera_yaw": 3.5,
            "distractors": (2, 3),
            "cube_colors": ("magenta", "teal", "lime"),
            "bowl_colors": ("tan", "black"),
        },
    }
    profile = profiles[split]
    object_xy = cast(float, profile["object_xy"])
    camera_xyz = cast(float, profile["camera_xyz"])
    light_min, light_max = cast(tuple[float, float], profile["light"])
    distractor_min, distractor_max = cast(tuple[int, int], profile["distractors"])
    return {
        "cube": {
            "xOffsetMeters": _rounded_uniform(rng, -object_xy, object_xy),
            "yOffsetMeters": _rounded_uniform(rng, -object_xy, object_xy),
            "yawDegrees": _rounded_uniform(
                rng,
                -cast(float, profile["cube_yaw"]),
                cast(float, profile["cube_yaw"]),
                digits=2,
            ),
            "colorName": rng.choice(cast(Sequence[str], profile["cube_colors"])),
        },
        "bowl": {
            "xOffsetMeters": _rounded_uniform(rng, -object_xy, object_xy),
            "yOffsetMeters": _rounded_uniform(rng, -object_xy, object_xy),
            "yawDegrees": _rounded_uniform(
                rng,
                -cast(float, profile["bowl_yaw"]),
                cast(float, profile["bowl_yaw"]),
                digits=2,
            ),
            "colorName": rng.choice(cast(Sequence[str], profile["bowl_colors"])),
        },
        "lighting": {
            "intensityMultiplier": _rounded_uniform(
                rng, light_min, light_max, digits=3
            ),
            "colorTemperatureKelvin": rng.randrange(3900, 6501, 100),
        },
        "cameraRig": {
            "xOffsetMeters": _rounded_uniform(rng, -camera_xyz, camera_xyz),
            "yOffsetMeters": _rounded_uniform(rng, -camera_xyz, camera_xyz),
            "zOffsetMeters": _rounded_uniform(rng, -camera_xyz, camera_xyz),
            "yawDegrees": _rounded_uniform(
                rng,
                -cast(float, profile["camera_yaw"]),
                cast(float, profile["camera_yaw"]),
                digits=2,
            ),
        },
        "appearance": {
            "tableTone": rng.choice(("neutral-light", "neutral-mid", "neutral-dark")),
            "backgroundTone": rng.choice(
                ("warm-neutral", "cool-neutral", "gray-neutral")
            ),
            "distractorCount": rng.randint(distractor_min, distractor_max),
        },
    }


def _render_prompt(
    *,
    base_environment: ImmutableEnvironmentVersion,
    variant_id: str,
    split: CurriculumSplit,
    seed: int,
    changes: Mapping[str, Any],
) -> str:
    change_json = json.dumps(changes, indent=2, sort_keys=True)
    return f"""Create one isolated DROID cube-in-bowl curriculum scene variant.

Immutable source version: {base_environment.uri}
Variant id: {variant_id}
Dataset split: {split}
Deterministic variant seed: {seed}

The workflow was launched from the exact source version above. Modify only this
workflow's isolated session, then publish a new ready environment version. Never
edit, overwrite, or relaunch from the environment's mutable default version.

Apply these concrete changes (all offsets are relative to the source scene):
{change_json}

Hard acceptance constraints:
- Preserve the DROID robot, articulation, controller, robot start pose, cube-in-
  bowl task, observation cameras, sensor names, task-state paths, and success
  predicate compatibility.
- Keep cube and bowl dimensions, mass, density, friction, collision geometry,
  rigid-body settings, gravity, physics timestep, solver settings, and contact
  behavior exactly unchanged. This curriculum varies appearance and reachable
  initial geometry, not dynamics.
- Place the cube and bowl stably on the table, separated and collision-free.
  Both must remain reachable by the robot from its unchanged start pose. Keep
  distractors outside the robot/cube/bowl workspace and out of camera occlusion.
- Camera offsets apply coherently to the existing camera rig; do not rename,
  add, or delete observation cameras. Verify all required camera views still
  contain the robot workspace, cube, and bowl.
- Use the named colors as visual targets without changing material friction or
  other physics properties. Keep lighting physically plausible and avoid
  overexposure, underexposure, or flicker.
- Inspect the final live scene and publish it. Report the new environment id,
  exact new version id, and ready version status. Do not claim completion if
  persistence or version materialization fails.
"""


def _derive_unique_seed(
    root_seed: int,
    split: CurriculumSplit,
    index: int,
    used: set[int],
) -> int:
    nonce = 0
    while True:
        digest = hashlib.sha256(
            f"{CURRICULUM_SCHEMA_VERSION}:{root_seed}:{split}:{index}:{nonce}".encode()
        ).digest()
        candidate = int.from_bytes(digest[:4], "big") & MAX_VARIANT_SEED
        if candidate not in used:
            used.add(candidate)
            return candidate
        nonce += 1


def _rounded_uniform(
    rng: random.Random,
    minimum: float,
    maximum: float,
    *,
    digits: int = 4,
) -> float:
    return round(rng.uniform(minimum, maximum), digits)


def _validate_source_binding(
    snapshot: WorkflowRunSnapshot,
    expected: ImmutableEnvironmentVersion,
) -> None:
    input_payload = snapshot.input
    expected_fields = {
        "sessionMode": "environment",
        "environmentId": expected.environment_id,
        "environmentVersionId": expected.version_id,
    }
    for key, value in expected_fields.items():
        if input_payload.get(key) != value:
            raise CurriculumWorkflowError(
                f"workflow {snapshot.run_id} did not preserve exact source field {key}"
            )


def _completed_output(
    snapshot: WorkflowRunSnapshot,
) -> tuple[ImmutableEnvironmentVersion, str]:
    if snapshot.result is None:
        raise CurriculumWorkflowError(
            f"completed workflow {snapshot.run_id} did not return a result"
        )
    environment_id = _require_string(snapshot.result, "environmentId")
    version_id = _require_string(snapshot.result, "environmentVersionId")
    version_status = _require_string(snapshot.result, "environmentVersionStatus")
    if version_status != "ready":
        raise CurriculumWorkflowError(
            f"workflow {snapshot.run_id} output version is {version_status!r}, not 'ready'"
        )
    return (
        ImmutableEnvironmentVersion.parse(
            f"cybernetics://envs/{environment_id}/versions/{version_id}"
        ),
        version_status,
    )


def _variant_from_dict(payload: Any) -> CurriculumVariant:
    if not isinstance(payload, dict):
        raise CurriculumError("manifest variant entries must be objects")
    split = _require_string(payload, "split")
    if split not in SPLITS:
        raise CurriculumError(f"unsupported curriculum split {split!r}")
    changes = _require_mapping(payload, "changes")
    execution_payload = _require_mapping(payload, "execution")
    status = _require_string(execution_payload, "status")
    if status not in {"planned", "running", "completed", "failed"}:
        raise CurriculumError(f"unsupported variant execution status {status!r}")
    execution = VariantExecution(
        status=cast(ExecutionStatus, status),
        workflow_run_id=_optional_string(execution_payload, "workflowRunId"),
        output_environment_uri=_optional_string(
            execution_payload, "outputEnvironmentUri"
        ),
        environment_version_status=_optional_string(
            execution_payload, "environmentVersionStatus"
        ),
        error=_optional_string(execution_payload, "error"),
    )
    if (
        execution.workflow_run_id is not None
        and not execution.workflow_run_id.startswith("wfr_")
    ):
        raise CurriculumError("variant workflowRunId must start with 'wfr_'")
    if execution.output_environment_uri is not None:
        ImmutableEnvironmentVersion.parse(execution.output_environment_uri)
    if execution.status == "running" and execution.workflow_run_id is None:
        raise CurriculumError("running variant execution requires workflowRunId")
    if execution.status == "completed":
        if (
            execution.workflow_run_id is None
            or execution.output_environment_uri is None
        ):
            raise CurriculumError(
                "completed variant execution requires workflowRunId and outputEnvironmentUri"
            )
        if execution.environment_version_status != "ready":
            raise CurriculumError("completed output environment version must be ready")
    return CurriculumVariant(
        variant_id=_require_string(payload, "variantId"),
        split=cast(CurriculumSplit, split),
        index=_require_manifest_int(payload, "index"),
        seed=_require_manifest_int(payload, "seed"),
        changes=dict(changes),
        prompt=_require_string(payload, "prompt"),
        execution=execution,
    )


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def _require_bounded_int(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CurriculumError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise CurriculumError(f"{name} must be between {minimum} and {maximum}")
    return value


def _require_manifest_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CurriculumError(f"manifest field {key!r} must be an integer")
    return value


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CurriculumError(f"field {key!r} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CurriculumError(f"field {key!r} must be null or a non-empty string")
    return value


def _require_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise CurriculumError(f"field {key!r} must be an object")
    return dict(value)


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message or 'no message'}"[:1000]


def _safe_http_error_detail(response: Any) -> str | None:
    """Return only the control plane's bounded public error code and message."""

    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, Mapping):
        return None
    raw_code = payload.get("code")
    raw_message = payload.get("message")
    code = raw_code if isinstance(raw_code, str) else None
    message = raw_message if isinstance(raw_message, str) else None
    if code is not None:
        code = re.sub(r"[^A-Za-z0-9_.-]", "", code)[:64] or None
    if message is not None:
        message = re.sub(r"\s+", " ", message).strip()[:300] or None
    if code and message:
        return f"{code}: {message}"
    return code or message


__all__ = [
    "CURRICULUM_SCHEMA_VERSION",
    "MAX_TOTAL_VARIANTS",
    "MAX_VARIANTS_PER_SPLIT",
    "CurriculumError",
    "CurriculumLaunchTimeout",
    "CurriculumManifest",
    "CurriculumPlanConfig",
    "CurriculumVariant",
    "CurriculumWorkflowError",
    "CyberneticsWorkflowRunClient",
    "ImmutableEnvironmentVersion",
    "VariantExecution",
    "WorkflowRunClient",
    "WorkflowRunSnapshot",
    "launch_curriculum",
    "load_manifest",
    "load_or_create_manifest",
    "plan_curriculum",
    "wait_for_workflow",
    "write_manifest",
]
