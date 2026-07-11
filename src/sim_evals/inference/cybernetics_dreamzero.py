"""Cybernetics DreamZero-DROID inference client."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping, Protocol

import numpy as np

from .abstract_client import InferenceClient
from .droid_observation import DroidObservation, camera_strip


class DroidObservationSamplingAPI(Protocol):
    """Narrow API expected from a typed/raw Cybernetics DROID sampler."""

    def sample_droid(
        self, observation: DroidObservation, *, timeout: float | None = None
    ) -> Any: ...

    def reset_sampling_session(self) -> None: ...

    def close(self) -> None: ...


class CyberneticsSDKDroidSamplingAPI:
    """Lazy bridge to the typed Cybernetics SDK sampling surface."""

    def __init__(
        self,
        *,
        base_model: str = "dreamzero-droid",
        model_path: str | None = None,
        session_timeout: float = 2400.0,
    ) -> None:
        try:
            from cybernetics import DroidObservation as SDKDroidObservation
            from cybernetics import ServiceClient
        except ImportError as exc:
            raise RuntimeError(
                "The Cybernetics backend requires the 'cybernetic-physics' SDK"
            ) from exc

        self._service_client_type = ServiceClient
        self._sdk_observation_type = SDKDroidObservation
        self._service_client: Any | None = None
        self._base_model = base_model
        self._model_path = model_path
        self._session_timeout = session_timeout
        self._sampling_client: Any | None = None

    def reset_sampling_session(self) -> None:
        self._close_active_session()
        self._service_client = self._service_client_type()
        create_kwargs: dict[str, Any] = {"timeout": self._session_timeout}
        if self._model_path is not None:
            create_kwargs["model_path"] = self._model_path
        else:
            create_kwargs["base_model"] = self._base_model
        self._sampling_client = self._service_client.create_sampling_client(
            **create_kwargs
        )

    def sample_droid(
        self, observation: DroidObservation, *, timeout: float | None = None
    ) -> Any:
        if self._sampling_client is None:
            self.reset_sampling_session()
        sample = getattr(self._sampling_client, "sample_droid", None)
        if not callable(sample):
            raise RuntimeError(
                "Cybernetics SamplingClient must expose sample_droid(observation)"
            )
        sdk_observation = self._sdk_observation_type.from_numpy(
            exterior_image_0_left=observation.exterior_image_1_left,
            exterior_image_1_left=observation.exterior_image_2_left,
            wrist_image_left=observation.wrist_image_left,
            joint_position=observation.joint_position,
            gripper_position=observation.gripper_position,
            instruction=observation.instruction,
        )
        result = sample(sdk_observation)
        if hasattr(result, "result"):
            return result.result(timeout=timeout)
        return result

    def _close_active_session(self) -> None:
        service_client = self._service_client
        self._sampling_client = None
        self._service_client = None
        if service_client is None:
            return
        try:
            rest_client = service_client.create_rest_client()
            rest_client.cancel_session(service_client.session_id).result(
                timeout=self._session_timeout
            )
        finally:
            service_client.holder.close()

    def close(self) -> None:
        self._close_active_session()


def _action_chunk(response: Any) -> np.ndarray:
    if isinstance(response, Mapping):
        value = response.get("action_chunk")
    else:
        value = getattr(response, "action_chunk", None)
    if value is None:
        raise ValueError("Cybernetics response did not include action_chunk")
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif isinstance(value, Mapping) and "data" in value:
        shape = value.get("shape")
        value = np.asarray(value["data"])
        if shape is not None:
            value = value.reshape(shape)
    chunk = np.asarray(value, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 8:
        raise ValueError(
            f"DreamZero-DROID action_chunk must have shape [N,8], got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise ValueError("DreamZero-DROID action_chunk must contain only finite values")
    return np.ascontiguousarray(chunk)


class Client(InferenceClient):
    """Consume validated DreamZero-DROID joint-position action chunks."""

    def __init__(
        self,
        *,
        sampling_api: DroidObservationSamplingAPI | None = None,
        base_model: str = "dreamzero-droid",
        model_path: str | None = None,
        request_timeout: float = 2400.0,
        session_timeout: float = 2400.0,
        open_loop_horizon: int = 8,
    ) -> None:
        self.sampling_api = sampling_api or CyberneticsSDKDroidSamplingAPI(
            base_model=base_model,
            model_path=model_path,
            session_timeout=session_timeout,
        )
        self.request_timeout = request_timeout
        if open_loop_horizon < 1:
            raise ValueError("open_loop_horizon must be at least 1")
        self.open_loop_horizon = open_loop_horizon
        self._action_chunk: np.ndarray | None = None
        self._action_index = 0
        self._sample_latencies_ms: list[float] = []
        self._errors: list[dict[str, str]] = []
        self._reset_latency_ms: float | None = None

    def reset(self) -> None:
        self._action_chunk = None
        self._action_index = 0
        self._sample_latencies_ms = []
        self._errors = []
        started = perf_counter()
        try:
            self.sampling_api.reset_sampling_session()
        except Exception as exc:
            self._errors.append(
                {"phase": "reset", "type": type(exc).__name__, "message": str(exc)}
            )
            raise
        finally:
            self._reset_latency_ms = (perf_counter() - started) * 1000

    def infer(self, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        observation = DroidObservation.from_sim_observation(obs, instruction)
        sampled_new_chunk = False
        latency_ms: float | None = None
        if self._action_chunk is None or self._action_index >= len(self._action_chunk):
            sampled_new_chunk = True
            started = perf_counter()
            try:
                response = self.sampling_api.sample_droid(
                    observation, timeout=self.request_timeout
                )
                self._action_chunk = _action_chunk(response)[: self.open_loop_horizon]
                self._action_index = 0
            except Exception as exc:
                self._errors.append(
                    {"phase": "sample", "type": type(exc).__name__, "message": str(exc)}
                )
                raise
            finally:
                latency_ms = (perf_counter() - started) * 1000
                self._sample_latencies_ms.append(latency_ms)

        assert self._action_chunk is not None
        action = self._action_chunk[self._action_index].copy()
        self._action_index += 1
        return {
            "action": action,
            "viz": camera_strip(observation),
            "sampled_new_chunk": sampled_new_chunk,
            "latency_ms": latency_ms,
        }

    def episode_metrics(self) -> dict[str, Any]:
        latency = self._sample_latencies_ms
        return {
            "sampling_requests": len(latency),
            "sampling_errors": list(self._errors),
            "sampling_latency_ms": {
                "count": len(latency),
                "min": min(latency) if latency else None,
                "max": max(latency) if latency else None,
                "mean": sum(latency) / len(latency) if latency else None,
                "total": sum(latency),
            },
            "session_reset_latency_ms": self._reset_latency_ms,
        }

    def close(self) -> None:
        self.sampling_api.close()
