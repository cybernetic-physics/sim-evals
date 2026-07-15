"""Prove hosted Isaac hard-body contact telemetry through the public SDK."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ROOT_PATH = "/World/ContactIntegritySmoke"
_PAIR_LABEL = "smoke-pair"
_MAXIMUM_PENETRATION_M = 0.001
_MAXIMUM_NORMAL_IMPULSE_NS = 0.5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-uri",
        default=os.environ.get("CYBERNETICS_DROID_ENV_URI"),
        help="environment URI required when --session-id is omitted",
    )
    parser.add_argument(
        "--session-id",
        help="attach to a caller-owned session instead of launching one",
    )
    parser.add_argument(
        "--runtime-provider",
        choices=("warm_pool", "vast"),
        default="warm_pool",
    )
    parser.add_argument("--launch-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--mcp-ttl-seconds", type=int, default=3600)
    parser.add_argument("--results-dir", type=Path)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument(
        "--keep-session",
        dest="keep_session",
        action="store_true",
        help="retain a session launched by this command",
    )
    lifecycle.add_argument(
        "--stop-session",
        dest="keep_session",
        action="store_false",
        help="stop a session launched by this command after the smoke test (default)",
    )
    parser.set_defaults(keep_session=False)
    return parser


def _timestamped_results_dir(now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return Path("runs") / "contact-integrity" / timestamp.strftime("%Y%m%dT%H%M%S.%fZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tool_data(result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise RuntimeError("MCP tool returned a non-object payload")
    payload = dict(result)
    if payload.get("ok") is False:
        raise RuntimeError(f"MCP tool failed: {payload.get('error')}")
    data = payload.get("data")
    if isinstance(data, Mapping):
        payload = dict(data)
        if payload.get("ok") is False:
            raise RuntimeError(f"MCP tool failed: {payload.get('error')}")
    status = str(payload.get("status", "")).lower()
    if payload.get("success") is False or status in {
        "error",
        "failed",
        "failure",
    }:
        raise RuntimeError(
            "MCP tool failed: "
            f"{payload.get('message') or payload.get('error') or status}"
        )
    return payload


def _contact_trace(step_result: Any) -> dict[str, Any]:
    payload = _tool_data(step_result)
    trace = payload.get("contact_integrity")
    if not isinstance(trace, Mapping):
        raise RuntimeError("step result omitted contact_integrity")
    return dict(trace)


def _violates(trace: Mapping[str, Any], metric: str) -> bool:
    violations = trace.get("violations")
    return isinstance(violations, list) and any(
        isinstance(item, Mapping) and item.get("metric") == metric
        for item in violations
    )


def _summary_number(trace: Mapping[str, Any], name: str) -> float:
    summary = trace.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("contact trace omitted summary")
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"contact trace summary omitted numeric {name}")
    return float(value)


def _evaluate_clean(trace: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "complete": trace.get("complete") is True,
        "all_updates_captured": trace.get("captured_updates")
        == trace.get("requested_updates"),
        "no_contact": _summary_number(trace, "updates_with_contact") == 0,
        "penetration_within_limit": _summary_number(trace, "maximum_penetration_m")
        <= _MAXIMUM_PENETRATION_M,
        "impulse_within_limit": _summary_number(trace, "maximum_normal_impulse_ns")
        <= _MAXIMUM_NORMAL_IMPULSE_NS,
        "within_configured_limits": trace.get("within_configured_limits") is True,
        "no_violations": trace.get("violations") == [],
    }


def _evaluate_penetration_fault(trace: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "complete": trace.get("complete") is True,
        "penetration_exceeds_limit": _summary_number(trace, "maximum_penetration_m")
        > _MAXIMUM_PENETRATION_M,
        "limit_verdict_failed": trace.get("within_configured_limits") is False,
        "penetration_violation_present": _violates(trace, "maximum_penetration_m"),
    }


def _evaluate_impulse_fault(trace: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "complete": trace.get("complete") is True,
        "impulse_exceeds_limit": _summary_number(trace, "maximum_normal_impulse_ns")
        > _MAXIMUM_NORMAL_IMPULSE_NS,
        "limit_verdict_failed": trace.get("within_configured_limits") is False,
        "impulse_violation_present": _violates(trace, "maximum_normal_impulse_ns"),
    }


def _evaluate_saturation_fault(trace: Mapping[str, Any]) -> dict[str, bool]:
    saturated = trace.get("saturated_pairs")
    return {
        "trace_failed_closed": trace.get("complete") is False,
        "pair_reported_saturated": isinstance(saturated, list)
        and _PAIR_LABEL in saturated,
        "limit_verdict_failed": trace.get("within_configured_limits") is False,
    }


def _state_number(state: Mapping[str, Any], name: str) -> float:
    value = state.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"simulation state omitted finite numeric {name}")
    return float(value)


def _simulation_state(mcp: Any) -> dict[str, Any]:
    state = _tool_data(mcp.call_tool("isaac.get_simulation_state", {}))
    physics_dt = _state_number(state, "physics_dt")
    current_time = _state_number(state, "current_time")
    timeline_state = state.get("timeline_state")
    if physics_dt <= 0:
        raise RuntimeError("simulation state physics_dt must be positive")
    if timeline_state not in {"playing", "paused", "stopped"}:
        raise RuntimeError("simulation state returned an invalid timeline_state")
    return {
        "physics_dt": physics_dt,
        "current_time": current_time,
        "timeline_state": timeline_state,
    }


def _evaluate_atomic_step(
    step: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    num_steps: int,
) -> dict[str, bool]:
    physics_dt = _state_number(before, "physics_dt")
    observed_seconds = _state_number(after, "current_time") - _state_number(
        before, "current_time"
    )
    expected_seconds = num_steps * physics_dt
    tolerance_seconds = max(1e-6, physics_dt * 0.25)
    return {
        "stepped_exactly": step.get("stepped") == num_steps,
        "requested_exactly": step.get("requested_steps") == num_steps,
        "exact_step_completed": step.get("exact_step_completed") is True,
        "pause_after_confirmed": step.get("pause_after") is True,
        "step_timeline_paused": step.get("timeline_state") == "paused",
        "state_timeline_paused": after.get("timeline_state") == "paused",
        "no_timeout": step.get("timed_out") is not True,
        "time_advanced_exactly": observed_seconds >= 0
        and abs(observed_seconds - expected_seconds) <= tolerance_seconds,
    }


def _fixture_script(*, scenario: str) -> str:
    scenario_root, sensor_path, filter_path = _scenario_paths(scenario)
    if scenario == "clean":
        sensor_position = (8.0, 0.0, 1.0)
        filter_position = (8.5, 0.0, 1.0)
        sensor_velocity = (0.0, 0.0, 0.0)
        max_depenetration_velocity = 3.0
    elif scenario in {"penetration", "saturation"}:
        base_x = 10.0 if scenario == "penetration" else 14.0
        sensor_position = (base_x, 0.0, 1.0)
        filter_position = (base_x + 0.08, 0.0, 1.0)
        sensor_velocity = (0.0, 0.0, 0.0)
        max_depenetration_velocity = 0.01
    elif scenario == "impulse":
        sensor_position = (12.0, 0.0, 1.0)
        filter_position = (12.099, 0.0, 1.0)
        sensor_velocity = (5.0, 0.0, 0.0)
        max_depenetration_velocity = 0.01
    else:
        raise ValueError(f"unsupported fixture scenario: {scenario}")

    return f"""
import json
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

timeline = omni.timeline.get_timeline_interface()
app = omni.kit.app.get_app()
old_physics_view = SimulationManager.get_physics_sim_view()
if timeline.is_stopped():
    timeline.play()
    timeline.commit()
timeline.stop()
timeline.commit()
app.update()
if SimulationManager.get_physics_sim_view() is not None:
    raise RuntimeError("contact smoke hard stop did not invalidate physics view")
stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, {_ROOT_PATH!r})
scenario_root = {scenario_root!r}
if stage.GetPrimAtPath(scenario_root).IsValid():
    stage.RemovePrim(scenario_root)
UsdGeom.Xform.Define(stage, scenario_root)

def define_cube(path, position, velocity, kinematic, max_depenetration_velocity):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(0.1)
    prim = cube.GetPrim()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.CollisionAPI.Apply(prim)
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateKinematicEnabledAttr(kinematic)
    rigid.CreateVelocityAttr(Gf.Vec3f(*velocity))
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(0.002)
    collision.CreateRestOffsetAttr(0.0)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rigid.CreateDisableGravityAttr(True)
    physx_rigid.CreateEnableCCDAttr(True)
    physx_rigid.CreateMaxDepenetrationVelocityAttr(max_depenetration_velocity)

define_cube(
    {sensor_path!r},
    {sensor_position!r},
    (0.0, 0.0, 0.0),
    False,
    {max_depenetration_velocity!r},
)
define_cube(
    {filter_path!r},
    {filter_position!r},
    (0.0, 0.0, 0.0),
    True,
    3.0,
)
app.update()
timeline.play()
timeline.commit()
app.update()
new_physics_view = SimulationManager.get_physics_sim_view()
if new_physics_view is None or new_physics_view is old_physics_view:
    raise RuntimeError("contact smoke physics view was not rebuilt from final USD")
timeline.pause()
timeline.commit()
if timeline.is_playing() or timeline.is_stopped():
    raise RuntimeError("contact smoke physics rebuild did not finish paused")

runtime_velocity = {sensor_velocity!r}
if runtime_velocity != (0.0, 0.0, 0.0):
    RigidPrim({sensor_path!r}).set_velocities(
        linear_velocities=[list(runtime_velocity)]
    )

print("CONTACT_INTEGRITY_FIXTURE=" + json.dumps({{
    "scenario": {scenario!r},
    "physics_view_rebuilt": True,
    "timeline_state": "paused",
    "runtime_sensor_velocity": list(runtime_velocity),
}}))
"""


def _cleanup_script() -> str:
    return f"""
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager

timeline = omni.timeline.get_timeline_interface()
timeline.stop()
timeline.commit()
omni.kit.app.get_app().update()
if SimulationManager.get_physics_sim_view() is not None:
    raise RuntimeError("contact smoke cleanup did not invalidate physics view")
stage = omni.usd.get_context().get_stage()
if stage.GetPrimAtPath({_ROOT_PATH!r}).IsValid():
    stage.RemovePrim({_ROOT_PATH!r})
omni.kit.app.get_app().update()
print("CONTACT_INTEGRITY_FIXTURE_REMOVED")
"""


def _scenario_paths(scenario: str) -> tuple[str, str, str]:
    if scenario not in {"clean", "penetration", "impulse", "saturation"}:
        raise ValueError(f"unsupported fixture scenario: {scenario}")
    root = f"{_ROOT_PATH}/{scenario.capitalize()}"
    return root, f"{root}/Sensor", f"{root}/Filter"


def _trace_config(max_contacts_per_pair: int, *, scenario: str) -> dict[str, Any]:
    _, sensor_path, filter_path = _scenario_paths(scenario)
    return {
        "pairs": [
            {
                "label": _PAIR_LABEL,
                "sensor_path": sensor_path,
                "filter_path": filter_path,
            }
        ],
        "max_contacts_per_pair": max_contacts_per_pair,
        "limits": {
            "maximum_penetration_m": _MAXIMUM_PENETRATION_M,
            "maximum_normal_impulse_ns": _MAXIMUM_NORMAL_IMPULSE_NS,
        },
    }


def _run_scenario(
    mcp: Any,
    *,
    scenario: str,
    num_steps: int,
    max_contacts_per_pair: int,
) -> dict[str, Any]:
    _, sensor_path, filter_path = _scenario_paths(scenario)
    setup = _tool_data(
        mcp.call_tool(
            "isaac.execute_script",
            {"code": _fixture_script(scenario=scenario)},
        )
    )
    before = _simulation_state(mcp)
    raw_step = mcp.call_tool(
        "isaac.step_simulation",
        {
            "num_steps": num_steps,
            "pause_after": True,
            "observe_prims": [sensor_path, filter_path],
            "contact_integrity": _trace_config(
                max_contacts_per_pair,
                scenario=scenario,
            ),
        },
    )
    step = _tool_data(raw_step)
    after = _simulation_state(mcp)
    return {
        "setup": setup,
        "before": before,
        "step": step,
        "after": after,
        "atomic_step_checks": _evaluate_atomic_step(
            step,
            before,
            after,
            num_steps=num_steps,
        ),
        "trace": _contact_trace(raw_step),
    }


def _run_matrix(mcp: Any) -> dict[str, Any]:
    scenarios = {
        "clean": _run_scenario(
            mcp,
            scenario="clean",
            num_steps=2,
            max_contacts_per_pair=64,
        ),
        "penetration": _run_scenario(
            mcp,
            scenario="penetration",
            num_steps=2,
            max_contacts_per_pair=64,
        ),
        "impulse": _run_scenario(
            mcp,
            scenario="impulse",
            num_steps=4,
            max_contacts_per_pair=64,
        ),
        "saturation": _run_scenario(
            mcp,
            scenario="saturation",
            num_steps=2,
            max_contacts_per_pair=1,
        ),
    }
    checks = {
        "clean": _evaluate_clean(scenarios["clean"]["trace"]),
        "penetration": _evaluate_penetration_fault(scenarios["penetration"]["trace"]),
        "impulse": _evaluate_impulse_fault(scenarios["impulse"]["trace"]),
        "saturation": _evaluate_saturation_fault(scenarios["saturation"]["trace"]),
    }
    for name, scenario in scenarios.items():
        checks[name].update(
            {
                f"atomic_{key}": value
                for key, value in scenario["atomic_step_checks"].items()
            }
        )
    passed = all(all(values.values()) for values in checks.values())
    return {"passed": passed, "checks": checks, "scenarios": scenarios}


def main() -> None:
    args = _parser().parse_args()
    if not args.session_id and not args.environment_uri:
        raise SystemExit("--environment-uri is required when --session-id is omitted")
    results_dir = (args.results_dir or _timestamped_results_dir()).resolve()
    results_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        results_dir / "config.json",
        {
            "schema_version": 1,
            "started_at": started_at,
            "environment_uri": args.environment_uri,
            "requested_session_id": args.session_id,
            "runtime_provider": args.runtime_provider,
            "keep_session": args.keep_session,
            "limits": _trace_config(64, scenario="clean")["limits"],
        },
    )

    try:
        from cybernetics.sim import (  # pyright: ignore[reportMissingImports]
            SimulationClient,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install a Cybernetics SDK release that provides "
            "cybernetics.sim.SimulationClient and mcp_session"
        ) from exc

    launched = False
    session_id = args.session_id
    client = SimulationClient()
    try:
        if session_id is None:
            launch = client.launch(
                args.environment_uri,
                runtime_provider=args.runtime_provider,
                wait=True,
                timeout_seconds=args.launch_timeout_seconds,
            )
            session_id = launch.session_id
            launched = True
        assert session_id is not None
        with client.mcp_session(
            session_id,
            ttl_seconds=args.mcp_ttl_seconds,
            name="contact-integrity-live-smoke",
        ) as mcp:
            try:
                matrix = _run_matrix(mcp)
            finally:
                cleanup = _tool_data(
                    mcp.call_tool(
                        "isaac.execute_script",
                        {"code": _cleanup_script()},
                    )
                )
        result = {
            "schema_version": 1,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "launched_session": launched,
            "cleanup": cleanup,
            **matrix,
        }
        _write_json(results_dir / "result.json", result)
        print(json.dumps({"results_dir": str(results_dir), **result["checks"]}))
        if not result["passed"]:
            raise SystemExit(1)
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            _write_json(
                results_dir / "error.json",
                {
                    "schema_version": 1,
                    "session_id": session_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        if launched and session_id and not args.keep_session:
            try:
                client.stop_session(session_id)
            finally:
                client.close()
        else:
            client.close()


if __name__ == "__main__":
    main()
