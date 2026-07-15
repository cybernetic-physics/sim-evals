from __future__ import annotations

import unittest

from run_contact_integrity_smoke import (
    _evaluate_atomic_step,
    _evaluate_clean,
    _evaluate_impulse_fault,
    _evaluate_penetration_fault,
    _evaluate_saturation_fault,
    _fixture_script,
    _parser,
    _scenario_paths,
    _tool_data,
)


def _trace(
    *,
    complete: bool = True,
    contact_updates: int = 0,
    penetration: float = 0.0,
    impulse: float = 0.0,
    saturated_pairs: list[str] | None = None,
    violations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    configured_violations = violations or []
    return {
        "complete": complete,
        "requested_updates": 2,
        "captured_updates": 2,
        "saturated_pairs": saturated_pairs or [],
        "within_configured_limits": complete and not configured_violations,
        "violations": configured_violations,
        "summary": {
            "updates_with_contact": contact_updates,
            "maximum_penetration_m": penetration,
            "maximum_normal_impulse_ns": impulse,
        },
    }


class ContactIntegritySmokeTests(unittest.TestCase):
    def test_stops_launched_session_by_default(self) -> None:
        self.assertFalse(_parser().parse_args([]).keep_session)

    def test_unwraps_gateway_data(self) -> None:
        self.assertEqual(
            _tool_data({"ok": True, "data": {"stepped": 2}}),
            {"stepped": 2},
        )

    def test_fixture_stops_physics_before_topology_change(self) -> None:
        script = _fixture_script(scenario="penetration")
        self.assertLess(
            script.index("timeline.stop()"), script.index("stage.RemovePrim")
        )

    def test_fixture_rebuilds_physics_after_final_usd(self) -> None:
        script = _fixture_script(scenario="penetration")
        self.assertLess(
            script.index("stage.RemovePrim"), script.index("timeline.play()", 500)
        )
        self.assertIn("new_physics_view is old_physics_view", script)
        self.assertIn("CreateThresholdAttr(0.0)", script)

    def test_impulse_velocity_is_applied_after_physics_rebuild(self) -> None:
        script = _fixture_script(scenario="impulse")
        self.assertLess(
            script.index("new_physics_view ="), script.index("set_velocities")
        )
        self.assertIn("runtime_velocity = (5.0, 0.0, 0.0)", script)
        self.assertIn("linear_velocities=[list(runtime_velocity)]", script)

    def test_scenarios_have_disjoint_rigid_body_paths(self) -> None:
        paths = [
            _scenario_paths(name)
            for name in ("clean", "penetration", "impulse", "saturation")
        ]
        self.assertEqual(len({path for group in paths for path in group}), 12)

    def test_rejects_inner_tool_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MCP tool failed"):
            _tool_data({"status": "error", "message": "runtime unavailable"})

    def test_atomic_step_requires_exact_paused_time_advance(self) -> None:
        checks = _evaluate_atomic_step(
            {
                "stepped": 2,
                "requested_steps": 2,
                "exact_step_completed": True,
                "pause_after": True,
                "timeline_state": "paused",
            },
            {"physics_dt": 1 / 120, "current_time": 1.0},
            {
                "physics_dt": 1 / 120,
                "current_time": 1.0 + 2 / 120,
                "timeline_state": "paused",
            },
            num_steps=2,
        )
        self.assertTrue(all(checks.values()))

    def test_clean_control_requires_complete_bounded_trace(self) -> None:
        checks = _evaluate_clean(_trace())
        self.assertTrue(all(checks.values()))

    def test_penetration_fault_requires_machine_violation(self) -> None:
        checks = _evaluate_penetration_fault(
            _trace(
                contact_updates=2,
                penetration=0.02,
                violations=[{"metric": "maximum_penetration_m"}],
            )
        )
        self.assertTrue(all(checks.values()))

    def test_impulse_fault_requires_machine_violation(self) -> None:
        checks = _evaluate_impulse_fault(
            _trace(
                contact_updates=1,
                impulse=2.0,
                violations=[{"metric": "maximum_normal_impulse_ns"}],
            )
        )
        self.assertTrue(all(checks.values()))

    def test_saturation_must_fail_closed(self) -> None:
        checks = _evaluate_saturation_fault(
            _trace(
                complete=False,
                contact_updates=2,
                saturated_pairs=["smoke-pair"],
            )
        )
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
