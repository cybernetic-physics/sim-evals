from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import run_curriculum
from sim_evals.curriculum import (
    MAX_TOTAL_VARIANTS,
    CyberneticsWorkflowRunClient,
    CurriculumError,
    CurriculumPlanConfig,
    CurriculumWorkflowError,
    ImmutableEnvironmentVersion,
    VariantExecution,
    WorkflowRunSnapshot,
    launch_curriculum,
    load_manifest,
    plan_curriculum,
    write_manifest,
)


BASE_URI = "cybernetics://envs/env_droid/versions/ver_immutable"


class FakeWorkflowClient:
    def __init__(self, *, terminal_status: str = "completed") -> None:
        self.terminal_status = terminal_status
        self.events: list[tuple[str, str]] = []
        self.create_inputs: list[dict[str, Any]] = []
        self._runs: dict[str, WorkflowRunSnapshot] = {}

    def create_simulation_from_prompt(
        self,
        *,
        prompt: str,
        environment: ImmutableEnvironmentVersion,
        budget_turns: int,
        budget_seconds: float,
        workspace_id: str | None,
    ) -> WorkflowRunSnapshot:
        ordinal = len(self.create_inputs)
        run_id = f"wfr_{ordinal}"
        input_payload = {
            "prompt": prompt,
            "sessionMode": "environment",
            "environmentId": environment.environment_id,
            "environmentVersionId": environment.version_id,
        }
        self.create_inputs.append(
            {
                "input": input_payload,
                "budgetTurns": budget_turns,
                "budgetSeconds": budget_seconds,
                "workspaceId": workspace_id,
            }
        )
        self.events.append(("create", run_id))
        snapshot = WorkflowRunSnapshot(
            run_id=run_id,
            status="queued",
            input=input_payload,
        )
        self._runs[run_id] = snapshot
        return snapshot

    def get_workflow_run(self, run_id: str) -> WorkflowRunSnapshot:
        self.events.append(("get", run_id))
        created = self._runs.get(run_id)
        if created is None:
            created = WorkflowRunSnapshot(
                run_id=run_id,
                status="running",
                input={
                    "sessionMode": "environment",
                    "environmentId": "env_droid",
                    "environmentVersionId": "ver_immutable",
                },
            )
        if self.terminal_status == "completed":
            ordinal = run_id.removeprefix("wfr_")
            return replace(
                created,
                status="completed",
                result={
                    "environmentId": f"env_variant_{ordinal}",
                    "environmentVersionId": f"ver_output_{ordinal}",
                    "environmentVersionStatus": "ready",
                },
            )
        return replace(
            created,
            status="failed",
            error_message="builder could not validate the scene",
        )


class CurriculumPlanTests(unittest.TestCase):
    def test_requires_an_exact_immutable_base_version(self) -> None:
        for invalid in (
            "cybernetics://envs/env_droid",
            "cybernetics://envs/env_droid/versions/",
            "env_droid",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(CurriculumError):
                CurriculumPlanConfig(base_environment_uri=invalid)

        parsed = ImmutableEnvironmentVersion.parse(BASE_URI)
        self.assertEqual(parsed.environment_id, "env_droid")
        self.assertEqual(parsed.version_id, "ver_immutable")
        self.assertEqual(parsed.uri, BASE_URI)

    def test_plan_is_deterministic_bounded_and_split(self) -> None:
        config = CurriculumPlanConfig(
            base_environment_uri=BASE_URI,
            root_seed=17,
            train_variants=3,
            validation_variants=2,
            held_out_variants=1,
        )
        first = plan_curriculum(config)
        second = plan_curriculum(config)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            first.split_counts, {"train": 3, "validation": 2, "held_out": 1}
        )
        self.assertEqual(
            [variant.split for variant in first.variants],
            ["train", "train", "train", "validation", "validation", "held_out"],
        )
        self.assertEqual(len({variant.seed for variant in first.variants}), 6)
        self.assertTrue(
            all(0 <= variant.seed < (1 << 31) for variant in first.variants)
        )
        self.assertNotEqual(
            first.plan_sha256,
            plan_curriculum(replace(config, root_seed=18)).plan_sha256,
        )

    def test_prompt_pins_source_and_preserves_dynamics_contract(self) -> None:
        manifest = plan_curriculum(
            CurriculumPlanConfig(
                base_environment_uri=BASE_URI,
                train_variants=1,
                validation_variants=1,
                held_out_variants=1,
            )
        )
        for variant in manifest.variants:
            with self.subTest(variant=variant.variant_id):
                self.assertIn(BASE_URI, variant.prompt)
                self.assertIn("exact source version", variant.prompt)
                self.assertIn("mass, density, friction", variant.prompt)
                self.assertIn("robot start pose", variant.prompt)
                self.assertIn("publish a new ready environment version", variant.prompt)
                self.assertIn(f"Dataset split: {variant.split}", variant.prompt)

    def test_counts_are_bounded_and_all_splits_are_required(self) -> None:
        with self.assertRaises(CurriculumError):
            CurriculumPlanConfig(
                base_environment_uri=BASE_URI,
                train_variants=MAX_TOTAL_VARIANTS,
                validation_variants=1,
                held_out_variants=1,
            )
        with self.assertRaises(CurriculumError):
            CurriculumPlanConfig(
                base_environment_uri=BASE_URI,
                train_variants=1,
                validation_variants=0,
                held_out_variants=1,
            )

    def test_manifest_round_trip_rejects_plan_mutation(self) -> None:
        manifest = plan_curriculum(
            CurriculumPlanConfig(
                base_environment_uri=BASE_URI,
                train_variants=1,
                validation_variants=1,
                held_out_variants=1,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_manifest(manifest, path)
            self.assertEqual(load_manifest(path), manifest)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["variants"][0]["changes"]["cube"]["xOffsetMeters"] = 99
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CurriculumError, "planSha256"):
                load_manifest(path)


class CurriculumLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = plan_curriculum(
            CurriculumPlanConfig(
                base_environment_uri=BASE_URI,
                root_seed=7,
                train_variants=2,
                validation_variants=1,
                held_out_variants=1,
            )
        )

    def test_launches_sequentially_and_records_exact_output_uris(self) -> None:
        client = FakeWorkflowClient()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            result = launch_curriculum(
                self.manifest,
                client,
                manifest_path=path,
                max_launches=3,
                poll_interval_seconds=0,
            )
            persisted = load_manifest(path)

        self.assertEqual(
            client.events,
            [
                ("create", "wfr_0"),
                ("get", "wfr_0"),
                ("create", "wfr_1"),
                ("get", "wfr_1"),
                ("create", "wfr_2"),
                ("get", "wfr_2"),
            ],
        )
        self.assertEqual(
            result.output_environment_uris,
            (
                "cybernetics://envs/env_variant_0/versions/ver_output_0",
                "cybernetics://envs/env_variant_1/versions/ver_output_1",
                "cybernetics://envs/env_variant_2/versions/ver_output_2",
            ),
        )
        self.assertEqual(persisted, result)
        self.assertEqual(result.variants[3].execution.status, "planned")
        for request in client.create_inputs:
            self.assertEqual(request["input"]["environmentId"], "env_droid")
            self.assertEqual(request["input"]["environmentVersionId"], "ver_immutable")

    def test_live_workflow_snapshot_requires_a_canonical_run_id(self) -> None:
        with self.assertRaisesRegex(CurriculumWorkflowError, "start with 'wfr_'"):
            WorkflowRunSnapshot.from_dict(
                {
                    "runId": "not-a-workflow-id",
                    "status": "queued",
                    "input": {},
                }
            )

    def test_rest_adapter_reports_only_bounded_public_error_fields(self) -> None:
        class _Response:
            status_code = 404

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "code": "NOT_FOUND",
                    "message": "Environment\nnot found",
                    "details": {"internal": "must not be rendered"},
                }

        class _HttpClient:
            @staticmethod
            def request(*_args: object, **_kwargs: object) -> _Response:
                return _Response()

        client = object.__new__(CyberneticsWorkflowRunClient)
        client._client = _HttpClient()
        client.api_key = "not-a-real-key"

        with self.assertRaisesRegex(
            CurriculumWorkflowError,
            r"HTTP 404 \(NOT_FOUND: Environment not found\)$",
        ) as raised:
            client._request("POST", "/v1/workflows/runs", json_body={})

        self.assertNotIn("internal", str(raised.exception))

    def test_resumes_recorded_run_before_creating_next(self) -> None:
        running = self.manifest.with_execution(
            0,
            VariantExecution(status="running", workflow_run_id="wfr_existing"),
        )
        client = FakeWorkflowClient()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_manifest(running, path)
            result = launch_curriculum(
                running,
                client,
                manifest_path=path,
                max_launches=1,
                poll_interval_seconds=0,
            )

        self.assertEqual(client.events[0], ("get", "wfr_existing"))
        self.assertEqual(client.events.count(("create", "wfr_0")), 1)
        self.assertEqual(result.variants[0].execution.workflow_run_id, "wfr_existing")
        self.assertEqual(result.variants[1].execution.workflow_run_id, "wfr_0")

    def test_resume_rejects_a_run_bound_to_a_different_source_version(self) -> None:
        running = self.manifest.with_execution(
            0,
            VariantExecution(status="running", workflow_run_id="wfr_existing"),
        )
        client = FakeWorkflowClient()
        client._runs["wfr_existing"] = WorkflowRunSnapshot(
            run_id="wfr_existing",
            status="running",
            input={
                "sessionMode": "environment",
                "environmentId": "env_droid",
                "environmentVersionId": "ver_different",
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_manifest(running, path)
            with self.assertRaisesRegex(CurriculumWorkflowError, "exact source"):
                launch_curriculum(
                    running,
                    client,
                    manifest_path=path,
                    max_launches=1,
                    poll_interval_seconds=0,
                )
            persisted = load_manifest(path)

        self.assertEqual(persisted.variants[0].execution.status, "failed")
        self.assertEqual(client.events, [("get", "wfr_existing")])

    def test_failed_workflow_stops_before_next_variant(self) -> None:
        client = FakeWorkflowClient(terminal_status="failed")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            with self.assertRaises(CurriculumWorkflowError):
                launch_curriculum(
                    self.manifest,
                    client,
                    manifest_path=path,
                    max_launches=4,
                    poll_interval_seconds=0,
                )
            persisted = load_manifest(path)

        self.assertEqual(client.events, [("create", "wfr_0"), ("get", "wfr_0")])
        self.assertEqual(persisted.variants[0].execution.status, "failed")
        self.assertEqual(persisted.variants[1].execution.status, "planned")


class CurriculumCliTests(unittest.TestCase):
    def test_cli_is_dry_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "curriculum.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = run_curriculum.main(
                    [
                        "--base-environment-uri",
                        BASE_URI,
                        "--train-variants",
                        "1",
                        "--validation-variants",
                        "1",
                        "--held-out-variants",
                        "1",
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            summary = json.loads(output.getvalue())

            self.assertEqual(result, 0)
            self.assertEqual(summary["mode"], "dry-run")
            self.assertEqual(summary["completed"], 0)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(
                all(
                    variant.execution.status == "planned"
                    for variant in load_manifest(manifest_path).variants
                )
            )


if __name__ == "__main__":
    unittest.main()
