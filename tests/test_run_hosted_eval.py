from __future__ import annotations

import unittest

from run_hosted_eval import _parser


class HostedEvalParserTests(unittest.TestCase):
    def test_keeps_launched_session_by_default(self) -> None:
        args = _parser().parse_args([])
        self.assertTrue(args.keep_session)

    def test_stop_session_is_explicit(self) -> None:
        args = _parser().parse_args(["--stop-session"])
        self.assertFalse(args.keep_session)

    def test_keep_session_remains_compatible(self) -> None:
        args = _parser().parse_args(["--keep-session"])
        self.assertTrue(args.keep_session)

    def test_scene1_task_success_predicate_is_explicit(self) -> None:
        default_args = _parser().parse_args([])
        selected_args = _parser().parse_args(
            ["--task-success-predicate", "scene1-cube-in-bowl"]
        )

        self.assertIsNone(default_args.task_success_predicate)
        self.assertEqual(
            selected_args.task_success_predicate,
            "scene1-cube-in-bowl",
        )


if __name__ == "__main__":
    unittest.main()
