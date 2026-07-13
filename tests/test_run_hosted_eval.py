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


if __name__ == "__main__":
    unittest.main()
