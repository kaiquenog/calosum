from __future__ import annotations

import unittest

from calosum.harness_checks import run_harness_checks


class HarnessChecksTests(unittest.TestCase):
    def test_harness_checks_pass(self) -> None:
        report = run_harness_checks()
        messages = [f"{issue.code}:{issue.path}:{issue.message}" for issue in report.issues]
        self.assertTrue(report.passed, "\n".join(messages))


if __name__ == "__main__":
    unittest.main()
