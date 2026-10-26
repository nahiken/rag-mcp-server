from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class BootstrapVerificationE2ETests(unittest.TestCase):
    def test_smoke_check_runs_from_project_root(self) -> None:
        root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, "scripts/smoke_check.py"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("smoke_check passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()