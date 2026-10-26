from __future__ import annotations

import unittest
from pathlib import Path


class BootstrapSpecTests(unittest.TestCase):
    def test_dev_spec_exists_and_names_first_loop(self) -> None:
        root = Path(__file__).resolve().parents[2]
        spec = (root / "DEV_SPEC.md").read_text(encoding="utf-8")
        self.assertIn("Minimum RAG Loop", spec)
        self.assertIn("data/index/chunks.jsonl", spec)
        self.assertIn("unittest", spec)


if __name__ == "__main__":
    unittest.main()