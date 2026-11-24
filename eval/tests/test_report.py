"""Tests for the run summary writer and its stable JSON and Markdown output."""

import json
import tempfile
import unittest
from pathlib import Path

from ninfer_eval.result import DatasetResult, ResultCounts, write_summary


class ReportTest(unittest.TestCase):
    """Coverage for write_summary's stable JSON and Markdown outputs."""
    def test_writes_stable_json_and_markdown(self):
        """Write a completed dataset result and assert the summary JSON fields and Markdown table row."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = DatasetResult(
                job_id="aime25",
                backend="mock",
                dataset="aime25",
                status="completed",
                primary_metric="accuracy",
                metrics={"accuracy": 0.5},
                counts=ResultCounts(planned=30, completed=30, scored=30),
                duration_seconds=12.5,
            )
            write_summary(root, "run", "completed", [result])
            payload = json.loads((root / "summary.json").read_text())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["results"][0]["counts"]["planned"], 30)
            self.assertIn(
                "| aime25 | completed | accuracy |", (root / "summary.md").read_text()
            )


if __name__ == "__main__":
    unittest.main()
