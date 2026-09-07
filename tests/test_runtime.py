"""Tests for shared runtime worker selection."""

import unittest
from unittest.mock import patch

from src.runtime import resolve_field_workers


class FieldWorkerTests(unittest.TestCase):
    def test_auto_workers_share_cpus_across_concurrent_pipelines(self) -> None:
        with patch("src.runtime.os.cpu_count", return_value=16):
            self.assertEqual(
                resolve_field_workers(0, concurrent_pipelines=4),
                4,
            )

    def test_workers_are_bounded_by_independent_field_count(self) -> None:
        self.assertEqual(resolve_field_workers(20), 6)
        self.assertEqual(resolve_field_workers(2), 2)

    def test_negative_worker_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            resolve_field_workers(-1)


if __name__ == "__main__":
    unittest.main()
