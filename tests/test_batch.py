from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import main as particle_main
from src.batch import (
    BatchFileResult,
    build_batch_metrics,
    discover_h5_files,
)
from src.runtime import read_json


class BatchPipelineTests(unittest.TestCase):
    def test_discovery_is_non_recursive_sorted_and_exact_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.h5").touch()
            (root / "a.h5").touch()
            (root / "ignored.H5").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "ignored.h5").touch()

            files = discover_h5_files(root)

            self.assertEqual(
                [path.name for path in files],
                ["a.h5", "b.h5"],
            )

    def test_batch_metrics_use_byte_weighted_cr_and_total_timings(self) -> None:
        results = [
            self._result(
                "a.h5",
                particles=10,
                original_bytes=100,
                compressed_bytes=20,
                wall_seconds=4.0,
                preprocess_seconds=1.0,
            ),
            self._result(
                "b.h5",
                particles=30,
                original_bytes=300,
                compressed_bytes=100,
                wall_seconds=6.0,
                preprocess_seconds=2.0,
            ),
        ]

        metrics = build_batch_metrics(
            Path("/inputs"),
            "roundtrip",
            2,
            results,
            batch_wall_seconds=7.0,
        )

        self.assertEqual(metrics["summary"]["total_particle_count"], 40)
        self.assertEqual(
            metrics["sizes"]["selected_original_payload_bytes_total"],
            400,
        )
        self.assertEqual(metrics["sizes"]["compressed_total_bytes"], 120)
        self.assertTrue(
            math.isclose(
                metrics["sizes"]["payload_compression_ratio"],
                400 / 120,
            )
        )
        self.assertEqual(metrics["timing"]["file_wall_seconds_total"], 10.0)
        self.assertEqual(
            metrics["timing"]["stage_seconds_total"][
                "preprocess_wall_seconds"
            ],
            3.0,
        )
        self.assertEqual(
            metrics["statistics"]["per_file_payload_compression_ratio"][
                "median"
            ],
            4.0,
        )

    def test_single_file_still_uses_the_existing_application(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_h5 = Path(temp_dir) / "input.h5"
            input_h5.touch()
            argv = ["preprocess", str(input_h5)]

            with patch.object(
                particle_main.PipelineApplication,
                "run",
                return_value=0,
            ) as single_run, patch.object(
                particle_main.DirectoryPipelineApplication,
                "run",
                return_value=0,
            ) as directory_run:
                result = particle_main.main(argv)

            self.assertEqual(result, 0)
            single_run.assert_called_once_with()
            directory_run.assert_not_called()

    def test_directory_preprocesses_files_in_isolated_work_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            for name, offset in (("b.h5", 10), ("a.h5", 0)):
                self._write_particle_h5(inputs / name, offset)

            argv = [
                "preprocess",
                str(inputs),
                "--work-dir",
                str(outputs),
                "--file-workers",
                "2",
                "--pos-compressor",
                "sz3",
                "--vel-compressor",
                "sz3",
                "--force",
            ]
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = particle_main.main(argv)

            self.assertEqual(result, 0, stderr.getvalue())
            self.assertLess(
                stdout.getvalue().index(str(inputs / "a.h5")),
                stdout.getvalue().index(str(inputs / "b.h5")),
            )
            for name in ("a.h5", "b.h5"):
                manifest = read_json(outputs / name / "manifest.json")
                self.assertEqual(
                    Path(manifest["input_h5"]),
                    (inputs / name).resolve(),
                )
            batch = read_json(outputs / "batch_metrics.json")
            self.assertEqual(batch["workers"], 2)
            self.assertEqual(batch["summary"]["successful_files"], 2)
            self.assertEqual(batch["summary"]["total_particle_count"], 8)
            self.assertIsNone(
                batch["sizes"]["payload_compression_ratio"]
            )

    @staticmethod
    def _result(
        name: str,
        particles: int,
        original_bytes: int,
        compressed_bytes: int,
        wall_seconds: float,
        preprocess_seconds: float,
    ) -> BatchFileResult:
        return BatchFileResult(
            input_h5=f"/inputs/{name}",
            work_dir=f"/outputs/{name}",
            wall_seconds=wall_seconds,
            report_path=f"/outputs/{name}/metrics.json",
            report={
                "sizes": {
                    "selected_particle_count": particles,
                    "selected_original_payload_bytes": original_bytes,
                    "compressed_total_bytes": compressed_bytes,
                    "payload_compression_ratio": (
                        original_bytes / compressed_bytes
                    ),
                },
                "timing": {
                    "preprocess_wall_seconds": preprocess_seconds,
                },
            },
        )

    @staticmethod
    def _write_particle_h5(path: Path, offset: int) -> None:
        with h5py.File(path, "w") as h5:
            h5.create_dataset(
                "id",
                data=np.arange(offset, offset + 4, dtype=np.uint64),
            )
            for index, logical in enumerate(
                ("x", "y", "z", "vx", "vy", "vz")
            ):
                h5.create_dataset(
                    logical,
                    data=(
                        np.arange(4, dtype=np.float32)
                        + offset
                        + index
                    ),
                )


if __name__ == "__main__":
    unittest.main()
