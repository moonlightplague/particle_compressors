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
    print_batch_summary,
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

    def test_batch_summary_reports_byte_weighted_field_group_crs(self) -> None:
        first = self._result(
            "a.h5",
            particles=10,
            original_bytes=320,
            compressed_bytes=100,
            wall_seconds=4.0,
            preprocess_seconds=1.0,
        )
        second = self._result(
            "b.h5",
            particles=30,
            original_bytes=960,
            compressed_bytes=300,
            wall_seconds=6.0,
            preprocess_seconds=2.0,
        )
        fields = {
            "id": {"dtype": "uint64"},
            **{
                logical: {"dtype": "float32"}
                for logical in ("x", "y", "z", "vx", "vy", "vz")
            },
        }
        first.report["fields"] = fields
        first.report["sizes"]["compressed_components_bytes"] = {
            "compressed/positions.lcp": 20,
            "compressed/order.pco": 10,
            "compressed/id.pco": 10,
            "compressed/velocities.lcp": 25,
            "compressed/velocity_order.pco": 5,
        }
        second.report["fields"] = fields
        second.report["sizes"]["compressed_components_bytes"] = {
            "compressed/x.psz": 40,
            "compressed/y.psz": 40,
            "compressed/z.psz": 40,
            "compressed/id.pco": 60,
            "compressed/vx.szo": 30,
            "compressed/vy.szo": 30,
            "compressed/vz.szo": 30,
        }

        metrics = build_batch_metrics(
            Path("/inputs"),
            "roundtrip",
            2,
            [first, second],
            batch_wall_seconds=7.0,
        )

        groups = metrics["sizes"]["field_groups"]
        self.assertEqual(groups["positions"]["original_bytes_total"], 480)
        self.assertEqual(groups["positions"]["compressed_bytes_total"], 150)
        self.assertTrue(
            math.isclose(groups["positions"]["compression_ratio"], 3.2)
        )
        self.assertTrue(
            math.isclose(
                groups["id"]["compression_ratio"],
                320 / 70,
            )
        )
        self.assertTrue(
            math.isclose(groups["velocities"]["compression_ratio"], 4.0)
        )

        output = StringIO()
        with redirect_stdout(output):
            print_batch_summary(metrics, Path("/outputs/batch_metrics.json"))

        summary = output.getvalue()
        self.assertIn("positions_CR_total = 3.2", summary)
        self.assertIn("id_CR_total = 4.57143", summary)
        self.assertIn("velocities_CR_total = 4", summary)

    def test_batch_metrics_include_each_files_field_quality_metrics(
        self,
    ) -> None:
        result = self._result(
            "a.h5",
            particles=10,
            original_bytes=100,
            compressed_bytes=20,
            wall_seconds=4.0,
            preprocess_seconds=1.0,
        )
        result.report["fields"] = {
            "x": {
                "max_absolute_error": 0.01,
                "mse": 0.0001,
                "psnr": 60.0,
            },
            "y": {
                "max_absolute_error": 0.02,
                "mse": 0.0002,
                "psnr": 55.0,
                "fixed_point_units": {
                    "max_absolute_error": 2.0,
                    "mse": 1.25,
                    "psnr": 48.0,
                },
            },
            "vx": {
                "max_absolute_error": 0.03,
                "mse": 0.0003,
                "psnr": 50.0,
            },
        }

        metrics = build_batch_metrics(
            Path("/inputs"),
            "roundtrip",
            1,
            [result],
            batch_wall_seconds=4.0,
        )

        quality = metrics["files"][0]["quality_metrics"]
        self.assertEqual(
            quality["x"],
            {
                "max_abs": 0.01,
                "mse": 0.0001,
                "psnr": 60.0,
                "units": "lcp_units",
            },
        )
        self.assertEqual(
            quality["y"],
            {
                "max_abs": 2.0,
                "mse": 1.25,
                "psnr": 48.0,
                "units": "source_units",
            },
        )
        self.assertEqual(
            quality["vx"],
            {
                "max_abs": 0.03,
                "mse": 0.0003,
                "psnr": 50.0,
                "units": "source_units",
            },
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
