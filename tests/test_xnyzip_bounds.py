"""Regressions for native XnYZip rounding and boundary-node failures."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import main as particle_main
from src.compress import CompressionPipeline
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS


class XnYZipBoundRetryTests(unittest.TestCase):
    def _pipeline(self, root):
        pipeline = object.__new__(CompressionPipeline)
        pipeline.preprocessed_dir = root
        pipeline.raw_paths = {"positions_xnyzip": "unused.raw"}
        archive = root / "positions.xnyzip"
        archive.write_bytes(b"native stream")
        pipeline.artifacts = {"positions": str(archive)}
        pipeline.manifest = {
            "error_bounds": {"positions_xnyzip_abs": 1e-5},
            "field_error_bounds": {"positions_xnyzip": {
                "abs": 1.1e-5, "compressor_abs": 1e-5,
                "preprocess_l2_max_abs": 1e-6,
            }},
            "fields": {key: {"dtype": "float32"} for key in POSITION_FIELDS},
        }
        pipeline.compressed_fields = {}
        pipeline.count = 3
        pipeline.tools = None
        pipeline.settings = SimpleNamespace(force=False)
        pipeline.structured_layout = None
        return pipeline

    def test_retries_preserve_requested_budget_and_use_final_order_and_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self._pipeline(Path(temp))
            orders = [np.array(v, dtype=np.uint64) for v in
                      ([0, 1, 2], [1, 0, 2], [2, 1, 0])]
            with patch("src.compress.compress_xnyzip_triplet", side_effect=orders) as encode:
                with patch.object(pipeline, "_measure_xnyzip_position_error",
                                  side_effect=[5000., 1.001e-5, .997e-5]) as measure:
                    canonical = pipeline._compress_canonical_xnyzip_positions()
            self.assertEqual(encode.call_count, 3)
            self.assertFalse(encode.call_args_list[0].args[6])
            self.assertTrue(encode.call_args.args[6])
            self.assertEqual(encode.call_args.kwargs["quantizer"], "cube")
            actual_bound = encode.call_args.args[4]
            self.assertAlmostEqual(actual_bound, .99e-5, places=15)
            self.assertEqual(measure.call_args.args[1], actual_bound)
            np.testing.assert_array_equal(canonical.values, orders[-1])
            metadata = pipeline.compressed_fields["positions"]
            self.assertEqual(metadata["l2_error_bound"], actual_bound)
            self.assertEqual(metadata["validation_l2_bound"], 1e-5)
            # The verified error may exceed the native scale, but never the
            # original budget. Rechecking against the tightened scale is wrong.
            self.assertGreater(metadata["validated_max_l2_error"], actual_bound)
            self.assertEqual(metadata["compression_attempts"], 3)
            self.assertEqual(pipeline.manifest["error_bounds"]["positions_xnyzip_abs"], actual_bound)
            bounds = pipeline.manifest["field_error_bounds"]["positions_xnyzip"]
            self.assertEqual(bounds["abs"], 1.1e-5)
            self.assertEqual(bounds["preprocess_l2_max_abs"], 1e-6)
            self.assertEqual(bounds["compressor_abs"], actual_bound)

    def test_unrecoverable_error_never_publishes_position_metadata(self):
        for maximum in (float("nan"), float("inf"), 1., 1.00001e-5):
            with self.subTest(maximum=maximum), tempfile.TemporaryDirectory() as temp:
                pipeline = self._pipeline(Path(temp))
                with patch("src.compress.compress_xnyzip_triplet", return_value=np.arange(3)) as encode:
                    with patch.object(pipeline, "_measure_xnyzip_position_error", return_value=maximum):
                        with self.assertRaisesRegex(RuntimeError, "failed L2 validation"):
                            pipeline._compress_canonical_xnyzip_positions()
                self.assertLessEqual(encode.call_count, 6)
                self.assertNotIn("positions", pipeline.compressed_fields)
                self.assertEqual(pipeline.manifest["error_bounds"]["positions_xnyzip_abs"], 1e-5)


class XnYZipSmallBoundRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(__file__).resolve().parents[1] / "tools/XnYZip/build/XnYZip"
        if not cls.binary.is_file():
            raise unittest.SkipTest("Build XnYZip to run native bound regressions")
        try:
            from src.runtime import load_pcodec, load_pyszo
            load_pcodec()
            load_pyszo()
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(str(exc))

    def _run(self, argv):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = particle_main.main(argv)
        self.assertEqual(result, 0, errors.getvalue() + output.getvalue())

    def test_snapshot_boundary_and_rounding_fixture_with_saved_package_decode(self):
        # Reduced from dat_7.1: TO emits a negative z node; cube overshoots
        # 8.568587749e-6 by 6.71e-9 on the second particle. No source data needed.
        bound = 8.568587749101733e-6
        offset = np.array([4.6566128730773926e-9, 4.190951585769653e-9,
                           7.078051567077637e-8])
        step = 2 * bound / np.sqrt(5)
        points = np.array([
            offset,
            [.4234301745891571, .054487086832523346, .37138697504997253],
            offset + [step, step, 0],
            [.49620944261550903, .49620935320854187, .4962094724178314],
        ], dtype=np.float32)
        for structured in (False, True):
            for requested in (bound, 1e-6):
                with self.subTest(structured=structured, bound=requested), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    source, work = root / "input.h5", root / "work"
                    with h5py.File(source, "w") as h5:
                        h5.attrs["nsidemesh"] = 2
                        h5["id"] = np.arange(4, dtype=np.uint64)
                        for axis, key in enumerate(POSITION_FIELDS):
                            h5[key] = points[:, axis]
                        for axis, key in enumerate(VELOCITY_FIELDS):
                            h5[key] = np.arange(4, dtype=np.float32) * (axis + 1)
                    argv = ["roundtrip", str(source), "--work-dir", str(work),
                            "--pos-compressor", "xnyzip", "--vel-compressor", "szo",
                            "--xnyzip", str(self.binary),
                            "--pos-abs-eb", str(requested), "--vel-abs-eb", "1e-6",
                            "--metrics", "--clean-raw"]
                    if structured:
                        argv.append("--xnyzip-structure-aware")
                    self._run(argv)
                    manifest = json.loads((work / "manifest.json").read_text())
                    metadata = manifest["compressed_fields"]["positions"]
                    self.assertEqual(bool(manifest.get("structured_layout", {}).get("enabled")), structured)
                    if requested == bound:
                        self.assertEqual(metadata["quantizer"], "cube")
                        self.assertLess(metadata["l2_error_bound"], requested)
                        self.assertGreaterEqual(metadata["compression_attempts"], 3)
                    # Strict independent check, without the metrics tolerance.
                    with h5py.File(work / "reconstructed.h5", "r") as h5:
                        ids = h5["id"][:]
                        decoded = np.column_stack([h5[key][:] for key in POSITION_FIELDS])
                        errors = np.linalg.norm(decoded.astype(np.float64) - points[ids].astype(np.float64), axis=1)
                        self.assertLessEqual(errors.max(), requested)
                        expected = {key: h5[key][:] for key in h5}
                        for axis, key in enumerate(VELOCITY_FIELDS):
                            np.testing.assert_allclose(h5[key][:], ids * (axis + 1), rtol=0, atol=1e-6)
                    source.unlink()
                    self.assertFalse((work / "preprocessed").exists())
                    self._run(["decompress", "--work-dir", str(work), "--force"])
                    with h5py.File(work / "reconstructed.h5", "r") as h5:
                        for key, values in expected.items():
                            np.testing.assert_array_equal(h5[key][:], values)
