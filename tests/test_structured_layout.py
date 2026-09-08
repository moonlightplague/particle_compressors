"""Reversibility, deterministic ordering, and real-codec package coverage."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import permutations
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import main as particle_main
from src.cli import build_parser
from src.compress import CompressionPipeline, CompressionSettings
from src.structured_layout import (
    StructuredParticleLayout, decode_lattice_ids, encode_lattice_ids,
    hilbert_to_morton_3d, hybrid_velocity_order, make_structured_layout,
    morton_decode_3d, morton_encode_3d, morton_to_hilbert_3d,
    validate_structured_package,
)


class StructuredLayoutTests(unittest.TestCase):
    def test_hilbert_inverse_all_supported_bit_widths(self):
        rng = np.random.default_rng(901)
        for bits in range(1, 11):
            with self.subTest(bits=bits):
                size = 1 << (3 * bits)
                codes = (np.arange(size, dtype=np.uint32) if bits <= 4 else
                         rng.integers(0, size, 30_000, dtype=np.uint32))
                np.testing.assert_array_equal(
                    hilbert_to_morton_3d(morton_to_hilbert_3d(codes, bits), bits), codes)
                np.testing.assert_array_equal(
                    morton_to_hilbert_3d(hilbert_to_morton_3d(codes, bits), bits), codes)
                np.testing.assert_array_equal(morton_encode_3d(*morton_decode_3d(codes)), codes)

    def test_ids_sparse_non_power_of_two_bases_axes_and_dtypes(self):
        rng = np.random.default_rng(48)
        for side in (3, 810, 1024):
            for base in (0, 1):
                for axes in permutations(range(3)):
                    layout = make_structured_layout(side, base, axes, 7)
                    self.assertEqual(StructuredParticleLayout.from_metadata(layout.metadata()), layout)
                    ids = np.concatenate(([base, side**3 - 1 + base],
                                          rng.integers(base, side**3 + base, 500)))
                    for dtype in (np.int32, np.uint32, np.int64, np.uint64):
                        values = ids.astype(dtype)
                        np.testing.assert_array_equal(
                            decode_lattice_ids(encode_lattice_ids(values, layout), layout, dtype), values)

    def test_hybrid_order_is_stable_periodic_and_reversible(self):
        layout = make_structured_layout(3, 0, (0, 2, 1), 7)
        ids = np.array([4, 0, 4, 12, 26], dtype=np.uint64)
        positions = {key: np.array([0., 1., 0., -0.01, 1.01], dtype=np.float32)
                     for key in ("x", "y", "z")}
        order = hybrid_velocity_order(ids, positions, layout)
        np.testing.assert_array_equal(np.sort(order), np.arange(ids.size))
        self.assertLess(list(order).index(0), list(order).index(2))
        decoded_ids = decode_lattice_ids(encode_lattice_ids(ids, layout), layout, ids.dtype)
        np.testing.assert_array_equal(order, hybrid_velocity_order(decoded_ids, positions, layout))
        values = np.arange(ids.size, dtype=np.float64) + .25
        restored = np.empty_like(values)
        restored[order] = values[order]
        np.testing.assert_array_equal(restored, values)

    def test_invalid_geometry_and_payloads_rejected(self):
        layout = make_structured_layout(3, 0, (0, 1, 2), 7)
        for side, base, axes, cells in ((1, 0, (0, 1, 2), 7),
                                      (1025, 0, (0, 1, 2), 7),
                                      (3, 2, (0, 1, 2), 7),
                                      (3, 0, (0, 0, 2), 7),
                                      (3, 0, (0, 1, 2), 11)):
            with self.assertRaises(RuntimeError):
                make_structured_layout(side, base, axes, cells)
        for codes in (np.array([64], dtype=np.uint32), np.array([], dtype=np.uint32),
                      np.array([0], dtype=np.uint64)):
            with self.assertRaises(RuntimeError):
                decode_lattice_ids(codes, layout, np.uint64)
        with self.assertRaises(RuntimeError):
            StructuredParticleLayout.from_metadata({**layout.metadata(), "name": "unknown"})
        for axis in (np.array([-1]), np.array([1024]), np.array([.5])):
            with self.assertRaises(RuntimeError):
                morton_encode_3d(axis, axis, axis)
        with self.assertRaises(RuntimeError):
            hybrid_velocity_order(np.array([1]), {key: np.array([np.nan])
                                  for key in ("x", "y", "z")}, layout)

    def test_cli_opt_in_and_conflicting_layout(self):
        argv = ["compress", "input.h5", "--pos-compressor", "xnyzip", "--vel-compressor", "szo"]
        parser = build_parser(argv)
        self.assertFalse(CompressionSettings.from_args(parser.parse_args(argv)).structure_aware)
        args = parser.parse_args(argv + ["--xnyzip-structure-aware"])
        self.assertTrue(CompressionSettings.from_args(args).structure_aware)
        args.lattice_layout = True
        with self.assertRaises(RuntimeError):
            CompressionSettings.from_args(args)

    def test_missing_mesh_metadata_falls_back_without_touching_raw_fields(self):
        argv = ["compress", "input.h5", "--pos-compressor", "xnyzip",
                "--vel-compressor", "szo", "--xnyzip-structure-aware"]
        pipeline = object.__new__(CompressionPipeline)
        pipeline.settings = CompressionSettings.from_args(build_parser(argv).parse_args(argv))
        pipeline.manifest = {}
        pipeline._prepare_structure_aware_layout()
        self.assertFalse(pipeline.manifest["structured_layout"]["enabled"])
        self.assertIn("nsidemesh", pipeline.manifest["structured_layout"]["reason"])

    def test_incomplete_structured_metadata_rejected(self):
        from src.structured_layout import HYBRID_VELOCITY_LAYOUT
        validate_structured_package({"compressed_fields": {"id": {"codec": "pcodec"}}})
        with self.assertRaises(RuntimeError):
            validate_structured_package({"compressed_fields": {
                "vx": {"spatial_layout": HYBRID_VELOCITY_LAYOUT}}})
        with self.assertRaises(RuntimeError):
            validate_structured_package({"structured_layout":
                make_structured_layout(3, 0, (0, 1, 2), 7).metadata()})

    def test_failed_native_validation_refuses_to_finish_package(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = object.__new__(CompressionPipeline)
            pipeline.preprocessed_dir = Path(temp)
            pipeline.raw_paths = {"positions_xnyzip": "unused.raw"}
            pipeline.artifacts = {"positions": "unused.xnyzip"}
            pipeline.manifest = {"error_bounds": {"positions_xnyzip_abs": .001}}
            pipeline.count = 1
            pipeline.tools = None
            pipeline.settings = SimpleNamespace(force=False)
            pipeline.structured_layout = make_structured_layout(3, 0, (0, 1, 2), 7)
            with patch("src.compress.compress_xnyzip_triplet", return_value=np.array([0])) as encode:
                with patch.object(pipeline, "_validate_structured_positions", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "L2 validation"):
                        pipeline._compress_canonical_xnyzip_positions()
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(encode.call_args.kwargs["quantizer"], "cube")


class StructuredNativeRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = Path(__file__).resolve().parents[1] / "tools/XnYZip/build/XnYZip"
        if not binary.is_file():
            raise unittest.SkipTest("Build XnYZip to run native structured roundtrip coverage")
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

    def test_mixed_dtype_roundtrip_and_decode_without_source_or_raws(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, work = root / "input.h5", root / "package"
            side = 16
            ids = np.random.default_rng(19).permutation(side**3).astype(np.uint64) + 1
            linear = ids - 1
            positions = ((linear // side**2 + .25) / side,
                         (linear % side + .125) / side,
                         ((linear // side) % side + .375) / side)
            with h5py.File(source, "w") as handle:
                handle.attrs["nsidemesh"] = side
                handle.attrs["bitwidth"] = 2**31 - 1
                handle["id"] = ids
                for key, values in zip(("x", "y", "z"), positions):
                    handle[key] = np.rint(values * (2**31 - 1)).astype(np.int32)
                handle["vx"] = np.sin(linear * .01).astype(np.float32)
                handle["vy"] = np.cos(linear * .02).astype(np.float64)
                handle["vz"] = np.full(ids.size, .125, dtype=np.float32)
            self._run(["roundtrip", str(source), "--work-dir", str(work),
                       "--pos-compressor", "xnyzip", "--vel-compressor", "szo",
                       "--pos-abs-eb", "0.0001", "--vel-abs-eb", "0.0001",
                       "--xnyzip-structure-aware",
                       "--field-workers", "2", "--metrics", "--clean-raw"])
            manifest = json.loads((work / "manifest.json").read_text())
            self.assertTrue(manifest["structured_layout"]["enabled"], manifest["structured_layout"])
            self.assertEqual(manifest["structured_layout"]["id_base"], 1)
            self.assertEqual(manifest["format_version"], 9)
            self.assertIn(manifest["compressed_fields"]["positions"]["curve"], ("-h", "-z"))
            self.assertEqual(manifest["compressed_fields"]["positions"]["quantizer"], "cube")
            self.assertEqual(len(list((work / "compressed").iterdir())), 5)
            metrics = json.loads((work / "metrics.json").read_text())
            self.assertTrue(metrics["fields"]["id"]["exact_match"])
            self.assertTrue(all(check["satisfied"]
                                for check in metrics["error_bound_consistency"].values()),
                            metrics["error_bound_consistency"])
            self.assertTrue(metrics["xnyzip_l2_error_bound_consistency"]["positions"]["satisfied"])
            with h5py.File(manifest["artifacts"]["reconstructed_h5"], "r") as handle:
                expected = {key: handle[key][:] for key in handle}
                indices = np.argsort(expected["id"])
                np.testing.assert_array_equal(expected["id"][indices], np.arange(1, side**3 + 1))
                for key, original in (("vx", np.sin(np.arange(side**3) * .01).astype(np.float32)),
                                      ("vy", np.cos(np.arange(side**3) * .02))):
                    self.assertLessEqual(np.max(np.abs(expected[key][indices] - original)), .0001)
                np.testing.assert_array_equal(expected["vz"], np.full(ids.size, .125))
                self.assertEqual(expected["vy"].dtype, np.dtype("float64"))
            source.unlink()
            self.assertFalse((work / "preprocessed").exists())
            self._run(["decompress", "--work-dir", str(work), "--force"])
            with h5py.File(manifest["artifacts"]["reconstructed_h5"], "r") as handle:
                for key, values in expected.items():
                    np.testing.assert_array_equal(handle[key][:], values)


if __name__ == "__main__":
    unittest.main()
