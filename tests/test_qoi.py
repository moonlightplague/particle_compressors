"""QoI bounds, actual codecs, canonical alignment, and portable corrections."""

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import zstandard as zstd

import main
from src.cli import build_parser
from src.hacc_snapshot import HACC_FIELDS
from src.models import CanonicalOrder
from src.qoi import (
    QOI_CHOICES, apply_qoi_corrections, evaluate_qoi, prepare_qoi,
    uniform_velocity_bound, validate_and_store_qoi,
)


class QoiTests(unittest.TestCase):
    def run_cli(self, argv, expected=0):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = main.main(argv)
        self.assertEqual(result, expected, output.getvalue())
        return output.getvalue()

    def make_hacc(self, root, count=1025):
        source = root / "hacc"
        source.mkdir()
        rng = np.random.default_rng(73)
        arrays = {}
        for key, name in HACC_FIELDS.items():
            values = rng.uniform(0, 60, count) if key in "xyz" else rng.normal(0, 120, count)
            arrays[key] = values.astype("<f4")
            arrays[key].tofile(source / f"{name}.f32")
        return source, arrays

    def load(self, path):
        return json.loads(path.read_text())

    def test_uniform_bounds_at_cube_corners(self):
        rng = np.random.default_rng(4)
        original = {key: rng.normal(0, 150, 1000) for key in ("vx", "vy", "vz")}
        for key in original:
            original[key][0] = 0
        max_speed = float(evaluate_qoi("speed", original).max())
        tolerance, mass = 0.3, 2.5
        for quantity in QOI_CHOICES:
            with self.subTest(quantity=quantity):
                expected = evaluate_qoi(quantity, original, mass)
                bound = uniform_velocity_bound(quantity, tolerance, max_speed, float(expected.max()), mass)
                for corner in range(8):
                    perturbed = {
                        key: values + (bound if corner & (1 << axis) else -bound)
                        for axis, (key, values) in enumerate(original.items())
                    }
                    error = np.abs(evaluate_qoi(quantity, perturbed, mass) - expected)
                    self.assertLessEqual(float(error.max()), tolerance + 1e-9)
                self.assertEqual(uniform_velocity_bound(quantity, 0, max_speed, float(expected.max()), mass), 0)
        self.assertEqual(uniform_velocity_bound("kinetic-energy", 1e-300, 0, 0, 1e300), 0)

    def test_actual_hacc_roundtrips_for_all_choices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, arrays = self.make_hacc(root)
            for quantity in QOI_CHOICES:
                with self.subTest(quantity=quantity):
                    work = root / quantity
                    output = self.run_cli([
                        "roundtrip", str(source), "--work-dir", str(work),
                        "--qoi", quantity, "--qoi-rel-eb", "1e-6", "--qoi-mass", "2.5",
                        "--limit", "513", "--xnyzip-tie-sort", "--metrics", "--field-workers", "2",
                    ])
                    self.assertIn("bound_ok=True", output)
                    manifest = self.load(work / "manifest.json")
                    qoi = manifest["qoi"]
                    self.assertEqual(manifest["format_version"], 11)
                    self.assertEqual(manifest["count"], 513)
                    self.assertEqual(qoi["state"], "validated")
                    self.assertTrue(qoi["validation"]["satisfied"])
                    order = np.fromfile(manifest["artifacts"]["preprocessed"]["position_order"], dtype="uint64")
                    expected = {key: values[:513][order] for key, values in arrays.items()}
                    original_q = evaluate_qoi(quantity, expected, 2.5)
                    self.assertEqual(qoi["tolerance"]["original_range"], float(original_q.max() - original_q.min()))
                    with h5py.File(work / "reconstructed.h5") as restored:
                        actual = {key: restored[HACC_FIELDS[key]][:] for key in ("vx", "vy", "vz")}
                        error = np.abs(evaluate_qoi(quantity, actual, 2.5) - original_q)
                        self.assertLessEqual(float(error.max()), qoi["tolerance"]["absolute"])
                        for key in actual:
                            data_error = np.abs(actual[key].astype(float) - expected[key].astype(float))
                            self.assertLessEqual(float(data_error.max()), qoi["requested_velocity_abs_bounds"][key])
                    metrics = self.load(work / "metrics.json")
                    final_qoi = metrics["qoi"]["reconstruction_validation"]
                    self.assertEqual(final_qoi["violations"], 0)
                    self.assertEqual(final_qoi["max_abs_error"], float(error.max()))
                    for key in ("vx", "vy", "vz"):
                        self.assertEqual(manifest["compressed_fields"][key]["abs_error_bound"],
                                         qoi["compressor_velocity_abs_bounds"][key])
                    # All original inputs and temporary permutations may disappear.
                    shutil.rmtree(work / "preprocessed")
                    shutil.rmtree(work / "input_adapters")
                    if quantity == QOI_CHOICES[-1]:
                        shutil.rmtree(source)
                    self.run_cli(["decompress", "--work-dir", str(work), "--force", "--clean-raw"])
                    with h5py.File(work / "reconstructed.h5") as restored:
                        for key, values in actual.items():
                            np.testing.assert_array_equal(restored[HACC_FIELDS[key]][:], values)

    def test_loose_estimator_is_repaired_with_real_codecs_and_no_sources(self):
        """Exercise the guarantee even if an estimator or codec misses the bound."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.h5"
            work = root / "work"
            rng = np.random.default_rng(89)
            count = 71
            arrays = {key: rng.uniform(1, 20, count).astype("float32") for key in "xyz"}
            arrays.update({key: rng.normal(0, 200, count).astype(dtype) for key, dtype in
                           (("vx", "float64"), ("vy", "float32"), ("vz", "float64"))})
            arrays["id"] = rng.permutation(count).astype("uint64") + 10
            with h5py.File(source, "w") as h5:
                for key, values in arrays.items():
                    h5[f"particles/{key}"] = values
            with patch("src.qoi.uniform_velocity_bound", return_value=1000), patch("src.qoi.CHUNK_ROWS", 7):
                self.run_cli([
                    "compress", str(source), "--work-dir", str(work), "--qoi", "kinetic-energy",
                    "--qoi-abs-eb", "1e-10", "--vel-abs-eb", "10", "--field-workers", "1",
                ])
            manifest = self.load(work / "manifest.json")
            self.assertGreater(manifest["qoi"]["validation"]["corrected_particles"], 0)
            self.assertFalse((work / "reconstructed.h5").exists())
            order = np.fromfile(manifest["artifacts"]["preprocessed"]["position_order"], dtype="uint64")
            expected = {key: values[order] for key, values in arrays.items()}
            correction_path = Path(manifest["qoi"]["corrections"]["path"])
            self.assertEqual(manifest["sizes"]["compressed_components_bytes"]["compressed/qoi.corrections.zst"],
                             correction_path.stat().st_size)
            source.unlink()
            shutil.rmtree(work / "preprocessed")
            with patch("src.qoi.CHUNK_ROWS", 7):
                self.run_cli(["decompress", "--work-dir", str(work), "--clean-raw"])
            with h5py.File(work / "reconstructed.h5") as restored:
                actual = {key: restored[f"particles/{key}"][:] for key in expected}
                for key in ("vx", "vy", "vz", "id"):
                    self.assertEqual(actual[key].tobytes(), expected[key].tobytes())
            # Missing, truncated, unsupported, or miscounted sidecars fail visibly.
            valid_payload = correction_path.read_bytes()
            for bad_payload in (b"not zstd", zstd.ZstdCompressor().compress(b"\0")):
                correction_path.write_bytes(bad_payload)
                with self.assertRaises(RuntimeError):
                    apply_qoi_corrections(manifest, work / "reconstructed.h5")
            correction_path.write_bytes(valid_payload)
            manifest["qoi"]["corrections"]["count"] += 1
            with self.assertRaisesRegex(RuntimeError, "count"):
                apply_qoi_corrections(manifest, work / "reconstructed.h5")
            correction_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "Could not read"):
                apply_qoi_corrections(manifest, work / "reconstructed.h5")

    def test_constant_range_means_exact_qoi_and_disabled_mode_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, arrays = self.make_hacc(root, 33)
            for key in ("vx", "vy", "vz"):
                arrays[key][:] = 1.25
                arrays[key].tofile(source / f"{key}.f32")
            work = root / "exact"
            self.run_cli(["roundtrip", str(source), "--work-dir", str(work),
                          "--qoi", "speed", "--qoi-rel-eb", "1e-4", "--metrics"])
            manifest = self.load(work / "manifest.json")
            self.assertEqual(manifest["qoi"]["tolerance"]["absolute"], 0)
            self.assertEqual(manifest["qoi"]["validation"]["max_abs_error"], 0)
            self.assertEqual(manifest["qoi"]["compressor_velocity_abs_bounds"]["vx"], 0)
            self.run_cli(["roundtrip", str(source), "--work-dir", str(work), "--force", "--qoi", "none"])
            self.assertNotIn("qoi", self.load(work / "manifest.json"))
            self.assertFalse((work / "compressed/qoi.corrections.zst").exists())

    def test_validator_repairs_nonfinite_and_data_only_violations(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / "preprocessed").mkdir()
            (work / "compressed").mkdir()
            count = 8
            original = {"vx": np.arange(count, dtype="float32") + 3,
                        "vy": np.full(count, 4, dtype="float32"),
                        "vz": np.zeros(count, dtype="float32")}
            permutation = np.arange(count)[::-1]
            expected = {key: values[permutation].copy() for key, values in original.items()}
            decoded = {key: values.copy() for key, values in expected.items()}
            # Swapping components preserves speed exactly but violates raw-data bounds.
            decoded["vx"][0], decoded["vy"][0] = decoded["vy"][0], decoded["vx"][0]
            decoded["vx"][1] = np.nan
            paths = {}
            for key, values in original.items():
                paths[key] = str(work / "preprocessed" / f"{key}.raw")
                values.tofile(paths[key])
            manifest = {
                "count": count,
                "fields": {key: {"dtype": "float32", "h5_path": f"group/{key}"} for key in original},
                "field_error_bounds": {key: {"abs": 0.01} for key in original},
                "compressed_fields": {key: {"field": key} for key in original},
                "artifacts": {"compressed": {}},
            }
            args = SimpleNamespace(qoi="speed", qoi_abs_eb=1e-5, qoi_rel_eb=None, qoi_mass=1)
            prepare_qoi(args, manifest, paths)

            def decode(field, path, force):
                decoded[field["field"]].tofile(path)

            with patch("src.qoi.decompress_lossy_raw", side_effect=decode), patch("src.qoi.CHUNK_ROWS", 3):
                validate_and_store_qoi(manifest, paths, CanonicalOrder(values=permutation), work, False)
            validation = manifest["qoi"]["validation"]
            self.assertEqual(validation["qoi_violations_before"], 1)
            self.assertEqual(validation["data_violations_before"], 2)
            self.assertEqual(validation["corrected_particles"], 2)
            output = work / "restored.h5"
            with h5py.File(output, "w") as h5:
                for key, values in decoded.items():
                    h5[f"group/{key}"] = values
            with patch("src.qoi.CHUNK_ROWS", 3):
                apply_qoi_corrections(manifest, output)
            with h5py.File(output) as h5:
                for key in expected:
                    np.testing.assert_array_equal(h5[f"group/{key}"][:], expected[key])

            corrections = manifest["qoi"]["corrections"]
            dtype = np.dtype([("row", "<u8"), ("vx", "<f4"), ("vy", "<f4"), ("vz", "<f4")])
            with open(corrections["path"], "rb") as source, zstd.ZstdDecompressor().stream_reader(source) as reader:
                records = np.frombuffer(reader.read(), dtype=dtype).copy()
            records["row"][1] = records["row"][0]
            Path(corrections["path"]).write_bytes(zstd.ZstdCompressor().compress(records.tobytes()))
            with self.assertRaisesRegex(RuntimeError, "indices"):
                apply_qoi_corrections(manifest, output)

    def test_yaml_and_cli_tolerance_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.yaml"
            config.write_text("advanced:\n  pos_compressor: xnyzip\n  vel_compressor: szo\n"
                              "qoi:\n  quantity: speed\n  rel_eb: 0.001\n")
            argv = ["roundtrip", "input", "--config", str(config), "--qoi-abs-eb", "0.25"]
            args = build_parser(argv).parse_args(argv)
            self.assertEqual(args.qoi, "speed")
            self.assertEqual(args.qoi_abs_eb, 0.25)
            self.assertIsNone(args.qoi_rel_eb)
            config.write_text("advanced: {}\nqoi:\n  quantity: vx-square\n  abs_eb: 0.5\n")
            argv = ["compress", "input", "--config", str(config), "--qoi-rel-eb", "0.01"]
            args = build_parser(argv).parse_args(argv)
            self.assertEqual(args.qoi_rel_eb, 0.01)
            self.assertIsNone(args.qoi_abs_eb)

    def test_bad_options_fail_before_workspace_creation(self):
        cases = [
            (["--qoi", "speed"], "exactly one"),
            (["--qoi-abs-eb", "1"], "require --qoi"),
            (["--qoi", "speed", "--qoi-abs-eb", "-1"], "non-negative"),
            (["--qoi", "speed", "--qoi-rel-eb", "nan"], "finite"),
            (["--qoi", "speed", "--qoi-abs-eb", "inf"], "finite"),
            (["--qoi", "kinetic-energy", "--qoi-abs-eb", "1", "--qoi-mass", "0"], "positive"),
            (["--qoi", "speed", "--qoi-abs-eb", "1", "--vel-compressor", "sz3"], "requires"),
            (["--qoi", "speed", "--qoi-abs-eb", "1", "--lattice-layout"], "flat"),
            (["--qoi", "speed", "--qoi-abs-eb", "1", "--xnyzip-structure-aware"], "flat"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "work"
            for extra, message in cases:
                with self.subTest(extra=extra):
                    output = self.run_cli(["compress", "missing-input", "--work-dir", str(work), *extra], expected=2)
                    self.assertIn(message, output)
                    self.assertFalse(work.exists())

    def test_nonfinite_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, arrays = self.make_hacc(root, 33)
            arrays["vx"][0] = np.nan
            arrays["vx"].tofile(source / "vx.f32")
            output = self.run_cli(["preprocess", str(source), "--work-dir", str(root / "work"),
                                   "--qoi", "speed", "--qoi-abs-eb", "1"], expected=2)
            self.assertIn("finite", output)


if __name__ == "__main__":
    unittest.main()
