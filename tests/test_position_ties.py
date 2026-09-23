"""No-sidecar tie sorting must preserve complete particle associations."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import h5py
import numpy as np

import main
from src.cli import build_parser
from src.compress import CompressionSettings
from src.position_ties import sort_position_ties


class PositionTieTests(unittest.TestCase):
    def test_exact_position_bits_and_source_permutation_across_blocks(self):
        rng = np.random.default_rng(221)
        for count in (0, 1, 2, 517):
            for block_size in (1, 7, 1024):
                with self.subTest(count=count, block=block_size):
                    order = rng.permutation(count).astype("uint64")
                    original_order = order.copy()
                    positions = {
                        key: (np.arange(count) // 23).astype("float32")
                        for key in ("x", "y", "z")
                    }
                    if count > 50:
                        # Adjacent floats and signed zero are NOT ties.
                        positions["y"][8] = np.nextafter(np.float32(0), np.float32(1))
                        positions["z"][9] = -0.0
                    velocities = {key: rng.normal(size=count) for key in ("vx", "vy", "vz")}
                    sort_position_ties(order, positions, velocities, block_size=block_size)
                    np.testing.assert_array_equal(np.sort(order), np.arange(count))
                    inverse = np.argsort(original_order)
                    for values in positions.values():
                        before = values.view("uint32")
                        np.testing.assert_array_equal(before[inverse[order]], before)
                    repeated = original_order.copy()
                    sort_position_ties(repeated, positions, velocities, block_size=block_size)
                    np.testing.assert_array_equal(repeated, order)

    def test_unique_positions_and_constant_velocities_are_stable(self):
        for unique in (True, False):
            order = np.arange(49, -1, -1, dtype="uint64")
            original = order.copy()
            pos = np.arange(50, dtype="float32") if unique else np.zeros(50, dtype="float32")
            stats = sort_position_ties(order, dict.fromkeys(("x", "y", "z"), pos),
                                       dict.fromkeys(("vx", "vy", "vz"), np.ones(50)))
            np.testing.assert_array_equal(order, original)
            self.assertEqual(stats["moved_particles"], 0)
            self.assertEqual(stats["sidecar_bytes"], 0)

    def test_rejects_nonfinite_velocity_in_a_tie(self):
        with self.assertRaisesRegex(RuntimeError, "finite velocities"):
            sort_position_ties(np.arange(3),
                               dict.fromkeys(("x", "y", "z"), np.zeros(3, dtype="float32")),
                               dict.fromkeys(("vx", "vy", "vz"), np.array([0., np.nan, 1.])))

    def test_cli_and_yaml_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text("advanced:\n  xnyzip_tie_sort: true\n")
            argv = ["compress", "unused.h5", "--config", str(config), "--pos-compressor", "szo"]
            args = build_parser(argv).parse_args(argv)
            with self.assertRaisesRegex(RuntimeError, "requires --pos-compressor xnyzip"):
                CompressionSettings.from_args(args)
            argv += ["--no-xnyzip-tie-sort"]
            self.assertFalse(CompressionSettings.from_args(build_parser(argv).parse_args(argv)).xnyzip_tie_sort)
            argv += ["--xnyzip-tie-sort", "--pos-compressor", "xnyzip", "--vel-compressor", "xnyzip"]
            with self.assertRaisesRegex(RuntimeError, "requires fieldwise"):
                CompressionSettings.from_args(build_parser(argv).parse_args(argv))


class PositionTieRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = Path(__file__).resolve().parents[1] / "tools/XnYZip/build/XnYZip"
        if not binary.is_file():
            raise unittest.SkipTest("Build XnYZip for native tie-order tests")
        from src.runtime import load_pcodec, load_pyszo
        try:
            load_pcodec()
            load_pyszo()
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(str(exc))

    def run_cli(self, argv):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = main.main(argv)
        self.assertEqual(result, 0, output.getvalue())

    def test_roundtrip_alignment_with_ids_and_without_ids(self):
        cases = ((False, "szo", 0), (True, "szo", 0), (True, "pcodec", 0),
                 (False, "sz3", 0))
        for with_ids, velocity_codec, chunk_size in cases:
            with self.subTest(ids=with_ids, codec=velocity_codec, chunk=chunk_size), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / "config.yaml"
                config.write_text("advanced: {}\n")
                rng = np.random.default_rng(77)
                count = 2049
                cell = rng.integers(0, 8, count)
                values = {
                    key: (10 + cell * (axis + 1) + rng.uniform(0, .001, count)).astype("float32")
                    for axis, key in enumerate(("x", "y", "z"))
                }
                values.update({key: rng.normal(0, 300, count).astype("float64")
                               for key in ("vx", "vy", "vz")})
                if with_ids:
                    values["id"] = rng.permutation(count).astype("uint64") + 100
                source = root / "source.h5"
                with h5py.File(source, "w") as h5:
                    h5.attrs["test_attribute"] = 77
                    for key, value in values.items():
                        h5[key] = value
                work = root / "work"
                common = ["--config", str(config), "--work-dir", str(work)]
                self.run_cli(["roundtrip", str(source), *common, "--pos-compressor", "xnyzip",
                              "--vel-compressor", velocity_codec, "--xnyzip-tie-sort",
                              "--vel-chunk-size", str(chunk_size),
                              "--pos-abs-eb", ".05", "--vel-abs-eb", ".5", "--metrics"])
                manifest = json.loads((work / "manifest.json").read_text())
                metrics = json.loads((work / "metrics.json").read_text())
                self.assertGreater(manifest["position_tie_sort"]["moved_particles"], 0)
                if velocity_codec != "xnyzip":
                    self.assertNotIn("velocity_order", manifest["compressed_fields"])
                self.assertTrue(all(v["satisfied"] for v in metrics["error_bound_consistency"].values()),
                                metrics["error_bound_consistency"])
                self.assertTrue(all(v["satisfied"] for v in metrics["xnyzip_l2_error_bound_consistency"].values()))
                order = np.fromfile(manifest["artifacts"]["preprocessed"]["position_order"], dtype="uint64")
                with h5py.File(work / "reconstructed.h5") as h5:
                    first_decode = {key: h5[key][:] for key in h5}
                    self.assertEqual(h5.attrs["test_attribute"], 77)
                    for key, source_values in values.items():
                        expected = source_values[order]
                        actual = first_decode[key]
                        self.assertEqual(actual.dtype, source_values.dtype)
                        if key == "id" or (key in ("vx", "vy", "vz") and velocity_codec == "pcodec"):
                            np.testing.assert_array_equal(actual, expected)
                        else:
                            bound = .05 if key in ("x", "y", "z") else .5
                            self.assertLessEqual(float(np.max(np.abs(actual - expected))), bound)
                # Only compressed files + manifest are needed for a fresh decode.
                source.unlink()
                shutil.rmtree(work / "preprocessed")
                shutil.rmtree(work / "decompressed")
                (work / "reconstructed.h5").unlink()
                self.run_cli(["decompress", *common])
                with h5py.File(work / "reconstructed.h5") as h5:
                    for key in first_decode:
                        np.testing.assert_array_equal(h5[key][:], first_decode[key])


if __name__ == "__main__":
    unittest.main()
