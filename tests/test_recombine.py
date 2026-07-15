import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.cli import AVAILABLE_COMPRESSORS, build_parser
from src.decompress import recombine_h5, velocity_compressor_from_manifest


FIELDS = {
    logical: {"h5_path": logical, "dtype": "uint64" if logical == "id" else "float32"}
    for logical in ("id", "x", "y", "z", "vx", "vy", "vz")
}


class RecombineTests(unittest.TestCase):
    def test_cli_accepts_lcp_velocity_compressor(self) -> None:
        argv = ["roundtrip", "input.h5", "--vel-compressor", "lcp"]
        args = build_parser(argv).parse_args(argv)
        self.assertEqual(args.vel_compressor, "lcp")

    def test_lcp_velocity_uses_its_own_permutation(self) -> None:
        self.assertIn("lcp", AVAILABLE_COMPRESSORS["vel_compressor"])
        count = 5
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.int32)
        velocity_order = np.array([1, 4, 0, 3, 2], dtype=np.int32)
        original = {
            "id": np.array([901, 117, 502, 330, 774], dtype=np.uint64),
            "x": np.array([10, 20, 30, 40, 50], dtype=np.float32),
            "y": np.array([11, 21, 31, 41, 51], dtype=np.float32),
            "z": np.array([12, 22, 32, 42, 52], dtype=np.float32),
            "vx": np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "vy": np.array([6, 7, 8, 9, 10], dtype=np.float32),
            "vz": np.array([11, 12, 13, 14, 15], dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dec_paths = {}
            for logical, values in original.items():
                path = root / f"{logical}.raw"
                if logical in ("x", "y", "z"):
                    values[position_order].tofile(path)
                elif logical in ("vx", "vy", "vz"):
                    values[velocity_order].tofile(path)
                else:
                    values.tofile(path)
                dec_paths[logical] = str(path)

            order_path = root / "order.raw"
            velocity_order_path = root / "velocity_order.raw"
            position_order.tofile(order_path)
            velocity_order.tofile(velocity_order_path)
            dec_paths["order"] = str(order_path)
            dec_paths["velocity_order"] = str(velocity_order_path)

            manifest = {
                "count": count,
                "position_scale": {"value": 1.0},
                "fields": FIELDS,
                "order_dtype": "int32",
                "compressors": {"velocities": "lcp"},
                "compressed_fields": {"velocity_order": {"dtype": "int32"}},
            }
            output = root / "reconstructed.h5"
            recombine_h5(manifest, dec_paths, output)

            with h5py.File(output, "r") as h5:
                for logical, expected in original.items():
                    np.testing.assert_array_equal(h5[logical][:], expected)

    def test_manifest_without_compressor_metadata_defaults_to_sz3(self) -> None:
        self.assertEqual(velocity_compressor_from_manifest({}), "sz3")


if __name__ == "__main__":
    unittest.main()
