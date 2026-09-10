"""Lossless field selection must preserve source bits through the pipeline."""
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import h5py
import numpy as np

import main
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS


class PcodecFieldTests(unittest.TestCase):
    def test_roundtrips_source_bits_and_ignores_bounds(self):
        cases = [
            ("pcodec", "pcodec", "float64", False, 1),
            ("pcodec", "pcodec", "uint64", True, 2),
            ("pcodec", "szo", "float32", False, 1),
            ("pcodec", "sz3", "float64", True, 1),
            ("szo", "pcodec", "float32", False, 1),
            ("sz3", "pcodec", "float32", True, 2),
            ("lcp", "pcodec", "float32", False, 1),
            ("xnyzip", "pcodec", "float32", False, 1),
        ]
        for pos, vel, dtype, sort, workers in cases:
            with self.subTest(pos=pos, vel=vel, dtype=dtype, sort=sort):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = {"id": np.arange(63, -1, -1, dtype=np.uint64)}
                    for i, field in enumerate(POSITION_FIELDS):
                        values = np.arange(64, dtype=np.dtype(dtype))
                        if dtype == "uint64":
                            values += np.uint64(2**60 + i)
                        else:
                            values = (values / 71 + i + 1e-10).astype(dtype)
                            values[0] = -0.0
                        source[field] = values
                    for i, field in enumerate(VELOCITY_FIELDS):
                        values = np.arange(64, dtype=np.float64) / 73 + i + 1e-12
                        values[0] = -0.0
                        source[field] = values
                    with h5py.File(root / "input.h5", "w") as h5:
                        h5.attrs["bitwidth"] = 123.0
                        for field, values in source.items():
                            h5.create_dataset(field, data=values)
                    # Use an explicit config so local user defaults cannot affect tests.
                    (root / "config.yaml").write_text("advanced: {}\n")
                    argv = ["roundtrip", str(root / "input.h5"),
                            "--config", str(root / "config.yaml"),
                            "--work-dir", str(root / "work"),
                            "--pos-compressor", pos, "--vel-compressor", vel,
                            "--field-workers", str(workers), "--abs-eb", "0.01"]
                    for prefix, codec in (("pos", pos), ("vel", vel)):
                        if codec == "pcodec":
                            argv += [f"--{prefix}-abs-eb", "-1", f"--{prefix}-rel-eb", "0.9"]
                    if sort:
                        argv += ["--sort", "--lattice-layout"]
                    output = StringIO()
                    with redirect_stdout(output), redirect_stderr(output):
                        result = main.main(argv)
                    self.assertEqual(result, 0, output.getvalue())
                    manifest = json.loads((root / "work/manifest.json").read_text())
                    with h5py.File(manifest["artifacts"]["reconstructed_h5"], "r") as h5:
                        order = 63 - h5["id"][:].astype(np.int64)
                        for fields, codec in ((POSITION_FIELDS, pos), (VELOCITY_FIELDS, vel)):
                            if codec != "pcodec":
                                continue
                            for field in fields:
                                self.assertEqual(h5[field][:].tobytes(), source[field][order].tobytes())
                                self.assertEqual(h5[field].dtype, source[field].dtype)
                                self.assertEqual(manifest["compressed_fields"][field]["codec"], "pcodec")
                                bound = manifest["field_error_bounds"][field]
                                self.assertEqual(bound["mode"], "lossless")
                                self.assertEqual(bound["compressor_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
