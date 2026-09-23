"""HACC adapters and ID-free compression must preserve particle associations."""
import json
import tempfile
import shutil
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import main
from src.hacc_snapshot import HACC_FIELDS, write_hacc_adapter


class HaccTests(unittest.TestCase):
    def make_source(self, root, count=257):
        source = root / "hacc"
        source.mkdir()
        rng = np.random.default_rng(21)
        arrays = {}
        for logical, name in HACC_FIELDS.items():
            values = rng.uniform(10, 100, count) if logical in "xyz" else rng.normal(0, 300, count)
            arrays[logical] = values.astype("<f4")
            arrays[logical].tofile(source / f"{name}.f32")
        (root / "config.yaml").write_text("advanced: {}\n")
        return source, arrays

    def run_cli(self, argv):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = main.main(argv)
        self.assertEqual(result, 0, output.getvalue())

    def test_codec_matrix(self):
        cases = [(p, v, 0, False) for p in ("sz3", "szo", "pcodec")
                 for v in ("sz3", "szo", "pcodec")]
        cases += [(p, v, 0, False) for p in ("lcp", "xnyzip")
                  for v in ("sz3", "szo", "pcodec", "xnyzip")]
        cases += [("lcp", "lcp", 0, False), ("lcp", "lcp", 64, False),
                  ("lcp", "lcp", 0, True), ("lcp", "xnyzip", 64, False),
                  ("xnyzip", "xnyzip", 64, False)]
        for pos, vel, chunk, blockwise in cases:
            with self.subTest(pos=pos, vel=vel, chunk=chunk, blockwise=blockwise), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, arrays = self.make_source(root)
                work = root / "work"
                argv = ["roundtrip", str(source), "--config", str(root / "config.yaml"),
                        "--work-dir", str(work), "--pos-compressor", pos,
                        "--vel-compressor", vel, "--pos-abs-eb", "0.05", "--vel-abs-eb", "0.5", "--metrics",
                        "--field-workers", "2", "--vel-chunk-size", str(chunk)]
                if blockwise:
                    argv += ["--blockwise-ord"]
                needs_lossless = "pcodec" in (pos, vel) or vel in ("lcp", "xnyzip")
                if needs_lossless:
                    self.run_cli(argv)
                else:
                    with patch("src.raw_codecs.load_pcodec", side_effect=AssertionError("Unexpected lossless codec")):
                        self.run_cli(argv)
                manifest = json.loads((work / "manifest.json").read_text())
                metrics = json.loads((work / "metrics.json").read_text())
                self.assertEqual(manifest["position_scale"]["value"], 1.0)
                self.assertEqual(manifest["sizes"]["selected_original_payload_bytes"], 257 * 24)
                for key in ("fields", "compressed_fields"):
                    self.assertNotIn("id", manifest[key])
                self.assertNotIn("id", manifest["artifacts"]["compressed"])
                self.assertTrue(all(b["satisfied"] for b in metrics["error_bound_consistency"].values()))
                self.assertTrue(all(b["satisfied"] for b in metrics.get("xnyzip_l2_error_bound_consistency", {}).values()))
                order = np.arange(257)
                if pos in ("lcp", "xnyzip"):
                    order = np.fromfile(manifest["artifacts"]["preprocessed"]["position_order"], dtype=manifest["order_dtype"])
                with h5py.File(work / "reconstructed.h5") as h5:
                    self.assertEqual(set(h5), set(HACC_FIELDS.values()))
                    for field, name in HACC_FIELDS.items():
                        actual = h5[name][:]
                        expected = arrays[field][order]
                        self.assertEqual(h5[name].dtype, np.dtype("<f4"))
                        if (pos if field in "xyz" else vel) == "pcodec":
                            self.assertEqual(actual.tobytes(), expected.tobytes())
                        else:
                            self.assertLessEqual(float(np.max(np.abs(actual - expected))),
                                                 0.05001 if field in "xyz" else 0.50001)
                # Packages decode without the input or temporary permutations.
                shutil.rmtree(source)
                shutil.rmtree(work / "preprocessed")
                shutil.rmtree(work / "input_adapters")
                self.run_cli(["decompress", "--work-dir", str(work), "--config", str(root / "config.yaml"), "--force"])

    def test_adapter_and_unscaled_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, arrays = self.make_source(root)
            arrays["x"][0] = -0.0
            arrays["x"].tofile(source / "xx.f32")
            adapter = write_hacc_adapter(source, root / "adapter")
            with h5py.File(adapter) as h5:
                for logical, name in HACC_FIELDS.items():
                    self.assertEqual(h5[name][:].tobytes(), arrays[logical].tobytes())
                    self.assertEqual(len(h5[name].external), 1)
            self.run_cli([
                "preprocess", str(source), "--config", str(root / "config.yaml"),
                "--work-dir", str(root / "work"), "--limit", "127",
            ])
            manifest = json.loads((root / "work/manifest.json").read_text())
            for logical in "xyz":
                raw = manifest["artifacts"]["preprocessed"][logical]
                self.assertEqual(Path(raw).read_bytes(), arrays[logical][:127].tobytes())
            self.assertEqual(manifest["source"]["byte_order"], "little_endian")
            self.assertNotIn("id", manifest["artifacts"]["preprocessed"])

    def test_validation(self):
        for problem in ("missing", "short", "unaligned", "empty"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, arrays = self.make_source(root)
                file = source / "vz.f32"
                if problem == "missing":
                    file.unlink()
                else:
                    file.write_bytes({"short": b"\0" * 4, "unaligned": b"\0", "empty": b""}[problem])
                with self.assertRaises(RuntimeError):
                    write_hacc_adapter(source, root / "adapter")

    def test_limit_and_id_options(self):
        for extra in (["--sort", "--lattice-layout"], ["--xnyzip-structure-aware"]):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, arrays = self.make_source(root)
                self.run_cli(["roundtrip", str(source), "--config", str(root / "config.yaml"),
                              "--work-dir", str(root / "work"), "--pos-compressor", "szo",
                              "--vel-compressor", "szo", "--limit", "127", "--metrics", "--clean-raw", *extra])
                manifest = json.loads((root / "work/manifest.json").read_text())
                self.assertEqual(manifest["count"], 127)
                self.assertEqual(manifest["sizes"]["selected_original_payload_bytes"], 127 * 24)
                self.assertFalse(manifest["particle_sort"]["enabled"])
                key = "lattice_layout" if "--lattice-layout" in extra else "structured_layout"
                self.assertFalse(manifest[key]["enabled"])
                self.assertIn("IDs", manifest[key]["reason"])


if __name__ == "__main__":
    unittest.main()
