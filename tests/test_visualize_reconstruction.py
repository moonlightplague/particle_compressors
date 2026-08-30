import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from experiments.visualize_reconstruction import build_visualization
from src.constants import LOGICAL_ORDER, POSITION_FIELDS


class ReconstructionVisualizationTests(unittest.TestCase):
    def test_dashboard_aligns_id_sorted_reconstruction(self) -> None:
        ids = np.array([30, 10, 20, 40], dtype=np.uint64)
        source = {
            "id": ids,
            "x": np.array([10, 20, 30, 40], dtype=np.int32),
            "y": np.array([50, 60, 70, 80], dtype=np.int32),
            "z": np.array([90, 100, 110, 120], dtype=np.int32),
            "vx": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "vy": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
            "vz": np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32),
        }
        order = np.argsort(ids, kind="stable")
        reconstructed = {
            logical: values[order].copy()
            for logical, values in source.items()
        }
        for logical in POSITION_FIELDS:
            reconstructed[logical] += 1
        for logical in ("vx", "vy", "vz"):
            reconstructed[logical] += np.float32(0.01)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.h5"
            reconstructed_path = root / "reconstructed.h5"
            _write_h5(original_path, source)
            _write_h5(reconstructed_path, reconstructed)
            manifest = _manifest(original_path, reconstructed_path, source)
            (root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            metrics = _metrics()
            (root / "metrics.json").write_text(
                json.dumps(metrics),
                encoding="utf-8",
            )
            output = root / "dashboard.html"

            payload = build_visualization(
                root,
                output=output,
                sample_size=source["id"].size,
                seed=4,
            )

            self.assertEqual(payload["alignment_source"], "particle_id")
            self.assertEqual(payload["sample_count"], source["id"].size)
            self.assertTrue(payload["id_exact"])
            self.assertEqual(payload["id_exact_scope"], "full")
            np.testing.assert_allclose(payload["fields"]["x"]["error"], 0.1)
            np.testing.assert_allclose(
                payload["fields"]["vx"]["error"],
                0.01,
                atol=1e-7,
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("const DATA =", rendered)
            self.assertIn("Spatial reconstruction error", rendered)
            self.assertIn('"codec":"qoz"', rendered)


def _write_h5(path: Path, fields: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as h5:
        for logical, values in fields.items():
            h5.create_dataset(logical, data=values)


def _manifest(
    original_path: Path,
    reconstructed_path: Path,
    source: dict[str, np.ndarray],
) -> dict[str, object]:
    return {
        "input_h5": str(original_path),
        "count": int(source["id"].size),
        "compressors": {"lossy": "qoz", "lossless": "pcodec"},
        "position_scale": {"value": 10.0},
        "fields": {
            logical: {"h5_path": logical, "dtype": str(values.dtype)}
            for logical, values in source.items()
        },
        "field_error_bounds": {
            logical: {"abs": 0.2 if logical != "id" else 0.0}
            for logical in LOGICAL_ORDER
        },
        "artifacts": {
            "reconstructed_h5": str(reconstructed_path),
            "preprocessed": {},
        },
        "ordering": {
            "reconstructed_rows": {
                "mapping": "id_sorted",
                "original_row_order_restored": False,
            }
        },
    }


def _metrics() -> dict[str, object]:
    return {
        "sizes": {"payload_compression_ratio": 4.5},
        "fields": {
            logical: {
                "max_absolute_error": 0.1 if logical in POSITION_FIELDS else 0.01,
                "rmse": 0.05 if logical in POSITION_FIELDS else 0.005,
                **({"exact_match": True} if logical == "id" else {}),
            }
            for logical in LOGICAL_ORDER
        },
        "error_bound_consistency": {
            logical: {
                "effective_final_abs_bound": 0.2 if logical != "id" else 0.0,
                "satisfied": True,
            }
            for logical in LOGICAL_ORDER
        },
    }


if __name__ == "__main__":
    unittest.main()
