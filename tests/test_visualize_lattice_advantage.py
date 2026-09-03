import json
import tempfile
import unittest
from pathlib import Path

from experiments.visualize_lattice_advantage import (
    build_lattice_advantage_visualization,
)


class LatticeAdvantageVisualizationTests(unittest.TestCase):
    def test_attributes_package_savings_and_loads_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flat = self._report(lattice=False)
            lattice = self._report(lattice=True)
            flat_path = root / "flat.json"
            lattice_path = root / "lattice.json"
            ablation_path = root / "ablation.json"
            output = root / "report.html"
            flat_path.write_text(json.dumps(flat), encoding="utf-8")
            lattice_path.write_text(json.dumps(lattice), encoding="utf-8")
            ablation_path.write_text(
                json.dumps(
                    {
                        "count": 100,
                        "results": [
                            {
                                "field": field,
                                "layout": layout,
                                "compressed_bytes": size,
                            }
                            for field in ("x", "y", "z", "vx", "vy", "vz")
                            for layout, size in (
                                ("id_sorted_1d:auto", 100),
                                ("dense_3d:auto", 70),
                                ("lattice_residual_dense_3d:auto", 30),
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_lattice_advantage_visualization(
                flat_path,
                lattice_path,
                output,
                ablation_json=ablation_path,
            )

            self.assertEqual(payload["comparison"]["bytes_saved"], 600)
            self.assertEqual(
                payload["comparison"]["group_savings"]["positions"],
                270,
            )
            self.assertEqual(
                payload["comparison"]["group_savings"]["velocities"],
                300,
            )
            self.assertEqual(
                payload["comparison"]["group_savings"]["metadata/other"],
                30,
            )
            self.assertEqual(
                payload["ablation"]["fields"]["x"][2]["stage"],
                "3-D + lattice residual",
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("Why the periodic lattice layout", rendered)
            self.assertNotIn("__EMBEDDED_DATA__", rendered)

    @staticmethod
    def _report(*, lattice: bool) -> dict:
        if lattice:
            components = {
                "compressed/id.pco": 50,
                "compressed/x.szo": 10,
                "compressed/x.lattice-wrap.pco": 1,
                "compressed/y.szo": 10,
                "compressed/y.lattice-wrap.pco": 1,
                "compressed/z.szo": 10,
                "compressed/z.lattice-wrap.pco": 1,
                "compressed/vx.szo": 100,
                "compressed/vy.szo": 100,
                "compressed/vz.szo": 100,
                "manifest.json": 117,
            }
            total = 500
            layout = {
                "enabled": True,
                "dense_count": 105,
                "dense_shape": [3, 5, 7],
                "occupancy": 100 / 105,
                "axis_search": True,
                "hole_fill": "linear_flat_index",
            }
        else:
            components = {
                "compressed/id.pco": 50,
                "compressed/x.szo": 101,
                "compressed/y.szo": 101,
                "compressed/z.szo": 101,
                "compressed/vx.szo": 200,
                "compressed/vy.szo": 200,
                "compressed/vz.szo": 200,
                "manifest.json": 47,
            }
            total = 1100
            layout = None
        return {
            "input_h5": "/source/example.h5",
            "count": 100,
            "compressors": {"lossy": "szo", "lossless": "pcodec"},
            "particle_sort": {"enabled": True},
            "lattice_layout": layout,
            "field_error_bounds": {
                field: {"abs": 0.01, "compressor_abs": 0.01}
                for field in ("x", "y", "z", "vx", "vy", "vz")
            },
            "sizes": {
                "selected_original_payload_bytes": 10_000,
                "compressed_total_bytes": total,
                "payload_compression_ratio": 10_000 / total,
                "compressed_components_bytes": components,
            },
        }


if __name__ == "__main__":
    unittest.main()
