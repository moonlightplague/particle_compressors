import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from experiments.visualize_xy_statistics import build_xy_statistics


class XYStatisticsVisualizationTests(unittest.TestCase):
    def test_aggregates_files_into_binned_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_h5(
                root / "rank0.h5",
                x=[1, 2],
                y=[1, 2],
                z=[1, 3],
                vx=[1, 3],
                vy=[0, 4],
                vz=[0, 0],
                rank=0,
            )
            _write_h5(
                root / "rank1.h5",
                x=[8, 9],
                y=[8, 9],
                z=[7, 9],
                vx=[-1, -3],
                vy=[0, -4],
                vz=[0, 0],
                rank=1,
            )
            output = root / "plots" / "xy.html"

            payload = build_xy_statistics(
                [root],
                output,
                bins=2,
                max_particles=0,
                chunk_size=1,
                extent=(0.0, 1.0, 0.0, 1.0),
            )

            self.assertEqual(payload["file_count"], 2)
            self.assertEqual(payload["total_count"], 4)
            self.assertEqual(payload["analyzed_count"], 4)
            self.assertTrue(payload["sampling_is_exact"])
            self.assertEqual(payload["grids"]["particle_count"], [2, 0, 0, 2])
            self.assertAlmostEqual(payload["grids"]["mean_z"][0], 0.2)
            self.assertAlmostEqual(payload["grids"]["mean_z"][3], 0.8)
            self.assertAlmostEqual(payload["grids"]["mean_vx"][0], 2.0)
            self.assertAlmostEqual(payload["grids"]["mean_vx"][3], -2.0)
            self.assertIsNone(payload["grids"]["mean_speed"][1])
            self.assertTrue(output.is_file())
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("const DATA =", rendered)
            self.assertIn("Mean planar velocity", rendered)

    def test_sampling_stride_is_global_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for rank in range(2):
                _write_h5(
                    root / f"rank{rank}.h5",
                    x=[1, 2],
                    y=[1, 2],
                    z=[1, 2],
                    vx=[rank + 1, rank + 1],
                    vy=[0, 0],
                    vz=[0, 0],
                    rank=rank,
                )

            payload = build_xy_statistics(
                [root],
                root / "xy.html",
                bins=2,
                max_particles=2,
                chunk_size=1,
                extent=(0.0, 1.0, 0.0, 1.0),
            )

            self.assertEqual(payload["sampling_stride"], 2)
            self.assertEqual(payload["analyzed_count"], 2)
            self.assertEqual(sum(payload["grids"]["particle_count"]), 2)
            self.assertAlmostEqual(payload["summaries"]["vx"]["mean"], 1.5)


def _write_h5(
    path: Path,
    *,
    x: list[int],
    y: list[int],
    z: list[int],
    vx: list[float],
    vy: list[float],
    vz: list[float],
    rank: int,
) -> None:
    with h5py.File(path, "w") as source:
        source.attrs["bitwidth"] = np.int32(10)
        source.attrs["rank"] = np.int32(rank)
        source.attrs["nsidemesh"] = np.int32(2)
        source.create_dataset("id", data=np.arange(len(x), dtype=np.uint64))
        for name, values in (("x", x), ("y", y), ("z", z)):
            source.create_dataset(name, data=np.asarray(values, dtype=np.int32))
        for name, values in (("vx", vx), ("vy", vy), ("vz", vz)):
            source.create_dataset(name, data=np.asarray(values, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
