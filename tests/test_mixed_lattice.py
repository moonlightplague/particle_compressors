from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

import main as particle_main
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.lattice_layout import LATTICE_LAYOUT_NAME


REPOSITORY = Path(__file__).resolve().parents[1]
LCP_EXE = REPOSITORY / "tools" / "LCP" / "build" / "bin" / "lcp"
XNYZIP_EXE = REPOSITORY / "tools" / "XnYZip" / "build" / "XnYZip"


@unittest.skipUnless(
    LCP_EXE.is_file() and XNYZIP_EXE.is_file(),
    "LCP and XnYZip executables have not both been built",
)
class MixedNativeLatticeRoundtripTests(unittest.TestCase):
    def test_lcp_and_xnyzip_position_orders_drive_lattice_velocities(self):
        source, side = self._source()
        for position_codec, velocity_codec in (
            ("lcp", "szo"),
            ("xnyzip", "sz3"),
        ):
            with self.subTest(
                position_codec=position_codec,
                velocity_codec=velocity_codec,
            ):
                self._assert_roundtrip(
                    source,
                    side,
                    position_codec,
                    velocity_codec,
                )

    @staticmethod
    def _source() -> tuple[dict[str, np.ndarray], int]:
        side = 50
        ids = np.arange(4 * side * side, dtype=np.uint64)
        high = ids // (side * side)
        middle = (ids // side) % side
        low = ids % side
        source = {
            "id": ids,
            "x": ((low + 0.125) / side).astype(np.float32),
            "y": ((high + 0.25) / side).astype(np.float32),
            "z": ((middle + 0.375) / side).astype(np.float32),
            "vx": (low * 0.03125 + high).astype(np.float32),
            "vy": (middle * -0.0625 + low * 0.01).astype(np.float32),
            "vz": (high * 0.5 + middle * 0.02).astype(np.float32),
        }
        source_order = np.random.default_rng(20260907).permutation(ids.size)
        return {
            logical: values[source_order]
            for logical, values in source.items()
        }, side

    def _assert_roundtrip(
        self,
        source: dict[str, np.ndarray],
        side: int,
        position_codec: str,
        velocity_codec: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_h5 = root / "input.h5"
            work_dir = root / "work"
            with h5py.File(input_h5, "w") as h5:
                h5.attrs["nsidemesh"] = np.int32(side)
                for logical, values in source.items():
                    h5.create_dataset(logical, data=values)

            argv = [
                "roundtrip",
                str(input_h5),
                "--work-dir",
                str(work_dir),
                "--lcp",
                str(LCP_EXE),
                "--xnyzip",
                str(XNYZIP_EXE),
                "--pos-compressor",
                position_codec,
                "--vel-compressor",
                velocity_codec,
                "--position-scale",
                "raw",
                "--pos-abs-eb",
                "1e-5",
                "--vel-abs-eb",
                "1e-3",
                "--lattice-layout",
                "--no-lattice-axis-search",
                "--field-workers",
                "2",
                "--metrics",
                "--force",
            ]
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = particle_main.main(argv)
            self.assertEqual(result, 0, stderr.getvalue())

            manifest = json.loads(
                (work_dir / "manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (work_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["lattice_layout"]["enabled"])
            self.assertEqual(
                manifest["lattice_layout"]["field_scope"],
                "velocities",
            )
            self.assertEqual(
                manifest["compressed_fields"]["positions"]["codec"],
                position_codec,
            )
            for logical in POSITION_FIELDS:
                self.assertNotIn(logical, manifest["compressed_fields"])
            for logical in VELOCITY_FIELDS:
                self.assertEqual(
                    manifest["compressed_fields"][logical]["spatial_layout"],
                    LATTICE_LAYOUT_NAME,
                )
            self.assertTrue(
                all(
                    field["satisfied"]
                    for field in metrics["error_bound_consistency"].values()
                )
            )

            with h5py.File(work_dir / "reconstructed.h5", "r") as rebuilt:
                rebuilt_ids = rebuilt["id"][:]
                source_rows = np.empty(source["id"].size, dtype=np.intp)
                source_rows[source["id"].astype(np.intp)] = np.arange(
                    source["id"].size,
                    dtype=np.intp,
                )
                for logical in (*POSITION_FIELDS, *VELOCITY_FIELDS):
                    expected = source[logical][source_rows[rebuilt_ids]]
                    np.testing.assert_allclose(
                        rebuilt[logical][:],
                        expected,
                        atol=1e-3,
                        rtol=0,
                    )
