from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import main as particle_main
from src.batch import discover_particle_files
from src.cli import build_parser
from src.native_snapshot import (
    BYTES_PER_PARTICLE,
    NATIVE_FORMAT,
    adapt_particle_input,
    read_native_header,
)
from src.preprocess import preprocess
from src.runtime import read_json


class NativeSnapshotTests(unittest.TestCase):
    def test_adapter_exposes_field_major_binary_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path, expected = self._write_partition(
                root,
                snapshot=7,
                rank=1,
                proc_size=2,
                total=4,
                ids=np.array([3, 1, 2, 0], dtype=np.uint64),
            )

            adapted = adapt_particle_input(data_path, root / "adapters")

            self.assertEqual(adapted.original_path, data_path.resolve())
            self.assertEqual(adapted.native_header.npart, 4)
            self.assertEqual(data_path.stat().st_size, 4 * BYTES_PER_PARTICLE)
            with h5py.File(adapted.h5_path, "r") as source:
                self.assertEqual(int(source.attrs["rank"]), 1)
                self.assertEqual(int(source.attrs["proc_size"]), 2)
                self.assertEqual(int(source.attrs["npart_total"]), 4)
                self.assertEqual(int(source.attrs["nsidemesh"]), 810)
                self.assertEqual(int(source.attrs["bitwidth"]), 2147483647)
                for name, values in expected.items():
                    np.testing.assert_array_equal(source[name][:], values)

    def test_header_rejects_payload_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path, _ = self._write_partition(
                root,
                snapshot=7,
                rank=0,
                proc_size=1,
                total=2,
                ids=np.array([0, 1], dtype=np.uint64),
            )
            data_path.write_bytes(data_path.read_bytes()[:-1])

            with self.assertRaisesRegex(RuntimeError, "expected 64"):
                read_native_header(data_path)

    def test_discovery_includes_native_files_in_natural_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_partition(
                root,
                snapshot=7,
                rank=10,
                proc_size=11,
                total=2,
                ids=np.array([1], dtype=np.uint64),
            )
            self._write_partition(
                root,
                snapshot=7,
                rank=2,
                proc_size=11,
                total=2,
                ids=np.array([0], dtype=np.uint64),
            )

            files = discover_particle_files(root)

            self.assertEqual(
                [path.name for path in files],
                ["dat_7.2", "dat_7.10"],
            )

    def test_discovery_rejects_missing_nonempty_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path, _ = self._write_partition(
                root,
                snapshot=7,
                rank=0,
                proc_size=1,
                total=1,
                ids=np.array([0], dtype=np.uint64),
            )
            data_path.unlink()

            with self.assertRaisesRegex(
                RuntimeError,
                "declares 1 particles.*data file.*is missing",
            ):
                discover_particle_files(root)

    def test_preprocess_accepts_native_file_and_records_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path, expected = self._write_partition(
                root,
                snapshot=7,
                rank=0,
                proc_size=1,
                total=4,
                ids=np.array([3, 1, 2, 0], dtype=np.uint64),
            )
            work_dir = root / "work"
            argv = [
                "preprocess",
                str(data_path),
                "--work-dir",
                str(work_dir),
                "--limit",
                "3",
                "--force",
            ]
            args = build_parser(argv).parse_args(argv)

            manifest, raw_paths, _ = preprocess(args)

            self.assertEqual(manifest["count"], 3)
            self.assertEqual(manifest["input_format"], NATIVE_FORMAT)
            self.assertEqual(Path(manifest["input_file"]), data_path.resolve())
            self.assertEqual(manifest["source"]["storage_order"], "field_major")
            self.assertEqual(manifest["position_scale"]["value"], 2147483647)
            np.testing.assert_array_equal(
                np.fromfile(raw_paths["id"], dtype=np.uint64),
                expected["id"][:3],
            )
            np.testing.assert_array_equal(
                np.fromfile(raw_paths["vx"], dtype=np.float32),
                expected["velx"][:3],
            )
            np.testing.assert_allclose(
                np.fromfile(raw_paths["x"], dtype=np.float32),
                expected["posx"][:3].astype(np.float64) / 2147483647,
            )

    def test_directory_merge_accepts_native_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = root / "snapshot"
            outputs = root / "outputs"
            inputs.mkdir()
            self._write_partition(
                inputs,
                snapshot=7,
                rank=0,
                proc_size=2,
                total=4,
                ids=np.array([0, 1], dtype=np.uint64),
            )
            self._write_partition(
                inputs,
                snapshot=7,
                rank=1,
                proc_size=2,
                total=4,
                ids=np.array([2, 3], dtype=np.uint64),
            )
            stdout = StringIO()
            stderr = StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = particle_main.main(
                    [
                        "preprocess",
                        str(inputs),
                        "--merge",
                        "--work-dir",
                        str(outputs),
                        "--force",
                    ]
                )

            self.assertEqual(result, 0, stderr.getvalue())
            manifest = read_json(outputs / "manifest.json")
            self.assertEqual(manifest["count"], 4)
            self.assertEqual(manifest["merge"]["source_file_count"], 2)
            self.assertEqual(len(manifest["merge"]["native_sources"]), 2)
            self.assertEqual(
                [Path(path).name for path in manifest["merge"]["input_files"]],
                ["dat_7.0", "dat_7.1"],
            )
            with h5py.File(outputs / "merged" / "merged.h5", "r") as merged:
                np.testing.assert_array_equal(
                    merged["id"][:],
                    np.arange(4, dtype=np.uint64),
                )

    @staticmethod
    def _write_partition(
        root: Path,
        snapshot: int,
        rank: int,
        proc_size: int,
        total: int,
        ids: np.ndarray,
    ) -> tuple[Path, dict[str, np.ndarray]]:
        count = int(ids.size)
        prefix = np.arange(count, dtype=np.int32) + rank * 100
        fields = {
            "posx": prefix + 1,
            "posy": prefix + 11,
            "posz": prefix + 21,
            "velx": prefix.astype(np.float32) + np.float32(0.25),
            "vely": prefix.astype(np.float32) + np.float32(0.5),
            "velz": prefix.astype(np.float32) + np.float32(0.75),
            "id": ids,
        }
        data_path = root / f"dat_{snapshot}.{rank}"
        with data_path.open("wb") as stream:
            for values in fields.values():
                values.tofile(stream)

        header = [
            str(rank),
            str(proc_size),
            str(total),
            "2.914170e-02",
            "1.215000e+05",
            "0.676600",
            "0.311100",
            "0.688900",
            "6.550993",
            str(count),
            *(["0"] * 31),
            "810",
            "2147483647",
        ]
        if len(header) != 43:
            raise AssertionError(len(header))
        (root / f"cfg_{snapshot}.{rank}").write_text(
            "\n".join(header) + "\n",
            encoding="ascii",
        )
        return data_path, fields


if __name__ == "__main__":
    unittest.main()
