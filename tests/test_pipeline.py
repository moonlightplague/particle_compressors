from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from src.cli import AVAILABLE_COMPRESSORS, build_parser
from src.compress import CompressionSettings, compress
from src.hdf5_io import recombine_h5
from src.manifest import lossy_compressor_from_manifest
from src.metrics import (
    comparison_order_for_reconstructed_rows,
    print_component_summary,
)
from src.preprocess import build_compressed_artifacts


LOGICAL_FIELDS = ("id", "x", "y", "z", "vx", "vy", "vz")
POSITION_FIELDS = ("x", "y", "z")
VELOCITY_FIELDS = ("vx", "vy", "vz")
FIELDS = {
    logical: {
        "h5_path": logical,
        "dtype": "uint64" if logical == "id" else "float32",
    }
    for logical in LOGICAL_FIELDS
}


class CompressorSelectionTests(unittest.TestCase):
    def test_cli_uses_one_lossy_compressor_for_all_fields(self) -> None:
        for codec in ("szo", "sz3", "sperr", "qoz"):
            argv = [
                "roundtrip",
                "input.h5",
                "--lossy-compressor",
                codec,
            ]
            args = build_parser(argv).parse_args(argv)
            self.assertEqual(args.lossy_compressor, codec)
            self.assertFalse(hasattr(args, "position_compressor"))
            self.assertFalse(hasattr(args, "velocity_compressor"))

        self.assertEqual(
            AVAILABLE_COMPRESSORS["lossy_compressor"],
            ("szo", "sz3", "sperr", "qoz"),
        )

    def test_cli_accepts_explicit_field_worker_count(self) -> None:
        argv = [
            "roundtrip",
            "input.h5",
            "--field-workers",
            "3",
        ]
        args = build_parser(argv).parse_args(argv)
        self.assertEqual(args.field_workers, 3)

    def test_settings_preserve_sort_and_lattice_options(self) -> None:
        sorted_settings = CompressionSettings.from_args(
            SimpleNamespace(
                lossy_compressor="szo",
                force=False,
                sort=True,
                lattice_layout=False,
                lattice_min_occupancy=0.8,
                lattice_axis_search=True,
            )
        )
        self.assertTrue(sorted_settings.sort_by_id)
        self.assertFalse(sorted_settings.lattice_requested)

        lattice_settings = CompressionSettings.from_args(
            SimpleNamespace(
                lossy_compressor="sz3",
                force=False,
                sort=False,
                lattice_layout=True,
                lattice_min_occupancy=0.9,
                lattice_axis_search=False,
                field_workers=4,
            )
        )
        self.assertTrue(lattice_settings.sort_by_id)
        self.assertTrue(lattice_settings.lattice_requested)
        self.assertEqual(lattice_settings.lattice_min_occupancy, 0.9)
        self.assertFalse(lattice_settings.lattice_axis_search)
        self.assertEqual(lattice_settings.field_workers, 4)

    def test_artifact_extensions_match_selected_codec(self) -> None:
        root = Path("/tmp/package/compressed")
        szo = build_compressed_artifacts(root, "szo")
        sz3 = build_compressed_artifacts(root, "sz3")
        sperr = build_compressed_artifacts(root, "sperr")
        qoz = build_compressed_artifacts(root, "qoz")

        self.assertEqual(Path(szo["id"]).name, "id.pco")
        self.assertEqual(Path(sz3["id"]).name, "id.pco")
        self.assertEqual(Path(sperr["id"]).name, "id.pco")
        self.assertEqual(Path(qoz["id"]).name, "id.pco")
        for field in (*POSITION_FIELDS, *VELOCITY_FIELDS):
            self.assertEqual(Path(szo[field]).name, f"{field}.szo")
            self.assertEqual(Path(sz3[field]).name, f"{field}.psz")
            self.assertEqual(Path(sperr[field]).name, f"{field}.sperr")
            self.assertEqual(Path(qoz[field]).name, f"{field}.qoz")

    def test_manifest_requires_one_codec_for_every_lossy_field(self) -> None:
        for configured in ("szo", "sz3", "sperr", "qoz"):
            self.assertEqual(
                lossy_compressor_from_manifest(
                    {"compressors": {"lossy": configured}}
                ),
                configured,
            )

        inferred = {
            "compressed_fields": {
                field: {"codec": "pysz"}
                for field in (*POSITION_FIELDS, *VELOCITY_FIELDS)
            }
        }
        self.assertEqual(lossy_compressor_from_manifest(inferred), "sz3")

        inferred["compressed_fields"] = {
            field: {"codec": "sperr"}
            for field in (*POSITION_FIELDS, *VELOCITY_FIELDS)
        }
        self.assertEqual(lossy_compressor_from_manifest(inferred), "sperr")

        inferred["compressed_fields"] = {
            field: {"codec": "qoz"}
            for field in (*POSITION_FIELDS, *VELOCITY_FIELDS)
        }
        self.assertEqual(lossy_compressor_from_manifest(inferred), "qoz")


class StableSortingTests(unittest.TestCase):
    def test_sort_applies_one_stable_id_order_to_every_field(self) -> None:
        count = 5
        source = {
            "id": np.array([30, 10, 20, 10, 40], dtype=np.uint64),
            "x": np.array([0, 1, 2, 3, 4], dtype=np.float32),
            "y": np.array([10, 11, 12, 13, 14], dtype=np.float32),
            "z": np.array([20, 21, 22, 23, 24], dtype=np.float32),
            "vx": np.array([30, 31, 32, 33, 34], dtype=np.float32),
            "vy": np.array([40, 41, 42, 43, 44], dtype=np.float32),
            "vz": np.array([50, 51, 52, 53, 54], dtype=np.float32),
        }
        expected_order = np.array([1, 3, 2, 0, 4])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "preprocessed"
            compressed_dir = root / "compressed"
            raw_dir.mkdir()
            compressed_dir.mkdir()
            raw_paths = {}
            for logical, values in source.items():
                path = raw_dir / f"{logical}.raw"
                values.tofile(path)
                raw_paths[logical] = str(path)

            artifacts = build_compressed_artifacts(compressed_dir, "szo")
            manifest = {
                "format_version": 2,
                "count": count,
                "fields": {
                    logical: {"dtype": str(values.dtype)}
                    for logical, values in source.items()
                },
                "field_error_bounds": {
                    **{
                        field: {"abs": 0.1, "compressor_abs": 0.1}
                        for field in (*POSITION_FIELDS, *VELOCITY_FIELDS)
                    },
                    "id": {"abs": 0.0, "compressor_abs": 0.0},
                },
                "artifacts": {
                    "preprocessed": raw_paths,
                    "compressed": artifacts,
                },
                "compressed_fields": {},
                "sizes": {"selected_original_payload_bytes": 160},
                "root_attrs": {},
            }
            args = SimpleNamespace(
                work_dir=str(root),
                force=False,
                lossless="pcodec",
                lossy_compressor="szo",
                sort=True,
                lattice_layout=False,
                lattice_min_occupancy=0.8,
                lattice_axis_search=True,
            )
            captured = {}

            def fake_integer(
                codec,
                raw_path,
                dtype,
                compressed_path,
                field_name,
                field_count,
                force,
            ):
                captured[field_name] = np.fromfile(raw_path, dtype=dtype)
                Path(compressed_path).write_bytes(b"id")
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": field_count,
                    "path": compressed_path,
                    "bytes": 2,
                }

            def fake_lossy(
                codec,
                raw_path,
                dtype,
                compressed_path,
                field_name,
                field_count,
                bound,
                force,
            ):
                captured[field_name] = np.fromfile(raw_path, dtype=dtype)
                Path(compressed_path).write_bytes(field_name.encode())
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": field_count,
                    "path": compressed_path,
                    "bytes": len(field_name),
                    "abs_error_bound": bound,
                }

            with patch(
                "src.compress.compress_integer_raw",
                side_effect=fake_integer,
            ), patch(
                "src.compress.compress_lossy_raw",
                side_effect=fake_lossy,
            ), patch("src.compress.update_compressed_size_metrics"):
                result = compress(args, manifest, raw_paths)

            for logical in LOGICAL_FIELDS:
                np.testing.assert_array_equal(
                    captured[logical],
                    source[logical][expected_order],
                )
            self.assertEqual(
                result["ordering"]["reconstructed_rows"]["mapping"],
                "id_sorted",
            )
            self.assertTrue(result["particle_sort"]["stable"])
            self.assertEqual(result["format_version"], 3)

    def test_metrics_read_recorded_sort_permutation(self) -> None:
        expected_order = np.array([2, 0, 3, 1], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            order_path = root / "id_sort_order.i64.raw"
            expected_order.tofile(order_path)
            manifest = {
                "ordering": {
                    "reconstructed_rows": {
                        "mapping": "id_sorted",
                        "original_row_order_restored": False,
                        "temporary_permutation_artifact": "id_sort_order",
                        "temporary_permutation_dtype": "int64",
                    },
                },
                "artifacts": {
                    "preprocessed": {"id_sort_order": str(order_path)},
                },
            }
            original_path = root / "original.h5"
            reconstructed_path = root / "reconstructed.h5"
            with h5py.File(original_path, "w") as original, h5py.File(
                reconstructed_path,
                "w",
            ) as reconstructed:
                order, source = comparison_order_for_reconstructed_rows(
                    original,
                    reconstructed,
                    manifest,
                    expected_order.size,
                )

            np.testing.assert_array_equal(order, expected_order)
            self.assertEqual(source, "temporary_id_sort_order")


class ReconstructionAndReportingTests(unittest.TestCase):
    def test_recombine_preserves_schema_attributes_and_position_scale(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            values = {
                "id": np.array([2, 1], dtype=np.uint64),
                "x": np.array([1.0, 2.0], dtype=np.float32),
                "y": np.array([3.0, 4.0], dtype=np.float32),
                "z": np.array([5.0, 6.0], dtype=np.float32),
                "vx": np.array([0.1, 0.2], dtype=np.float32),
                "vy": np.array([0.3, 0.4], dtype=np.float32),
                "vz": np.array([0.5, 0.6], dtype=np.float32),
            }
            for logical, data in values.items():
                path = root / f"{logical}.raw"
                data.tofile(path)
                paths[logical] = str(path)
            fields = {
                logical: {
                    "h5_path": f"particles/{logical}",
                    "dtype": (
                        "uint64"
                        if logical == "id"
                        else ("int32" if logical in POSITION_FIELDS else "float32")
                    ),
                    "attrs": {},
                }
                for logical in LOGICAL_FIELDS
            }
            manifest = {
                "count": 2,
                "position_scale": {"value": 10.0},
                "fields": fields,
                "root_attrs": {
                    "step": {"dtype": "int32", "shape": [], "value": 7}
                },
            }
            output = root / "reconstructed.h5"
            recombine_h5(manifest, paths, output)

            with h5py.File(output, "r") as h5:
                self.assertEqual(h5.attrs["step"], 7)
                np.testing.assert_array_equal(
                    h5["particles/x"][:],
                    np.array([10, 20], dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    h5["particles/id"][:],
                    values["id"],
                )

    def test_component_summary_reports_each_lossy_field(self) -> None:
        report = {
            "count": 4,
            "fields": FIELDS,
            "sizes": {
                "compressed_components_bytes": {
                    "compressed/x.psz": 4,
                    "compressed/y.psz": 8,
                    "compressed/z.psz": 16,
                    "compressed/id.pco": 8,
                    "compressed/vx.psz": 4,
                    "compressed/vy.psz": 8,
                    "compressed/vz.psz": 16,
                },
            },
        }
        output = StringIO()
        with redirect_stdout(output):
            print_component_summary(report)

        lines = output.getvalue().splitlines()
        self.assertIn(
            "  x: CR=4, original_bytes=16, compressed_bytes=4",
            lines,
        )
        self.assertIn(
            "  vx: CR=4, original_bytes=16, compressed_bytes=4",
            lines,
        )


if __name__ == "__main__":
    unittest.main()
