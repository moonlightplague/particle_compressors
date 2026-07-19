from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

import main as particle_main
from src.cli import AVAILABLE_COMPRESSORS, build_parser
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.compress import CompressionSettings
from src.compress import compress as compress_pipeline
from src.compress import (
    compress_chunked_lcp_triplet,
    compress_lossy_raw,
    compress_pcodec_raw,
    compress_szo_raw,
    reorder_raw,
)
from src.decompress import (
    decompress_lossy_raw,
    decompress_pcodec_raw,
)
from src.decompress import (
    position_compressor_from_manifest,
    recombine_h5,
    velocity_compressor_from_manifest,
)
from src.metrics import (
    comparison_order_for_reconstructed_rows,
    print_component_summary,
)
from src.preprocess import preprocess as preprocess_pipeline


FIELDS = {
    logical: {"h5_path": logical, "dtype": "uint64" if logical == "id" else "float32"}
    for logical in ("id", "x", "y", "z", "vx", "vy", "vz")
}


class CompressionOrderingTests(unittest.TestCase):
    def test_velocity_lcp_requires_lcp_positions_at_compression_entry(
        self,
    ) -> None:
        for position_codec in ("sz3", "szo"):
            with self.subTest(position_codec=position_codec):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "--vel-compressor lcp requires --pos-compressor lcp",
                ):
                    compress_pipeline(
                        SimpleNamespace(
                            work_dir="unused",
                            pos_compressor=position_codec,
                            vel_compressor="lcp",
                            vel_chunk_size=0,
                            vel_chunk_workers=0,
                            force=False,
                            sort=False,
                        ),
                        {},
                        {},
                        SimpleNamespace(lcp=Path("lcp")),
                    )

    def test_cli_rejects_velocity_lcp_with_fieldwise_positions(
        self,
    ) -> None:
        for position_codec in ("sz3", "szo"):
            with self.subTest(position_codec=position_codec):
                stderr = StringIO()
                with redirect_stderr(stderr):
                    result = particle_main.main(
                        [
                            "compress",
                            "input.h5",
                            "--pos-compressor",
                            position_codec,
                            "--vel-compressor",
                            "lcp",
                        ]
                    )

                self.assertEqual(result, 2)
                self.assertIn(
                    "error: --vel-compressor lcp requires "
                    "--pos-compressor lcp.",
                    stderr.getvalue(),
                )

    def test_preprocess_rejects_deprecated_combination_before_writing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            work_dir = root / "work"
            argv = [
                "preprocess",
                str(root / "missing.h5"),
                "--work-dir",
                str(work_dir),
                "--pos-compressor",
                "sz3",
                "--vel-compressor",
                "lcp",
            ]
            args = build_parser(argv).parse_args(argv)

            with self.assertRaisesRegex(
                RuntimeError,
                "--vel-compressor lcp requires --pos-compressor lcp",
            ):
                preprocess_pipeline(args)

            self.assertFalse(work_dir.exists())

    def test_compression_settings_accepts_all_remaining_combinations(
        self,
    ) -> None:
        combinations = (
            ("lcp", "sz3"),
            ("lcp", "szo"),
            ("lcp", "lcp"),
            ("sz3", "sz3"),
            ("sz3", "szo"),
            ("szo", "sz3"),
            ("szo", "szo"),
        )
        for position_codec, velocity_codec in combinations:
            with self.subTest(
                position_codec=position_codec,
                velocity_codec=velocity_codec,
            ):
                settings = CompressionSettings.from_args(
                    SimpleNamespace(
                        pos_compressor=position_codec,
                        vel_compressor=velocity_codec,
                        vel_chunk_size=0,
                        vel_chunk_workers=0,
                        force=False,
                        sort=False,
                    )
                )
                self.assertEqual(settings.position_codec, position_codec)
                self.assertEqual(settings.velocity_codec, velocity_codec)

    def test_compress_reorders_id_and_sz3_velocities_and_omits_position_order(self) -> None:
        count = 5
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.int32)
        source = {
            "x": np.array([10, 20, 30, 40, 50], dtype=np.float32),
            "y": np.array([11, 21, 31, 41, 51], dtype=np.float32),
            "z": np.array([12, 22, 32, 42, 52], dtype=np.float32),
            "id": np.array([901, 117, 502, 330, 774], dtype=np.uint64),
            "vx": np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "vy": np.array([6, 7, 8, 9, 10], dtype=np.float32),
            "vz": np.array([11, 12, 13, 14, 15], dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preprocessed = root / "preprocessed"
            compressed = root / "compressed"
            preprocessed.mkdir()
            compressed.mkdir()
            raw_paths = {}
            for logical, values in source.items():
                path = preprocessed / f"{logical}.raw"
                values.tofile(path)
                raw_paths[logical] = str(path)

            artifacts = {
                "positions": str(compressed / "positions.lcp"),
                "id": str(compressed / "id.pco"),
                "vx": str(compressed / "vx.psz"),
                "vy": str(compressed / "vy.psz"),
                "vz": str(compressed / "vz.psz"),
            }
            manifest = {
                "count": count,
                "fields": {
                    logical: {"dtype": str(values.dtype)}
                    for logical, values in source.items()
                },
                "error_bounds": {"positions_lcp_abs": 0.01},
                "field_error_bounds": {
                    logical: {"abs": 0.01}
                    for logical in ("vx", "vy", "vz")
                },
                "artifacts": {
                    "preprocessed": raw_paths,
                    "compressed": artifacts,
                },
                "compressed_fields": {},
                "sizes": {"selected_original_payload_bytes": 160},
            }
            args = SimpleNamespace(
                work_dir=str(root),
                force=False,
                lossless="pcodec",
                vel_compressor="sz3",
            )
            captured = {}

            def fake_run_command(argv):
                Path(argv[argv.index("-z") + 1]).write_bytes(b"lcp")
                position_order.tofile(argv[-1])

            def fake_integer(codec, raw_path, dtype, compressed_path, field_name, *unused):
                captured[field_name] = np.fromfile(raw_path, dtype=np.dtype(dtype))
                Path(compressed_path).write_bytes(b"integer")
                return {"field": field_name, "codec": codec, "dtype": dtype, "count": count}

            def fake_velocity(codec, raw_path, dtype, compressed_path, field_name, *unused):
                self.assertEqual(codec, "sz3")
                captured[field_name] = np.fromfile(raw_path, dtype=np.dtype(dtype))
                Path(compressed_path).write_bytes(b"velocity")
                return {"field": field_name, "codec": "pysz", "dtype": dtype, "count": count}

            with patch("src.lcp_codec.run_command", side_effect=fake_run_command), patch(
                "src.compress.compress_integer_raw", side_effect=fake_integer
            ), patch("src.compress.compress_lossy_raw", side_effect=fake_velocity), patch(
                "src.compress.update_compressed_size_metrics"
            ):
                result = compress_pipeline(
                    args,
                    manifest,
                    raw_paths,
                    SimpleNamespace(lcp=Path("lcp")),
                )

            self.assertNotIn("order", result["compressed_fields"])
            self.assertNotIn("order", result["artifacts"]["compressed"])
            self.assertFalse(
                result["ordering"]["reconstructed_rows"][
                    "position_permutation_stored"
                ]
            )
            for logical in ("id", "vx", "vy", "vz"):
                np.testing.assert_array_equal(captured[logical], source[logical][position_order])

    def test_reorder_raw_applies_lcp_position_permutation_without_casting(self) -> None:
        order = np.array([2, 0, 3, 1], dtype=np.intp)
        cases = {
            "uint64": np.array([2**63, 7, 2**64 - 1, 11], dtype=np.uint64),
            "float64": np.array([0.25, -3.5, 9.75, 1.0], dtype=np.float64),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for dtype_name, values in cases.items():
                with self.subTest(dtype=dtype_name):
                    source_path = root / f"{dtype_name}.raw"
                    output_path = root / f"{dtype_name}.ordered.raw"
                    values.tofile(source_path)
                    reorder_raw(
                        str(source_path),
                        dtype_name,
                        output_path,
                        values.size,
                        order,
                        False,
                    )
                    np.testing.assert_array_equal(
                        np.fromfile(output_path, dtype=np.dtype(dtype_name)),
                        values[order],
                    )

    def test_id_sort_reorders_every_field_before_fieldwise_compression(self) -> None:
        count = 5
        source = {
            "id": np.array([30, 10, 30, 20, 10], dtype=np.uint64),
            "x": np.array([10, 20, 30, 40, 50], dtype=np.float32),
            "y": np.array([11, 21, 31, 41, 51], dtype=np.float32),
            "z": np.array([12, 22, 32, 42, 52], dtype=np.float32),
            "vx": np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "vy": np.array([6, 7, 8, 9, 10], dtype=np.float32),
            "vz": np.array([11, 12, 13, 14, 15], dtype=np.float32),
        }
        expected_order = np.array([1, 4, 3, 0, 2], dtype=np.intp)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preprocessed = root / "preprocessed"
            compressed = root / "compressed"
            preprocessed.mkdir()
            compressed.mkdir()
            raw_paths = {}
            for logical, values in source.items():
                path = preprocessed / f"{logical}.raw"
                values.tofile(path)
                raw_paths[logical] = str(path)

            artifacts = {
                "id": str(compressed / "id.pco"),
                **{
                    logical: str(
                        compressed
                        / f"{logical}.{'psz' if logical in POSITION_FIELDS else 'szo'}"
                    )
                    for logical in POSITION_FIELDS + VELOCITY_FIELDS
                },
            }
            manifest = {
                "count": count,
                "fields": {
                    logical: {"dtype": str(values.dtype)}
                    for logical, values in source.items()
                },
                "field_error_bounds": {
                    logical: {"abs": 0.01, "compressor_abs": 0.01}
                    for logical in POSITION_FIELDS + VELOCITY_FIELDS
                },
                "artifacts": {
                    "preprocessed": raw_paths,
                    "compressed": artifacts,
                },
                "compressed_fields": {},
                "sizes": {"selected_original_payload_bytes": 160},
            }
            args = SimpleNamespace(
                work_dir=str(root),
                force=False,
                lossless="pcodec",
                pos_compressor="sz3",
                vel_compressor="szo",
                vel_chunk_size=0,
                vel_chunk_workers=0,
                sort=True,
            )
            captured = {}

            def fake_compress(
                codec,
                raw_path,
                dtype,
                compressed_path,
                field_name,
                *unused,
            ):
                captured[field_name] = np.fromfile(
                    raw_path,
                    dtype=np.dtype(dtype),
                )
                Path(compressed_path).write_bytes(field_name.encode())
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": count,
                }

            with patch(
                "src.compress.compress_integer_raw",
                side_effect=fake_compress,
            ), patch(
                "src.compress.compress_lossy_raw",
                side_effect=fake_compress,
            ), patch("src.compress.update_compressed_size_metrics"):
                result = compress_pipeline(
                    args,
                    manifest,
                    raw_paths,
                    SimpleNamespace(lcp=Path("lcp")),
                )

            for logical, values in source.items():
                np.testing.assert_array_equal(
                    captured[logical],
                    values[expected_order],
                )
            np.testing.assert_array_equal(
                np.fromfile(
                    raw_paths["id_sort_order"],
                    dtype=np.int64,
                ),
                expected_order,
            )
            self.assertEqual(
                result["ordering"]["reconstructed_rows"]["mapping"],
                "id_sorted",
            )
            self.assertFalse(
                result["ordering"]["reconstructed_rows"][
                    "original_row_order_restored"
                ]
            )
            self.assertEqual(
                result["ordering"]["reconstructed_rows"][
                    "temporary_permutation_dtype"
                ],
                "int64",
            )
            self.assertEqual(
                result["particle_sort"],
                {
                    "requested": True,
                    "enabled": True,
                    "key": "id",
                    "direction": "ascending",
                    "stable": True,
                },
            )

    def test_id_sort_is_disabled_when_positions_use_lcp(self) -> None:
        for position_codec, velocity_codec in (
            ("lcp", "sz3"),
            ("lcp", "szo"),
            ("lcp", "lcp"),
        ):
            with self.subTest(
                position_codec=position_codec,
                velocity_codec=velocity_codec,
            ):
                settings = CompressionSettings.from_args(
                    SimpleNamespace(
                        pos_compressor=position_codec,
                        vel_compressor=velocity_codec,
                        vel_chunk_size=0,
                        vel_chunk_workers=0,
                        force=False,
                        sort=True,
                    )
                )
                self.assertTrue(settings.sort_requested)
                self.assertFalse(settings.sort_by_id)

class RecombinationTests(unittest.TestCase):
    def test_recombine_accepts_shared_lcp_position_order_without_sidecar(self) -> None:
        count = 5
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.intp)
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
                values[position_order].tofile(path)
                dec_paths[logical] = str(path)

            manifest = {
                "count": count,
                "position_scale": {"value": 1.0},
                "fields": FIELDS,
                "compressors": {"velocities": "sz3"},
                "compressed_fields": {
                    logical: {"dtype": str(original[logical].dtype)}
                    for logical in ("id", "vx", "vy", "vz")
                },
                "ordering": {
                    "reconstructed_rows": {
                        "mapping": "lcp_position_sorted",
                        "original_row_order_restored": False,
                        "position_permutation_stored": False,
                    }
                },
            }
            output = root / "reconstructed.h5"
            recombine_h5(manifest, dec_paths, output)

            with h5py.File(output, "r") as h5:
                for logical, values in original.items():
                    np.testing.assert_array_equal(h5[logical][:], values[position_order])

    def test_recombine_keeps_legacy_velocity_lcp_packages_readable(
        self,
    ) -> None:
        count = 5
        velocity_order = np.array([1, 4, 0, 3, 2], dtype=np.intp)
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
                values[velocity_order].tofile(path)
                dec_paths[logical] = str(path)

            manifest = {
                "count": count,
                "position_scale": {"value": 1.0},
                "fields": FIELDS,
                "compressors": {"positions": "sz3", "velocities": "lcp"},
                "compressed_fields": {
                    "velocities": {"codec": "lcp", "dtype": "float32"},
                },
                "ordering": {
                    "reconstructed_rows": {
                        "mapping": "lcp_velocity_sorted",
                        "original_row_order_restored": False,
                        "velocity_permutation_stored": False,
                    }
                },
            }
            output = root / "reconstructed.h5"
            recombine_h5(manifest, dec_paths, output)

            with h5py.File(output, "r") as h5:
                for logical, values in original.items():
                    np.testing.assert_array_equal(h5[logical][:], values[velocity_order])

class SZoPipelineTests(unittest.TestCase):
    def test_cli_exposes_szo_only_as_a_lossy_compressor(self) -> None:
        argv = [
            "roundtrip",
            "input.h5",
            "--pos-compressor",
            "szo",
            "--vel-compressor",
            "szo",
            "--lossless",
            "pcodec",
        ]
        args = build_parser(argv).parse_args(argv)
        self.assertEqual(args.pos_compressor, "szo")
        self.assertEqual(args.vel_compressor, "szo")
        self.assertEqual(args.lossless, "pcodec")
        self.assertFalse(args.sort)
        self.assertIsInstance(args.rel_eb, float)
        self.assertIn("szo", AVAILABLE_COMPRESSORS["pos_compressor"])
        self.assertIn("szo", AVAILABLE_COMPRESSORS["vel_compressor"])
        self.assertEqual(AVAILABLE_COMPRESSORS["lossless"], ("pcodec",))

    def test_cli_accepts_id_sort_for_fieldwise_compressors(self) -> None:
        argv = [
            "roundtrip",
            "input.h5",
            "--pos-compressor",
            "sz3",
            "--vel-compressor",
            "szo",
            "--sort",
        ]

        args = build_parser(argv).parse_args(argv)

        self.assertTrue(args.sort)

    def test_preprocess_assigns_szo_only_to_lossy_field_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_h5 = root / "input.h5"
            work_dir = root / "work"
            with h5py.File(input_h5, "w") as h5:
                h5.create_dataset("id", data=np.arange(16, dtype=np.uint64))
                for offset, logical in enumerate(("x", "y", "z", "vx", "vy", "vz")):
                    h5.create_dataset(
                        logical,
                        data=np.arange(16, dtype=np.float32) + offset,
                    )

            argv = [
                "preprocess",
                str(input_h5),
                "--work-dir",
                str(work_dir),
                "--pos-compressor",
                "szo",
                "--vel-compressor",
                "szo",
                "--lossless",
                "pcodec",
                "--force",
            ]
            args = build_parser(argv).parse_args(argv)
            manifest, _, _ = preprocess_pipeline(args)
            artifacts = manifest["artifacts"]["compressed"]

            self.assertEqual(Path(artifacts["id"]).name, "id.pco")
            for logical in POSITION_FIELDS + VELOCITY_FIELDS:
                self.assertEqual(Path(artifacts[logical]).suffix, ".szo")
            self.assertEqual(
                manifest["compressors"],
                {"positions": "szo", "velocities": "szo", "lossless": "pcodec"},
            )

    def test_szo_roundtrips_lossy_float_fields(self) -> None:
        class FakeConfig:
            def __init__(self, shape):
                self.shape = shape
                self.errorBoundMode = None
                self.absErrorBound = None

        class FakeErrorBoundMode:
            ABS = "abs"

        class FakeAlgorithm:
            LORENZO_REG = "lorenzo_reg"

        test_case = self

        class FakeSZo:
            @staticmethod
            def compress(data, config, copy):
                test_case.assertTrue(copy)
                test_case.assertEqual(config.errorBoundMode, FakeErrorBoundMode.ABS)
                test_case.assertEqual(config.cmprAlgo, FakeAlgorithm.LORENZO_REG)
                payload = np.ascontiguousarray(data).view(np.uint8)
                return payload, data.nbytes / payload.size

            @staticmethod
            def decompress(compressed, dtype, shape):
                data = np.frombuffer(compressed.tobytes(), dtype=dtype).copy()
                return data.reshape(shape), FakeConfig(shape)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for dtype_name in ("float32", "float64"):
                with self.subTest(dtype=dtype_name), patch(
                    "src.raw_codecs.load_pyszo",
                    return_value=(FakeSZo, FakeConfig, FakeErrorBoundMode, FakeAlgorithm),
                ):
                    dtype = np.dtype(dtype_name)
                    source = np.array([0.125, -3.5, 9.75, 1.0], dtype=dtype)
                    raw_path = root / f"{dtype_name}.raw"
                    compressed_path = root / f"{dtype_name}.szo"
                    output_path = root / f"{dtype_name}.out.raw"
                    source.tofile(raw_path)
                    field = compress_lossy_raw(
                        "szo",
                        str(raw_path),
                        dtype_name,
                        str(compressed_path),
                        dtype_name,
                        source.size,
                        0.01,
                        False,
                    )
                    self.assertEqual(field["codec"], "szo")
                    self.assertEqual(field["abs_error_bound"], 0.01)
                    decompress_lossy_raw(field, str(output_path), False)
                    np.testing.assert_array_equal(np.fromfile(output_path, dtype=dtype), source)

            integers = np.arange(4, dtype=np.int32)
            integer_path = root / "integer.raw"
            integers.tofile(integer_path)
            with self.assertRaisesRegex(RuntimeError, "expected float32/float64"):
                compress_szo_raw(
                    str(integer_path),
                    "int32",
                    str(root / "integer.szo"),
                    "id",
                    integers.size,
                    0.01,
                    False,
                )

class ChunkedLCPTests(unittest.TestCase):
    def test_cli_accepts_lcp_velocity_compressor(self) -> None:
        argv = [
            "roundtrip",
            "input.h5",
            "--pos-compressor",
            "lcp",
            "--vel-compressor",
            "lcp",
            "--vel-chunk-size",
            "32768",
            "--vel-chunk-workers",
            "4",
        ]
        args = build_parser(argv).parse_args(argv)
        self.assertEqual(args.vel_compressor, "lcp")
        self.assertEqual(args.vel_chunk_size, 32768)
        self.assertEqual(args.vel_chunk_workers, 4)

    def test_parallel_chunk_compression_is_byte_deterministic(self) -> None:
        count = 12
        chunk_size = 3
        values = tuple(
            np.arange(count, dtype=np.float32) + axis * 100
            for axis in range(3)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = []
            for axis, data in enumerate(values):
                path = root / f"input-{axis}.raw"
                data.tofile(path)
                inputs.append(str(path))

            def fake_run_command(argv):
                input_index = argv.index("-i") + 1
                if "-2" in argv:
                    shape_index = argv.index("-2")
                    chunks = int(argv[shape_index + 1])
                    values_per_chunk = int(argv[shape_index + 2])
                    encoded_count = chunks * values_per_chunk
                    order = np.concatenate(
                        [
                            np.arange(values_per_chunk, dtype=np.int32)
                            + chunk * values_per_chunk
                            for chunk in range(chunks)
                        ]
                    )
                else:
                    encoded_count = int(argv[argv.index("-1") + 1])
                    order = np.arange(encoded_count, dtype=np.int32)
                payload = b"".join(
                    np.fromfile(argv[input_index + axis], dtype=np.float32, count=encoded_count)
                    .tobytes()
                    for axis in range(3)
                )
                Path(argv[argv.index("-z") + 1]).write_bytes(payload)
                order.tofile(argv[-1])

            outputs = []
            with patch("src.lcp_codec.LCP_CHUNK_BATCH_VALUES", 6), patch(
                "src.lcp_codec.run_command", side_effect=fake_run_command
            ):
                for workers in (1, 2):
                    run = root / f"workers-{workers}"
                    run.mkdir()
                    archive = run / "velocities.lcp"
                    order = run / "order.raw"
                    metadata = compress_chunked_lcp_triplet(
                        SimpleNamespace(lcp=Path("lcp")),
                        tuple(inputs),
                        str(archive),
                        count,
                        chunk_size,
                        0.01,
                        order,
                        False,
                        workers,
                    )
                    outputs.append((archive.read_bytes(), order.read_bytes(), metadata))

            self.assertEqual(outputs[0][0], outputs[1][0])
            self.assertEqual(outputs[0][1], outputs[1][1])
            self.assertEqual(outputs[0][2]["lcp_workers"], 1)
            self.assertEqual(outputs[1][2]["lcp_workers"], 2)

    def test_all_lcp_chunking_uses_position_order_and_chunk_local_velocity_orders(self) -> None:
        count = 5
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.int32)
        local_orders = (
            np.array([2, 0, 1], dtype=np.int32),
            np.array([1, 0], dtype=np.int32),
        )
        source = {
            "x": np.array([10, 20, 30, 40, 50], dtype=np.float32),
            "y": np.array([11, 21, 31, 41, 51], dtype=np.float32),
            "z": np.array([12, 22, 32, 42, 52], dtype=np.float32),
            "id": np.array([901, 117, 502, 330, 774], dtype=np.uint64),
            "vx": np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "vy": np.array([6, 7, 8, 9, 10], dtype=np.float32),
            "vz": np.array([11, 12, 13, 14, 15], dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preprocessed = root / "preprocessed"
            compressed = root / "compressed"
            preprocessed.mkdir()
            compressed.mkdir()
            raw_paths = {}
            for logical, values in source.items():
                path = preprocessed / f"{logical}.raw"
                values.tofile(path)
                raw_paths[logical] = str(path)
                if logical in ("vx", "vy", "vz"):
                    raw_paths[f"{logical}_lcp"] = str(path)

            artifacts = {
                "positions": str(compressed / "positions.lcp"),
                "id": str(compressed / "id.pco"),
                "velocities": str(compressed / "velocities.lcp"),
                "velocity_order": str(compressed / "velocity_order.pco"),
            }
            manifest = {
                "count": count,
                "fields": {
                    logical: {"dtype": str(values.dtype)}
                    for logical, values in source.items()
                },
                "error_bounds": {
                    "positions_lcp_abs": 0.01,
                    "velocities_lcp_abs": 0.01,
                },
                "artifacts": {
                    "preprocessed": raw_paths,
                    "compressed": artifacts,
                },
                "compressed_fields": {},
                "sizes": {"selected_original_payload_bytes": 160},
            }
            args = SimpleNamespace(
                work_dir=str(root),
                force=False,
                lossless="pcodec",
                pos_compressor="lcp",
                vel_compressor="lcp",
                vel_chunk_size=3,
                vel_chunk_workers=1,
            )
            captured_integer = {}
            captured_velocity_chunks = []

            def fake_run_command(argv):
                compressed_path = Path(argv[argv.index("-z") + 1])
                order_path = Path(argv[-1])
                chunk_count = int(argv[argv.index("-1") + 1])
                if compressed_path == Path(artifacts["positions"]):
                    compressed_path.write_bytes(b"positions")
                    position_order.tofile(order_path)
                    return

                input_index = argv.index("-i") + 1
                inputs = tuple(
                    np.fromfile(argv[input_index + axis], dtype=np.float32, count=chunk_count)
                    for axis in range(3)
                )
                chunk_index = len(captured_velocity_chunks)
                captured_velocity_chunks.append(inputs)
                compressed_path.write_bytes(f"chunk-{chunk_index}".encode())
                local_orders[chunk_index].tofile(order_path)

            def fake_integer(codec, raw_path, dtype, compressed_path, field_name, *unused):
                captured_integer[field_name] = np.fromfile(raw_path, dtype=np.dtype(dtype))
                payload = b"integer"
                Path(compressed_path).write_bytes(payload)
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": count,
                    "path": compressed_path,
                    "bytes": len(payload),
                }

            with patch("src.lcp_codec.run_command", side_effect=fake_run_command), patch(
                "src.compress.compress_integer_raw", side_effect=fake_integer
            ), patch("src.compress.update_compressed_size_metrics"):
                result = compress_pipeline(
                    args,
                    manifest,
                    raw_paths,
                    SimpleNamespace(lcp=Path("lcp")),
                )

            np.testing.assert_array_equal(captured_integer["id"], source["id"][position_order])
            np.testing.assert_array_equal(
                captured_integer["velocity_order"], np.concatenate(local_orders)
            )
            for axis, logical in enumerate(("vx", "vy", "vz")):
                expected = source[logical][position_order]
                np.testing.assert_array_equal(captured_velocity_chunks[0][axis], expected[:3])
                np.testing.assert_array_equal(captured_velocity_chunks[1][axis], expected[3:])
            velocity_order_field = result["compressed_fields"]["velocity_order"]
            velocity_field = result["compressed_fields"]["velocities"]
            self.assertEqual(
                velocity_order_field["index_scope"],
                "chunk_local",
            )
            self.assertEqual(
                velocity_order_field["order_bits_per_particle"],
                2,
            )
            self.assertEqual(
                velocity_field["container"],
                "chunked_lcp_v2",
            )
            self.assertEqual(result["format_version"], 4)

    def test_chunk_local_velocity_order_recombines_each_particle_correspondence(self) -> None:
        count = 5
        chunk_size = 3
        local_order = np.array([2, 0, 1, 1, 0], dtype=np.int32)
        global_order = np.array([2, 0, 1, 4, 3], dtype=np.intp)
        canonical = {
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
            for logical, values in canonical.items():
                path = root / f"{logical}.raw"
                encoded = values[global_order] if logical in ("vx", "vy", "vz") else values
                encoded.tofile(path)
                dec_paths[logical] = str(path)
            order_path = root / "velocity_order.raw"
            local_order.tofile(order_path)
            dec_paths["velocity_order"] = str(order_path)

            manifest = {
                "count": count,
                "position_scale": {"value": 1.0},
                "fields": FIELDS,
                "compressors": {"positions": "lcp", "velocities": "lcp"},
                "compressed_fields": {
                    "velocities": {"codec": "lcp", "dtype": "float32"},
                    "velocity_order": {"dtype": "int32", "chunk_size": chunk_size},
                },
            }
            output = root / "reconstructed.h5"
            recombine_h5(manifest, dec_paths, output)

            with h5py.File(output, "r") as h5:
                for logical, expected in canonical.items():
                    np.testing.assert_array_equal(h5[logical][:], expected)

class CodecAndManifestTests(unittest.TestCase):
    def test_pcodec_bitpacks_int32_chunk_order_to_theoretical_width(self) -> None:
        count = 1 << 15
        order = np.random.default_rng(1234).permutation(count).astype(np.int32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "order.i32.raw"
            compressed = root / "order.pco"
            restored = root / "order.restored.raw"
            order.tofile(raw)
            field = compress_pcodec_raw(
                str(raw), "int32", str(compressed), "velocity_order", count, False
            )
            decompress_pcodec_raw(field, str(restored), False)

            theoretical_bytes = count * 15 // 8
            self.assertLessEqual(field["bytes"], theoretical_bytes + 128)
            np.testing.assert_array_equal(np.fromfile(restored, dtype=np.int32), order)

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
        self.assertEqual(position_compressor_from_manifest({}), "lcp")
        self.assertEqual(
            position_compressor_from_manifest({"compressors": {"positions": "sz3"}}),
            "sz3",
        )
        self.assertEqual(
            velocity_compressor_from_manifest(
                {
                    "compressed_fields": {
                        logical: {"codec": "szo"} for logical in VELOCITY_FIELDS
                    }
                }
            ),
            "szo",
        )
        self.assertEqual(
            position_compressor_from_manifest(
                {
                    "compressed_fields": {
                        logical: {"codec": "szo"} for logical in POSITION_FIELDS
                    }
                }
            ),
            "szo",
        )
        self.assertEqual(
            position_compressor_from_manifest(
                {
                    "compressors": {"positions": "sz3"},
                    "artifacts": {"compressed": {"positions": "positions.lcp"}},
                }
            ),
            "lcp",
        )


class MetricReportingTests(unittest.TestCase):
    def test_fieldwise_triplets_report_separate_position_ratios(self) -> None:
        report = self._report("sz3", "szo")

        output = StringIO()
        with redirect_stdout(output):
            print_component_summary(report)

        lines = output.getvalue().splitlines()
        self.assertIn("  x: CR=4, original_bytes=16, compressed_bytes=4", lines)
        self.assertIn("  y: CR=2, original_bytes=16, compressed_bytes=8", lines)
        self.assertIn("  z: CR=1, original_bytes=16, compressed_bytes=16", lines)
        self.assertFalse(any(line.startswith("  xyz:") for line in lines))

    def test_lcp_triplet_keeps_combined_position_ratio(self) -> None:
        report = self._report("lcp", "sz3")
        report["sizes"]["compressed_components_bytes"][
            "compressed/positions.lcp"
        ] = 24

        output = StringIO()
        with redirect_stdout(output):
            print_component_summary(report)

        lines = output.getvalue().splitlines()
        self.assertTrue(any(line.startswith("  xyz:") for line in lines))
        for logical in POSITION_FIELDS:
            self.assertFalse(
                any(line.startswith(f"  {logical}:") for line in lines)
            )

    def test_metrics_read_the_recorded_id_sort_permutation_dtype(self) -> None:
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
                    "preprocessed": {
                        "id_sort_order": str(order_path),
                    },
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

    @staticmethod
    def _report(position_codec: str, velocity_codec: str):
        return {
            "count": 4,
            "fields": FIELDS,
            "compressors": {
                "positions": position_codec,
                "velocities": velocity_codec,
            },
            "sizes": {
                "compressed_components_bytes": {
                    "compressed/x.psz": 4,
                    "compressed/y.psz": 8,
                    "compressed/z.psz": 16,
                    "compressed/id.pco": 8,
                    "compressed/vx.szo": 4,
                    "compressed/vy.szo": 8,
                    "compressed/vz.szo": 16,
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
