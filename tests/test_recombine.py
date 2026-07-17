import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

import src.helpers as hp
from src.cli import AVAILABLE_COMPRESSORS, build_parser
from src.compress import compress as compress_pipeline
from src.compress import (
    compress_chunked_lcp_triplet,
    compress_pcodec_raw,
    compress_szo_raw,
    reorder_raw,
)
from src.decompress import decompress_pcodec_raw, decompress_szo_raw
from src.decompress import (
    position_compressor_from_manifest,
    recombine_h5,
    velocity_compressor_from_manifest,
)
from src.preprocess import preprocess as preprocess_pipeline


FIELDS = {
    logical: {"h5_path": logical, "dtype": "uint64" if logical == "id" else "float32"}
    for logical in ("id", "x", "y", "z", "vx", "vy", "vz")
}


class RecombineTests(unittest.TestCase):
    def test_velocity_lcp_reorders_id_and_sz3_positions_and_omits_velocity_order(self) -> None:
        count = 5
        velocity_order = np.array([1, 4, 0, 3, 2], dtype=np.int32)
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
                "x": str(compressed / "x.psz"),
                "y": str(compressed / "y.psz"),
                "z": str(compressed / "z.psz"),
                "id": str(compressed / "id.pco"),
                "velocities": str(compressed / "velocities.lcp"),
            }
            manifest = {
                "count": count,
                "fields": {
                    logical: {"dtype": str(values.dtype)}
                    for logical, values in source.items()
                },
                "error_bounds": {"velocities_lcp_abs": 0.01},
                "field_error_bounds": {
                    logical: {"abs": 0.01, "compressor_abs": 0.01}
                    for logical in ("x", "y", "z")
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
                vel_compressor="lcp",
            )
            captured = {}

            def fake_run_command(argv):
                Path(argv[argv.index("-z") + 1]).write_bytes(b"lcp")
                velocity_order.tofile(argv[-1])

            def fake_integer(codec, raw_path, dtype, compressed_path, field_name, *unused):
                captured[field_name] = np.fromfile(raw_path, dtype=np.dtype(dtype))
                Path(compressed_path).write_bytes(b"integer")
                return {"field": field_name, "codec": codec, "dtype": dtype, "count": count}

            def fake_position(raw_path, dtype, compressed_path, field_name, *unused):
                captured[field_name] = np.fromfile(raw_path, dtype=np.dtype(dtype))
                Path(compressed_path).write_bytes(b"position")
                return {"field": field_name, "codec": "pysz", "dtype": dtype, "count": count}

            with patch("src.compress.hp.run_command", side_effect=fake_run_command), patch(
                "src.compress.compress_integer_raw", side_effect=fake_integer
            ), patch("src.compress.compress_pysz_raw", side_effect=fake_position), patch(
                "src.compress.hp.update_compressed_size_metrics"
            ):
                result = compress_pipeline(
                    args,
                    manifest,
                    raw_paths,
                    SimpleNamespace(lcp=Path("lcp")),
                )

            self.assertNotIn("velocity_order", result["compressed_fields"])
            self.assertNotIn("velocity_order", result["artifacts"]["compressed"])
            self.assertFalse(result["ordering"]["reconstructed_rows"]["velocity_permutation_stored"])
            for logical in ("id", "x", "y", "z"):
                np.testing.assert_array_equal(captured[logical], source[logical][velocity_order])

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

            def fake_velocity(raw_path, dtype, compressed_path, field_name, *unused):
                captured[field_name] = np.fromfile(raw_path, dtype=np.dtype(dtype))
                Path(compressed_path).write_bytes(b"velocity")
                return {"field": field_name, "codec": "pysz", "dtype": dtype, "count": count}

            with patch("src.compress.hp.run_command", side_effect=fake_run_command), patch(
                "src.compress.compress_integer_raw", side_effect=fake_integer
            ), patch("src.compress.compress_pysz_raw", side_effect=fake_velocity), patch(
                "src.compress.hp.update_compressed_size_metrics"
            ):
                result = compress_pipeline(
                    args,
                    manifest,
                    raw_paths,
                    SimpleNamespace(lcp=Path("lcp")),
                )

            self.assertNotIn("order", result["compressed_fields"])
            self.assertNotIn("order", result["artifacts"]["compressed"])
            self.assertFalse(result["ordering"]["reconstructed_rows"]["position_permutation_stored"])
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

    def test_recombine_accepts_shared_lcp_velocity_order_without_sidecar(self) -> None:
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
        self.assertIn("szo", AVAILABLE_COMPRESSORS["pos_compressor"])
        self.assertIn("szo", AVAILABLE_COMPRESSORS["vel_compressor"])
        self.assertNotIn("szo", AVAILABLE_COMPRESSORS["lossless"])

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
            for logical in hp.POSITION_FIELDS + hp.VELOCITY_FIELDS:
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

        test_case = self

        class FakeSZo:
            @staticmethod
            def compress(data, config, copy):
                test_case.assertTrue(copy)
                test_case.assertEqual(config.errorBoundMode, FakeErrorBoundMode.ABS)
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
                    "src.compress.hp.load_pyszo",
                    return_value=(FakeSZo, FakeConfig, FakeErrorBoundMode, object),
                ), patch(
                    "src.decompress.hp.load_pyszo",
                    return_value=(FakeSZo, FakeConfig, FakeErrorBoundMode, object),
                ):
                    dtype = np.dtype(dtype_name)
                    source = np.array([0.125, -3.5, 9.75, 1.0], dtype=dtype)
                    raw_path = root / f"{dtype_name}.raw"
                    compressed_path = root / f"{dtype_name}.szo"
                    output_path = root / f"{dtype_name}.out.raw"
                    source.tofile(raw_path)
                    field = compress_szo_raw(
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
                    decompress_szo_raw(field, str(output_path), False)
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
            with patch("src.compress.hp.LCP_CHUNK_BATCH_VALUES", 6), patch(
                "src.compress.hp.run_command", side_effect=fake_run_command
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

            with patch("src.compress.hp.run_command", side_effect=fake_run_command), patch(
                "src.compress.compress_integer_raw", side_effect=fake_integer
            ), patch("src.compress.hp.update_compressed_size_metrics"):
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
            self.assertEqual(result["compressed_fields"]["velocity_order"]["index_scope"], "chunk_local")
            self.assertEqual(result["compressed_fields"]["velocity_order"]["order_bits_per_particle"], 2)
            self.assertEqual(result["compressed_fields"]["velocities"]["container"], "chunked_lcp_v2")
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
                        logical: {"codec": "szo"} for logical in hp.VELOCITY_FIELDS
                    }
                }
            ),
            "szo",
        )
        self.assertEqual(
            position_compressor_from_manifest(
                {
                    "compressed_fields": {
                        logical: {"codec": "szo"} for logical in hp.POSITION_FIELDS
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


if __name__ == "__main__":
    unittest.main()
