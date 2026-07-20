from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import main as particle_main
from src.cli import AVAILABLE_COMPRESSORS
from src.compress import CompressionSettings, compress
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.hdf5_io import recombine_h5
from src.models import ToolPaths
from src.preprocess import build_compressed_artifacts
from src.xnyzip_codec import (
    compress_chunked_xnyzip_triplet,
    read_xnyzip_permutation,
    run_chunked_xnyzip_decompress,
)


XNYZIP_EXE = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "XnYZip"
    / "build"
    / "XnYZip"
)
LCP_EXE = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "LCP"
    / "build"
    / "bin"
    / "lcp"
)


def _source_fields() -> dict[str, np.ndarray]:
    return {
        "x": np.array([10, 20, 30, 40, 50], dtype=np.float32),
        "y": np.array([11, 21, 31, 41, 51], dtype=np.float32),
        "z": np.array([12, 22, 32, 42, 52], dtype=np.float32),
        "id": np.array([901, 117, 502, 330, 774], dtype=np.uint64),
        "vx": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "vy": np.array([6, 7, 8, 9, 10], dtype=np.float32),
        "vz": np.array([11, 12, 13, 14, 15], dtype=np.float32),
    }


def _write_preprocessed(
    directory: Path,
    source: dict[str, np.ndarray],
) -> dict[str, str]:
    raw_paths = {}
    for logical, values in source.items():
        path = directory / f"{logical}.raw"
        values.tofile(path)
        raw_paths[logical] = str(path)
        if logical in VELOCITY_FIELDS:
            raw_paths[f"{logical}_xnyzip"] = str(path)

    positions = directory / "positions.xnyzip.f32.raw"
    np.column_stack(
        [source[field] for field in POSITION_FIELDS]
    ).astype(np.float32).tofile(positions)
    raw_paths["positions_xnyzip"] = str(positions)
    return raw_paths


def _manifest(
    root: Path,
    source: dict[str, np.ndarray],
    velocity_codec: str,
) -> dict:
    return {
        "count": len(source["id"]),
        "fields": {
            logical: {"dtype": str(values.dtype)}
            for logical, values in source.items()
        },
        "error_bounds": {
            "positions_xnyzip_abs": 0.1,
            "velocities_xnyzip_abs": 0.05,
        },
        "field_error_bounds": {
            logical: {"abs": 0.05, "compressor_abs": 0.05}
            for logical in VELOCITY_FIELDS
        },
        "artifacts": {
            "preprocessed": {},
            "compressed": build_compressed_artifacts(
                root / "compressed",
                "xnyzip",
                velocity_codec,
            ),
        },
        "compressed_fields": {},
        "sizes": {"selected_original_payload_bytes": 160},
    }


def _args(root: Path, velocity_codec: str) -> SimpleNamespace:
    return SimpleNamespace(
        work_dir=str(root),
        force=False,
        lossless="pcodec",
        pos_compressor="xnyzip",
        vel_compressor=velocity_codec,
        vel_chunk_size=0,
        vel_chunk_workers=0,
        sort=True,
    )


class XnYZipConfigurationTests(unittest.TestCase):
    def test_supported_xnyzip_combinations_are_accepted(self) -> None:
        self.assertIn("xnyzip", AVAILABLE_COMPRESSORS["pos_compressor"])
        self.assertIn("xnyzip", AVAILABLE_COMPRESSORS["vel_compressor"])
        for velocity_codec in ("sz3", "szo", "xnyzip"):
            settings = CompressionSettings.from_args(
                _args(Path("unused"), velocity_codec)
            )
            self.assertEqual(settings.position_codec, "xnyzip")
            self.assertEqual(settings.velocity_codec, velocity_codec)
            self.assertFalse(settings.sort_by_id)

        lcp_xnyzip = CompressionSettings.from_args(
            SimpleNamespace(
                **{
                    **vars(_args(Path("unused"), "xnyzip")),
                    "pos_compressor": "lcp",
                }
            )
        )
        self.assertEqual(lcp_xnyzip.position_codec, "lcp")
        self.assertEqual(lcp_xnyzip.velocity_codec, "xnyzip")
        self.assertFalse(lcp_xnyzip.sort_by_id)

        for position_codec in ("sz3", "szo"):
            with self.subTest(position_codec=position_codec):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "--vel-compressor xnyzip requires "
                    "--pos-compressor lcp or xnyzip",
                ):
                    CompressionSettings.from_args(
                        SimpleNamespace(
                            **{
                                **vars(_args(Path("unused"), "xnyzip")),
                                "pos_compressor": position_codec,
                            }
                        )
                    )

    def test_xnyzip_velocities_accept_chunk_settings_after_vector_positions(
        self,
    ) -> None:
        for position_codec in ("lcp", "xnyzip"):
            with self.subTest(position_codec=position_codec):
                args = SimpleNamespace(
                    **{
                        **vars(_args(Path("unused"), "xnyzip")),
                        "pos_compressor": position_codec,
                        "vel_chunk_size": 32768,
                        "vel_chunk_workers": 4,
                    }
                )
                settings = CompressionSettings.from_args(args)
                self.assertEqual(settings.velocity_chunk_size, 32768)
                self.assertEqual(settings.configured_chunk_workers, 4)
                self.assertEqual(settings.effective_chunk_workers, 4)

    def test_cli_rejects_xnyzip_velocities_with_fieldwise_positions(
        self,
    ) -> None:
        for position_codec in ("sz3", "szo"):
            with self.subTest(position_codec=position_codec):
                stderr = StringIO()
                with redirect_stderr(stderr):
                    result = particle_main.main(
                        [
                            "compress",
                            "missing.h5",
                            "--pos-compressor",
                            position_codec,
                            "--vel-compressor",
                            "xnyzip",
                        ]
                    )
                self.assertEqual(result, 2)
                self.assertIn(
                    "error: --vel-compressor xnyzip requires "
                    "--pos-compressor lcp or xnyzip.",
                    stderr.getvalue(),
                )

    def test_artifacts_use_combined_xnyzip_streams(self) -> None:
        root = Path("/tmp/compressed")
        fieldwise = build_compressed_artifacts(root, "xnyzip", "sz3")
        self.assertEqual(
            Path(fieldwise["positions"]).name,
            "positions.xnyzip",
        )
        self.assertNotIn("velocity_order", fieldwise)
        for logical in VELOCITY_FIELDS:
            self.assertEqual(Path(fieldwise[logical]).suffix, ".psz")

        all_xnyzip = build_compressed_artifacts(
            root,
            "xnyzip",
            "xnyzip",
        )
        self.assertEqual(
            Path(all_xnyzip["velocities"]).name,
            "velocities.xnyzip",
        )
        self.assertEqual(
            Path(all_xnyzip["velocity_order"]).name,
            "velocity_order.pco",
        )

        lcp_xnyzip = build_compressed_artifacts(
            root,
            "lcp",
            "xnyzip",
        )
        self.assertEqual(Path(lcp_xnyzip["positions"]).name, "positions.lcp")
        self.assertEqual(
            Path(lcp_xnyzip["velocities"]).name,
            "velocities.xnyzip",
        )
        self.assertEqual(
            Path(lcp_xnyzip["velocity_order"]).name,
            "velocity_order.pco",
        )


class XnYZipPermutationTests(unittest.TestCase):
    def test_uint64_permutation_is_returned_without_dtype_conversion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "order.u64.raw"
            expected = np.array([2, 0, 3, 1], dtype=np.uint64)
            expected.tofile(path)

            actual = read_xnyzip_permutation(str(path), expected.size)

            self.assertEqual(actual.dtype, np.dtype("uint64"))
            np.testing.assert_array_equal(actual, expected)

    def test_duplicate_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "order.u64.raw"
            np.array([0, 1, 1], dtype=np.uint64).tofile(path)

            with self.assertRaisesRegex(
                RuntimeError,
                "not a permutation",
            ):
                read_xnyzip_permutation(str(path), 3)

    def test_out_of_range_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "order.u64.raw"
            np.array([0, 1, 3], dtype=np.uint64).tofile(path)

            with self.assertRaisesRegex(
                RuntimeError,
                "not a valid index range",
            ):
                read_xnyzip_permutation(str(path), 3)


class ChunkedXnYZipCodecTests(unittest.TestCase):
    def test_chunk_container_is_deterministic_and_roundtrips_local_orders(
        self,
    ) -> None:
        count = 5
        chunk_size = 3
        values = np.arange(count * 3, dtype=np.float32).reshape(count, 3)

        def fake_compress(
            tools,
            input_path,
            compressed_path,
            particle_count,
            bound,
            order_path,
            force,
        ):
            chunk = np.fromfile(input_path, dtype=np.float32).reshape(-1, 3)
            self.assertEqual(chunk.shape[0], particle_count)
            order = np.arange(particle_count - 1, -1, -1, dtype=np.uint64)
            Path(compressed_path).write_bytes(
                np.ascontiguousarray(chunk[order]).tobytes()
            )
            order.tofile(order_path)
            return order

        def fake_decompress(
            tools,
            compressed_path,
            output_paths,
            fields,
            particle_count,
            bound,
            interleaved_path,
            force,
        ):
            chunk = np.frombuffer(
                Path(compressed_path).read_bytes(),
                dtype=np.float32,
            ).reshape(particle_count, 3)
            for axis, field in enumerate(fields):
                np.ascontiguousarray(chunk[:, axis]).tofile(
                    output_paths[field]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "velocities.f32.raw"
            values.tofile(input_path)
            outputs = []

            with patch(
                "src.xnyzip_codec.compress_xnyzip_triplet",
                side_effect=fake_compress,
            ):
                for workers in (1, 2):
                    run = root / f"workers-{workers}"
                    run.mkdir()
                    archive = run / "velocities.xnyzip"
                    order = run / "velocity_order.u64.raw"
                    metadata = compress_chunked_xnyzip_triplet(
                        ToolPaths(Path("lcp"), Path("xnyzip")),
                        str(input_path),
                        str(archive),
                        count,
                        chunk_size,
                        0.05,
                        order,
                        False,
                        workers,
                    )
                    outputs.append(
                        (archive.read_bytes(), order.read_bytes(), metadata)
                    )

            self.assertEqual(outputs[0][0], outputs[1][0])
            self.assertEqual(outputs[0][1], outputs[1][1])
            self.assertEqual(outputs[0][2]["xnyzip_workers"], 1)
            self.assertEqual(outputs[1][2]["xnyzip_workers"], 2)
            expected_local_order = np.array(
                [2, 1, 0, 1, 0],
                dtype=np.uint64,
            )
            np.testing.assert_array_equal(
                np.frombuffer(outputs[0][1], dtype=np.uint64),
                expected_local_order,
            )

            decoded_paths = {
                field: str(root / f"{field}.decoded.raw")
                for field in VELOCITY_FIELDS
            }
            with patch(
                "src.xnyzip_codec.run_xnyzip_decompress",
                side_effect=fake_decompress,
            ):
                run_chunked_xnyzip_decompress(
                    ToolPaths(Path("lcp"), Path("xnyzip")),
                    str(root / "workers-2" / "velocities.xnyzip"),
                    decoded_paths,
                    VELOCITY_FIELDS,
                    count,
                    chunk_size,
                    0.05,
                    2,
                )

            encoded_order = np.array([2, 1, 0, 4, 3])
            for axis, field in enumerate(VELOCITY_FIELDS):
                np.testing.assert_array_equal(
                    np.fromfile(decoded_paths[field], dtype=np.float32),
                    values[encoded_order, axis],
                )


class XnYZipOrderingTests(unittest.TestCase):
    def test_lcp_position_order_is_applied_before_id_and_xnyzip_compression(
        self,
    ) -> None:
        source = _source_fields()
        count = len(source["id"])
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.int32)
        velocity_order = np.array([1, 4, 0, 3, 2], dtype=np.uint64)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "preprocessed").mkdir()
            (root / "compressed").mkdir()
            raw_paths = _write_preprocessed(
                root / "preprocessed",
                source,
            )
            artifacts = build_compressed_artifacts(
                root / "compressed",
                "lcp",
                "xnyzip",
            )
            manifest = {
                "count": count,
                "fields": {
                    logical: {"dtype": str(values.dtype)}
                    for logical, values in source.items()
                },
                "error_bounds": {
                    "positions_lcp_abs": 0.1,
                    "velocities_xnyzip_abs": 0.05,
                },
                "artifacts": {
                    "preprocessed": raw_paths,
                    "compressed": artifacts,
                },
                "compressed_fields": {},
                "sizes": {"selected_original_payload_bytes": 160},
            }
            captured_integer = {}

            def fake_lcp(
                tools,
                input_paths,
                compressed_path,
                particle_count,
                bound,
                order_path,
                force,
            ):
                self.assertEqual(particle_count, count)
                Path(compressed_path).write_bytes(b"positions")
                position_order.tofile(order_path)

            def fake_xnyzip(
                tools,
                input_path,
                compressed_path,
                particle_count,
                bound,
                order_path,
                force,
            ):
                self.assertEqual(particle_count, count)
                np.testing.assert_array_equal(
                    np.fromfile(input_path, dtype=np.float32).reshape(-1, 3),
                    np.column_stack(
                        [
                            source[field][position_order]
                            for field in VELOCITY_FIELDS
                        ]
                    ),
                )
                Path(compressed_path).write_bytes(b"velocities")
                velocity_order.tofile(order_path)
                return velocity_order

            def fake_integer(
                codec,
                raw_path,
                dtype,
                compressed_path,
                field_name,
                *unused,
            ):
                captured_integer[field_name] = np.fromfile(
                    raw_path,
                    dtype=np.dtype(dtype),
                )
                Path(compressed_path).write_bytes(b"integer")
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": count,
                    "path": compressed_path,
                    "bytes": 7,
                }

            args = SimpleNamespace(
                work_dir=str(root),
                force=False,
                lossless="pcodec",
                pos_compressor="lcp",
                vel_compressor="xnyzip",
                vel_chunk_size=0,
                vel_chunk_workers=0,
                sort=False,
            )
            with patch(
                "src.compress.compress_lcp_triplet",
                side_effect=fake_lcp,
            ), patch(
                "src.compress.compress_xnyzip_triplet",
                side_effect=fake_xnyzip,
            ), patch(
                "src.compress.compress_integer_raw",
                side_effect=fake_integer,
            ), patch("src.compress.update_compressed_size_metrics"):
                result = compress(
                    args,
                    manifest,
                    raw_paths,
                    ToolPaths(Path("lcp"), Path("xnyzip")),
                )

            np.testing.assert_array_equal(
                captured_integer["id"],
                source["id"][position_order],
            )
            np.testing.assert_array_equal(
                captured_integer["velocity_order"],
                velocity_order,
            )
            self.assertEqual(
                result["ordering"]["reconstructed_rows"]["mapping"],
                "lcp_position_sorted",
            )
            self.assertEqual(
                result["ordering"]["velocities"]["mapping"],
                "xnyzip_velocity_sorted_index_to_lcp_position_sorted_row",
            )
            self.assertEqual(
                result["compressed_fields"]["velocity_order"][
                    "order_mapping"
                ],
                "xnyzip_velocity_sorted_index_to_lcp_position_sorted_row",
            )

    def test_fieldwise_velocities_receive_the_xnyzip_position_order(
        self,
    ) -> None:
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.uint64)
        for velocity_codec in ("sz3", "szo"):
            with self.subTest(velocity_codec=velocity_codec):
                self._assert_fieldwise_order(position_order, velocity_codec)

    def _assert_fieldwise_order(
        self,
        position_order: np.ndarray,
        velocity_codec: str,
    ) -> None:
        source = _source_fields()
        count = len(source["id"])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "preprocessed").mkdir()
            (root / "compressed").mkdir()
            raw_paths = _write_preprocessed(
                root / "preprocessed",
                source,
            )
            manifest = _manifest(root, source, velocity_codec)
            manifest["artifacts"]["preprocessed"] = raw_paths
            captured = {}

            def fake_xnyzip(
                tools,
                input_path,
                compressed_path,
                particle_count,
                bound,
                order_path,
                force,
            ):
                self.assertEqual(particle_count, count)
                np.testing.assert_array_equal(
                    np.fromfile(input_path, dtype=np.float32).reshape(-1, 3),
                    np.column_stack(
                        [source[field] for field in POSITION_FIELDS]
                    ),
                )
                Path(compressed_path).write_bytes(b"positions")
                position_order.tofile(order_path)

            def fake_integer(
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
                Path(compressed_path).write_bytes(b"integer")
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": count,
                    "path": compressed_path,
                    "bytes": 7,
                }

            def fake_lossy(
                codec,
                raw_path,
                dtype,
                compressed_path,
                field_name,
                *unused,
            ):
                self.assertEqual(codec, velocity_codec)
                captured[field_name] = np.fromfile(
                    raw_path,
                    dtype=np.dtype(dtype),
                )
                Path(compressed_path).write_bytes(b"lossy")
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": count,
                }

            with patch(
                "src.compress.compress_xnyzip_triplet",
                side_effect=fake_xnyzip,
            ), patch(
                "src.compress.compress_integer_raw",
                side_effect=fake_integer,
            ), patch(
                "src.compress.compress_lossy_raw",
                side_effect=fake_lossy,
            ), patch("src.compress.update_compressed_size_metrics"):
                result = compress(
                    _args(root, velocity_codec),
                    manifest,
                    raw_paths,
                    ToolPaths(Path("lcp"), Path("xnyzip")),
                )

            for logical in ("id", *VELOCITY_FIELDS):
                np.testing.assert_array_equal(
                    captured[logical],
                    source[logical][position_order],
                )
            self.assertNotIn("velocity_order", result["compressed_fields"])
            self.assertEqual(result["format_version"], 5)
            self.assertEqual(
                result["ordering"]["reconstructed_rows"]["mapping"],
                "xnyzip_position_sorted",
            )

    def test_all_xnyzip_stores_the_secondary_velocity_order(self) -> None:
        source = _source_fields()
        count = len(source["id"])
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.uint64)
        velocity_order = np.array([1, 4, 0, 3, 2], dtype=np.uint64)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "preprocessed").mkdir()
            (root / "compressed").mkdir()
            raw_paths = _write_preprocessed(
                root / "preprocessed",
                source,
            )
            manifest = _manifest(root, source, "xnyzip")
            manifest["artifacts"]["preprocessed"] = raw_paths
            captured_integer = {}
            compressed_calls = 0

            def fake_xnyzip(
                tools,
                input_path,
                compressed_path,
                particle_count,
                bound,
                order_path,
                force,
            ):
                nonlocal compressed_calls
                compressed_calls += 1
                values = np.fromfile(
                    input_path,
                    dtype=np.float32,
                ).reshape(-1, 3)
                if compressed_calls == 1:
                    expected = np.column_stack(
                        [source[field] for field in POSITION_FIELDS]
                    )
                    order = position_order
                else:
                    expected = np.column_stack(
                        [
                            source[field][position_order]
                            for field in VELOCITY_FIELDS
                        ]
                    )
                    order = velocity_order
                np.testing.assert_array_equal(values, expected)
                Path(compressed_path).write_bytes(
                    f"triplet-{compressed_calls}".encode()
                )
                order.tofile(order_path)

            def fake_integer(
                codec,
                raw_path,
                dtype,
                compressed_path,
                field_name,
                *unused,
            ):
                captured_integer[field_name] = np.fromfile(
                    raw_path,
                    dtype=np.dtype(dtype),
                )
                Path(compressed_path).write_bytes(b"integer")
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": count,
                    "path": compressed_path,
                    "bytes": 7,
                }

            with patch(
                "src.compress.compress_xnyzip_triplet",
                side_effect=fake_xnyzip,
            ), patch(
                "src.compress.compress_integer_raw",
                side_effect=fake_integer,
            ), patch("src.compress.update_compressed_size_metrics"):
                result = compress(
                    _args(root, "xnyzip"),
                    manifest,
                    raw_paths,
                    ToolPaths(Path("lcp"), Path("xnyzip")),
                )

            self.assertEqual(compressed_calls, 2)
            np.testing.assert_array_equal(
                captured_integer["id"],
                source["id"][position_order],
            )
            np.testing.assert_array_equal(
                captured_integer["velocity_order"],
                velocity_order,
            )
            self.assertEqual(
                result["compressed_fields"]["velocity_order"]["dtype"],
                "uint64",
            )
            self.assertEqual(
                result["compressed_fields"]["velocities"]["codec"],
                "xnyzip",
            )
            self.assertEqual(
                result["ordering"]["velocities"]["index_scope"],
                "global",
            )

    def test_all_xnyzip_chunking_stores_chunk_local_velocity_orders(
        self,
    ) -> None:
        source = _source_fields()
        count = len(source["id"])
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.uint64)
        local_order = np.array([2, 0, 1, 1, 0], dtype=np.uint64)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "preprocessed").mkdir()
            (root / "compressed").mkdir()
            raw_paths = _write_preprocessed(
                root / "preprocessed",
                source,
            )
            manifest = _manifest(root, source, "xnyzip")
            manifest["artifacts"]["preprocessed"] = raw_paths
            captured_integer = {}

            def fake_xnyzip(
                tools,
                input_path,
                compressed_path,
                particle_count,
                bound,
                order_path,
                force,
            ):
                self.assertEqual(particle_count, count)
                Path(compressed_path).write_bytes(b"positions")
                position_order.tofile(order_path)
                return position_order

            def fake_chunked_xnyzip(
                tools,
                input_path,
                compressed_path,
                particle_count,
                chunk_size,
                bound,
                order_path,
                force,
                workers,
            ):
                self.assertEqual(particle_count, count)
                self.assertEqual(chunk_size, 3)
                self.assertEqual(workers, 2)
                np.testing.assert_array_equal(
                    np.fromfile(input_path, dtype=np.float32).reshape(-1, 3),
                    np.column_stack(
                        [
                            source[field][position_order]
                            for field in VELOCITY_FIELDS
                        ]
                    ),
                )
                Path(compressed_path).write_bytes(b"chunked-velocities")
                local_order.tofile(order_path)
                return {
                    "chunk_size": 3,
                    "chunk_count": 2,
                    "xnyzip_workers": 2,
                    "order_bits_per_particle": 2,
                }

            def fake_integer(
                codec,
                raw_path,
                dtype,
                compressed_path,
                field_name,
                *unused,
            ):
                captured_integer[field_name] = np.fromfile(
                    raw_path,
                    dtype=np.dtype(dtype),
                )
                Path(compressed_path).write_bytes(b"integer")
                return {
                    "field": field_name,
                    "codec": codec,
                    "dtype": dtype,
                    "count": count,
                    "path": compressed_path,
                    "bytes": 7,
                }

            args = SimpleNamespace(
                **{
                    **vars(_args(root, "xnyzip")),
                    "vel_chunk_size": 3,
                    "vel_chunk_workers": 2,
                }
            )
            with patch(
                "src.compress.compress_xnyzip_triplet",
                side_effect=fake_xnyzip,
            ), patch(
                "src.compress.compress_chunked_xnyzip_triplet",
                side_effect=fake_chunked_xnyzip,
            ), patch(
                "src.compress.compress_integer_raw",
                side_effect=fake_integer,
            ), patch("src.compress.update_compressed_size_metrics"):
                result = compress(
                    args,
                    manifest,
                    raw_paths,
                    ToolPaths(Path("lcp"), Path("xnyzip")),
                )

            np.testing.assert_array_equal(
                captured_integer["velocity_order"],
                local_order,
            )
            velocity_field = result["compressed_fields"]["velocities"]
            order_field = result["compressed_fields"]["velocity_order"]
            self.assertEqual(velocity_field["chunk_size"], 3)
            self.assertEqual(
                velocity_field["container"],
                "chunked_xnyzip_v1",
            )
            self.assertEqual(order_field["dtype"], "uint64")
            self.assertEqual(order_field["index_scope"], "chunk_local")
            self.assertEqual(order_field["xnyzip_workers"], 2)
            self.assertEqual(
                result["ordering"]["velocities"]["index_scope"],
                "chunk_local",
            )
            self.assertEqual(result["format_version"], 6)

    def test_recombination_applies_uint64_xnyzip_velocity_order(self) -> None:
        source = _source_fields()
        count = len(source["id"])
        position_order = np.array([2, 0, 4, 1, 3], dtype=np.intp)
        velocity_order = np.array([1, 4, 0, 3, 2], dtype=np.uint64)
        canonical = {
            logical: values[position_order]
            for logical, values in source.items()
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            for logical, values in canonical.items():
                path = root / f"{logical}.raw"
                (
                    values[velocity_order]
                    if logical in VELOCITY_FIELDS
                    else values
                ).tofile(path)
                paths[logical] = str(path)
            order_path = root / "velocity_order.u64.raw"
            velocity_order.tofile(order_path)
            paths["velocity_order"] = str(order_path)

            fields = {
                logical: {
                    "h5_path": logical,
                    "dtype": str(values.dtype),
                }
                for logical, values in source.items()
            }
            manifest = {
                "count": count,
                "position_scale": {"value": 1.0},
                "fields": fields,
                "compressors": {
                    "positions": "xnyzip",
                    "velocities": "xnyzip",
                },
                "compressed_fields": {
                    "positions": {
                        "codec": "xnyzip",
                        "dtype": "float32",
                    },
                    "velocities": {
                        "codec": "xnyzip",
                        "dtype": "float32",
                    },
                    "velocity_order": {
                        "dtype": "uint64",
                        "chunk_size": 0,
                    },
                },
            }
            output = root / "reconstructed.h5"
            recombine_h5(manifest, paths, output)

            with h5py.File(output, "r") as reconstructed:
                for logical, expected in canonical.items():
                    np.testing.assert_array_equal(
                        reconstructed[logical][:],
                        expected,
                    )


@unittest.skipUnless(
    XNYZIP_EXE.is_file(),
    "XnYZip executable has not been built",
)
class XnYZipNativeRoundtripTests(unittest.TestCase):
    def test_all_xnyzip_pipeline_preserves_rows_and_bounds_with_optional_chunks(
        self,
    ) -> None:
        count = 257
        rng = np.random.default_rng(20260719)
        source = {
            "x": rng.uniform(-10.0, 10.0, count).astype(np.float32),
            "y": rng.uniform(-20.0, 20.0, count).astype(np.float32),
            "z": rng.uniform(-30.0, 30.0, count).astype(np.float32),
            "vx": rng.normal(size=count).astype(np.float32),
            "vy": rng.normal(size=count).astype(np.float32),
            "vz": rng.normal(size=count).astype(np.float32),
            "id": np.arange(1000, 1000 + count, dtype=np.uint64),
        }
        for chunk_size in (0, 64):
            with self.subTest(chunk_size=chunk_size):
                self._assert_native_roundtrip(source, chunk_size)

    @unittest.skipUnless(
        LCP_EXE.is_file(),
        "LCP executable has not been built",
    )
    def test_lcp_positions_and_xnyzip_velocities_preserve_rows(
        self,
    ) -> None:
        count = 257
        rng = np.random.default_rng(20260720)
        source = {
            "x": rng.uniform(-10.0, 10.0, count).astype(np.float32),
            "y": rng.uniform(-20.0, 20.0, count).astype(np.float32),
            "z": rng.uniform(-30.0, 30.0, count).astype(np.float32),
            "vx": rng.normal(size=count).astype(np.float32),
            "vy": rng.normal(size=count).astype(np.float32),
            "vz": rng.normal(size=count).astype(np.float32),
            "id": np.arange(2000, 2000 + count, dtype=np.uint64),
        }
        for chunk_size in (0, 64):
            with self.subTest(chunk_size=chunk_size):
                self._assert_native_roundtrip(
                    source,
                    chunk_size,
                    position_codec="lcp",
                )

    def _assert_native_roundtrip(
        self,
        source: dict[str, np.ndarray],
        chunk_size: int,
        position_codec: str = "xnyzip",
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_h5 = root / "input.h5"
            work_dir = root / "work"
            with h5py.File(input_h5, "w") as h5:
                for logical, values in source.items():
                    h5.create_dataset(logical, data=values)

            argv = [
                "roundtrip",
                str(input_h5),
                "--work-dir",
                str(work_dir),
                "--xnyzip",
                str(XNYZIP_EXE),
                "--pos-compressor",
                position_codec,
                "--vel-compressor",
                "xnyzip",
                "--pos-abs-eb",
                "0.1",
                "--vel-abs-eb",
                "0.05",
            ]
            if chunk_size:
                argv.extend(
                    [
                        "--vel-chunk-size",
                        str(chunk_size),
                        "--vel-chunk-workers",
                        "2",
                    ]
                )
            argv.append("--force")

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
            order_field = manifest["compressed_fields"]["velocity_order"]
            velocity_field = manifest["compressed_fields"]["velocities"]
            self.assertEqual(
                manifest["format_version"],
                6 if chunk_size else 5,
            )
            self.assertEqual(order_field["dtype"], "uint64")
            self.assertEqual(
                order_field["index_scope"],
                "chunk_local" if chunk_size else "global",
            )
            self.assertEqual(
                velocity_field["container"],
                "chunked_xnyzip_v1" if chunk_size else "native_xnyzip",
            )
            self.assertNotIn("order", manifest["artifacts"]["compressed"])
            self.assertTrue(
                all(
                    entry["satisfied"]
                    for entry in metrics[
                        "xnyzip_l2_error_bound_consistency"
                    ].values()
                )
            )

            with h5py.File(
                work_dir / "reconstructed.h5",
                "r",
            ) as reconstructed:
                reconstructed_ids = reconstructed["id"][:]
                source_rows = reconstructed_ids - source["id"][0]
                np.testing.assert_array_equal(
                    reconstructed_ids,
                    source["id"][source_rows],
                )
                position_difference = np.column_stack(
                    [
                        reconstructed[field][:]
                        - source[field][source_rows]
                        for field in POSITION_FIELDS
                    ]
                )
                if position_codec == "xnyzip":
                    self.assertLessEqual(
                        float(
                            np.linalg.norm(
                                position_difference,
                                axis=1,
                            ).max(initial=0.0)
                        ),
                        0.1 + 1e-6,
                    )
                else:
                    self.assertLessEqual(
                        float(np.abs(position_difference).max(initial=0.0)),
                        0.1 + 1e-6,
                    )

                velocity_difference = np.column_stack(
                    [
                        reconstructed[field][:]
                        - source[field][source_rows]
                        for field in VELOCITY_FIELDS
                    ]
                )
                self.assertLessEqual(
                    float(
                        np.linalg.norm(
                            velocity_difference,
                            axis=1,
                        ).max(initial=0.0)
                    ),
                    0.05 + 1e-6,
                )
