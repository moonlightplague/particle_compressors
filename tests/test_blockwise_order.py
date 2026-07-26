from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import main as particle_main
from src.cli import build_parser
from src.compress import CompressionSettings
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.huffman_encode import huffman_decode_file, huffman_encode_file
from src.lcp_codec import compress_lcp_triplet, run_lcp_decompress
from src.metrics import component_compression_ratios
from src.models import ToolPaths
from src.preprocess import build_compressed_artifacts
from src.raw_codecs import compress_pcodec_raw, decompress_pcodec_raw


LCP_EXE = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "LCP"
    / "build"
    / "bin"
    / "lcp"
)


class BlockIdHuffmanTests(unittest.TestCase):
    def test_huffman_and_pcodec_layers_roundtrip_uint32_block_ids(self) -> None:
        block_ids = np.repeat(
            np.array([3, 10, 91, 500], dtype=np.uint32),
            [6000, 2500, 1000, 500],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "block_ids.uint32.raw"
            huffman = root / "block_ids.huffman.raw"
            compressed = root / "block_ids.pco"
            restored_huffman = root / "block_ids.restored.huffman.raw"
            restored = root / "block_ids.restored.uint32.raw"
            duplicate = root / "block_ids.duplicate.huffman.raw"
            block_ids.tofile(raw)

            first = huffman_encode_file(
                raw,
                huffman,
                False,
                expected_count=block_ids.size,
            )
            huffman_encode_file(
                raw,
                duplicate,
                False,
                expected_count=block_ids.size,
            )
            self.assertEqual(huffman.read_bytes(), duplicate.read_bytes())
            self.assertLess(first["encoded_bytes"], raw.stat().st_size)

            field = compress_pcodec_raw(
                str(huffman),
                "uint8",
                str(compressed),
                "velocity_block_ids",
                huffman.stat().st_size,
                False,
            )
            decompress_pcodec_raw(field, str(restored_huffman), False)
            huffman_decode_file(
                restored_huffman,
                restored,
                False,
                expected_count=block_ids.size,
            )
            np.testing.assert_array_equal(
                np.fromfile(restored, dtype=np.uint32),
                block_ids,
            )

    def test_huffman_roundtrips_uint64_block_ids(self) -> None:
        block_ids = np.array(
            [
                0,
                2**32 + 17,
                2**32 + 17,
                2**32 + 18,
                2**33 + 5,
            ],
            dtype=np.uint64,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "block_ids.uint64.raw"
            encoded = root / "block_ids.huffman.raw"
            restored = root / "block_ids.restored.uint64.raw"
            block_ids.tofile(raw)

            metadata = huffman_encode_file(
                raw,
                encoded,
                False,
                expected_count=block_ids.size,
            )
            decoded = huffman_decode_file(
                encoded,
                restored,
                False,
                expected_count=block_ids.size,
            )

            self.assertEqual(metadata["symbol_dtype"], "uint64")
            self.assertEqual(decoded["symbol_dtype"], "uint64")
            np.testing.assert_array_equal(
                np.fromfile(restored, dtype=np.uint64),
                block_ids,
            )

    def test_huffman_decoder_rejects_truncated_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "block_ids.raw"
            encoded = root / "block_ids.huffman"
            truncated = root / "block_ids.truncated.huffman"
            output = root / "decoded.raw"
            np.arange(64, dtype=np.uint32).tofile(raw)
            huffman_encode_file(raw, encoded, False, expected_count=64)
            truncated.write_bytes(encoded.read_bytes()[:-1])

            with self.assertRaisesRegex(RuntimeError, "payload size"):
                huffman_decode_file(
                    truncated,
                    output,
                    False,
                    expected_count=64,
                )


class BlockwiseOrderConfigurationTests(unittest.TestCase):
    def test_cli_enables_blockwise_order_for_all_lcp(self) -> None:
        argv = [
            "compress",
            "input.h5",
            "--pos-compressor",
            "lcp",
            "--vel-compressor",
            "lcp",
            "--blockwise-ord",
        ]
        args = build_parser(argv).parse_args(argv)
        settings = CompressionSettings.from_args(args)
        self.assertTrue(settings.blockwise_order)

    def test_blockwise_order_rejects_other_velocity_modes_and_chunks(
        self,
    ) -> None:
        for velocity_codec, chunk_size in (("sz3", 0), ("lcp", 128)):
            with self.subTest(
                velocity_codec=velocity_codec,
                chunk_size=chunk_size,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "--blockwise-ord",
                ):
                    CompressionSettings.from_args(
                        build_parser(
                            [
                                "compress",
                                "input.h5",
                                "--pos-compressor",
                                "lcp",
                                "--vel-compressor",
                                velocity_codec,
                                "--vel-chunk-size",
                                str(chunk_size),
                                "--blockwise-ord",
                            ]
                        ).parse_args(
                            [
                                "compress",
                                "input.h5",
                                "--pos-compressor",
                                "lcp",
                                "--vel-compressor",
                                velocity_codec,
                                "--vel-chunk-size",
                                str(chunk_size),
                                "--blockwise-ord",
                            ]
                        )
                    )

    def test_blockwise_artifacts_include_block_id_sidecar(self) -> None:
        artifacts = build_compressed_artifacts(
            Path("/tmp/compressed"),
            "lcp",
            "lcp",
            blockwise_order=True,
        )
        self.assertEqual(
            Path(artifacts["velocity_block_ids"]).name,
            "velocity_block_ids.pco",
        )


class BlockwiseLCPCommandTests(unittest.TestCase):
    def test_native_commands_use_both_blockwise_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = tuple(str(root / f"{axis}.raw") for axis in "xyz")
            outputs = {
                field: str(root / f"{field}.out")
                for field in VELOCITY_FIELDS
            }
            archive = root / "velocities.lcp"
            order = root / "velocity_order.raw"
            block_ids = root / "velocity_block_ids.raw"
            commands = []

            with patch(
                "src.lcp_codec.run_command",
                side_effect=lambda argv: commands.append(argv),
            ):
                compress_lcp_triplet(
                    ToolPaths(lcp=Path("lcp")),
                    inputs,
                    str(archive),
                    100,
                    0.01,
                    order,
                    False,
                    block_ids,
                )
                run_lcp_decompress(
                    ToolPaths(lcp=Path("lcp")),
                    str(archive),
                    outputs,
                    VELOCITY_FIELDS,
                    100,
                    0.01,
                    order,
                    block_ids,
                )

            self.assertEqual(
                commands[0][-3:],
                [str(order), "--blockwise-ord", str(block_ids)],
            )
            self.assertEqual(
                commands[1][-5:],
                [
                    "--decompress-with-order",
                    "32",
                    str(order),
                    "--blockwise-ord",
                    str(block_ids),
                ],
            )


@unittest.skipUnless(LCP_EXE.is_file(), "LCP executable has not been built")
class BlockwiseLCPNativeRoundtripTests(unittest.TestCase):
    def test_pipeline_restores_velocity_rows_with_blockwise_order(self) -> None:
        help_text = subprocess.run(
            [str(LCP_EXE), "--help"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        if "--blockwise-ord" not in help_text:
            self.skipTest("Built LCP does not support blockwise order")

        count = 1024
        rng = np.random.default_rng(20260726)
        source = {
            "id": np.arange(5000, 5000 + count, dtype=np.uint64),
            "x": rng.uniform(-8.0, 8.0, count).astype(np.float32),
            "y": rng.uniform(-6.0, 6.0, count).astype(np.float32),
            "z": rng.uniform(-4.0, 4.0, count).astype(np.float32),
            "vx": rng.normal(size=count).astype(np.float32),
            "vy": rng.normal(size=count).astype(np.float32),
            "vz": rng.normal(size=count).astype(np.float32),
        }
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
                "--lcp",
                str(LCP_EXE),
                "--pos-compressor",
                "lcp",
                "--vel-compressor",
                "lcp",
                "--blockwise-ord",
                "--position-scale",
                "raw",
                "--pos-abs-eb",
                "0.01",
                "--vel-abs-eb",
                "0.01",
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
            self.assertEqual(manifest["format_version"], 7)
            self.assertEqual(
                manifest["compressed_fields"]["velocity_order"][
                    "order_encoding"
                ],
                "lcp_blockwise_packed",
            )
            self.assertIn(
                "velocity_block_ids",
                manifest["compressed_fields"],
            )
            sidecar_ratio = component_compression_ratios(metrics)[
                "velocity_order"
            ]
            self.assertEqual(
                sidecar_ratio["original_bytes"],
                (
                    manifest["compressed_fields"]["velocity_order"][
                        "uncompressed_bytes"
                    ]
                    + manifest["compressed_fields"]["velocity_block_ids"][
                        "decoded_bytes"
                    ]
                ),
            )

            with h5py.File(
                work_dir / "reconstructed.h5",
                "r",
            ) as reconstructed:
                reconstructed_ids = reconstructed["id"][:]
                source_rows = reconstructed_ids - source["id"][0]
                for logical in POSITION_FIELDS + VELOCITY_FIELDS:
                    max_error = np.max(
                        np.abs(
                            reconstructed[logical][:]
                            - source[logical][source_rows]
                        ),
                        initial=0.0,
                    )
                    self.assertLessEqual(float(max_error), 0.010001)


if __name__ == "__main__":
    unittest.main()
