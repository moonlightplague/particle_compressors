"""Tests for flat field codec adapters."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.raw_codecs import (
    compress_lossy_raw,
    decompress_lossy_raw,
    sperr_pwe_quality,
)


class RawCodecTests(unittest.TestCase):
    def test_qoz_roundtrip_uses_absolute_error_mode(self) -> None:
        class FakeQoZ:
            encoded = None
            error_bound = None
            mode = None

            @classmethod
            def compress(cls, data, error_bound, mode):
                cls.encoded = np.asarray(data).copy()
                cls.error_bound = error_bound
                cls.mode = mode
                return b"qoz-payload"

            @classmethod
            def decompress(cls, payload):
                self.assertEqual(payload, b"qoz-payload")
                return cls.encoded.copy()

        values = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "input.raw"
            compressed_path = root / "field.qoz"
            output_path = root / "decoded.raw"
            values.tofile(raw_path)

            with patch("src.raw_codecs.load_qoz", return_value=FakeQoZ):
                field = compress_lossy_raw(
                    "qoz",
                    str(raw_path),
                    "float32",
                    str(compressed_path),
                    "vx",
                    values.size,
                    0.01,
                    False,
                )
                decompress_lossy_raw(field, str(output_path), False)

            self.assertEqual(field["codec"], "qoz")
            self.assertEqual(field["encoded_count"], 10_000)
            self.assertEqual(FakeQoZ.error_bound, 0.01)
            self.assertEqual(FakeQoZ.mode, "abs")
            np.testing.assert_array_equal(
                np.fromfile(output_path, dtype=np.float32),
                values,
            )

    def test_qoz_zero_bound_constant_roundtrips_exactly(self) -> None:
        class FakeQoZ:
            @staticmethod
            def compress(data, error_bound, mode):
                np.testing.assert_array_equal(data, np.zeros_like(data))
                self.assertEqual(error_bound, 1.0)
                self.assertEqual(mode, "abs")
                return b"constant-qoz"

            @staticmethod
            def decompress(payload):
                self.assertEqual(payload, b"constant-qoz")
                return np.zeros(10_000, dtype=np.float64)

        values = np.full(23, -7.25, dtype=np.float64)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "input.raw"
            compressed_path = root / "field.qoz"
            output_path = root / "decoded.raw"
            values.tofile(raw_path)

            with patch("src.raw_codecs.load_qoz", return_value=FakeQoZ):
                field = compress_lossy_raw(
                    "qoz",
                    str(raw_path),
                    "float64",
                    str(compressed_path),
                    "vy",
                    values.size,
                    0.0,
                    False,
                )
                decompress_lossy_raw(field, str(output_path), False)

            self.assertEqual(field["constant_value"], -7.25)
            np.testing.assert_array_equal(
                np.fromfile(output_path, dtype=np.float64),
                values,
            )

    def test_sperr_zero_bound_is_supported_only_for_constant_data(self) -> None:
        quality = sperr_pwe_quality(
            np.ones(4, dtype=np.float32),
            0.0,
            "x",
        )
        self.assertGreater(quality, 0.0)

        with self.assertRaisesRegex(RuntimeError, "positive finite"):
            sperr_pwe_quality(
                np.arange(4, dtype=np.float32),
                0.0,
                "x",
            )

    def test_sperr_roundtrip_uses_pointwise_error_mode(self) -> None:
        class FakeSPERR:
            encoded = None
            quality = None
            mode = None

            @classmethod
            def compress(cls, data, quality, mode):
                cls.encoded = np.asarray(data).copy()
                cls.quality = quality
                cls.mode = mode
                return b"sperr-payload"

            @classmethod
            def decompress(cls, payload):
                self.assertEqual(payload, b"sperr-payload")
                return cls.encoded.copy()

        values = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "input.raw"
            compressed_path = root / "field.sperr"
            output_path = root / "decoded.raw"
            values.tofile(raw_path)

            with patch("src.raw_codecs.load_sperr", return_value=FakeSPERR):
                field = compress_lossy_raw(
                    "sperr",
                    str(raw_path),
                    "float32",
                    str(compressed_path),
                    "vx",
                    values.size,
                    0.01,
                    False,
                )
                decompress_lossy_raw(field, str(output_path), False)

            self.assertEqual(field["codec"], "sperr")
            self.assertEqual(field["encoded_shape"], [1, values.size])
            self.assertEqual(FakeSPERR.quality, 0.01)
            self.assertEqual(FakeSPERR.mode, "pwe")
            np.testing.assert_array_equal(
                np.fromfile(output_path, dtype=np.float32),
                values,
            )


if __name__ == "__main__":
    unittest.main()
