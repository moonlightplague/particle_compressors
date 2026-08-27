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
