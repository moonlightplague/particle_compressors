"""Tests for dense shaped codec orientation metadata."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.shaped_codecs import (
    compress_shaped_lossy_raw,
    decompress_shaped_lossy_raw,
)


class ShapedCodecTests(unittest.TestCase):
    def test_adaptive_orientation_roundtrips_axis_reversals(self):
        class FakeConfig:
            def __init__(self, shape):
                self.shape = tuple(shape)
                self.errorBoundMode = None
                self.absErrorBound = None
                self.cmprAlgo = None

        class FakeErrorBoundMode:
            ABS = "abs"

        class FakeAlgorithms:
            INTERP_LORENZO = "interp_lorenzo"

        class FakeSZo:
            decoded_by_token = {}
            call_count = 0

            @classmethod
            def compress(cls, data, config, copy):
                del config, copy
                token = cls.call_count
                cls.call_count += 1
                cls.decoded_by_token[token] = np.asarray(data).copy()
                if data.shape == (2, 3, 4) and float(data.flat[0]) == 23.0:
                    size = 4
                elif data.shape == (2, 3, 4):
                    size = 20
                else:
                    size = 30
                return np.full(size, token, dtype=np.uint8), 1.0

            @classmethod
            def decompress(cls, payload, dtype, shape):
                token = int(np.asarray(payload, dtype=np.uint8)[0])
                decoded = cls.decoded_by_token[token].astype(dtype, copy=True)
                return decoded.reshape(shape), FakeConfig(shape)

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "input.raw"
            compressed_path = root / "field.szo"
            output_path = root / "decoded.raw"
            values.tofile(raw_path)
            with patch(
                "src.shaped_codecs.load_pyszo",
                return_value=(
                    FakeSZo,
                    FakeConfig,
                    FakeErrorBoundMode,
                    FakeAlgorithms,
                ),
            ):
                field = compress_shaped_lossy_raw(
                    "szo",
                    str(raw_path),
                    "float32",
                    str(compressed_path),
                    "vx",
                    values.size,
                    0.01,
                    False,
                    values.shape,
                    True,
                )
                self.assertEqual(field["axis_permutation"], [0, 1, 2])
                self.assertEqual(field["axis_flips"], [True, True, True])
                decompress_shaped_lossy_raw(
                    field,
                    str(output_path),
                    False,
                )

            restored = np.fromfile(output_path, dtype=np.float32).reshape(
                values.shape
            )
            np.testing.assert_array_equal(restored, values)

    def test_sperr_shaped_adapter_roundtrips_selected_orientation(self):
        class FakeSPERR:
            decoded_by_token = {}
            call_count = 0

            @classmethod
            def compress(cls, data, quality, mode):
                self.assertEqual(quality, 0.01)
                self.assertEqual(mode, "pwe")
                token = cls.call_count
                cls.call_count += 1
                cls.decoded_by_token[token] = np.asarray(data).copy()
                size = 4 if (
                    data.shape == (2, 3, 4)
                    and float(data.flat[0]) == 23.0
                ) else 20 + token
                return bytes([token]) + bytes(size - 1)

            @classmethod
            def decompress(cls, payload):
                token = int(np.asarray(payload, dtype=np.uint8)[0])
                return cls.decoded_by_token[token].copy()

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "input.raw"
            compressed_path = root / "field.sperr"
            output_path = root / "decoded.raw"
            values.tofile(raw_path)
            with patch(
                "src.shaped_codecs.load_sperr",
                return_value=FakeSPERR,
            ):
                field = compress_shaped_lossy_raw(
                    "sperr",
                    str(raw_path),
                    "float32",
                    str(compressed_path),
                    "vx",
                    values.size,
                    0.01,
                    False,
                    values.shape,
                    True,
                )
                decompress_shaped_lossy_raw(
                    field,
                    str(output_path),
                    False,
                )

            self.assertEqual(field["codec"], "sperr")
            self.assertEqual(field["axis_permutation"], [0, 1, 2])
            self.assertEqual(field["axis_flips"], [True, True, True])
            restored = np.fromfile(output_path, dtype=np.float32).reshape(
                values.shape
            )
            np.testing.assert_array_equal(restored, values)


if __name__ == "__main__":
    unittest.main()
