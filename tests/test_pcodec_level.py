"""Pcodec levels reach the encoder, including transformed and byte streams."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.cli import build_parser, load_config
from src.compress import CompressionSettings
from src.raw_codecs import compress_integer_raw, compress_lossy_raw, compress_lattice_hilbert_ids
from src.runtime import load_pcodec
from src.structured_layout import make_structured_layout, encode_lattice_ids


class PcodecLevelTests(unittest.TestCase):
    def test_config_default_override_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.yaml'
            argv = ['roundtrip', 'unused.h5', '--config', str(config)]
            for value, expected in [('{}', 12), ('{pcodec_level: 0}', 0), ('{pcodec_level: 12}', 12)]:
                config.write_text(f'advanced: {value}\n')
                args = build_parser(argv).parse_args(argv)
                self.assertEqual(CompressionSettings.from_args(args).pcodec_level, expected)
                override = argv + ['--pcodec-level', '4']
                self.assertEqual(build_parser(override).parse_args(override).pcodec_level, 4)
            for value in ('-1', '13', '1.5', 'true', 'null', '"3"'):
                config.write_text(f'advanced: {{pcodec_level: {value}}}\n')
                with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'pcodec_level'):
                    load_config(str(config))
            config.write_text('advanced: {}\n')
            for value in ('-1', '13', '1.5'):
                with self.subTest(cli=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    build_parser(argv).parse_args(argv + ['--pcodec-level', value])

    def test_payload_matches_explicit_native_level(self):
        standalone, Config = load_pcodec()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(2026)
            for level in (0, 3, 12):
                for dtype in ('uint64', 'int8', 'uint8', 'float64'):
                    with self.subTest(level=level, dtype=dtype):
                        values = rng.integers(0, 100, 10001).astype(dtype)
                        raw, packed = root / 'input.raw', root / 'output.pco'
                        values.tofile(raw)
                        args = ('pcodec', str(raw), dtype, str(packed), 'test', values.size)
                        if dtype == 'float64':
                            metadata = compress_lossy_raw(*args, 0.0, True, pcodec_level=level)
                        else:
                            metadata = compress_integer_raw(*args, True, pcodec_level=level)
                        config = Config(compression_level=level)
                        config.enable_8_bit = values.dtype.itemsize == 1
                        self.assertEqual(packed.read_bytes(), standalone.simple_compress(values, config))
                        self.assertEqual(metadata['pcodec_compression_level'], level)
                        np.testing.assert_array_equal(standalone.simple_decompress(packed.read_bytes()), values)
                layout = make_structured_layout(16, 0, (0, 1, 2), 7)
                ids = rng.permutation(16**3).astype('uint64')
                packed = root / 'ids.pco'
                metadata = compress_lattice_hilbert_ids(ids, 'uint64', str(packed), 'id', layout, True,
                                                       pcodec_level=level)
                self.assertEqual(packed.read_bytes(), standalone.simple_compress(
                    encode_lattice_ids(ids, layout), Config(compression_level=level)))
                self.assertEqual(metadata['pcodec_compression_level'], level)


if __name__ == '__main__':
    unittest.main()
