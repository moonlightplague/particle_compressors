"""Strict source-dtype bounds, compact corrections, and particle associations."""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import main
from src.cli import build_parser, load_config
from src.compress import CompressionSettings
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.hacc_snapshot import HACC_FIELDS
from src.xnyzip_linf import (
    apply_outlier_codes, estimate_linf_l2_bound, make_outlier_codes,
    reconstruct_source_positions, restore_linf_outliers, write_linf_outliers,
    linf_axis_scales,
)


class LinfCorrectionTests(unittest.TestCase):
    def test_geometry_estimate_accounts_for_smallest_axis_and_preprocessing(self):
        bounds = dict(zip(POSITION_FIELDS, (1., 2., 4.)))
        preprocess = dict(zip(POSITION_FIELDS, (.1, .2, .3)))
        ranges = dict(zip(POSITION_FIELDS, (100., 200., 400.)))
        self.assertAlmostEqual(estimate_linf_l2_bound(bounds, preprocess, ranges), .9 * math.sqrt(5) / 2)
        bounds['x'] = 0.
        self.assertGreater(estimate_linf_l2_bound(bounds, preprocess, ranges), 0.)

    def test_axis_normalization_spends_each_coordinate_budget(self):
        bounds = dict(zip(POSITION_FIELDS, (.064, .256, .256)))
        factors = linf_axis_scales(bounds)
        self.assertEqual(factors, {'x': 1., 'y': .25, 'z': .25})
        for key in POSITION_FIELDS:
            self.assertEqual(bounds[key] * factors[key], .064)

    def test_signed_steps_and_exact_escapes_for_source_dtypes(self):
        rng = np.random.default_rng(810)
        for dtype, scale, bound in [('float32', 1., .1), ('float64', 7., .001),
                                    ('int32', 1024., 1e-5), ('float64', 1., 0.)]:
            with self.subTest(dtype=dtype, scale=scale, bound=bound):
                original = rng.uniform(-10, 10, 10003).astype(dtype)
                decoded = (original.astype('float64') / scale + rng.normal(0, 10 * max(bound, 1e-6), len(original))).astype('float32')
                codes, escape, exact, stats = make_outlier_codes(decoded, original, bound, scale)
                base = reconstruct_source_positions(decoded, original.dtype, scale)
                actual = apply_outlier_codes(base, codes, escape, exact, 2 * bound * scale)
                error = np.abs(actual.astype('float64') / scale - original.astype('float64') / scale)
                self.assertLessEqual(float(error.max()), bound)
                self.assertEqual(actual.dtype, original.dtype)
                self.assertEqual(stats['validated_max_abs_error'], float(error.max()))
                if dtype == 'float32':
                    self.assertGreater(stats['outlier_count'], 0)
                    self.assertLess(stats['exact_count'], stats['outlier_count'] // 100)
                if bound == 0:
                    np.testing.assert_array_equal(actual, original)

    def test_preprocessing_errors_too_small_for_a_step_preserve_source_bits(self):
        original = np.array([1.000000001, 3.000000004, -4.000000003], dtype='float64')
        decoded = original.astype('float32')
        codes, escape, exact, stats = make_outlier_codes(decoded, original, 1e-12, 1.)
        self.assertEqual(stats['exact_count'], len(original))
        actual = apply_outlier_codes(decoded.astype('float64'), codes, escape, exact, 2e-12)
        self.assertEqual(actual.tobytes(), original.tobytes())

    def test_multiframe_sidecar_roundtrip_and_rejects_truncation(self):
        rng = np.random.default_rng(82)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.outliers'
            source = rng.normal(size=115).astype('float64')
            order = rng.permutation(len(source)).astype('uint64')
            decoded = (source[order] + rng.normal(0, .3, len(source))).astype('float32')
            with patch('src.xnyzip_linf.OUTLIER_BLOCK_VALUES', 17):
                metadata, stats = write_linf_outliers(path, decoded, source, order, .01, 1.)
                self.assertGreater(stats['outlier_count'], 0)
                actual = reconstruct_source_positions(decoded, source.dtype, 1.)
                restore_linf_outliers(actual, metadata)
                self.assertLessEqual(float(np.abs(actual - source[order]).max()), .01)
                self.assertEqual(metadata['bytes'], path.stat().st_size)
                payload = path.read_bytes()
                for damaged in (payload[:-1], b'bad', payload + b'extra'):
                    path.write_bytes(damaged)
                    with self.assertRaises(RuntimeError):
                        restore_linf_outliers(actual.copy(), metadata)

    def test_zero_outliers_produce_no_stream_and_force_removes_stale_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.outliers'
            values = np.arange(20, dtype='float32')
            path.write_bytes(b'stale')
            metadata, stats = write_linf_outliers(path, values, values, np.arange(20), .01, 1., True)
            self.assertIsNone(metadata)
            self.assertFalse(path.exists())
            self.assertEqual(stats['outlier_count'], 0)

    def test_cli_yaml_enable_disable_alias_and_codec_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.yaml'
            config.write_text('advanced:\n  xnyzip_linf_bound: true\n')
            base = ['roundtrip', 'unused.h5', '--config', str(config), '--pos-compressor', 'xnyzip']
            self.assertTrue(CompressionSettings.from_args(build_parser(base).parse_args(base)).xnyzip_linf_bound)
            for flag in ('--no-xnyzip-Linf-bound', '--no-xnyzip-linf-bound'):
                args = build_parser(base + [flag]).parse_args(base + [flag])
                self.assertFalse(CompressionSettings.from_args(args).xnyzip_linf_bound)
            args = base + ['--pos-compressor', 'szo']
            with self.assertRaisesRegex(RuntimeError, 'requires --pos-compressor xnyzip'):
                CompressionSettings.from_args(build_parser(args).parse_args(args))
            config.write_text('advanced:\n  xnyzip_linf_bound: 1\n')
            with self.assertRaisesRegex(RuntimeError, 'must be a boolean'):
                load_config(str(config))


class LinfRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (Path(__file__).resolve().parents[1] / 'tools/XnYZip/build/XnYZip').is_file():
            raise unittest.SkipTest('Build XnYZip for native Linf tests')
        from src.runtime import load_pcodec, load_pyszo
        try:
            load_pcodec()
            load_pyszo()
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(str(exc))

    def run_cli(self, argv):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = main.main(argv)
        self.assertEqual(result, 0, output.getvalue())

    def check_package(self, root, source, arrays, extra, native=False):
        work = root / 'work'
        config = root / 'config.yaml'
        config.write_text('advanced: {}\n')
        self.run_cli(['roundtrip', str(source), '--config', str(config), '--work-dir', str(work),
                      '--pos-compressor', 'xnyzip', '--vel-compressor', 'szo',
                      '--xnyzip-Linf-bound', '--metrics', *extra])
        manifest = json.loads((work / 'manifest.json').read_text())
        metrics = json.loads((work / 'metrics.json').read_text())
        self.assertEqual(manifest['compressed_fields']['positions']['error_bound_norm'], 'linf')
        self.assertEqual(manifest['format_version'], 10)
        self.assertTrue(all(row['satisfied'] for row in metrics['error_bound_consistency'].values()))
        self.assertTrue(metrics['xnyzip_l2_error_bound_consistency']['positions']['satisfied'])
        order = np.fromfile(manifest['artifacts']['preprocessed']['position_order'], dtype='uint64')
        scale = manifest['position_scale']['value']
        with h5py.File(work / 'reconstructed.h5') as h5:
            expected_decode = {key: h5[key][:] for key in h5}
            for key, original in arrays.items():
                name = HACC_FIELDS[key] if native else key
                actual = h5[name][:]
                self.assertEqual(actual.dtype, original.dtype)
                if key == 'id':
                    np.testing.assert_array_equal(actual, original[order])
                else:
                    divisor = scale if key in POSITION_FIELDS else 1.
                    error = np.abs(actual.astype('float64') / divisor - original[order].astype('float64') / divisor)
                    self.assertLessEqual(float(error.max()), manifest['field_error_bounds'][key]['abs'])
            self.assertNotIn('id', h5) if 'id' not in arrays else None
        if '--xnyzip-tie-sort' in extra:
            self.assertGreater(manifest['position_tie_sort']['moved_particles'], 0)
            self.assertEqual(manifest['position_tie_sort']['sidecar_bytes'], 0)
        source.unlink() if source.is_file() else shutil.rmtree(source)
        shutil.rmtree(work / 'preprocessed')
        shutil.rmtree(work / 'decompressed')
        if (work / 'input_adapters').exists():
            shutil.rmtree(work / 'input_adapters')
        (work / 'reconstructed.h5').unlink()
        self.run_cli(['decompress', '--config', str(config), '--work-dir', str(work)])
        with h5py.File(work / 'reconstructed.h5') as h5:
            for key, expected in expected_decode.items():
                np.testing.assert_array_equal(h5[key][:], expected)
        return manifest

    def test_hacc_relative_bounds_with_and_without_tie_sort_and_sample_tuning(self):
        rng = np.random.default_rng(178)
        for ties in (False, True):
            with self.subTest(ties=ties), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'hacc'
                source.mkdir()
                cells = rng.integers(0, 8, 12001)
                arrays = {k: (cells * (axis + 1) + rng.uniform(0, .02, len(cells))).astype('float32')
                          for axis, k in enumerate(POSITION_FIELDS)}
                arrays.update({k: rng.normal(0, 100, len(cells)).astype('float32') for k in VELOCITY_FIELDS})
                for k, values in arrays.items():
                    values.tofile(source / f'{HACC_FIELDS[k]}.f32')
                extra = ['--rel-eb', '.001', '--vel-rel-eb', '.01']
                if ties:
                    extra.append('--xnyzip-tie-sort')
                manifest = self.check_package(root, source, arrays, extra, True)
                tuning = manifest['xnyzip_linf_tuning']
                self.assertEqual(tuning['selected']['total_bytes'], min(row['total_bytes'] for row in tuning['candidates']))
                self.assertEqual(tuning['sample_count'], len(cells))

    def test_corrections_after_ties_with_float64_and_scaled_integer_positions(self):
        rng = np.random.default_rng(180)
        for dtype, bound, scale in [('float32', .02, 1.), ('float64', 1e-9, 1.), ('int32', 1e-6, 1024.)]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'input.h5'
                cells = rng.integers(0, 8, 4099)
                arrays = {k: ((10 + cells + rng.uniform(0, .01, len(cells))) * scale).astype(dtype) for k in POSITION_FIELDS}
                arrays.update({k: rng.normal(0, 100, len(cells)).astype('float64') for k in VELOCITY_FIELDS})
                arrays['id'] = rng.permutation(len(cells)).astype('uint64') + 100
                with h5py.File(source, 'w') as h5:
                    h5.attrs['bitwidth'] = scale
                    for k, a in arrays.items():
                        h5[k] = a
                # Force a useful coarse native lattice to exercise corrections
                # despite tiny requested bounds and preprocessing precision loss.
                # IDs bypass sample tuning. Alter only the native estimate, not
                # the requested per-axis budget or the real codec/validation.
                with patch('src.xnyzip_linf.estimate_linf_l2_bound', return_value=.05):
                    manifest = self.check_package(root, source, arrays,
                                                  ['--pos-abs-eb', str(bound), '--vel-abs-eb', '.1', '--xnyzip-tie-sort'])
                fields = manifest['compressed_fields']['positions']
                self.assertGreater(sum(row['outlier_count'] for row in fields['linf_validation'].values()), 0)
                if dtype != 'float32':
                    self.assertGreater(sum(row['exact_count'] for row in fields['linf_validation'].values()), 0)

    def test_zero_bounds_and_constant_axis_with_no_metrics(self):
        for zero in (False, True):
            with self.subTest(zero=zero), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, work = root / 'input.h5', root / 'work'
                count = 197
                arrays = {k: (np.linspace(0, axis + 1, count) if axis else np.full(count, 3.25)).astype('float32')
                          for axis, k in enumerate(POSITION_FIELDS)}
                with h5py.File(source, 'w') as h5:
                    for k, a in arrays.items():
                        h5[k] = a
                    for axis, k in enumerate(VELOCITY_FIELDS):
                        h5[k] = np.linspace(0, axis + 1, count).astype('float32')
                self.run_cli(['roundtrip', str(source), '--work-dir', str(work), '--pos-compressor', 'xnyzip',
                              '--vel-compressor', 'szo', '--xnyzip-Linf-bound', '--pos-rel-eb', '0' if zero else '.001',
                              '--vel-abs-eb', '.001'])
                self.assertFalse((work / 'metrics.json').exists())
                m = json.loads((work / 'manifest.json').read_text())
                order = np.fromfile(m['artifacts']['preprocessed']['position_order'], dtype='uint64')
                with h5py.File(work / 'reconstructed.h5') as h5:
                    for k, a in arrays.items():
                        self.assertLessEqual(float(np.abs(h5[k][:].astype('float64') - a[order].astype('float64')).max()),
                                             m['field_error_bounds'][k]['abs'])


if __name__ == '__main__':
    unittest.main()
