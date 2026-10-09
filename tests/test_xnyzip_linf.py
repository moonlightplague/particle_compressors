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
from src.compress import CompressionPipeline, CompressionSettings
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.hacc_snapshot import HACC_FIELDS
from src.metrics import component_compression_ratios
from src.xnyzip_linf import (
    apply_outlier_codes, estimate_linf_l2_bound, make_outlier_codes,
    reconstruct_source_positions, restore_linf_outliers, write_linf_outliers,
    linf_axis_scales, _velocity_sample_rows,
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

    def test_velocity_geometry_and_bounded_chunk_sampling(self):
        bounds = dict(zip(VELOCITY_FIELDS, (.5, 2., 4.)))
        factors = linf_axis_scales(bounds)
        self.assertEqual(factors, {'vx': 1., 'vy': .25, 'vz': .125})
        self.assertAlmostEqual(estimate_linf_l2_bound(
            bounds, dict.fromkeys(bounds, 0.), dict.fromkeys(bounds, 100.), factors), math.sqrt(5) / 4)
        count, chunk = 12341, 257
        sample = _velocity_sample_rows(count, chunk)
        self.assertLessEqual(len(sample), 8 * chunk)
        self.assertEqual(sample[-1], count - 1)
        for start in range(0, len(sample), chunk):
            block = sample[start:start + chunk]
            self.assertEqual(block[0] % chunk, 0)
            np.testing.assert_array_equal(block, np.arange(block[0], block[0] + len(block)))
        self.assertEqual(len(_velocity_sample_rows(10_000_000, 1)), 8)

    def test_outliers_compose_velocity_and_input_permutations(self):
        rng = np.random.default_rng(345)
        with tempfile.TemporaryDirectory() as tmp:
            original = rng.normal(size=117).astype('float64')
            source_order = rng.permutation(len(original))
            native_order = rng.permutation(len(original))
            decoded = original[source_order[native_order]].astype('float32')
            with patch('src.xnyzip_linf.OUTLIER_BLOCK_VALUES', 19):
                metadata, stats = write_linf_outliers(Path(tmp) / 'vx.outliers', decoded, original,
                                                     native_order, 0., 1., source_order=source_order)
                self.assertEqual(stats['exact_count'], len(original))
                restored = decoded.astype('float64')
                restore_linf_outliers(restored, metadata)
                self.assertEqual(restored.tobytes(), original[source_order[native_order]].tobytes())

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
            with self.assertRaisesRegex(RuntimeError, 'requires an XnYZip position or velocity compressor'):
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


class VelocityLinfRoundtripTests(unittest.TestCase):
    setUpClass = classmethod(LinfRoundtripTests.setUpClass.__func__)
    run_cli = LinfRoundtripTests.run_cli

    def check_package(self, root, source, arrays, extra=(), native=False, position_codec='xnyzip'):
        work, config = root / 'work', root / 'config.yaml'
        config.write_text('advanced: {}\n')
        self.run_cli(['roundtrip', str(source), '--config', str(config), '--work-dir', str(work),
                      '--pos-compressor', position_codec, '--vel-compressor', 'xnyzip',
                      '--xnyzip-linf-bound', '--metrics', *extra])
        manifest = json.loads((work / 'manifest.json').read_text())
        metrics = json.loads((work / 'metrics.json').read_text())
        fields = manifest['compressed_fields']
        self.assertEqual(fields['velocities']['error_bound_norm'], 'linf')
        if position_codec == 'xnyzip':
            self.assertEqual(fields['positions']['error_bound_norm'], 'linf')
        else:
            self.assertNotEqual(fields['positions'].get('error_bound_norm'), 'linf')
        self.assertTrue(all(row['satisfied'] for row in metrics['error_bound_consistency'].values()), metrics)
        self.assertTrue(metrics['xnyzip_l2_error_bound_consistency']['velocities']['satisfied'])
        order = np.fromfile(manifest['artifacts']['preprocessed']['position_order'], dtype='uint64' if position_codec == 'xnyzip' else 'int32')
        scale = manifest['position_scale']['value']
        with h5py.File(work / 'reconstructed.h5') as h5:
            expected_decode = {key: h5[key][:] for key in h5}
            for key, original in arrays.items():
                actual = h5[HACC_FIELDS[key] if native else key][:]
                self.assertEqual(actual.dtype, original.dtype)
                if key == 'id':
                    np.testing.assert_array_equal(actual, original[order])
                else:
                    divisor = scale if key in POSITION_FIELDS else 1.
                    error = np.abs(actual.astype('float64') / divisor - original[order].astype('float64') / divisor)
                    self.assertLessEqual(float(error.max()), manifest['field_error_bounds'][key]['abs'])
        velocity_field = fields['velocities']
        velocity_bytes = velocity_field['bytes'] + fields['velocity_order']['bytes']
        velocity_bytes += sum(row['bytes'] for row in velocity_field['linf_outliers'].values())
        self.assertEqual(component_compression_ratios(manifest)['vxyz']['compressed_bytes'], velocity_bytes)
        tuning = manifest['xnyzip_velocity_linf_tuning']
        if tuning['candidates']:
            self.assertEqual(tuning['selected']['total_bytes'], min(row['total_bytes'] for row in tuning['candidates']))
            for row in tuning['candidates']:
                self.assertGreater(row['velocity_order_bytes'], 0)
                self.assertEqual(row['total_bytes'], sum(row[key] for key in (
                    'velocity_bytes', 'velocity_order_bytes', 'outlier_bytes', 'metadata_bytes')))
        source.unlink() if source.is_file() else shutil.rmtree(source)
        for directory in ('preprocessed', 'decompressed', 'input_adapters'):
            if (work / directory).exists():
                shutil.rmtree(work / directory)
        (work / 'reconstructed.h5').unlink()
        self.run_cli(['decompress', '--config', str(config), '--work-dir', str(work)])
        with h5py.File(work / 'reconstructed.h5') as h5:
            for key, expected in expected_decode.items():
                self.assertEqual(h5[key][:].tobytes(), expected.tobytes())
        return manifest

    def test_hacc_relative_bounds_single_stream_and_parallel_partial_chunks(self):
        rng = np.random.default_rng(702)
        for chunk in (0, 257):
            with self.subTest(chunk=chunk), tempfile.TemporaryDirectory() as tmp:
                root, count = Path(tmp), 4099
                source = root / 'hacc'
                source.mkdir()
                cells = rng.integers(0, 12, count)
                arrays = {k: (cells * (axis + 1) + rng.uniform(0, .02, count)).astype('float32')
                          for axis, k in enumerate(POSITION_FIELDS)}
                arrays.update({k: rng.normal(0, sigma, count).astype('float32')
                               for k, sigma in zip(VELOCITY_FIELDS, (10., 1000., 7.))})
                for k, values in arrays.items():
                    values.tofile(source / f'{HACC_FIELDS[k]}.f32')
                manifest = self.check_package(root, source, arrays, ['--rel-eb', '.001', '--vel-chunk-size', str(chunk),
                                            '--vel-chunk-workers', '2', '--pcodec-level', '3'], True)
                self.assertEqual(manifest['compressed_fields']['velocity_order']['index_scope'], 'chunk_local' if chunk else 'global')
                self.assertEqual(manifest['compressed_fields']['velocity_order']['pcodec_compression_level'], 3)
                self.assertLessEqual(manifest['xnyzip_velocity_linf_tuning']['sample_count'], 8 * chunk if chunk else count)
                self.assertTrue(all('velocity_score' in row for row in manifest['xnyzip_linf_tuning']['candidates']))

    def test_zero_and_tiny_float64_velocity_bounds_with_scaled_integer_positions_and_ids(self):
        rng = np.random.default_rng(703)
        for bound, chunk in ((0., 0), (1e-12, 97)):
            with self.subTest(bound=bound, chunk=chunk), tempfile.TemporaryDirectory() as tmp:
                root, count, scale = Path(tmp), 353, 1024.
                source = root / 'input.h5'
                arrays = {k: (rng.uniform(10, 20, count) * scale).astype('int32') for k in POSITION_FIELDS}
                arrays.update({k: rng.normal(0, 3, count).astype('float64') for k in VELOCITY_FIELDS})
                arrays['id'] = rng.permutation(count).astype('uint64') + 100
                with h5py.File(source, 'w') as h5:
                    h5.attrs['bitwidth'] = scale
                    for k, values in arrays.items():
                        h5[k] = values
                # Real native codec with a coarse tolerance forces exact escapes
                # and exercises corrections before both ordering permutations.
                with patch('src.compress.tune_linf_velocity_bound', return_value=(
                        'cube', .05, dict.fromkeys(VELOCITY_FIELDS, 1.), {'sample_count': count, 'candidates': []})):
                    manifest = self.check_package(root, source, arrays, ['--pos-abs-eb', '.001', '--vel-abs-eb', str(bound),
                                                  '--vel-chunk-size', str(chunk), '--vel-chunk-workers', '2'])
                stats = manifest['compressed_fields']['velocities']['linf_validation']
                self.assertEqual(sum(row['exact_count'] for row in stats.values()), 3 * count)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'tools/LCP/build/bin/lcp').is_file(), 'Build LCP for mixed codec test')
    def test_lcp_positions_receive_only_xnyzip_velocity_corrections(self):
        rng = np.random.default_rng(704)
        with tempfile.TemporaryDirectory() as tmp:
            root, count = Path(tmp), 257
            source = root / 'input.h5'
            arrays = {k: rng.uniform(1, 8, count).astype('float32') for k in POSITION_FIELDS}
            arrays.update({k: rng.normal(0, 3, count).astype('float32') for k in VELOCITY_FIELDS})
            with h5py.File(source, 'w') as h5:
                for k, values in arrays.items():
                    h5[k] = values
            manifest = self.check_package(root, source, arrays, ['--rel-eb', '.001', '--vel-chunk-size', '97'],
                                          position_codec='lcp')
            self.assertNotIn('linf_outliers', manifest['compressed_fields']['positions'])

    def test_zero_velocity_bounds_and_constant_axis_without_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, count = Path(tmp), 197
            source, work = root / 'input.h5', root / 'work'
            arrays = {k: np.linspace(1, axis + 2, count).astype('float32') for axis, k in enumerate(POSITION_FIELDS)}
            arrays.update({k: (np.linspace(-1, axis + 1, count) if axis else np.full(count, 3.250000001)).astype('float64')
                           for axis, k in enumerate(VELOCITY_FIELDS)})
            arrays['id'] = np.arange(count, dtype='uint64')
            with h5py.File(source, 'w') as h5:
                for k, values in arrays.items():
                    h5[k] = values
            self.run_cli(['roundtrip', str(source), '--work-dir', str(work), '--pos-compressor', 'xnyzip',
                          '--vel-compressor', 'xnyzip', '--xnyzip-linf-bound', '--rel-eb', '.001', '--vel-abs-eb', '0'])
            self.assertFalse((work / 'metrics.json').exists())
            manifest = json.loads((work / 'manifest.json').read_text())
            order = np.fromfile(manifest['artifacts']['preprocessed']['position_order'], dtype='uint64')
            with h5py.File(work / 'reconstructed.h5') as h5:
                for key in VELOCITY_FIELDS:
                    self.assertEqual(h5[key][:].tobytes(), arrays[key][order].tobytes())

    def test_velocity_native_validation_retries_and_rejects_corruption(self):
        rng = np.random.default_rng(705)
        actual_decode = CompressionPipeline._decode_and_measure_linf_velocities
        for recover in (True, False):
            with self.subTest(recover=recover), tempfile.TemporaryDirectory() as tmp:
                root, count = Path(tmp), 257
                source, work = root / 'input.h5', root / 'work'
                with h5py.File(source, 'w') as h5:
                    for key in POSITION_FIELDS:
                        h5[key] = rng.uniform(1, 8, count).astype('float32')
                    for key in VELOCITY_FIELDS:
                        h5[key] = rng.normal(0, 3, count).astype('float32')
                    h5['id'] = np.arange(count, dtype='uint64')
                calls = []

                def measured(pipeline, *args):
                    result = actual_decode(pipeline, *args)
                    calls.append(args[3])
                    return result if recover and len(calls) > 1 else (math.inf, *result[1:])

                output = StringIO()
                with patch('src.compress.tune_linf_velocity_bound', return_value=(
                        'to', .05, dict.fromkeys(VELOCITY_FIELDS, 1.), {'sample_count': count, 'candidates': []})), \
                     patch.object(CompressionPipeline, '_decode_and_measure_linf_velocities', measured), \
                     redirect_stdout(output), redirect_stderr(output):
                    result = main.main(['roundtrip', str(source), '--work-dir', str(work), '--pos-compressor', 'xnyzip',
                                        '--vel-compressor', 'xnyzip', '--xnyzip-linf-bound', '--rel-eb', '.001'])
                self.assertEqual(calls, ['to', 'cube'])
                self.assertEqual(result, 0 if recover else 2, output.getvalue())
                if recover:
                    field = json.loads((work / 'manifest.json').read_text())['compressed_fields']['velocities']
                    self.assertEqual(field['quantizer'], 'cube')
                    self.assertEqual(field['compression_attempts'], 2)
                else:
                    self.assertIn('XnYZip velocities failed native L2 validation', output.getvalue())

    def test_force_removes_velocity_corrections_when_disabling_mode_or_changing_codec(self):
        rng = np.random.default_rng(706)
        for codec, enabled in (('xnyzip', False), ('szo', True)):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as tmp:
                root, count = Path(tmp), 257
                source, work = root / 'input.h5', root / 'work'
                with h5py.File(source, 'w') as h5:
                    for key in POSITION_FIELDS:
                        h5[key] = rng.uniform(1, 8, count).astype('float32')
                    for key in VELOCITY_FIELDS:
                        h5[key] = rng.normal(0, 3, count).astype('float32')
                    h5['id'] = np.arange(count, dtype='uint64')
                base = ['roundtrip', str(source), '--work-dir', str(work), '--pos-compressor', 'xnyzip',
                        '--pos-abs-eb', '.01', '--vel-abs-eb', '.01']
                with patch('src.compress.tune_linf_velocity_bound', return_value=(
                        'cube', .05, dict.fromkeys(VELOCITY_FIELDS, 1.), {'sample_count': count, 'candidates': []})):
                    self.run_cli(base + ['--vel-compressor', 'xnyzip', '--xnyzip-linf-bound'])
                self.assertTrue(list((work / 'compressed').glob('v*.outliers')))
                self.run_cli(base + ['--vel-compressor', codec, '--force',
                                    '--xnyzip-linf-bound' if enabled else '--no-xnyzip-linf-bound'])
                self.assertFalse(list((work / 'compressed').glob('v*.outliers')))
                manifest = json.loads((work / 'manifest.json').read_text())
                self.assertFalse(manifest['compressed_fields'].get('velocities', {}).get('linf_outliers'))


if __name__ == '__main__':
    unittest.main()
