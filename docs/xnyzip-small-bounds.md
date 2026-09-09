# XnYZip position failures at small L2 bounds

The failure originates in the native XnYZip position codec. SZO is not involved:
standalone XnYZip compression and decompression reproduce it. The integration's
strict structure-aware check correctly detected the problem, but its retry with
cube quantization at the same bound did not account for native rounding error.

Two independent native behaviors matter:

- `TruncatedOctahedronQuantizer::quantize` can choose a negative odd lattice node
  on a zero-boundary tie. The unsigned block encoding does not preserve that node.
- Both TO and cube consume the full mathematical L2 budget during quantization.
  The float32 shift, quantization/recovery arithmetic, and offset restoration can
  push the actual error beyond it. Small bounds and large particle counts make
  a violating particle more likely. This is distinct from the input conversion
  allowance already reserved by `src/error_bounds.py`.

The relevant native implementations are under
`tools/XnYZip/source/quantizer/{truncated_octahedron,cube}_quantizer.hpp`,
`tools/XnYZip/source/preprocessor/shifting.hpp`, and
`tools/XnYZip/source/composer/block_{compressor,decompressor}_rle.hpp`.

## Reproduction

The retained failed `snapshot_7/dat_7.1` run contained 16,876,851 particles.
Its cube codec budget was `8.568587749101733e-6`; one particle reconstructed with
L2 error `8.57529864996768e-6` against the preprocessed float32 positions.

`tests/test_xnyzip_bounds.py` contains a four-particle reduction from that run,
including a boundary point that triggers the TO failure. With the existing
native binary and Hilbert/RLE block mode:

| Quantizer | Native bound | Maximum L2 error |
| --- | ---: | ---: |
| TO | 8.568587749e-6 | 4829.308614 |
| Cube | 8.568587749e-6 | 8.575298650e-6 |
| Cube | 8.482901872e-6 | 8.482902041e-6 |

The last row passes the original budget, even though rounding still causes a
small overshoot of the tighter native scale.

## Integration fix

Every XnYZip position stream is now decoded and measured in float64 before its
permutation is used for IDs and velocities. This applies to ordinary and
structure-aware layouts. A failed TO stream is retried with cube. If cube also
fails, the next native bound reserves the larger of 1% of the original budget
and twice the measured excess. Every attempt is checked against the original
available budget, which already excludes the source conversion allowance.

Only a verified stream is accepted. Its final permutation, quantizer, and bound
are used throughout the package. Both `error_bounds.positions_xnyzip_abs` and
`compressed_fields.positions.l2_error_bound` record the accepted native bound;
`field_error_bounds.positions_xnyzip.abs` keeps the user's requested bound.
The manifest also records the safety margin, attempt count, validation budget,
and measured float32-source error. Decompression honors the stored quantizer
for ordinary layouts as well as structure-aware layouts.

This is an integration workaround using the existing native binary. It does not
patch the XnYZip submodule or change the package format. Ordinary position
compression gains a validation decode; failed attempts add compression/decode
work. XnYZip velocity compression is outside this change.

## Validation

All 93 repository tests pass, including strict independent L2 checks, final row
ordering, refusal after unsuccessful retries, and decoding saved packages after
removing the source and preprocessed files.

Full `dat_7.1` roundtrips with XnYZip positions, SZO velocities, and
`--xnyzip-structure-aware --metrics --clean-raw` produced:

| Relative bound | Requested final L2 bound | Observed final L2 error | Attempts |
| --- | ---: | ---: | ---: |
| 1e-5 | 8.594598952e-6 | 8.478990610e-6 | 3 |
| 1e-6 | 8.594598952e-7 | 8.109254344e-7 | 3 |

All position, velocity, and ID checks passed in both runs. Final errors above
include reconstruction to the original integer position representation.

Retries cannot guarantee arbitrary precision from the native float32 codec.
At `--rel-eb 1e-7 --limit 200000`, a subset of the same partition still failed:
the measured error was `7.02886368e-8`, versus an available codec budget of
`2.4951704e-8`. The pipeline refuses such a result and reports the measured and
allowed errors. Retries are limited to six attempts, and stop earlier if no
positive next bound remains. Neither the requested bound nor the validation
tolerance is relaxed.
