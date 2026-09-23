# HACC position-tie ordering

Measurements on 2026-09-23 using the repository's native XnYZip and SZO
bindings and `data/EXASKY-HACC-data-medium-size`. No external dependencies or
submodule changes are required.

## Full dataset

The input has 280,953,867 particles, six float32 fields, no IDs, and
6,742,892,808 source payload bytes. All runs use `--rel-eb 1e-3`.

| Pipeline | Packaged bytes | Packaged CR |
| --- | ---: | ---: |
| XnYZip positions / SZO velocities | 538,645,943 | 12.518228 |
| XnYZip positions / XnYZip velocities + order | 618,663,813 | 10.899123 |
| XnYZip positions / SZO velocities + tie sort | 371,587,418 | 18.146182 |

Totals include the manifest and every compressed file. The optimized package
uses 31.0145% fewer bytes than the strongest baseline, increasing CR by
44.9580%. Byte totals can vary slightly with manifest paths and timing strings.

| Component | Baseline bytes | Optimized bytes |
| --- | ---: | ---: |
| Positions | 13,606,720 | 13,606,720 |
| VX | 176,519,648 | 121,766,544 |
| VY | 174,313,038 | 118,630,254 |
| VZ | 174,193,780 | 117,570,483 |

The all-XnYZip velocity stream is only 3,529,407 bytes, but its permutation
costs 601,513,479 bytes. The proposed mode avoids that independent velocity
sort and its sidecar. An exploratory collapse of duplicate decoded velocity
vectors into group IDs still needed 536,246,018 bytes at pcodec level 12,
before position/velocity streams; it was not incorporated.

## Why the ordering is valid

The XnYZip position decoder produces identical float32 triples for many
successive rows. Any permutation within one such run leaves all three decoded
position arrays bit-for-bit identical. The encoder applies that permutation
to complete particle records, including IDs when available. Therefore the
position error for each source particle and every particle association are
preserved. Velocities still use the original per-component error bounds.

The implementation identifies ties by exact uint32 position bits, including
the sign of zero, after the existing position decode/error validation. It
normalizes velocities within fixed 1,048,576-row work blocks, derives a 10-bit
per-axis Hilbert key, and sorts stably within each run. These coordinates are
only sorting keys: they do not quantize the stored velocities. Run identity
is checked again after sorting. Runs crossing work-block boundaries may be
split; this limits potential CR gain, not correctness.

There were 257,617,910 equal adjacent pairs inside these blocks, and
257,354,720 particle rows moved. The refined temporary source permutation is
used by all subsequent fields and metrics. No decoder change is necessary.

SZO retains the smaller of its default stream and a first-order Lorenzo
stream per field. Default SZO after tie sorting used 369,435,939 velocity
bytes; Lorenzo reduced that to 357,967,281. Both use identical requested
bounds. Production metadata records both candidate sizes and the selection.

## Accuracy and tests

All three full-data roundtrips passed their component and triplet bounds.
For the optimized and baseline XnYZip/SZO runs, position maximum L2 error is
0.36746224101030767 against 0.3676519874646664. Velocity bounds are
6.908747802734375, 7.2435625, and 7.26553662109375, unchanged by the optimization.

The position archives have the same SHA-256:
`a178a9958dc0208658d4eb35e203d6a52c673bddcf42dbfd85869e0deb990f0d`.
The optimized full package was also decoded with its preprocessing directory
and input adapters unavailable; all six reconstructed field hashes matched.

`python -m unittest discover -s tests` passes 105 tests. New coverage checks
exact permutations, block boundaries, unique positions, constant velocities,
signed zeros, adjacent floats, deterministic ties, nonfinite inputs, CLI/YAML
validation, exact IDs, float64 velocity fields, SZO/SZ3/pcodec, and standalone
decoding after removing the source and temporary files.

One-million-particle prefix roundtrips also passed all bounds:

| Relative bound | Baseline CR | Optimized CR |
| --- | ---: | ---: |
| 1e-4 | 4.688159 | 4.739428 |
| 1e-3 | 9.058696 | 9.722555 |
| 1e-2 | 25.084792 | 50.157369 |

Prefix bounds are derived from prefix ranges, so their absolute bounds differ
from those of the full dataset. The baseline and optimized run in each row
use the same bounds. Native XnYZip velocity relative bounds are L2 bounds,
whereas SZO velocity bounds are per-component; the all-XnYZip size comparison
does not imply identical distortion contracts across codecs.

Exploratory chunked-XnYZip tests also exposed an existing native velocity
corruption case: 2,049 synthetic float64 velocity triples with 64-row chunks
and absolute bound 0.5 returned errors above 1e8, including with tie sorting
disabled. This change does not repair that native path. The new option is
explicitly limited to fieldwise velocities; the existing XnYZip velocity
path and sidecar format are unchanged.

## Cost and reproduction

Observed full-data compression-stage times were 43.94 s for the baseline and
82.54 s for the optimized run; tie sorting took 33.08 s. Decode/reconstruction
took 10.26 s and 11.36 s respectively. Runs overlapped other work, so these
are illustrative wall times rather than controlled throughput measurements.
Full source-aligned metrics took several additional minutes.

```bash
python main.py roundtrip data/EXASKY-HACC-data-medium-size \
  --work-dir /tmp/hacc-opt/baseline-full \
  --pos-compressor xnyzip --vel-compressor szo --rel-eb 1e-3 --metrics
python main.py roundtrip data/EXASKY-HACC-data-medium-size \
  --work-dir /tmp/hacc-opt/ties-full \
  --pos-compressor xnyzip --vel-compressor szo --rel-eb 1e-3 \
  --xnyzip-tie-sort --metrics
python main.py roundtrip data/EXASKY-HACC-data-medium-size \
  --work-dir /tmp/hacc-opt/xnyzip-full \
  --pos-compressor xnyzip --vel-compressor xnyzip --rel-eb 1e-3 --metrics
```

Use fresh work directories, or `--force` to repeat a run. Add `--limit 1000000`
for prefix runs. `--metrics --clean-raw` calculates alignment-aware metrics
before removing scratch files. The [checked-in JSON summary](results/hacc_tie_sort.json)
records measured sizes, bounds, timings, and validation hashes.
