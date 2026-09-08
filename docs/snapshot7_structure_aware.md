# `snapshot_7`: structure-aware compression

## Data and bottleneck

The 32 nonempty native partitions contain 531,441,000 particles (`810³`) and
17,006,112,000 raw field bytes. IDs span 0–531,440,999; their high, low, and middle
base-810 digits correspond to physical X, Y, and Z. Positions are fixed-point
int32 normalized by 2,147,483,647, velocities are float32, and IDs are uint64.

Existing merged XnYZip/SZO output at relative bound `1e-4` occupied
4,097,201,748 bytes (CR 4.150665). Its ID stream alone was 1,581,105,826 bytes,
positions 485,079,794 bytes, and velocities 2,030,940,159 bytes. Thus improving
only the position codec would miss most of the remaining storage cost.

## Experiments and selected design

The initial parameter search used the complete `dat_7.1` partition, not a short
prefix. These are exploratory component measurements, excluding manifest bytes:

| Experiment | Result |
| --- | --- |
| Dense periodic SZO/SZO layout, merged snapshot | CR 3.6016; worse than XnYZip/SZO |
| Hilbert rather than Morton XnYZip order | Position stream 23,299,395 vs 23,490,015 bytes |
| Mesh Morton codes for IDs in baseline order | 38,007,384 vs 48,676,203 bytes |
| Mesh Hilbert codes for IDs in baseline order | 37,004,298 bytes at pcodec default level |
| Hybrid velocity cells: 4/5/6/7/8/9 bits per axis | 61.85/60.75/59.82/59.41/60.58/63.08 MB total, default SZO |
| First-order Lorenzo on hybrid VX | 19,801,586 vs 20,106,030 bytes, baseline position order |

The implementation combines Hilbert XnYZip position order, reversible physical-axis
Hilbert IDs with pcodec level 12, and 7-bit Eulerian-cell/Lagrangian-Morton
velocity ordering with first-order SZO Lorenzo prediction. Individual exploratory
gains should not be added: changing position order changes the other streams.
Outer Zstandard repacking and velocity displacement predictors were explored but
not included; their gains did not justify extra machinery.

The encoder decodes its position archive before constructing velocity keys.
The decoder recreates exactly the same stable permutation from those decoded
float32 positions and exact IDs, then scatters velocities back into canonical
XnYZip rows. This adds no lossy transform, changes no error bound, and stores
no permutation sidecar. Duplicate sort keys retain canonical order.

### Native boundary issue and correctness guard

Full-data auditing exposed a pre-existing native TO quantizer boundary case:
at a shifted Z coordinate of zero, an odd-lattice tie can select Z = -1.
The native unsigned block encoder cannot represent this correctly. Both Morton
and Hilbert order can be affected; it also reproduces with a small regular mesh
at tight bounds. The initial unguarded batch failed on `dat_7.14` and `dat_7.24`.
Those outputs are **not** accepted as correct compression results.

The final structure-aware encoder decodes and checks every position before
encoding IDs and velocities. A failed TO check triggers a cube-quantizer retry
at the identical L2 bound; if that also fails, compression raises an error.
The selected quantizer is stored in the manifest and used at decode time.
No native submodule change or rebuild is needed. The regression suite includes
the small tight-bound case and standalone decoding of the fallback package.

## Measured results

All CR figures include the compressed files and manifest; neither reconstructed
HDF5 nor temporary raw exports count as compressed payload. Baselines are the
existing repository run artifacts; new runs explicitly enable `--metrics`.

| Scope | Baseline bytes | New bytes | Baseline CR | New CR |
| --- | ---: | ---: | ---: | ---: |
| `dat_7.1`, 16,876,851 particles | 137,829,873 | 118,452,334 | 3.9183 | 4.5593 |
| Merged snapshot, 531,441,000 particles | 4,097,201,748 | 3,417,999,761 | 4.1507 | 4.9755 |
| All 32 partitions, final guarded mode | 4,090,900,249 | 3,583,847,961 | 4.1571 | 4.7452 |

The final independent-partition run reduces bytes by **12.3946%**. All 32
packages pass exact-ID, seven-field, and position-vector L2 checks. Only
`dat_7.14` and `dat_7.24` require the cube fallback. The run completed in
448 s with four file workers. Its audited [batch comparison](snapshot7_batch_comparison.json)
includes component sizes, bounds, observed errors, and per-package timings.

The merged result reduces bytes by **16.5772%** and increases CR by approximately
19.9%. IDs drop from 1,581,105,826 to 1,194,012,700 bytes; velocities from
2,030,940,159 to 1,742,168,282 bytes; positions from 485,079,794 to 481,742,688 bytes.
Every ID is exact, all seven field checks pass, and the position-vector L2 check
passes. The full merged run took 789 s to compress, 409 s to decode/recombine,
and 635 s for detailed quality metrics on the validation host (concurrent batch
work was also running, so these are not isolated performance measurements).

Machine-readable audits: [merged comparison](snapshot7_merged_comparison.json)
and [final guarded partition comparison](snapshot7_partition_comparison.json).
The merged run used the same transforms before the encoder-side guard was added;
its complete post-decode metrics independently verify its correctness. The final
guarded `dat_7.1` run reproduced identical compressed component bytes; its manifest
size differs slightly, giving 118,452,265 total bytes.

The partition result reduces packaged bytes by approximately 14.1%. Its new
component sizes are: positions 23,299,395; IDs 36,592,823; VX 19,814,649;
VY 19,340,832; VZ 19,387,756 bytes. The position L2 error was
0.0000858584119633 against a requested bound of 0.0000859459895229. Velocity
maximum errors were 0.0143364668, 0.0183186531, and 0.0165285319, each below its
unchanged component bound. IDs were exact.

## Reproduction

```bash
# Independent partition mode; remove --xnyzip-structure-aware for a fresh baseline.
python main.py roundtrip data/new_data/snapshot_7 \
  --work-dir /tmp/snapshot7_structured_batch \
  --pos-compressor xnyzip --vel-compressor szo --rel-eb 1e-4 \
  --xnyzip-structure-aware --file-workers 4 --field-workers 1 \
  --metrics --clean-raw

# Single merged package.
python main.py roundtrip data/new_data/snapshot_7 --merge \
  --work-dir /tmp/snapshot7_structured_merged \
  --pos-compressor xnyzip --vel-compressor szo --rel-eb 1e-4 \
  --xnyzip-structure-aware --field-workers 3 --metrics --clean-raw

python -m unittest discover -s tests -q
```

The regression suite passes 90 tests, including all existing codec-ordering
tests, Hilbert inverses for every supported bit width, sparse/non-power-of-two
meshes, zero/one-based IDs, all six axis mappings, mixed velocity dtypes,
constant fields, metadata rejection, failed-native validation, and standalone
decoding with source/preprocessing files removed. The final guarded mode was
also checked on `dat_7.1` at `1e-3`: CR 6.9118, exact IDs, and all bounds satisfied
(no equal-bound baseline was rerun for that additional check).

Use distinct work directories; existing outputs are protected by default.
`python -m experiments.compare_structured BASELINE_DIR OPTIMIZED_DIR --output REPORT.json`
checks identical particle counts and error-bound metadata, exact IDs, all seven
field-bound checks, and the position-vector L2 check before reporting actual
on-disk CR. It accepts either one package or directories of matching partition
packages. Decompression needs the package's manifest and compressed files, not
the original snapshot or preprocessing outputs.
Use `--batch` for the comparison tool when a directory contains both a merged
root package and per-partition subdirectories, as the existing snapshot baseline
does. Reports refuse any optimized package with missing or failed quality checks.

## Trade-offs

This is an opt-in data-layout optimization, not a universally better codec.
Cell resolution was tuned on one partition at `1e-4`; other snapshots or bounds
can favor different settings. Full merging and independent partitioning have
different data ranges and therefore different absolute bounds at the same
relative setting; compare each only against its own equal-bound baseline.
Extra sorting and transforms increase runtime and peak RAM. Native-codec
defaults, existing layouts, and existing package decoding remain available.
