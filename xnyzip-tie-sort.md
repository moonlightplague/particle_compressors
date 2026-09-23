# How `--xnyzip-tie-sort` works

`--xnyzip-tie-sort` improves velocity compression by changing particle order
**only within runs of exactly identical decoded positions**. It uses the
freedom created by lossy position quantization to put similar velocity vectors
near each other. The position archive stays unchanged, and the package needs
no additional permutation sidecar.

The intended HACC configuration is XnYZip for positions and SZO for velocities.
On all 280,953,867 particles in `data/EXASKY-HACC-data-medium-size`, this mode
improved packaged CR from **12.5182 to 18.1462** at relative error `1e-3`, using
**31.01% fewer compressed bytes** with the same requested bounds.

## The alignment problem

Each particle has a position vector `(x, y, z)` and a velocity vector
`(vx, vy, vz)`, optionally accompanied by an ID. These fields must describe
the same particle in every reconstructed row.

XnYZip sorts positions during compression. The pipeline adopts that decoded
position order as its canonical row order and rearranges velocities and IDs
to follow it. Its position permutation is temporary encoder metadata; it is
not included in the compressed package.

Compressing velocities independently with XnYZip introduces another sort.
That velocity order generally differs from the position order, so the package
must retain an alignment sidecar. On the full HACC benchmark, the XnYZip
velocity stream was only 3.53 MB, but its order sidecar cost 601.51 MB.

With fieldwise SZO velocities, there is no independent velocity sort and no
such sidecar. Tie sorting improves the order that SZO receives while retaining
this property.

## What makes a position tie useful?

Lossy position quantization can map several distinct source positions to the
same decoded float32 triple. If consecutive rows all decode to `P`, changing
which particle occupies each of those rows leaves the position output intact.

For example, suppose a velocity ordering prefers particles B, C, A:

| Canonical row | Decoded position | Particle before sorting | Particle after sorting |
| --- | --- | --- | --- |
| 0 | P | A, with velocity VA | B, with velocity VB |
| 1 | P | B, with velocity VB | C, with velocity VC |
| 2 | P | C, with velocity VC | A, with velocity VA |
| 3 | Q | D, with velocity VD | D, with velocity VD |

Each particle still receives its original decoded position: A, B, and C
receive P; D receives Q. IDs, when present, move with the velocity vectors.
All three velocity components use the **same** permutation.

The implementation requires exact equality of all three decoded position
bit patterns. It does not group merely nearby positions. It also distinguishes
`+0.0` from `-0.0`. Only contiguous runs are eligible; equal positions in
separate runs are not brought together.

## Encoder procedure

1. **Compress and validate positions.** The existing XnYZip path compresses
   positions, decodes them, and checks the per-particle L2 error. Its existing
   quantizer/bound retries run before tie sorting. The resulting native order
   maps each decoded row to a source particle.

2. **Process fixed work blocks.** Tie sorting visits 1,048,576 canonical rows
   at a time. It views the decoded float32 positions as uint32 bit patterns
   and starts a new run whenever any component changes. Blocks containing
   no ties are skipped. A run crossing a work-block boundary is split.

3. **Build velocity keys.** The encoder gathers source velocities through
   the current position permutation. Within the work block, it normalizes
   each velocity component independently and assigns a 10-bit coordinate:

   ```text
   coordinate = floor(1023 * (velocity - block_min) / (block_max - block_min))
   ```

   A constant component receives coordinate zero. The implementation performs
   these calculations in float64 and requires finite velocities in blocks
   being sorted. Normalization is per work block, not per position run.

4. **Sort within each run.** The three coordinates are combined into a
   Morton code and converted to a 3-D Hilbert key. A stable lexicographic sort
   uses the position-run number as its primary key and the velocity Hilbert
   key as its secondary key. Equal Hilbert keys retain their prior order.
   An explicit invariant checks that no row crossed a run boundary.

5. **Update the temporary particle mapping.** The refined source permutation
   replaces the native position permutation in `preprocessed/order.u64.raw`.
   IDs and velocities are exported using that mapping. Roundtrip metrics use
   the same mapping to compare reconstructed rows with the correct source
   particles.

6. **Compress the reordered fields.** In the flat SZO velocity path, each
   component is compressed twice: once with SZO's default configuration and
   once with first-order Lorenzo prediction. The smaller stream is retained.
   Both candidates use the same absolute error bound for that component.

The 10-bit coordinates are **sorting keys only**. They do not replace or
quantize the velocity values passed to SZO. Hilbert order tends to bring nearby
vectors together in all three velocity components, which can make prediction
residuals easier to compress. It is a heuristic, not a guarantee of the best
possible ordering.

The core operation can be summarized as:

```python
# order[j] identifies the source particle assigned to decoded position row j.
for block in canonical_row_blocks:
    groups = exact_position_run_ids(decoded_positions[block])
    source_rows = order[block].copy()
    keys = velocity_hilbert_keys(source_velocities[source_rows])

    local_order = stable_sort(primary=groups, secondary=keys)
    assert groups[local_order] == groups
    order[block] = source_rows[local_order]

# Export every associated field through the updated order.
```

## Why the position error bound is preserved

Let `O[j]` be the source particle assigned to decoded row `j`, and let
`P_hat[j]` be that row's decoded position. Before tie sorting, position
validation establishes:

```text
norm(P_hat[j] - P[O[j]]) <= position_L2_bound
```

Let `s[j]` select another row in the same exact-position run. The refined
source mapping is `O_new[j] = O[s[j]]`. Because the two decoded positions
are identical:

```text
P_hat[j] = P_hat[s[j]]

norm(P_hat[j] - P[O_new[j]])
    = norm(P_hat[s[j]] - P[O[s[j]]])
    <= position_L2_bound
```

Thus every source particle keeps its previously validated decoded position
and its position error. Velocity prediction changes with the ordering, so
individual velocity errors may differ from the baseline, but the requested
velocity bounds are unchanged. IDs remain lossless.

## Why decoding needs no tie permutation

The position decoder emits the same position sequence it always emitted.
The fieldwise velocity decoders emit the sequence chosen by the encoder.
Those sequences already align: within each tie run, every position row is
identical. The decoder neither reconstructs the Hilbert keys nor reverses the
tie sort.

Existing package formats and decoders therefore remain sufficient. The
manifest records tie-sort statistics and the selected SZO profiles, but no
new order stream is stored. In particular, `sidecar_bytes: 0` refers to the
additional tie-sort sidecar; the manifest itself still contributes bytes.

As with ordinary XnYZip position compression in this repository, the package
does **not** restore the original input row order. It preserves complete
particle records in the chosen canonical order. For ID-free HACC input,
source-aligned metrics must run before the temporary source permutation is
deleted. Decoding itself needs neither that permutation nor the source data.

## Using the option

```bash
python main.py roundtrip data/EXASKY-HACC-data-medium-size \
  --work-dir particle_pipeline_runs/hacc_ties \
  --pos-compressor xnyzip --vel-compressor szo \
  --xnyzip-tie-sort --rel-eb 1e-3 --metrics --clean-raw
```

The YAML setting is `advanced.xnyzip_tie_sort: true`. It defaults to `false`;
`--no-xnyzip-tie-sort` disables it explicitly. The HACC example in `run.sh`
enables it.

The option requires XnYZip positions and fieldwise SZO, SZ3, or pcodec
velocities. The adaptive SZO predictor selection applies to the flat velocity
path; lattice and structure-aware velocity paths have their own ordering and
prediction choices. HACC requires no IDs, lattice inference, or extra libraries.

## Results and tradeoffs

Full HACC measurements at relative bound `1e-3`, including the manifest and
all compressed sidecars:

| Method | Packaged bytes | CR |
| --- | ---: | ---: |
| XnYZip positions + SZO velocities | 538,645,943 | 12.5182 |
| Same codecs + tie sorting and adaptive SZO | 371,587,418 | 18.1462 |

The position archive was byte-identical between these runs. All component
bounds and the position L2 bound passed. A fresh full-data decode with the
temporary ordering files unavailable reproduced all six reconstructed fields
bit-for-bit.

Tie sorting took about 33 seconds in the measured full run; total compression
time increased from about 44 to 83 seconds. Those runs overlapped other work,
so the timings are illustrative. The fixed blocks bound additional sorting
scratch space; the surrounding pipeline still retains full-field arrays and
the full source permutation.

Larger position ties offer more ordering freedom. Tight position bounds or
sparse data may leave few ties and little benefit. Splitting a long run at a
work-block boundary can reduce the achievable gain. Sorting may also make
an ID stream harder to compress. The implementation selects the smaller SZO
profile after reordering; it does not compare entire reordered packages with
the original-order package or guarantee a CR improvement on every dataset.

Implementation and supporting evidence:

- [Tie grouping and Hilbert sorting](src/position_ties.py): `sort_position_ties`.
- [Pipeline integration](src/compress.py): `_compress_canonical_xnyzip_positions`
  and `_compress_velocities`.
- [SZO profile selection](src/raw_codecs.py): `compress_szo_raw`.
- [Alignment and standalone-decode tests](tests/test_position_ties.py).
- [Detailed HACC benchmark report](experiments/hacc_tie_sort.md) and
  [machine-readable results](experiments/results/hacc_tie_sort.json).
- [Reproducible ordering/predictor search](experiments/hacc_tie_search.py).
