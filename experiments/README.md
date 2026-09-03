# Payload-search experiments

These drivers record the searches behind the optional periodic lattice package
layout. They extend the ID/position displacement, occupancy, and spatial
velocity analysis in `analysis/particle_metrics.py`. Run them from the
repository root so they can load the local codec bindings:

```bash
python -m experiments.payload_search data/new_data/dat_2.15.h5 \
  --id-base 1 --sz3 --output /tmp/payload-search.json
python -m experiments.layout_search data/new_data/dat_2.15.h5 \
  --id-base 1 --output /tmp/layout-search.json
python -m experiments.periodic_layout_search data/dat_2.1.h5 \
  --id-base 0 --axes --output /tmp/periodic-search.json
python -m experiments.orientation_search data/dat_2.1.h5 \
  --output /tmp/orientation-search.json
python -m experiments.coupled_search data/new_data/dat_2.15.h5 \
  --id-base 1 --output /tmp/coupled-search.json
python -m experiments.velocity_id_relationship data/dat_2.1.h5 \
  --baseline-manifest particle_pipeline_runs/dat_2.1.h5/manifest.json \
  --output /tmp/velocity-id-relationship.json
```

## Lattice-layout advantage visualization

Compare a sort-only package with a lattice package built from the same input,
codec, and error bounds:

```bash
python -m experiments.visualize_lattice_advantage \
  /tmp/sort-only/manifest.json particle_pipeline_runs/lattice/manifest.json \
  --ablation-json /tmp/payload-search.json \
  --output /tmp/lattice-advantage.html
```

The self-contained dashboard attributes package bytes saved to positions and
velocities, compares every field, and displays the flat-to-dense and
dense-to-position-residual interventions recorded by `payload_search.py`.
Supplying the ablation JSON is optional.

## Reconstruction visualization

Create a self-contained HTML dashboard for any completed roundtrip package:

```bash
python -m experiments.visualize_reconstruction \
  particle_pipeline_runs/sample-qoz
```

The dashboard plots signed reconstruction error against the original value
for every lossy field, maps 3-D position-error magnitude over the sampled
`x-y` particle locations, and summarizes the full metrics and requested
bounds. It uses the manifest to align sorted reconstructions with their source
rows. Use `--sample-size` and `--seed` to control the embedded deterministic
sample, `--output` to select the HTML path, or `--original-h5` when the source
file has moved since compression. No plotting package is required.

## X-y particle statistics

Create a self-contained HTML dashboard of particle statistics projected onto
the x-y plane. Inputs may be individual HDF5 files, directories, or a mixture;
all selected rows are accumulated into one common grid:

```bash
python -m experiments.visualize_xy_statistics data/new_data \
  --output /tmp/new-data-xy.html
```

The dashboard maps particle density, mean z, mean speed, 3-D velocity
dispersion, each mean velocity component, and the mean planar velocity vector.
Integer positions are normalized using each file's `bitwidth` attribute. The
default deterministic systematic sample analyzes at most five million rows;
use `--max-particles 0` for exact full-data statistics, `--bins` to change the
grid resolution, or `--extent XMIN XMAX YMIN YMAX` to compare several reports
on identical axes. No plotting package is required.

## Velocity scalar versus ID

The velocity relationship driver tests the requested signed scalar
`cbrt(vx^3 + vy^3 + vz^3)`. This is not the conventional nonnegative L3
magnitude. It stably sorts by particle ID, reconstructs the periodic dense ID
lattice, compares flat and lattice compression, and verifies the decoded
absolute-error bound.

On all 53,957,517 particles in `data/dat_2.1.h5`, SZO at relative error
`1e-3` produced these particle-based compression ratios:

| Field and layout | Compressed bytes | Compression ratio |
| --- | ---: | ---: |
| `vx`, dense lattice | 19,824,481 | 10.8870 |
| `vy`, dense lattice | 21,475,569 | 10.0500 |
| `vz`, dense lattice | 20,434,282 | 10.5622 |
| Derived scalar, ID-sorted flat | 32,271,332 | 6.6880 |
| Derived scalar, dense lattice | 25,353,084 | 8.5130 |

The lattice reduced the scalar's compressed size by 21.44% compared with the
already ID-sorted flat stream, and adjacent ID-sorted scalar values had a
sampled Pearson correlation of 0.99084. This demonstrates strong local
ID/lattice coherence. The scalar has little global linear relationship with
the numeric ID itself (`r = 0.05889`), however, and its dense compression ratio
ranked last: it was 15.29% below even the least-compressible component (`vy`).
Thus the tested scalar follows local ID structure, but does not compress better
than `vx`, `vy`, or `vz` separately.

The searches established the following choices used by the production path:

- A periodic dense ID lattice exposes the spatial coherence that a flat
  ID-sorted stream misses. The full `dat_2.1.h5` partition unwraps to
  `152 x 600 x 600` cells at 98.606% occupancy.
- Position residuals against their inferred lattice coordinate reduce the
  three position streams to about 137 KB before package metadata and seam
  sidecars on that file.
- Independent dense SZO velocity streams outperform dense SZ3, flat SZO,
  stacked vector streams, and independently coded slabs at the tested
  relative error of `1e-3`.
- Linear flat-index interpolation was the best tested fill for missing cells.
  Fill values affect prediction only and are never returned as particles.
- Axis order materially affects SZO. Reversing the best axis order yields a
  smaller additional gain, so the production search tests those reversals too.
- A displacement-to-velocity affine predictor can shrink its unconstrained
  residual, but preserving the velocity bound requires a much tighter position
  stream. Its best error-safe `x + vx` result was roughly 13.49 MB versus about
  6.58 MB for independent dense fields, so it is intentionally excluded.

All reported production CRs come from `main.py roundtrip`, whose payload total
includes the manifest and every compressed sidecar and whose quality report
checks each field's requested bound.
