# Payload-search experiments

For the HACC optimization, see [the full-data report](hacc_tie_sort.md).
After a baseline XnYZip/SZO roundtrip **without** `--clean-raw` or
`--xnyzip-tie-sort`, reproduce the velocity ordering/predictor sweep with:

```bash
python -m experiments.hacc_tie_search /tmp/hacc-opt/baseline-1m \
  --limit 1000000 --output /tmp/hacc-tie-search.json
```

This compares native order, velocity-X sorting, Morton and Hilbert keys at
6/8/10 bits, each with default SZO and first-order Lorenzo. It decodes each
candidate and checks its error bound. Its reported sizes are velocity-only;
use full roundtrips to compare packaged CR, including metadata and sidecars.

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
```

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
