# HACC XnYZip Linf bounds

The complete `data/EXASKY-HACC-data-medium-size` input contains 280,953,867
particles, six float32 fields, and no IDs. These runs use XnYZip positions,
SZO velocities, relative error `1e-3`, and `--xnyzip-tie-sort`. Total CR includes
the manifest and every compressed stream. Full numeric results are in
[results/hacc_linf_bound.json](results/hacc_linf_bound.json).

| Method | Total CR | Package bytes | Outlier bytes |
| --- | ---: | ---: | ---: |
| Existing L2 mode | 18.1461 | 371,589,469 | 0 |
| Linf corrections, unscaled coordinates | 14.4336 | 467,166,424 | 73,963,565 |
| Optimized `--xnyzip-Linf-bound` | **15.5254** | **434,313,322** | **82,687** |

The baseline exceeds every requested per-axis bound. Its maximum coordinate
errors are approximately 0.32884, 0.32886, and 0.32886, against bounds of about
0.064, 0.256, and 0.256. Its higher CR therefore comes with a weaker guarantee.
The unscaled Linf row is the initial implementation, retained as an optimization
comparison; the final CLI automatically searches scaled and unscaled candidates.

Scaling reduces package size by **7.03%** compared with unscaled Linf corrections.
The final CR is **14.44% lower** than the existing L2 mode. The selected native
quantizer is cube, with L2 tolerance `0.11085124507715867` and axis factors
`x=1, y=0.25, z=0.25`. The deterministic sample selects the same candidate after
including estimated variable manifest costs. This sample search estimates the
best package among its candidates; it does not prove a global optimum.

| Axis | Requested absolute bound | Observed maximum error | Outlier coordinates | Exact escapes |
| --- | ---: | ---: | ---: | ---: |
| x | 0.06399999618530273 | 0.06399999558925629 | 2,674 | 310 |
| y | 0.25599998474121094 | 0.25599998235702515 | 2,567 | 241 |
| z | 0.25599998474121094 | 0.25599998235702515 | 2,706 | 255 |

Every coordinate passes a strict comparison, without a metrics tolerance.
Corrections are computed after tie sorting; the final run moved 232,345,698
particles inside identical decoded-position runs. SZO retains its existing
default/Lorenzo predictor selection. Native position floats remain available
for structure-aware ordering, with output corrections applied during HDF5
reconstruction.

The full saved package was decoded again after `--clean-raw`. All six fields
matched the first reconstruction **bit for bit**, across the complete dataset.
Native fixture tests also remove both the source and input adapters before
decoding, and cover IDs, float64 and scaled integer positions, zero bounds,
constant axes, tiny bounds, structure-aware SZO, and chunked XnYZip velocities.

Validation command:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m unittest discover -s tests
```

119 of 120 tests pass. The remaining test,
`test_batch_summary_reports_byte_weighted_field_group_crs`, also fails on an
unmodified checkout of HEAD: batch summaries insert ANSI color codes into text
the test expects without codes. All new Linf tests and existing native codec
roundtrip tests pass.

Reproduce the optimized full run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python main.py roundtrip \
  data/EXASKY-HACC-data-medium-size \
  --work-dir particle_pipeline_runs/hacc_linf \
  --pos-compressor xnyzip --vel-compressor szo --rel-eb 1e-3 \
  --xnyzip-Linf-bound --xnyzip-tie-sort --metrics --clean-raw
```
