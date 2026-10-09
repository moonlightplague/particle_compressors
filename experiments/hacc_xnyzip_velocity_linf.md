# HACC XnYZip position and velocity Linf bounds

`--xnyzip-linf-bound` now enforces the requested per-axis error on both triplets
when `--pos-compressor xnyzip --vel-compressor xnyzip` is selected. It also
supports LCP positions with XnYZip velocities, independent position/velocity
bounds, velocity chunking, and structure-aware velocity orders.

The complete HACC input contains 280,953,867 particles, six float32 fields,
and no IDs. These runs use relative error `1e-3` and pcodec level 12. Total CR
includes both native streams, the lossless velocity permutation, every outlier
stream, and the manifest. Full measurements and tuning candidates are in
[results/hacc_xnyzip_velocity_linf.json](results/hacc_xnyzip_velocity_linf.json).

| Mode | Total CR | Package bytes | Position outlier bytes | Velocity outlier bytes |
| --- | ---: | ---: | ---: | ---: |
| Existing L2 | 10.94098496 | 616,296,689 | 0 | 0 |
| Both triplets Linf | **9.92873835** | **679,128,865** | 82,648 | 102,176 |

Enforcing all six bounds reduces total CR by **9.25%** and increases
package bytes by **10.20%**. Corrections total
**184,824 bytes**. Most additional bytes come from
finer native quantization and the velocity-order sidecar, rather than corrections.
The original L2 run exceeds all six requested per-axis bounds.

| Axis | Requested per-axis bound | L2 maximum coordinate error | Linf maximum coordinate error |
| --- | ---: | ---: | ---: |
| x | 0.063999996185302735 | 0.3288421630859375 | 0.063999995589256287 |
| y | 0.25599998474121094 | 0.328857421875 | 0.25599998235702515 |
| z | 0.25599998474121094 | 0.328857421875 | 0.25599998235702515 |
| vx | 6.9087478027343749 | 11.063720703125 | 6.908747673034668 |
| vy | 7.2435625000000003 | 11.063720703125 | 7.2435622215270996 |
| vz | 7.2655366210937498 | 11.063720703125 | 7.2655364871025085 |

Every Linf reconstruction passes the strict source comparison without a
numerical tolerance. Both diagnostic vector envelopes also pass. The full
reconstructed maxima exactly match the encoder's independently checked
source-dtype correction maxima. Positions use cube quantization with native
L2 tolerance `0.11085124507715867` and factors `x=1, y=.25, z=.25`.
Velocities use cube quantization with tolerance `11.966302211015782`
and factors `{"vx": 1.0, "vy": 0.9537776201605735, "vz": 0.9508929846525687}`.

The position sample compares combined position and velocity costs, using a
fixed geometry-based velocity candidate to compare position orders. A second
sample search tunes the velocities in the chosen canonical or hybrid input
order. Its score includes actual native bytes, pcodec permutation bytes at the
requested compression level, compact correction bytes, and estimated variable
manifest bytes. Both searches select the smallest measured candidate; they
estimate cost and do not prove a global optimum.

Velocity corrections are stored in native decoded row order and applied before
restoring the velocity permutation. Most store signed multiples of twice the
axis bound; exact source escapes handle rounding and preprocessing precision
loss. On the full input, 12,775 velocity coordinates need corrections, including
319 exact escapes. Temporary raw files are removed by `--clean-raw`.

A second HACC run uses 1,048,593 particles, velocity chunks of 65,536, and two
chunk workers, including a 17-particle final chunk. All six bounds pass; total
CR is 7.52149551. Decoding this package
again after raw cleanup reproduces all six fields **bit for bit**; SHA-256 values
are included in the results JSON. Native fixture tests additionally remove the
source itself before standalone decoding.

Regression coverage includes zero/tiny bounds, constant velocity axes, float64
exact escapes, scaled integer positions, IDs, global/chunk-local/hybrid velocity
orders, non-default pcodec levels, native TO-to-cube retry and corruption
rejection, and removing stale corrections on forced reruns. Existing XnYZip/SZO
position tie sorting also passes. `--xnyzip-tie-sort` retains its existing SZO,
SZ3, and pcodec velocity support; it does not support XnYZip velocities.

127 of 128 tests pass. The sole failure is the existing
`test_batch_summary_reports_byte_weighted_field_group_crs`: ANSI color codes
interrupt a plain-text assertion. The same failure was reproduced on an
unmodified archive of HEAD.

Reproduce the complete Linf run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python main.py roundtrip \
  data/EXASKY-HACC-data-medium-size \
  --work-dir particle_pipeline_runs/hacc_both_linf \
  --pos-compressor xnyzip --vel-compressor xnyzip --rel-eb 1e-3 \
  --xnyzip-linf-bound --metrics --clean-raw
```

Omit `--xnyzip-linf-bound` and use another work directory for the L2 comparison.
Add `--limit 1048593 --vel-chunk-size 65536 --vel-chunk-workers 2` for the chunked
validation. Run the regression suite with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
```
