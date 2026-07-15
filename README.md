# Particle Compressors

Particle Compressors is a command-line pipeline for compressing particle data
stored in HDF5. It combines specialized codecs for each component of a particle
record, reconstructs the original HDF5 layout, and can calculate roundtrip error
and compression metrics. The pipeline preserves dataset names, dtypes, dataset attributes, and root HDF5
attributes in the reconstructed file.

## Input Format

The input must be an HDF5 file containing seven one-dimensional datasets with
the same length. Dataset basenames are matched case-insensitively using these
aliases:

| Logical field | Accepted dataset basenames | Required dtype |
| --- | --- | --- |
| ID | `id`, `particle_id`, `pid` | Any integer dtype |
| X | `x`, `posx`, `position_x` | Numeric |
| Y | `y`, `posy`, `position_y` | Numeric |
| Z | `z`, `posz`, `position_z` | Numeric |
| VX | `vx`, `velx`, `velocity_x` | `float32` or `float64` |
| VY | `vy`, `vely`, `velocity_y` | `float32` or `float64` |
| VZ | `vz`, `velz`, `velocity_z` | `float32` or `float64` |

Datasets may be located in HDF5 groups; matching uses only the final component
of each dataset path.

## Requirements

- Python with development headers; Python 3.13 is known to work
- A C++17 compiler, CMake, and Make for LCP
- Rust and Cargo for the local pcodec Python extension
- Git submodules initialized for `tools/LCP` and `tools/pcodec`

## Installation

Clone or initialize the native-code submodules:

```bash
git submodule update --init --recursive
```

From the repository root, the included installation script builds LCP and
installs the Python dependencies into the active Python environment:

```bash
bash install.sh
```


## Quick Start

Start with a limited roundtrip to validate the environment and input schema:


Run the full file with relative error bound of `1e-3`:

```bash
python main.py roundtrip data/sample.h5 \
  --config config.yaml \
  --work-dir particle_pipeline_runs \
  --rel-eb 1e-3 \
  --force 
```

Use a distinct work directory for each input/error-bound combination. Existing
outputs are rejected unless `--force` is supplied.

With the default `--vel-compressor sz3`, the compressed directory contains
`positions.lcp`, `order.pco`, `id.pco`, `vx.psz`, `vy.psz`, and `vz.psz`.
With `--vel-compressor lcp`, the three velocity files are replaced by
`velocities.lcp` and `velocity_order.pco`. The pipeline does not split fields
into parts. As a result, preprocessing,
compression, decompression, reconstruction, and metrics load a complete field
into memory at once. Manifests produced by the older part-based format are not
accepted by this format.

LCP sorts each input triplet independently. Position and velocity LCP streams
therefore require separate permutations to restore the original HDF5 row
order. Particle IDs cannot replace those permutations under the input contract:
IDs identify records, but they are not required to equal row indices or occur
in a canonical order. Storing IDs in each LCP-sorted order would still require
two sidecars, guaranteed-unique IDs, and an ID-to-original-row join during
reconstruction. Direct 32-bit permutations provide bounded memory use and
linear-time scatter reconstruction.


## Error Bounds

The default `--rel-eb 1e-3` applies to positions and velocities.
Position- and velocity-specific bounds can be supplied through
`--pos-abs-eb`, `--pos-rel-eb`, `--vel-abs-eb`, and `--vel-rel-eb`. A
field-class-specific value takes precedence over the global `--abs-eb` or
`--rel-eb`. LCP accepts one bound for the velocity triplet, so the strictest
derived `vx`/`vy`/`vz` bound is used. Do not set both absolute and relative
bounds for the same field class. IDs are always compressed losslessly.
`id_abs_eb` only defines the expected ID error used by metrics and defaults to
zero.
