# Installation and verification

Run from the repository root, inside the Python environment that will execute
`main.py`. Python 3.13 is tested. Required build tools are a C++20 compiler,
CMake, Make, Git, and Rust/Cargo. Conan is installed by `requirements.txt`.

```bash
git submodule update --init --recursive
conda create -n compressor python=3.13
conda activate compressor
bash install.sh
python -m unittest discover -s tests -q
```

`install.sh` builds LCP and XnYZip and installs the pinned Python requirements,
including the local SZO and pcodec bindings. Initial setup requires network
access for dependencies. Python development headers must be available.

## Structure-aware XnYZip mode

No new dependencies or native-source changes are required. Existing working
installations only need the updated Python sources. If bindings are missing,
install the repository versions:

```bash
python -m pip install -e tools/pcodec/pco_python
python -m pip install ./tools/SZo/tools/pyszo
python -m unittest tests.test_structured_layout -v
```

The real-codec integration test is skipped if XnYZip or its Python dependencies
are unavailable; check that it actually runs when verifying an installation.
It tests mixed velocity dtypes, exact IDs, bounded errors, constant fields, and
decoding after source and temporary raw files have been removed.

Use `--xnyzip-structure-aware` with `--pos-compressor xnyzip` and either
`--vel-compressor szo` or `--vel-compressor xnyzip`. XnYZip velocities also
support `--vel-chunk-size` in this mode and retain their codec-order sidecar.
The existing configuration stays opt-out. Format-9 packages need the updated
decoder; old packages remain supported.

Start with one native partition and `--field-workers 1`. Each partition contains
roughly 16 million particles; the new order construction uses several temporary
arrays proportional to that count. Increase `--file-workers` only as RAM and
temporary disk permit. Merging all 531 million particles requires much more
memory; full-snapshot validation was run on a 251-GiB host. Temporary exports,
the merged source, and reconstructed HDF5 are much larger than the compressed
package. `--clean-raw` removes temporary raw directories after a successful run,
but retains the merged source and reconstructed HDF5.

See [README.md](README.md#higher-compression-for-snapshot_7) for commands and
[the benchmark notes](docs/snapshot7_structure_aware.md) for quality checks.
