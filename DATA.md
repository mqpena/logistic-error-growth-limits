# Data and reproducibility

The public dataset is **Surface-quasigeostrophic forecast-error growth:
two-regime simulation dataset and reproducibility materials**, by Malaquias
Peña: [10.5281/zenodo.22822514](https://doi.org/10.5281/zenodo.22822514).

## Download and verify

From the Zenodo record, download both `sqg_logistic_data_v1.zip` and
`sqg_logistic_reproduction_v1.zip`, together with the eight companion files.
Keep all ten files in a download directory. Verify the uploaded files there:

```sh
shasum -a 256 -c SHA256SUMS.txt
mkdir extracted
unzip sqg_logistic_data_v1.zip -d extracted
unzip sqg_logistic_reproduction_v1.zip -d extracted
cd extracted
shasum -a 256 -c ../EXTRACTED_FILES.sha256
```

Both ZIP files merge into `SQG_Logistic/v1/`. The original scientific audit
outputs are in `SQG_Logistic/audit_20260911/`. Keep the ZIPs and their checksum
files together as the immutable published reference.

The data archive contains 19 NPZ files: eight trajectory/diagnostic archives,
eight initial truth snapshots, two stationarity archives, and one compact
figure dataset. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) and
[FILE_INVENTORY.json](FILE_INVENTORY.json). The original data archive is
1,688,544,194 bytes, with SHA-256
`2e00b7a5df7bc27e4d8fdaf1a6c26d4c00cfd73ada5ac4e61dea4ff3007724b0`.

The reproduction archive includes the exact model sources, supporting code,
provenance and offline dependency wheels. The offline installer targets
CPython 3.13/macOS arm64; Python itself must be installed separately. Its
`dependencies/bootstrap.py` restores a new environment with verified wheels.

## Repository checks with the raw data

Return to the repository root and activate the CPU environment:

```sh
PRL_SQG_DATA=/path/to/extracted/SQG_Logistic/v1/data/raw \
  python scripts/check_release.py
```

This enables the archived cluster hash, shape and metadata checks, in addition
to the convention and frozen-null regression tests. The full NPG audit command
is in [README.md](README.md).

To reconstruct the compact Figure 1 input from raw fields, use the documented
portable figure script in the extracted reproduction bundle. The raw-to-compact workflow reads saved scalar arrays and is CPU-only. The
repository's default figure commands operate directly on the existing compact
input; neither figure workflow requires MLX.

## Evidence and provenance

The repository's `data/prl_figure1_data_v1.npz` contains only compact derived
trajectory vectors; `results/` contains frozen preprint evidence. Their
scientific contents are preserved in this update. Raw fields, wheels and
generated audit caches are excluded from GitHub.

The current audit sources come from the published reproduction bundle. Their
exact source and delivered hashes are in
[docs/audit_source_provenance.json](docs/audit_source_provenance.json).
Only the relocation-guard test's dummy forbidden path is changed to remove a
workstation-specific directory. Numerical code and tolerances are unchanged.

Some historical files retain the publication-pending wording and local paths
from when they were frozen. The Zenodo record establishes publication status;
the licenses at this repository's root establish the author-selected reuse
terms. Historical paths in evidence are not active dependencies of the current
audit. Never replace a scientific input hash just to make a check pass.
