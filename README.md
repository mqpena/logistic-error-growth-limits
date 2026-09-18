# SQG logistic error-growth limits

Simulation and analysis code for studying when forecast-error energy grows
logistically in surface-quasigeostrophic (SQG) turbulence. The repository supports
the preprint [When Does Forecast-Error Energy Grow Logistically in Geophysical
Turbulence?](https://doi.org/10.48550/arXiv.2608.26492) and its NPG revision,
*Spectral conditions for interpreting logistic forecast-error growth in
geophysical turbulence*. The journal revision has not yet been submitted.

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22822514.svg)](https://doi.org/10.5281/zenodo.22822514)

The published [dataset and reproduction bundle](https://doi.org/10.5281/zenodo.22822514)
contains the saved model fields, exact archived solver and generator, initial
conditions, stationarity diagnostics, configurations, offline dependencies, and
the completed September 2026 existing-data audit. Large data files remain on
Zenodo. See [DATA.md](DATA.md) for download and verification instructions.

## Reproduce the compact preprint figures

The tested CPU environment uses CPython 3.13 on macOS arm64. The default
requirements do not install MLX. Other operating systems require compatible
packages and are not covered by the archived platform validation.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_release.py
MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig python scripts/make_prl_validity_figures_v2.py --force
MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig python scripts/make_prl_figure1_data_v1.py --force
```

The figure commands use the compact files already in this repository. They
regenerate the versioned files under `figures/` and the Figure 1 record under
`results/`; PDF metadata and rendering can vary with the environment. Verify
`MANIFEST.sha256` before regenerating files if checking the delivered snapshot.
The checks run all ten existing tests, including three function-style tests
omitted by plain `unittest discover`; one raw-data test is skipped unless
`PRL_SQG_DATA` is set.

## Reproduce the NPG existing-data audit

Download and extract both Zenodo ZIP files as described in [DATA.md](DATA.md).
From this repository, using the CPU environment above:

```sh
python scripts/npg_audit_v1/run_audit.py \
  --model-root /path/to/extracted/SQG_Logistic/v1 \
  --output /path/to/new_audit_output
```

Choose an output directory that does not yet exist. Run this script directly;
its imports expect its own directory on the Python search path. It processes
the eight truth groups and sixteen perturbed trajectories without new model
integrations or statistical resampling. It blocks runtime access to the old
project directory and checks the dependencies used by each analysis.

**Expected scientific status: completed with partial failures.** Five N512
highest-band allocated-increment budget checks fail the original tolerance.
The primary-band dependencies and frozen headline summaries pass. A zero runner
exit code means this documented workflow completed; it does not mean every
scientific check passed. See [the audit guide](docs/NPG_AUDIT.md).

## Scientific scope

The analysis distinguishes the exact full-field error-energy budget, a
conditional invariant-ray reduction yielding logistic growth, and a moving-front
similarity reduction yielding a spectrum-dependent power law. A good logistic
fit alone does not establish the reduction's physical conditions.

- This is a two-regime, nonconfirmatory pilot: four separated truth states and
  two perturbed forecasts per truth state in each regime. Those four states
  are not four independent model campaigns.
- The frozen cross-truth reassignment null did not reject (`p=113/384`), and
  the original intersection-union mechanism gate failed. The pilot does not
  identify that mechanism; this result does not falsify covariance coupling.
- The five highest-band budget failures are retained. Primary-band results
  do not certify all bands, integration convergence, or an arbitrary new band.
- Post-audit matched-lifecycle diagnostics are descriptive. Saved diagnostics
  do not supply independent tangent clocks for arbitrary spectral shells,
  and sampled bound calculations do not certify continuous trajectories.

## Repository contents

- `scripts/npg_audit_v1/`: current portable audit, dated specifications and
  mathematical verification records.
- `scripts/validity_limit_v1/`: original conventions, validity gates, similarity
  checks and regression tests.
- `scripts/make_prl_*.py`, `data/`, `results/`, `figures/`: compact preprint
  figure workflow and frozen scientific evidence.
- `scripts/jas_sqg_jfm.py`, `scripts/prl_v8_*.py`: historical solver and campaign
  sources. Historical certification runners retain their original execution
  bindings; they are not the supported entrypoint for the current audit.
- `DATA_DICTIONARY.md`, `FILE_INVENTORY.json`: descriptions of the Zenodo
  dataset, including files that are intentionally absent from this repository.

For future model generation, use the exact archived implementation and portable
configuration supplied in the Zenodo reproduction bundle. Optional MLX requires
Apple silicon and, for the archived wheels, macOS 26 or later. See
`requirements-model-macos-arm64.txt` and the bundle's dependency README. Merely
installing the solver dependencies does not launch a simulation.

## Citation and licenses

Cite the [dataset](https://doi.org/10.5281/zenodo.22822514), the
[preprint](https://doi.org/10.48550/arXiv.2608.26492), and the repository version
or commit used. [CITATION.cff](CITATION.cff) provides software citation metadata.
The dataset DOI identifies the dataset and is not a software-release DOI.

Original code is [MIT licensed](LICENSE). Original data, results, figures and
scientific documentation are [CC BY 4.0](LICENSE-DATA.txt). Third-party material
retains its own terms; see [LICENSES.md](LICENSES.md).
