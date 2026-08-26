# When Does Forecast-Error Energy Grow Logistically in Geophysical Turbulence?

Algorithms, frozen compact evidence records, and figure-generation code for
the preprint *When Does Forecast-Error Energy Grow Logistically in Geophysical Turbulence?*.

This repository distinguishes three claims that should not be conflated:

1. an exact full-field error-energy budget;
2. a conditional invariant-ray reduction that yields logistic growth; and
3. a moving-front similarity reduction that yields a spectrum-dependent power
   law.

The stored SQG audit records test the one-shape and one-clock conditions and
report the frozen relational-null result. They do not authorize or contain the
unlaunched confirmatory ensemble.

## Layout

- `scripts/validity_limit_v1/`: budget conventions, similarity verification,
  ordered validity gates, clock-shape decomposition, and unit tests.
- `scripts/prl_v8_*.py`: frozen two-regime generator/evaluator and null-gate
  algorithms.
- `scripts/make_prl_figure1_data_v1.py`: reproduces manuscript Figure 1 from
  the compact, hash-traceable derived data in `data/`.
- `scripts/make_prl_validity_figures_v2.py`: reproduces the supplemental audit
  figure from compact records in `results/`.
- `scripts/measure_finite_band_truth_slopes_v1.py`: recomputes the descriptive
  `k=10--20` truth-spectrum slopes reported in the supplement when the raw
  cluster archives are available.
- `results/`: dated machine-readable audit and verification records.
- `figures/`: released vector figures.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MPLCONFIGDIR=.mplconfig python scripts/make_prl_validity_figures_v2.py --force
MPLCONFIGDIR=.mplconfig python scripts/make_prl_figure1_data_v1.py --force
python -m unittest discover -s scripts/validity_limit_v1/tests
```

Both figure commands are CPU-only and require no raw SQG caches. The full SQG
generator and an optional raw-to-compact provenance reconstruction require
Apple silicon, MLX, and the large cluster artifacts described in `DATA.md`.

## Evidence boundary

The cross-truth reassignment null did not reject (`p=113/384`), so the frozen
intersection-union mechanism gate failed. This is non-identifiability in the
pilot, not falsification of covariance coupling.

## License

No software license has yet been selected. All rights remain reserved until a
LICENSE file is added by the principal investigator.
