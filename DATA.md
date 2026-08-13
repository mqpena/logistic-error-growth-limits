# Data and artifact policy

The small JSON records needed to reproduce the supplemental audit figure are
included in `results/`. The 76-KB `data/prl_figure1_data_v1.npz` file contains
only the derived trajectory vectors used by manuscript Figure 1. It contains
no full SQG fields and can be loaded with `allow_pickle=False`. Its SHA-256 is
recorded in `MANIFEST.sha256`.

The raw N=256 and N=512 SQG truth-cluster NPZ files total approximately 1.6 GB
and are excluded. Their expected SHA-256 hashes and original relative locations
are recorded in the machine-readable evidence, including
`results/finite_band_truth_slopes_v1.json`. With those archives available,
the compact input can be reconstructed and checked using:

```bash
MPLCONFIGDIR=.mplconfig python scripts/make_prl_figure1_data_v1.py \
  --cluster-root /path/to/prl_v8_gate5_nonconfirmatory_20260715 \
  --write-compact --force
```

Set `PRL_SQG_DATA` to the directory containing the two regime subdirectories
before running the large-artifact provenance test. Never replace a hash merely
to make a check pass.
