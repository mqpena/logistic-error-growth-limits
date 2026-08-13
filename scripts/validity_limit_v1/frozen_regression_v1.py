"""Recompute the three frozen p-values from their stored null arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_v1 import load_json_verified

NULL_SHA = "3c56548db321c352df687bfd4adb90c518046a8c12782f30d7ba926347bdbf53"
PRIMARY = "n512_re940_marginal"


def recompute(null_path: Path) -> dict:
    data = load_json_verified(null_path, NULL_SHA)
    regime = data["regimes"][PRIMARY]
    scalar = regime["scalar"]
    phase = regime["phase_scramble"]
    observed_scalar = float(scalar["observed_slope"])
    observed_field = float(phase["observed_field_slope"])
    tb = np.asarray(scalar["time_block"]["slopes"], dtype=float)
    ct = np.asarray(scalar["cross_truth_reassignment"]["slopes"], dtype=float)
    ps = np.asarray(phase["slopes"], dtype=float)
    if (tb.size, ct.size, ps.size) != (9999, 384, 999):
        raise RuntimeError("unexpected null distribution sizes")
    counts = {
        "time_block": int(np.count_nonzero(tb >= observed_scalar)),
        "cross_truth": int(np.count_nonzero(ct >= observed_scalar)),
        "phase_scramble": int(np.count_nonzero(ps >= observed_field)),
    }
    return {
        "counts_ge_observed": counts,
        "p_time_block": (1 + counts["time_block"]) / (tb.size + 1),
        "p_cross_truth": counts["cross_truth"] / ct.size,
        "p_phase_scramble": (1 + counts["phase_scramble"]) / (ps.size + 1),
    }
