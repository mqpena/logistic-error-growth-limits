from __future__ import annotations

import numpy as np

from scripts.validity_limit_v1.audit_mtb_ordered_gates_v1 import (
    clock_tier,
    geometry_tier,
    rankdata,
    spearman,
    weighted_line,
    weighted_sd,
)


def test_frozen_tier_boundaries() -> None:
    assert geometry_tier(0.05) == "ONE_SHAPE_LIMIT"
    assert geometry_tier(0.05001) == "MIXED"
    assert geometry_tier(0.25) == "MIXED"
    assert geometry_tier(0.25001) == "MOVING_FRONT"
    assert clock_tier(0.10) == "ONE_CLOCK"
    assert clock_tier(0.10001) == "MIXED_CLOCKS"
    assert clock_tier(0.40) == "MIXED_CLOCKS"
    assert clock_tier(0.40001) == "MANY_CLOCKS"


def test_weighted_line_and_dispersion() -> None:
    x = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 1.0])
    intercept, slope = weighted_line(x, 2.0 + 3.0 * x, weights)
    assert np.isclose(intercept, 2.0)
    assert np.isclose(slope, 3.0)
    assert np.isclose(weighted_sd(np.ones(3), weights), 0.0)


def test_rank_and_spearman_with_ties() -> None:
    assert np.allclose(rankdata(np.array([1.0, 1.0, 3.0])), [1.5, 1.5, 3.0])
    assert np.isclose(spearman(np.arange(4.0), np.arange(4.0)[::-1]), -1.0)
