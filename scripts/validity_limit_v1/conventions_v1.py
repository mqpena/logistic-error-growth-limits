"""Single source of truth for the PRL half-energy conventions."""
from __future__ import annotations

import numpy as np


def _real_inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.real(np.vdot(a, b)))


def Et(truth: np.ndarray) -> float:
    return 0.5 * _real_inner(truth, truth)


def Ef(forecast: np.ndarray) -> float:
    return 0.5 * _real_inner(forecast, forecast)


def C(truth: np.ndarray, forecast: np.ndarray) -> float:
    return 0.5 * _real_inner(truth, forecast)


def epsilon(truth: np.ndarray, forecast: np.ndarray) -> float:
    delta = forecast - truth
    return 0.5 * _real_inner(delta, delta)


def rho_from_moments(et: float, ef: float, covariance: float) -> float:
    if et <= 0 or ef <= 0:
        raise ValueError("Et and Ef must be positive")
    return covariance / np.sqrt(et * ef)


def x_C_from_moments(et: float, ef: float, covariance: float) -> float:
    denominator = et + ef
    if denominator <= 0:
        raise ValueError("Et + Ef must be positive")
    return 2.0 * covariance / denominator


def x_C_from_epsilon(et: float, ef: float, error: float) -> float:
    denominator = et + ef
    if denominator <= 0:
        raise ValueError("Et + Ef must be positive")
    return 1.0 - error / denominator


def normalized_production(production: np.ndarray, rate: np.ndarray,
                          error: np.ndarray) -> np.ndarray:
    denominator = np.asarray(rate) * np.asarray(error)
    if np.any(denominator == 0):
        raise ValueError("lambda * epsilon must be nonzero")
    return np.asarray(production) / denominator


def full_norm_error_from_half_energy(error: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(error)


def validate_metric_identity(et: np.ndarray, ef: np.ndarray, covariance: np.ndarray,
                             error: np.ndarray, *, rtol: float = 1e-12,
                             atol: float = 1e-14) -> None:
    expected = np.asarray(et) + np.asarray(ef) - 2.0 * np.asarray(covariance)
    if not np.allclose(np.asarray(error), expected, rtol=rtol, atol=atol):
        raise RuntimeError("half-energy metric identity failed")
