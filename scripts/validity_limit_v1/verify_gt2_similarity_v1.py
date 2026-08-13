#!/usr/bin/env python3
"""Controlled numerical checks for the M-T GT2 similarity reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad


def profile(x: float | np.ndarray, m: float) -> float | np.ndarray:
    return np.asarray(x) ** m / (1.0 + np.asarray(x) ** m)


def inertial_integral(n: float, m: float) -> tuple[float, float]:
    numerical = quad(lambda x: x ** (-n) * float(profile(x, m)), 0.0, np.inf, epsabs=1e-12)[0]
    analytical = np.pi / (m * np.sin(np.pi * (n - 1.0) / m))
    return numerical, analytical


def finite_energy(ke: float, n: float, m: float, klo: float, khi: float, amplitude: float = 1.0) -> float:
    return 2.0 * amplitude * quad(
        lambda k: k ** (-n) * float(profile(k / ke, m)), klo, khi, epsabs=1e-12
    )[0]


def finite_tendency_formula(ke: float, n: float, beta: float, gamma: float,
                            m: float, klo: float, khi: float,
                            amplitude: float = 1.0) -> float:
    zlo, zhi = klo / ke, khi / ke
    integral = quad(lambda x: x ** (-n) * float(profile(x, m)), zlo, zhi, epsabs=1e-12)[0]
    bracket = (
        (n - 1.0) * integral
        - zlo ** (1.0 - n) * float(profile(zlo, m))
        + zhi ** (1.0 - n) * float(profile(zhi, m))
    )
    return 2.0 * amplitude * gamma * ke ** (beta + 1.0 - n) * bracket


def finite_tendency_difference(ke: float, n: float, beta: float, gamma: float,
                               m: float, klo: float, khi: float) -> float:
    h = 1.0e-5 * ke
    derivative = (finite_energy(ke + h, n, m, klo, khi) - finite_energy(ke - h, n, m, klo, khi)) / (2.0 * h)
    return derivative * (-gamma * ke ** (beta + 1.0))


def heterogeneous_identity(seed: int = 20260811) -> float:
    rng = np.random.default_rng(seed)
    weights = rng.random(17)
    weights /= np.sum(weights)
    rates = np.exp(rng.normal(size=17))
    amplitudes = rng.uniform(0.05, 0.95, size=17)
    abar = float(np.sum(weights * amplitudes))
    lbar = float(np.sum(weights * rates))
    variance = float(np.sum(weights * (amplitudes - abar) ** 2))
    factor = amplitudes * (1.0 - amplitudes)
    covariance = float(np.sum(weights * (rates - lbar) * (factor - np.sum(weights * factor))))
    direct = float(np.sum(weights * rates * factor))
    decomposed = lbar * (abar * (1.0 - abar) - variance) + covariance
    return abs(direct - decomposed)


def verify() -> dict:
    integral_rows = []
    for n, m in ((5.0 / 3.0, 4.0), (2.0, 5.0), (3.0, 6.0)):
        numerical, analytical = inertial_integral(n, m)
        integral_rows.append({
            "n": n,
            "m": m,
            "numerical": numerical,
            "analytical": analytical,
            "relative_error": abs(numerical - analytical) / abs(analytical),
        })
    derivative_rows = []
    for n, beta, ke in ((5.0 / 3.0, 2.0 / 3.0, 14.0), (2.0, 0.5, 9.0), (3.0, 0.0, 18.0)):
        formula = finite_tendency_formula(ke, n, beta, 0.7, 5.0, 10.0, 20.0)
        difference = finite_tendency_difference(ke, n, beta, 0.7, 5.0, 10.0, 20.0)
        derivative_rows.append({
            "n": n,
            "beta": beta,
            "ke": ke,
            "formula": formula,
            "finite_difference": difference,
            "relative_error": abs(formula - difference) / max(abs(formula), 1e-30),
        })
    exponents = []
    for n in (5.0 / 3.0, 2.0, 3.0):
        beta = (3.0 - n) / 2.0
        p = (beta + 1.0 - n) / (1.0 - n)
        exponents.append({"n": n, "beta": beta, "p": p})
    identity_error = heterogeneous_identity()
    max_integral_error = max(row["relative_error"] for row in integral_rows)
    max_derivative_error = max(row["relative_error"] for row in derivative_rows)
    passed = max_integral_error < 1e-9 and max_derivative_error < 1e-8 and identity_error < 1e-14
    return {
        "schema": "prl-logistic-gt2-verification-1",
        "date": "2026-08-11",
        "passed": passed,
        "integral_checks": integral_rows,
        "finite_band_derivative_checks": derivative_rows,
        "local_clock_decomposition_absolute_error": identity_error,
        "inertial_exponents": exponents,
        "tolerances": {"integral_relative": 1e-9, "derivative_relative": 1e-8, "identity_absolute": 1e-14},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite GT2 verification output")
    result = verify()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError(f"GT2 verification failed: {result}")


if __name__ == "__main__":
    main()
