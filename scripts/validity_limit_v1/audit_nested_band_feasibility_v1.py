#!/usr/bin/env python3
"""M2a-only audit of arbitrary nested-band reconstructability.

This script reads the eight hash-frozen cluster artifacts.  It does not fit a
closure, select a band by skill, or run new tangent/nonlinear integrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "scripts/validity_limit_v1/config_v1.json"
PILOT_CONFIG = (
    ROOT.parent / "03_JFM/codex_work/experiments/"
    "prl_v8_gate5_nonconfirmatory_config_20260715.json"
)
NESTED_BANDS = [(10, 20), (11, 19), (12, 18), (13, 17), (14, 16)]
RHO_INTERVAL = (0.15, 0.85)
VALIDATION_TOLERANCE = 1.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def radial_mask(n: int, band: tuple[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kmag = np.hypot(kx, ky)
    square = (np.abs(kx) <= n // 3) & (np.abs(ky) <= n // 3)
    lo, hi = band
    lower = 0.0 if lo == 1 else lo - 0.5
    upper = hi + 0.5
    mask = (kmag >= lower) & (kmag < upper) & square
    safe = np.where(kmag > 0, kmag, 1.0)
    return mask.astype(float), {
        "kx": kx, "ky": ky, "kmag": kmag, "square": square.astype(float),
        "safe": safe,
    }


def geometry(n: int, nu: float, mu: float, band: tuple[int, int]) -> dict[str, np.ndarray]:
    mask, result = radial_mask(n, band)
    kmag, safe, square = result["kmag"], result["safe"], result["square"]
    result["mask"] = mask
    result["inv"] = np.where(kmag > 0, 1.0 / safe, 0.0) * square
    result["rate"] = (
        nu * kmag**2 + np.where(kmag > 0, mu / safe**2, 0.0)
    ) * square
    return result


def jacobian_rhs(field_hat: np.ndarray, geo: dict[str, np.ndarray]) -> np.ndarray:
    psi_hat = geo["inv"] * field_hat
    psi_x = np.fft.ifft2(1j * geo["kx"] * psi_hat).real
    psi_y = np.fft.ifft2(1j * geo["ky"] * psi_hat).real
    field_x = np.fft.ifft2(1j * geo["kx"] * field_hat).real
    field_y = np.fft.ifft2(1j * geo["ky"] * field_hat).real
    return -np.fft.fft2(psi_x * field_y - psi_y * field_x) * geo["square"]


def inner(a: np.ndarray, b: np.ndarray, mask: np.ndarray, n: int) -> float:
    return float(np.sum(np.real(np.conj(a) * b) * mask) / n**4)


def diagnostics(truth: np.ndarray, forecasts: np.ndarray,
                geo: dict[str, np.ndarray], *,
                precomputed_rhs: tuple[np.ndarray, list[np.ndarray], list[np.ndarray]] | None = None
                ) -> dict[str, np.ndarray]:
    n, mask = truth.shape[-1], geo["mask"]
    if precomputed_rhs is None:
        truth_j = jacobian_rhs(truth, geo)
        forecast_js = [jacobian_rhs(forecast, geo) for forecast in forecasts]
        delta_js = [jacobian_rhs(forecast - truth, geo) for forecast in forecasts]
    else:
        truth_j, forecast_js, delta_js = precomputed_rhs
    et = 0.5 * inner(truth, truth, mask, n)
    rows = {key: [] for key in (
        "Et", "Ef", "C", "epsilon", "rho", "x_C", "P_linear",
        "P_tl_nonlinear", "P_pure_error", "P_linearized", "P_complete",
    )}
    for forecast, forecast_j, delta_j in zip(forecasts, forecast_js, delta_js):
        delta = forecast - truth
        ef = 0.5 * inner(forecast, forecast, mask, n)
        cross = 0.5 * inner(truth, forecast, mask, n)
        error = 0.5 * inner(delta, delta, mask, n)
        rho = cross / np.sqrt(max(et * ef, 1e-30))
        x_c = 2.0 * cross / max(et + ef, 1e-30)
        linear = inner(delta, -geo["rate"] * delta, mask, n)
        pure = inner(delta, delta_j, mask, n)
        tangent = inner(delta, forecast_j - truth_j - delta_j, mask, n)
        values = (
            et, ef, cross, error, rho, x_c, linear, tangent, pure,
            linear + tangent, linear + tangent + pure,
        )
        for key, value in zip(rows, values):
            rows[key].append(value)
    return {key: np.asarray(value, dtype=float) for key, value in rows.items()}


def rel_l2(reference: list[float], candidate: list[float]) -> float:
    ref, cand = np.asarray(reference), np.asarray(candidate)
    return float(np.linalg.norm(cand - ref) / max(np.linalg.norm(ref), 1e-30))


def matching_indices(scalar: np.ndarray, field: np.ndarray) -> np.ndarray:
    lookup = {round(float(value), 12): i for i, value in enumerate(scalar)}
    return np.asarray([lookup[round(float(value), 12)] for value in field])


def availability_matrix() -> list[dict[str, str]]:
    return [
        {"quantity": "Et, Ef, C, epsilon, rho, x_C",
         "pre_existing_bands": "available at 733 scalar times",
         "arbitrary_nested_bands": "reconstructable from fields at 125 field times",
         "method": "hash-verified Fourier inner products; half-energy convention"},
        {"quantity": "P_linear, P_tl_nonlinear, P_pure_error",
         "pre_existing_bands": "available at 733 scalar times",
         "arbitrary_nested_bands": "reconstructable from fields at 125 field times",
         "method": "solver-equivalent NumPy spectral operators; common forcing cancels"},
        {"quantity": "complete instantaneous production",
         "pre_existing_bands": "available as sum of three stored components",
         "arbitrary_nested_bands": "reconstructable at field times",
         "method": "P_linear + P_tl_nonlinear + P_pure_error"},
        {"quantity": "manuscript linearized production",
         "pre_existing_bands": "available",
         "arbitrary_nested_bands": "reconstructable at field times",
         "method": "P_linear + P_tl_nonlinear"},
        {"quantity": "independent lambda_b",
         "pre_existing_bands": "available for five frozen bands plus full support",
         "arbitrary_nested_bands": "missing except outer k10_20",
         "method": "requires new independently initialized tangent-linear integration"},
        {"quantity": "nested-band lifecycle at scalar cadence",
         "pre_existing_bands": "available only for k10_20",
         "arbitrary_nested_bands": "missing",
         "method": "cannot recover 733-time arbitrary-band series from 125 snapshots"},
        {"quantity": "nested-band saturation ceiling",
         "pre_existing_bands": "estimable at scalar cadence",
         "arbitrary_nested_bands": "reconstructable but weakly sampled late",
         "method": "only four field snapshots after t=12 (20, 40, 80, 120 TU)"},
    ]


def audit(config_path: Path) -> dict[str, Any]:
    freeze, pilot = load_json(config_path), load_json(PILOT_CONFIG)
    root = (ROOT / freeze["cluster_root"]).resolve()
    regimes = {row["id"]: row for row in pilot["design"]["regimes"]}
    records: list[dict[str, Any]] = []
    validation_reference = {key: [] for key in (
        "Et", "Ef", "C", "epsilon", "rho", "P_linear",
        "P_tl_nonlinear", "P_pure_error",
    )}
    validation_candidate = {key: [] for key in validation_reference}
    coverage = {f"k{lo}_{hi}": [] for lo, hi in NESTED_BANDS}

    for relative, expected_hash in freeze["clusters"]:
        path = root / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"hash mismatch: {path}")
        regime_id = Path(relative).parts[0]
        setup = regimes[regime_id]
        geos = {
            f"k{lo}_{hi}": geometry(
                int(setup["n"]), float(setup["nu"]), float(pilot["model"]["mu"]), (lo, hi)
            ) for lo, hi in NESTED_BANDS
        }
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            field_time = np.asarray(data["field_time"], dtype=float)
            indices = matching_indices(data["scalar_time"], field_time)
            diagnostic_keys = (
                "Et", "Ef", "C", "epsilon", "rho", "x_C", "P_linear",
                "P_tl_nonlinear", "P_pure_error", "P_linearized", "P_complete",
            )
            band_diags = {
                name: {key: [] for key in diagnostic_keys} for name in geos
            }
            # NPZ members are compressed; load each large field array once rather
            # than causing a full decompression for every indexed snapshot.
            truth_fields = np.asarray(data["field_truth_hat"])
            forecast_fields = np.asarray(data["field_forecast_hat"])
            base_geo = next(iter(geos.values()))
            for field_index in range(len(field_time)):
                truth = truth_fields[field_index]
                forecasts = forecast_fields[field_index]
                rhs = (
                    jacobian_rhs(truth, base_geo),
                    [jacobian_rhs(forecast, base_geo) for forecast in forecasts],
                    [jacobian_rhs(forecast - truth, base_geo) for forecast in forecasts],
                )
                for name, geo in geos.items():
                    row = diagnostics(truth, forecasts, geo, precomputed_rhs=rhs)
                    for key, value in row.items():
                        band_diags[name][key].append(value)
            for name in band_diags:
                band_diags[name] = {
                    key: np.asarray(value) for key, value in band_diags[name].items()
                }

            primary = int(meta["primary_band_index"])
            outer = band_diags["k10_20"]
            for key in validation_reference:
                validation_reference[key].extend(
                    np.asarray(data[key][indices, :, primary], dtype=float).ravel().tolist()
                )
                validation_candidate[key].extend(outer[key].ravel().tolist())

            for name, values in band_diags.items():
                rho = values["rho"]
                for member in range(rho.shape[1]):
                    eligible = np.isfinite(rho[:, member]) & (rho[:, member] > RHO_INTERVAL[0]) & (rho[:, member] < RHO_INTERVAL[1])
                    coverage[name].append({
                        "regime": regime_id,
                        "truth_index": int(meta["truth_index"]),
                        "member_index": member,
                        "eligible_field_points": int(np.count_nonzero(eligible)),
                        "rho_min": float(np.nanmin(rho[:, member])),
                        "rho_max": float(np.nanmax(rho[:, member])),
                        "brackets_interval": bool(
                            np.nanmin(rho[:, member]) <= RHO_INTERVAL[0]
                            and np.nanmax(rho[:, member]) >= RHO_INTERVAL[1]
                        ),
                    })
            records.append({"path": str(path), "sha256": actual_hash})

    validation = {
        key: {"relative_l2": rel_l2(validation_reference[key], validation_candidate[key]),
              "tolerance": VALIDATION_TOLERANCE}
        for key in validation_reference
    }
    if any(row["relative_l2"] > row["tolerance"] for row in validation.values()):
        raise ValueError(f"field reconstruction validation failed: {validation}")
    coverage_summary = {}
    for name, rows in coverage.items():
        counts = [row["eligible_field_points"] for row in rows]
        coverage_summary[name] = {
            "trajectories": len(rows),
            "eligible_points_min": min(counts),
            "eligible_points_median": float(np.median(counts)),
            "eligible_points_max": max(counts),
            "trajectories_bracketing_0p15_0p85": sum(row["brackets_interval"] for row in rows),
            "details": rows,
        }
    return {
        "schema": "prl-logistic-m2a-feasibility-1",
        "date": "2026-08-09",
        "scope": "M2a only; no closure fit and no new model integration",
        "input_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "pilot_config": {"path": str(PILOT_CONFIG), "sha256": sha256(PILOT_CONFIG)},
        "artifacts": records,
        "nested_band_family": {
            "bands": [list(band) for band in NESTED_BANDS],
            "rule": "concentric one-shell narrowing of frozen primary k10_20, declared without response inspection",
        },
        "cadence": {
            "field_times": 125, "dense_field_interval": [0.0, 12.0],
            "dense_field_dt": 0.1, "late_field_times": [20.0, 40.0, 80.0, 120.0],
            "scalar_times_pre_existing_bands": 733,
            "lambda_times_pre_existing_bands": 1200,
        },
        "availability": availability_matrix(),
        "outer_band_reconstruction_validation": validation,
        "coverage": coverage_summary,
        "decision": {
            "arbitrary_nested_band_closure_test_now": "NO",
            "reason": "independent lambda_b is missing for all inner nested bands and 733-time arbitrary-band lifecycles are not reconstructable",
            "without_new_integration": "Only the existing k10_20 response is fully testable at scalar cadence; it is not a nested family.",
            "conditional_path_b": "A true nested-band test requires new tangent-linear integrations and new higher-cadence diagnostics; this is not authorized at D1.",
        },
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M2a nested-band feasibility exit record",
        "",
        "**Date:** 2026-08-09  ",
        "**Decision:** **Not feasible from frozen artifacts alone as a nested closure test.**",
        "",
        "The fields do permit arbitrary-band moments and instantaneous production to be",
        "reconstructed and validated. The blocker is independent arbitrary-band `lambda_b`,",
        "plus the absence of the dense 733-time lifecycle for inner bands. No new integration",
        "was run.",
        "",
        "## Machine-checked reconstruction",
        "",
        "| Quantity | Relative L2 error | Tolerance |",
        "|---|---:|---:|",
    ]
    for key, row in payload["outer_band_reconstruction_validation"].items():
        lines.append(f"| `{key}` | {row['relative_l2']:.3e} | {row['tolerance']:.1e} |")
    lines += [
        "",
        "## Frozen nested family and coverage",
        "",
        "The response-blind family is `k10_20`, `k11_19`, `k12_18`, `k13_17`,",
        "and `k14_16`: concentric one-shell narrowing of the pre-existing primary band.",
        "",
        "| Band | Eligible points per trajectory (min / median / max) | Trajectories bracketing rho=0.15–0.85 |",
        "|---|---:|---:|",
    ]
    for name, row in payload["coverage"].items():
        lines.append(
            f"| `{name}` | {row['eligible_points_min']} / {row['eligible_points_median']:.1f} / {row['eligible_points_max']} | "
            f"{row['trajectories_bracketing_0p15_0p85']}/{row['trajectories']} |"
        )
    lines += [
        "",
        "Field cadence is 0.1 TU from 0–12 TU, followed only by 20, 40, 80, and 120 TU.",
        "This samples transition but provides weak late-time ceiling support for arbitrary bands.",
        "",
        "## Availability verdict",
        "",
        "- `Et`, `Ef`, `C`, `epsilon`, `rho`, `x_C`, and all three production components:",
        "  reconstructable at the 125 field times.",
        "- Existing five bands plus full support: stored at 733 scalar times with independent",
        "  tangent rates at 1200 times.",
        "- Inner nested bands: independent `lambda_b` missing; it cannot be estimated from",
        "  the same finite-error lifecycle without violating the independence rule.",
        "- A true nested-band test therefore triggers conditional Path B (new tangent-linear",
        "  integration and higher-cadence diagnostics). Path B is not authorized.",
        "",
        "## D1 consequence",
        "",
        "M2 cannot enter an arbitrary nested-band branch from the frozen artifacts alone.",
        "At D1 the PI may either keep the analysis on pre-existing bands, authorize a tightly",
        "scoped Path B acquisition, or decline M2. This record does not authorize any option.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.config.resolve())
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    args.markdown.write_text(markdown(payload))


if __name__ == "__main__":
    main()
