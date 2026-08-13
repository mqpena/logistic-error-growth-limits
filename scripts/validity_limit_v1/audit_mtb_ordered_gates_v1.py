#!/usr/bin/env python3
"""Execute the ratified M-Tb ordered GT1 interpretability audit.

The implementation is response-blind by construction until ``main`` calls
``audit``: all thresholds, aggregation rules, and bootstrap settings are module
constants.  It reads only the eight hash-frozen cluster files and performs no
model or tangent-linear integration.
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
DEFAULT_FREEZE = (
    ROOT / "scripts/validity_limit_v1/analysis_freeze_mtb_v1_DRAFT_2026-08-09.json"
)
PRIMARY_BAND = (10, 20)
SENSITIVITY_BANDS = ((6, 9), (21, 32))
RHO_INTERVAL = (0.15, 0.85)
ENERGY_FLOOR = 1.0e-3
MIN_ELIGIBLE_TIMES = 5
VALIDATION_TOLERANCE = 1.0e-5
DENSE_DT = 0.1
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260811


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def matching_indices(scalar: np.ndarray, field: np.ndarray) -> np.ndarray:
    lookup = {round(float(value), 12): i for i, value in enumerate(scalar)}
    return np.asarray([lookup[round(float(value), 12)] for value in field])


def shell_labels(n: int) -> np.ndarray:
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kmag = np.hypot(kx, ky)
    square = (np.abs(kx) <= n // 3) & (np.abs(ky) <= n // 3)
    labels = np.floor(kmag + 0.5).astype(int)
    return np.where(square, labels, -1)


def shell_inner(a: np.ndarray, b: np.ndarray, labels: np.ndarray,
                max_shell: int, n: int) -> np.ndarray:
    valid = (labels >= 0) & (labels <= max_shell)
    weights = np.real(np.conj(a) * b)[valid] / n**4
    return np.bincount(labels[valid], weights=weights, minlength=max_shell + 1)


def shell_moments(truth: np.ndarray, forecasts: np.ndarray,
                  labels: np.ndarray, max_shell: int) -> dict[str, np.ndarray]:
    nt, nm, n = forecasts.shape[0], forecasts.shape[1], truth.shape[-1]
    et = np.empty((nt, max_shell + 1))
    ef = np.empty((nt, nm, max_shell + 1))
    cross = np.empty_like(ef)
    for ti in range(nt):
        et[ti] = 0.5 * shell_inner(truth[ti], truth[ti], labels, max_shell, n)
        for mi in range(nm):
            ef[ti, mi] = 0.5 * shell_inner(
                forecasts[ti, mi], forecasts[ti, mi], labels, max_shell, n
            )
            cross[ti, mi] = 0.5 * shell_inner(
                truth[ti], forecasts[ti, mi], labels, max_shell, n
            )
    rho = cross / np.sqrt(np.maximum(et[:, None, :] * ef, 1.0e-30))
    return {"Et": et, "Ef": ef, "C": cross, "rho": rho}


def rel_l2(reference: list[float], candidate: list[float]) -> float:
    ref, cand = np.asarray(reference), np.asarray(candidate)
    return float(np.linalg.norm(cand - ref) / max(np.linalg.norm(ref), 1.0e-30))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values) / np.sum(weights))


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / np.sum(weights)))


def weighted_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    xbar, ybar = weighted_mean(x, weights), weighted_mean(y, weights)
    denom = float(np.sum(weights * (x - xbar) ** 2))
    if denom <= 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum(weights * (x - xbar) * (y - ybar)) / denom)
    return ybar - slope * xbar, slope


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def geometry_tier(value: float) -> str:
    if value <= 0.05:
        return "ONE_SHAPE_LIMIT"
    if value <= 0.25:
        return "MIXED"
    return "MOVING_FRONT"


def clock_tier(value: float) -> str:
    if value <= 0.10:
        return "ONE_CLOCK"
    if value <= 0.40:
        return "MIXED_CLOCKS"
    return "MANY_CLOCKS"


def bootstrap_ci(truth_values: list[float], rng: np.random.Generator) -> list[float]:
    values = np.asarray(truth_values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    draws = np.median(values[indices], axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def band_summary(moments: dict[str, np.ndarray], times: np.ndarray,
                 band: tuple[int, int]) -> list[dict[str, Any]]:
    lo, hi = band
    shells = np.arange(lo, hi + 1)
    x = np.log10(shells.astype(float))
    wband = np.log10(hi / lo)
    et = moments["Et"][:, shells]
    ef = moments["Ef"][:, :, shells]
    cross = moments["C"][:, :, shells]
    rho_k = moments["rho"][:, :, shells]
    et_band = np.sum(et, axis=1)
    ef_band = np.sum(ef, axis=2)
    cross_band = np.sum(cross, axis=2)
    rho_band = cross_band / np.sqrt(np.maximum(et_band[:, None] * ef_band, 1.0e-30))
    rows: list[dict[str, Any]] = []
    for member in range(ef.shape[1]):
        lifecycle = (
            (times <= 12.0 + 1.0e-12)
            & (rho_band[:, member] >= RHO_INTERVAL[0])
            & (rho_band[:, member] <= RHO_INTERVAL[1])
        )
        eligible_indices = np.flatnonzero(lifecycle)
        r_values, slopes, k_halves, q_values, q_lower_bounds = [], [], [], [], []
        censored_half = censored_q = excluded_pairs = total_pairs = 0
        for ti in eligible_indices:
            weights_all = et[ti]
            include = (
                np.isfinite(rho_k[ti, member])
                & (weights_all >= ENERGY_FLOOR * max(et_band[ti], 1.0e-30))
            )
            total_pairs += len(shells)
            excluded_pairs += int(np.count_nonzero(~include))
            if np.count_nonzero(include) < 2:
                continue
            weights = weights_all[include]
            a = 1.0 - rho_k[ti, member, include]
            abar = weighted_mean(a, weights)
            variance = weighted_mean((a - abar) ** 2, weights)
            denom = abar * (1.0 - abar)
            if np.isfinite(denom) and denom > 0.0:
                r_values.append(variance / denom)
            intercept, slope = weighted_line(x[include], a, weights)
            slopes.append(slope)
            if np.isfinite(slope) and slope != 0.0:
                xhalf = (0.5 - intercept) / slope
                if x[0] <= xhalf <= x[-1]:
                    k_halves.append((times[ti], 10.0**xhalf))
                else:
                    censored_half += 1
                x25, x75 = (0.25 - intercept) / slope, (0.75 - intercept) / slope
                if x[0] <= min(x25, x75) and max(x25, x75) <= x[-1]:
                    q_values.append(abs(x75 - x25) / wband)
                else:
                    censored_q += 1
                    qlo, qhi = min(x25, x75), max(x25, x75)
                    if qlo < x[0] and qhi > x[-1]:
                        q_lower_bounds.append(1.0)
                    elif qlo < x[0] <= qhi <= x[-1]:
                        q_lower_bounds.append((qhi - x[0]) / wband)
                    elif x[0] <= qlo <= x[-1] < qhi:
                        q_lower_bounds.append((x[-1] - qlo) / wband)
                    else:
                        q_lower_bounds.append(0.0)
            else:
                censored_half += 1
                censored_q += 1
        half_times = np.asarray([row[0] for row in k_halves])
        half_values = np.asarray([row[1] for row in k_halves])
        rows.append({
            "member_index": member,
            "eligible_times": int(len(eligible_indices)),
            "included_R_times": int(len(r_values)),
            "R_median": float(np.median(r_values)) if r_values else float("nan"),
            "b_median_per_decade": float(np.median(slopes)) if slopes else float("nan"),
            "k_half_spearman_time": spearman(half_times, half_values),
            "k_half_uncensored": int(len(k_halves)),
            "k_half_censored": int(censored_half),
            "Q_median_uncensored": float(np.median(q_values)) if q_values else float("nan"),
            "Q_uncensored": int(len(q_values)),
            "Q_right_censored": int(censored_q),
            "Q_censored_lower_bound_median": float(np.median(q_lower_bounds)) if q_lower_bounds else float("nan"),
            "Q_censoring_rule": "when a threshold lies outside the band, report the in-band interval as a conservative lower bound; never extrapolate the crossing",
            "excluded_shell_time_pairs": int(excluded_pairs),
            "total_shell_time_pairs": int(total_pairs),
            "energy_floor_flag": bool(total_pairs and excluded_pairs / total_pairs > 0.05),
            "eligible": bool(len(eligible_indices) >= MIN_ELIGIBLE_TIMES and r_values),
        })
    return rows


def clock_summary(moments: dict[str, np.ndarray], times: np.ndarray,
                  band: tuple[int, int]) -> list[dict[str, Any]]:
    lo, hi = band
    shells = np.arange(lo, hi + 1)
    et = moments["Et"][:, shells]
    ef = moments["Ef"][:, :, shells]
    cross = moments["C"][:, :, shells]
    a_raw = 1.0 - moments["rho"][:, :, shells]
    et_band = np.sum(et, axis=1)
    ef_band = np.sum(ef, axis=2)
    cross_band = np.sum(cross, axis=2)
    rho_band = cross_band / np.sqrt(np.maximum(et_band[:, None] * ef_band, 1.0e-30))
    dense = np.isclose(np.diff(times), DENSE_DT, rtol=0.0, atol=1.0e-10)
    rows: list[dict[str, Any]] = []
    for member in range(ef.shape[1]):
        a = a_raw[:, member]
        smooth = np.full_like(a, np.nan)
        smooth[1:-1] = (a[:-2] + a[1:-1] + a[2:]) / 3.0
        adot = np.full_like(a, np.nan)
        valid_center = dense[:-1] & dense[1:]
        adot[1:-1][valid_center] = (smooth[2:][valid_center] - smooth[:-2][valid_center]) / (2.0 * DENSE_DT)
        dispersion, dispersion_raw = [], []
        nonpositive = endpoint_excluded = low_energy = 0
        for ti in range(2, len(times) - 2):
            if not (times[ti] <= 12.0 + 1.0e-12 and RHO_INTERVAL[0] <= rho_band[ti, member] <= RHO_INTERVAL[1]):
                continue
            weights = et[ti]
            energy_ok = weights >= ENERGY_FLOOR * max(et_band[ti], 1.0e-30)
            endpoint_ok = (a[ti] >= RHO_INTERVAL[0]) & (a[ti] <= RHO_INTERVAL[1])
            rate = adot[ti] / (a[ti] * (1.0 - a[ti]))
            positive = np.isfinite(rate) & (rate > 0.0)
            low_energy += int(np.count_nonzero(~energy_ok))
            endpoint_excluded += int(np.count_nonzero(energy_ok & ~endpoint_ok))
            nonpositive += int(np.count_nonzero(energy_ok & endpoint_ok & ~positive))
            include = energy_ok & endpoint_ok & positive
            if np.count_nonzero(include) >= 2:
                dispersion.append(weighted_sd(np.log(rate[include]), weights[include]))
            raw_adot = (a[ti + 1] - a[ti - 1]) / (2.0 * DENSE_DT)
            raw_rate = raw_adot / (a[ti] * (1.0 - a[ti]))
            raw_include = energy_ok & endpoint_ok & np.isfinite(raw_rate) & (raw_rate > 0.0)
            if np.count_nonzero(raw_include) >= 2:
                dispersion_raw.append(weighted_sd(np.log(raw_rate[raw_include]), weights[raw_include]))
        rows.append({
            "member_index": member,
            "Lambda_disp_median": float(np.median(dispersion)) if dispersion else float("nan"),
            "Lambda_disp_unsmoothed_median": float(np.median(dispersion_raw)) if dispersion_raw else float("nan"),
            "included_times": int(len(dispersion)),
            "excluded_nonpositive_lambda_eff_pairs": int(nonpositive),
            "excluded_endpoint_pairs": int(endpoint_excluded),
            "excluded_low_energy_pairs": int(low_energy),
            "log_domain_rule": "lambda_eff must be finite and positive; nonpositive values are excluded and counted",
        })
    return rows


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result: dict[str, Any] = {}
    for regime in sorted({row["regime"] for row in records}):
        selected = [row for row in records if row["regime"] == regime]
        truths = []
        for row in sorted(selected, key=lambda item: item["truth_index"]):
            primary = row["bands"]["k10_20"]
            r_members = [m["R_median"] for m in primary["D1"] if m["eligible"]]
            d_members = [m["Lambda_disp_median"] for m in primary["D2"] if np.isfinite(m["Lambda_disp_median"])]
            r_value = float(np.mean(r_members)) if len(r_members) == 2 else float("nan")
            d_value = float(np.mean(d_members)) if len(d_members) == 2 else float("nan")
            truths.append({
                "truth_index": row["truth_index"],
                "R": r_value,
                "R_tier": geometry_tier(r_value) if np.isfinite(r_value) else "EXCLUDED",
                "Lambda_disp": d_value,
                "clock_tier": clock_tier(d_value) if np.isfinite(d_value) else "EXCLUDED",
                "members": primary,
                "sensitivity_bands": {key: value for key, value in row["bands"].items() if key != "k10_20"},
            })
        r_values = [row["R"] for row in truths if np.isfinite(row["R"])]
        d_values = [row["Lambda_disp"] for row in truths if np.isfinite(row["Lambda_disp"])]
        r_tiers = {row["R_tier"] for row in truths}
        d_tiers = {row["clock_tier"] for row in truths}
        result[regime] = {
            "truths": truths,
            "D1": {
                "regime_median_R": float(np.median(r_values)) if r_values else float("nan"),
                "truth_bootstrap_95pct_CI": bootstrap_ci(r_values, rng),
                "tier": next(iter(r_tiers)) if len(r_tiers) == 1 else "HETEROGENEOUS",
            },
            "D2": {
                "regime_median_Lambda_disp": float(np.median(d_values)) if d_values else float("nan"),
                "truth_bootstrap_95pct_CI": bootstrap_ci(d_values, rng),
                "tier": next(iter(d_tiers)) if len(d_tiers) == 1 else "HETEROGENEOUS",
            },
        }
        gates = result[regime]
        gates["residual_control"] = {
            "tier": "NOT_ESTABLISHED",
            "reason": "M-Tb measures spectral geometry and clock dispersion; the finite-band I1, dissipation, forcing, and closure-memory residual accounting remains an M-T obligation.",
        }
        gates["D3"] = {
            "tier": "NOT_INTERPRETABLE_FOR_GT1",
            "computed": False,
            "reason": "The residual-control gate is not established; ordered logic forbids treating a lifecycle fit as script-G/E.",
        }
    return result


def audit(config_path: Path, freeze_path: Path) -> dict[str, Any]:
    config, freeze = load_json(config_path), load_json(freeze_path)
    if not freeze["authorization"]["authorized"]:
        raise RuntimeError("M-Tb freeze is not authorized")
    root = (ROOT / config["cluster_root"]).resolve()
    max_shell = max(band[1] for band in (PRIMARY_BAND,) + SENSITIVITY_BANDS)
    validation_reference = {"Et": [], "Ef": [], "C": []}
    validation_candidate = {"Et": [], "Ef": [], "C": []}
    records: list[dict[str, Any]] = []
    artifact_rows = []
    for relative, expected_hash in config["clusters"]:
        path = root / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"hash mismatch: {path}")
        regime = Path(relative).parts[0]
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            times = np.asarray(data["field_time"], dtype=float)
            indices = matching_indices(np.asarray(data["scalar_time"]), times)
            truth = np.asarray(data["field_truth_hat"])
            forecasts = np.asarray(data["field_forecast_hat"])
            labels = shell_labels(truth.shape[-1])
            moments = shell_moments(truth, forecasts, labels, max_shell)
            primary = int(metadata["primary_band_index"])
            shells = np.arange(PRIMARY_BAND[0], PRIMARY_BAND[1] + 1)
            candidates = {
                "Et": np.repeat(np.sum(moments["Et"][:, shells], axis=1)[:, None], forecasts.shape[1], axis=1),
                "Ef": np.sum(moments["Ef"][:, :, shells], axis=2),
                "C": np.sum(moments["C"][:, :, shells], axis=2),
            }
            for key in validation_reference:
                validation_reference[key].extend(np.asarray(data[key][indices, :, primary]).ravel().tolist())
                validation_candidate[key].extend(candidates[key].ravel().tolist())
            bands = {}
            for band in (PRIMARY_BAND,) + SENSITIVITY_BANDS:
                name = f"k{band[0]}_{band[1]}"
                bands[name] = {
                    "D1": band_summary(moments, times, band),
                    "D2": clock_summary(moments, times, band),
                }
            records.append({"regime": regime, "truth_index": int(metadata["truth_index"]), "bands": bands})
            artifact_rows.append({"path": str(path), "sha256": actual_hash})
    validation = {
        key: {"relative_l2": rel_l2(validation_reference[key], validation_candidate[key]), "tolerance": VALIDATION_TOLERANCE}
        for key in validation_reference
    }
    if any(row["relative_l2"] > VALIDATION_TOLERANCE for row in validation.values()):
        raise RuntimeError(f"reconstruction validation failed: {validation}")
    return {
        "schema": "prl-logistic-mtb-result-1",
        "date": "2026-08-11",
        "evidence_status": freeze["evidence_status"],
        "scope": "M-Tb only; stored-artifact post-processing; no new integration and no E-axis claim",
        "provenance": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path), "canonical_sha256": freeze["hashes"]["self_sha256"]},
            "code_sha256": sha256(Path(__file__)),
            "artifacts": artifact_rows,
        },
        "implementation_freeze": {
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "nonpositive_lambda_eff": "excluded from log dispersion and counted",
            "D3_ordering": "not computed unless D1, D2, and residual control all pass",
        },
        "reconstruction_validation": validation,
        "regimes": aggregate(records),
        "t_code": "NONE — M-Tb supplies ordered derivation inputs only",
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M-Tb ordered GT1 interpretability audit — exit record",
        "",
        "**Date:** 2026-08-11  ",
        "**Evidence:** prospective diagnostic on stored artifacts; no new integration  ",
        "**T-code:** none; M-Tb only selects the derivation obligation.",
        "",
        "## Reconstruction gate",
        "",
        "| Quantity | Relative L2 | Tolerance |",
        "|---|---:|---:|",
    ]
    for key, row in payload["reconstruction_validation"].items():
        lines.append(f"| `{key}` | {row['relative_l2']:.3e} | {row['tolerance']:.1e} |")
    lines += ["", "## Ordered gates", "", "| Regime | D1: R (95% CI) | Geometry | D2: dispersion (95% CI) | Clock | Residuals | D3 |", "|---|---:|---|---:|---|---|---|"]
    for regime, row in payload["regimes"].items():
        rci, dci = row["D1"]["truth_bootstrap_95pct_CI"], row["D2"]["truth_bootstrap_95pct_CI"]
        lines.append(
            f"| `{regime}` | {row['D1']['regime_median_R']:.4f} [{rci[0]:.4f}, {rci[1]:.4f}] | {row['D1']['tier']} | "
            f"{row['D2']['regime_median_Lambda_disp']:.4f} [{dci[0]:.4f}, {dci[1]:.4f}] | {row['D2']['tier']} | "
            f"{row['residual_control']['tier']} | {row['D3']['tier']} |"
        )
    lines += [
        "",
        "D3 was not computed because residual control is not yet established. A lifecycle",
        "logistic coefficient would therefore be only a best-fit number, not an estimate of",
        "the GT1 kernel coefficient. The measured D1/D2 branch is an input to M-T, not a",
        "claim reduction and not an E-axis result.",
        "",
        "## Per-truth audit",
        "",
    ]
    for regime, row in payload["regimes"].items():
        lines += [f"### `{regime}`", "", "| Truth | R | Tier | Clock dispersion | Tier |", "|---:|---:|---|---:|---|"]
        for truth in row["truths"]:
            lines.append(f"| {truth['truth_index']} | {truth['R']:.4f} | {truth['R_tier']} | {truth['Lambda_disp']:.4f} | {truth['clock_tier']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists() or args.markdown.exists():
        raise FileExistsError("M-Tb refuses to overwrite an existing result")
    payload = audit(args.config.resolve(), args.freeze.resolve())
    args.json.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    args.markdown.write_text(markdown(payload) + "\n")


if __name__ == "__main__":
    main()
