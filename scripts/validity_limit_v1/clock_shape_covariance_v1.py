#!/usr/bin/env python3
"""Exploratory finite-band shape and clock-shape correction decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.validity_limit_v1.audit_mtb_ordered_gates_v1 import (
        DENSE_DT, ENERGY_FLOOR, PRIMARY_BAND, RHO_INTERVAL, matching_indices,
        sha256, shell_labels, shell_moments, weighted_mean,
    )
except ModuleNotFoundError:  # Direct execution from this script's directory.
    from audit_mtb_ordered_gates_v1 import (
        DENSE_DT, ENERGY_FLOOR, PRIMARY_BAND, RHO_INTERVAL, matching_indices,
        sha256, shell_labels, shell_moments, weighted_mean,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "scripts/validity_limit_v1/config_v1.json"
DEFAULT_FREEZE = ROOT / "scripts/validity_limit_v1/analysis_freeze_clock_shape_covariance_v1_2026-08-12.json"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260812
BASELINE_FLOOR = 1.0e-12
IDENTITY_TOLERANCE = 1.0e-10
METRICS = (
    "shape_fraction",
    "clock_shape_fraction",
    "abs_clock_shape_fraction",
    "net_correction_fraction",
    "direct_fraction",
    "baseline",
    "shape_term",
    "clock_shape_term",
    "direct_tendency",
)


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


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPS, len(array)))
    draws = np.median(array[indices], axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def member_decomposition(moments: dict[str, np.ndarray], times: np.ndarray,
                         member: int) -> dict[str, Any]:
    shells = np.arange(PRIMARY_BAND[0], PRIMARY_BAND[1] + 1)
    et = moments["Et"][:, shells]
    ef = moments["Ef"][:, member, shells]
    cross = moments["C"][:, member, shells]
    rho_shell = moments["rho"][:, member, shells]
    et_band = np.sum(et, axis=1)
    ef_band = np.sum(ef, axis=1)
    cross_band = np.sum(cross, axis=1)
    rho_band = cross_band / np.sqrt(np.maximum(et_band * ef_band, 1.0e-30))
    a_raw = 1.0 - rho_shell
    a = np.full_like(a_raw, np.nan)
    a[1:-1] = (a_raw[:-2] + a_raw[1:-1] + a_raw[2:]) / 3.0
    adot = np.full_like(a_raw, np.nan)
    dense = np.isclose(np.diff(times), DENSE_DT, rtol=0.0, atol=1.0e-10)
    valid_center = dense[:-1] & dense[1:]
    adot[1:-1][valid_center] = (a[2:][valid_center] - a[:-2][valid_center]) / (2.0 * DENSE_DT)

    rows: list[dict[str, float]] = []
    denominator_excluded = insufficient_shells = 0
    max_identity_error = 0.0
    for ti in range(2, len(times) - 2):
        if not (
            times[ti] <= 12.0 + 1.0e-12
            and RHO_INTERVAL[0] <= rho_band[ti] <= RHO_INTERVAL[1]
        ):
            continue
        weights_all = et[ti]
        include = (
            np.isfinite(a[ti])
            & np.isfinite(adot[ti])
            & (a[ti] >= RHO_INTERVAL[0])
            & (a[ti] <= RHO_INTERVAL[1])
            & (weights_all >= ENERGY_FLOOR * max(et_band[ti], 1.0e-30))
        )
        if np.count_nonzero(include) < 2:
            insufficient_shells += 1
            continue
        weights = weights_all[include]
        amplitude = a[ti, include]
        amplitude_dot = adot[ti, include]
        factor = amplitude * (1.0 - amplitude)
        rates = amplitude_dot / factor
        abar = weighted_mean(amplitude, weights)
        lambda_bar = weighted_mean(rates, weights)
        variance = weighted_mean((amplitude - abar) ** 2, weights)
        factor_bar = weighted_mean(factor, weights)
        covariance = weighted_mean((rates - lambda_bar) * (factor - factor_bar), weights)
        baseline = lambda_bar * abar * (1.0 - abar)
        if not np.isfinite(baseline) or abs(baseline) <= BASELINE_FLOOR:
            denominator_excluded += 1
            continue
        shape_term = -lambda_bar * variance
        direct = weighted_mean(amplitude_dot, weights)
        identity_error = abs(direct - (baseline + shape_term + covariance))
        max_identity_error = max(max_identity_error, identity_error)
        rows.append({
            "time": float(times[ti]),
            "included_shells": int(np.count_nonzero(include)),
            "abar": abar,
            "lambda_bar": lambda_bar,
            "variance": variance,
            "baseline": baseline,
            "shape_term": shape_term,
            "clock_shape_term": covariance,
            "direct_tendency": direct,
            "shape_fraction": shape_term / baseline,
            "clock_shape_fraction": covariance / baseline,
            "abs_clock_shape_fraction": abs(covariance / baseline),
            "net_correction_fraction": (shape_term + covariance) / baseline,
            "direct_fraction": direct / baseline,
            "identity_absolute_error": identity_error,
        })
    if max_identity_error > IDENTITY_TOLERANCE:
        raise RuntimeError(f"local-clock identity failed: {max_identity_error}")
    summary = {
        metric: float(np.median([row[metric] for row in rows])) if rows else float("nan")
        for metric in METRICS
    }
    return {
        "member_index": member,
        "eligible_times": len(rows),
        "denominator_excluded_times": denominator_excluded,
        "insufficient_shell_times": insufficient_shells,
        "maximum_identity_absolute_error": max_identity_error,
        "time_median": summary,
        "timeseries": rows,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: dict[str, Any] = {}
    for regime in sorted({row["regime"] for row in records}):
        truth_rows = []
        for record in sorted(
            (row for row in records if row["regime"] == regime),
            key=lambda row: row["truth_index"],
        ):
            truth_summary = {
                metric: float(np.mean([member["time_median"][metric] for member in record["members"]]))
                for metric in METRICS
            }
            truth_rows.append({
                "truth_index": record["truth_index"],
                "truth_summary": truth_summary,
                "members": record["members"],
            })
        regime_summary = {}
        for metric in METRICS:
            values = [row["truth_summary"][metric] for row in truth_rows]
            regime_summary[metric] = {
                "median": float(np.median(values)),
                "truth_bootstrap_95pct_CI": bootstrap_ci(values, rng),
            }
        output[regime] = {"truths": truth_rows, "regime_summary": regime_summary}
    return output


def compute(config_path: Path, freeze_path: Path) -> dict[str, Any]:
    config, freeze = load_json(config_path), load_json(freeze_path)
    if not freeze["authorization"]["authorized"]:
        raise RuntimeError("exploratory covariance diagnostic is not authorized")
    root = (ROOT / config["cluster_root"]).resolve()
    records = []
    artifacts = []
    for relative, expected_hash in config["clusters"]:
        path = root / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            times = np.asarray(data["field_time"], dtype=float)
            matching_indices(np.asarray(data["scalar_time"]), times)
            truth = np.asarray(data["field_truth_hat"])
            forecasts = np.asarray(data["field_forecast_hat"])
            moments = shell_moments(
                truth, forecasts, shell_labels(truth.shape[-1]), PRIMARY_BAND[1]
            )
            members = [member_decomposition(moments, times, member) for member in range(forecasts.shape[1])]
            records.append({
                "regime": Path(relative).parts[0],
                "truth_index": int(metadata["truth_index"]),
                "members": members,
            })
            artifacts.append({"path": str(path), "sha256": actual_hash})
    regimes = aggregate(records)
    maximum_error = max(
        member["maximum_identity_absolute_error"]
        for regime in regimes.values()
        for truth in regime["truths"]
        for member in truth["members"]
    )
    return {
        "schema": "prl-logistic-exploratory-clock-shape-result-1",
        "date": "2026-08-12",
        "evidence_status": "exploratory_descriptive",
        "scope": "finite-band local-clock decomposition on stored k10_20 fields; no new integration",
        "non_claim": "This algebraic diagnostic is not a closure validation, hypothesis test, T-code, E-code, or C0 modification.",
        "provenance": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
            "code_sha256": sha256(Path(__file__)),
            "artifacts": artifacts,
        },
        "identity": freeze["identity"],
        "maximum_identity_absolute_error": maximum_error,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "regimes": regimes,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exploratory finite-band clock-shape covariance — exit record",
        "",
        "**Date:** 2026-08-12  ",
        "**Evidence:** exploratory descriptive magnitude on previously inspected stored data  ",
        "**Scope:** exact local-clock diagnostic identity; no closure validation, hypothesis test, T-code, E-code, or C0 change",
        "",
        "## Result",
        "",
        "All corrections below are fractions of `lambdabar*abar*(1-abar)`. Negative values reduce the baseline logistic tendency; positive values increase it.",
        "",
        "| Regime | Shape correction | Clock-shape correction | |Clock-shape| | Net correction | Direct/baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for regime, row in payload["regimes"].items():
        summary = row["regime_summary"]
        lines.append(
            f"| `{regime}` | {summary['shape_fraction']['median']:.4f} | "
            f"{summary['clock_shape_fraction']['median']:.4f} | "
            f"{summary['abs_clock_shape_fraction']['median']:.4f} | "
            f"{summary['net_correction_fraction']['median']:.4f} | "
            f"{summary['direct_fraction']['median']:.4f} |"
        )
    lines += ["", "Truth-bootstrap 95% intervals and all member/time values are retained in the machine record.", "", "## Per-truth values", ""]
    for regime, row in payload["regimes"].items():
        lines += [f"### `{regime}`", "", "| Truth | Shape | Clock-shape | |Clock-shape| | Net |", "|---:|---:|---:|---:|---:|"]
        for truth in row["truths"]:
            s = truth["truth_summary"]
            lines.append(f"| {truth['truth_index']} | {s['shape_fraction']:.4f} | {s['clock_shape_fraction']:.4f} | {s['abs_clock_shape_fraction']:.4f} | {s['net_correction_fraction']:.4f} |")
        lines.append("")
    lines += [
        "## Interpretation boundary",
        "",
        "Clock dispersion established that the GT1 eigencondition is not uniformly realised; it did not establish non-logistic band-mean behaviour. The covariance above is the relevant correction within the local-clock identity. It remains exploratory and does not evaluate the nonlocal I1, forcing, dissipation, or memory residuals required for a finite-band closure theorem.",
        "",
        f"Maximum algebraic identity error: `{payload['maximum_identity_absolute_error']:.3e}` (tolerance `{payload['identity_tolerance']:.1e}`).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists() or args.markdown.exists():
        raise FileExistsError("refusing to overwrite exploratory covariance output")
    payload = compute(args.config.resolve(), args.freeze.resolve())
    args.json.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    args.markdown.write_text(markdown(payload) + "\n")


if __name__ == "__main__":
    main()
