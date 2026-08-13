#!/usr/bin/env python3
"""Exact-recipe PRL v8 pilot collapse and locality decision analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import prl_v8_two_regime as generator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPT = Path(__file__).resolve()
DEFAULT_CONFIG = HERE / "prl_v8_gate5_nonconfirmatory_config_20260715.json"
DEFAULT_OUTPUT = HERE / "out" / "prl_v8_gate5_nonconfirmatory_20260715"
DEFAULT_EVALUATION = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_gate5_nonconfirmatory_20260715_per_cluster"
    / "prl_v8_pilot_evaluation_v2_8of8_f5e91c6d608c_be29ceeaa19f.json"
)
DEFAULT_REPORT_DIR = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_pilot_collapse_analysis_20260717"
)
DEFAULT_SPEC = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_pilot_collapse_decision_analysis_spec_20260717.md"
)
EVALUATION_SHA256 = "6fe14839d3426d0e4a09dde306fa425746a11c4e14d76cc2ef210b95577e2fa9"
SCHEMA = "prl-v8-pilot-collapse-decision-analysis-4"
PRIMARY_BINS = 8
BIN_COUNTS = (6, 8, 10, 12)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260717


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_fit(x: np.ndarray, y: np.ndarray,
               weights: np.ndarray | None = None) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.asarray(weights, dtype=float)
        mask &= np.isfinite(weights) & (weights > 0)
    x, y, weights = x[mask], y[mask], weights[mask]
    if x.size < 3:
        return {"n": int(x.size), "intercept": float("nan"),
                "slope": float("nan"), "r2": float("nan")}
    design = np.column_stack([np.ones(x.size), x])
    root_weight = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(design * root_weight[:, None],
                               y * root_weight, rcond=None)
    prediction = design @ beta
    mean = float(np.average(y, weights=weights))
    residual = float(np.sum(weights * (y - prediction) ** 2))
    total = float(np.sum(weights * (y - mean) ** 2))
    return {
        "n": int(x.size),
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(1.0 - residual / max(total, np.finfo(float).tiny)),
    }


def predictive_r2(y: np.ndarray, prediction: np.ndarray,
                  weights: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if weights is None:
        weights = np.ones_like(y)
    else:
        weights = np.asarray(weights, dtype=float)
    mean = float(np.average(y, weights=weights))
    residual = float(np.sum(weights * (y - prediction) ** 2))
    total = float(np.sum(weights * (y - mean) ** 2))
    return float(1.0 - residual / max(total, np.finfo(float).tiny))


def bin_trajectory(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bx = np.full(len(edges) - 1, np.nan)
    by = np.full(len(edges) - 1, np.nan)
    for index in range(len(edges) - 1):
        upper = edges[index + 1]
        mask = (x >= edges[index]) & (x < upper)
        if index == len(edges) - 2:
            mask = (x >= edges[index]) & (x <= upper)
        if np.any(mask):
            bx[index] = float(np.mean(x[mask]))
            by[index] = float(np.mean(y[mask]))
    return bx, by


def binned_curve(draws: list[list[dict[str, Any]]], bins: int,
                 member_weighting: str = "equal") -> dict[str, Any]:
    edges = np.linspace(0.15, 0.85, bins + 1)
    truth_x, truth_y = [], []
    for members in draws:
        if member_weighting == "pooled":
            x = np.concatenate([member["x"] for member in members])
            y = np.concatenate([member["y"] for member in members])
            bx, by = bin_trajectory(x, y, edges)
        elif member_weighting == "equal":
            member_bins = [
                bin_trajectory(member["x"], member["y"], edges)
                for member in members
            ]
            member_x = np.asarray([item[0] for item in member_bins])
            member_y = np.asarray([item[1] for item in member_bins])
            bx = np.mean(member_x, axis=0)
            by = np.mean(member_y, axis=0)
        else:
            raise ValueError(f"unknown member weighting: {member_weighting}")
        truth_x.append(bx)
        truth_y.append(by)
    truth_x_array = np.asarray(truth_x)
    truth_y_array = np.asarray(truth_y)
    coverage = np.sum(np.isfinite(truth_y_array), axis=0)
    valid = coverage == len(draws)
    mean_x = np.mean(truth_x_array[:, valid], axis=0)
    mean_y = np.mean(truth_y_array[:, valid], axis=0)
    sem_y = (np.std(truth_y_array[:, valid], axis=0, ddof=1)
             / np.sqrt(len(draws))) if len(draws) > 1 else np.zeros_like(mean_y)
    fit = linear_fit(mean_x, mean_y)
    return {
        "edges": edges.tolist(),
        "valid_bins": np.flatnonzero(valid).tolist(),
        "coverage": coverage.tolist(),
        "mean_x": mean_x.tolist(),
        "mean_y": mean_y.tolist(),
        "sem_y": sem_y.tolist(),
        "truth_x": truth_x_array.tolist(),
        "truth_y": truth_y_array.tolist(),
        "member_weighting": member_weighting,
        "fit": fit,
    }


def point_binned_curve(truths: list[list[dict[str, Any]]], bins: int) -> dict[str, Any]:
    edges = np.linspace(0.15, 0.85, bins + 1)
    x = np.concatenate([member["x"] for truth in truths for member in truth])
    y = np.concatenate([member["y"] for truth in truths for member in truth])
    bx, by = bin_trajectory(x, y, edges)
    valid = np.isfinite(by)
    return {
        "mean_x": bx[valid].tolist(),
        "mean_y": by[valid].tolist(),
        "fit": linear_fit(bx[valid], by[valid]),
    }


def component_audit(truths: list[list[dict[str, Any]]],
                    component: str) -> dict[str, Any]:
    x = np.concatenate([member["x"] for truth in truths for member in truth])
    y = np.concatenate([member[component] for truth in truths for member in truth])
    sensitivity = {}
    for bins in BIN_COUNTS:
        edges = np.linspace(0.15, 0.85, bins + 1)
        bx, by = bin_trajectory(x, y, edges)
        valid = np.isfinite(by)
        sensitivity[str(bins)] = linear_fit(bx[valid], by[valid])
    return {
        "point_pooled_fit": linear_fit(x, y),
        "point_binned_fit_sensitivity": sensitivity,
    }


def balanced_raw_fit(truths: list[list[dict[str, Any]]], unit: str) -> dict[str, float | int]:
    x_values, y_values, weights = [], [], []
    if unit == "truth":
        for truth in truths:
            x = np.concatenate([member["x"] for member in truth])
            y = np.concatenate([member["y"] for member in truth])
            x_values.append(x)
            y_values.append(y)
            weights.append(np.full(x.size, 1.0 / x.size))
    elif unit == "member":
        for truth in truths:
            for member in truth:
                x, y = member["x"], member["y"]
                x_values.append(x)
                y_values.append(y)
                weights.append(np.full(x.size, 1.0 / x.size))
    else:
        raise ValueError(f"unknown balance unit: {unit}")
    return linear_fit(np.concatenate(x_values), np.concatenate(y_values),
                      np.concatenate(weights))


def load_regime(config: dict[str, Any], output: Path,
                regime: dict[str, Any]) -> list[list[dict[str, Any]]]:
    rid = regime["id"]
    bands = generator.expanded_bands(config, int(regime["n"]))
    primary = bands.index(tuple(config["design"]["primary_band"]))
    centers = np.sqrt(np.asarray([lo * hi for lo, hi in bands], dtype=float))
    local = np.abs(np.log2(centers / centers[primary])) <= 1.0
    local_matrix = local[:, None] & local[None, :]
    truths: list[list[dict[str, Any]]] = []
    for truth_index in range(config["design"]["truths_per_regime"]):
        path = output / rid / "clusters" / f"truth_{truth_index:02d}.npz"
        members: list[dict[str, Any]] = []
        with np.load(path, allow_pickle=False) as data:
            burn = data["lambda_time"] >= float(config["lyapunov"]["burnin_tu"])
            lam = float(np.mean(data["lambda_energy_rate"][burn, primary]))
            et = data["Et"][:, :, primary]
            ef = data["Ef"][:, :, primary]
            cross = data["C"][:, :, primary]
            eps = data["epsilon"][:, :, primary]
            rho = data["rho"][:, :, primary]
            x = 2.0 * cross / np.maximum(et + ef, 1e-30)
            y = (data["P_linear"][:, :, primary]
                 + data["P_tl_nonlinear"][:, :, primary]) / (lam * eps)
            y_linear = data["P_linear"][:, :, primary] / (lam * eps)
            y_tangent_nonlinear = data["P_tl_nonlinear"][:, :, primary] / (lam * eps)
            source = data["locality_tl_source"][:, :, :, :, primary]
            local_abs = np.sum(np.abs(source[:, :, local_matrix]), axis=-1)
            total_abs = np.sum(np.abs(source), axis=(-2, -1))
            locality_by_member = np.mean(
                local_abs / np.maximum(total_abs, 1e-30), axis=0)
            for member_index in range(x.shape[1]):
                mask = (np.isfinite(x[:, member_index])
                        & np.isfinite(y[:, member_index])
                        & (rho[:, member_index] > 0.15)
                        & (rho[:, member_index] < 0.85))
                members.append({
                    "truth_index": truth_index,
                    "member_index": member_index,
                    "x": np.asarray(x[mask, member_index], dtype=float),
                    "y": np.asarray(y[mask, member_index], dtype=float),
                    "y_linear_only": np.asarray(
                        y_linear[mask, member_index], dtype=float),
                    "y_tangent_nonlinear_only": np.asarray(
                        y_tangent_nonlinear[mask, member_index], dtype=float),
                    "locality_fraction": float(locality_by_member[member_index]),
                    "lambda_energy": lam,
                    "artifact_sha256": sha256(path),
                })
        truths.append(members)
    return truths


def leave_one_truth_out(truths: list[list[dict[str, Any]]], bins: int,
                        member_weighting: str) -> dict[str, Any]:
    folds = []
    edges = np.linspace(0.15, 0.85, bins + 1)
    for held_index in range(len(truths)):
        training = [truth for index, truth in enumerate(truths)
                    if index != held_index]
        raw_fit = balanced_raw_fit(
            training, "member" if member_weighting == "equal" else "truth")
        held_x = np.concatenate([member["x"] for member in truths[held_index]])
        held_y = np.concatenate([member["y"] for member in truths[held_index]])
        if member_weighting == "equal":
            held_weights = np.concatenate([
                np.full(member["y"].size, 1.0 / member["y"].size)
                for member in truths[held_index]
            ])
        else:
            held_weights = np.ones_like(held_y)
        raw_prediction = raw_fit["intercept"] + raw_fit["slope"] * held_x

        train_curve = binned_curve(training, bins, member_weighting)
        if member_weighting == "equal":
            held_member_bins = [
                bin_trajectory(member["x"], member["y"], edges)
                for member in truths[held_index]
            ]
            held_bx = np.mean(np.asarray([item[0] for item in held_member_bins]), axis=0)
            held_by = np.mean(np.asarray([item[1] for item in held_member_bins]), axis=0)
        else:
            held_bx, held_by = bin_trajectory(held_x, held_y, edges)
        valid = np.isfinite(held_by)
        binned_prediction = (train_curve["fit"]["intercept"]
                             + train_curve["fit"]["slope"] * held_bx[valid])
        folds.append({
            "held_truth": held_index,
            "training_raw_fit": raw_fit,
            "raw_predictive_r2": predictive_r2(
                held_y, raw_prediction, held_weights),
            "raw_rmse": float(np.sqrt(np.average(
                (held_y - raw_prediction) ** 2, weights=held_weights))),
            "training_binned_fit": train_curve["fit"],
            "held_binned_points": int(np.count_nonzero(valid)),
            "binned_predictive_r2": predictive_r2(held_by[valid], binned_prediction),
            "binned_rmse": float(np.sqrt(np.mean((held_by[valid] - binned_prediction) ** 2))),
        })
    return {
        "member_weighting": member_weighting,
        "folds": folds,
        "mean_raw_predictive_r2": float(np.mean([fold["raw_predictive_r2"] for fold in folds])),
        "median_raw_predictive_r2": float(np.median([fold["raw_predictive_r2"] for fold in folds])),
        "mean_binned_predictive_r2": float(np.mean([fold["binned_predictive_r2"] for fold in folds])),
        "median_binned_predictive_r2": float(np.median([fold["binned_predictive_r2"] for fold in folds])),
    }


def percentile_interval(values: np.ndarray) -> list[float]:
    return np.percentile(values, [2.5, 50.0, 97.5]).astype(float).tolist()


def hierarchical_bootstrap(truths_by_regime: dict[str, list[list[dict[str, Any]]]],
                           replicates: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    regime_values: dict[str, dict[str, list[float]]] = {
        rid: {
            "slope": [],
            "binned_r2_equal_member": [],
            "binned_r2_pooled_member": [],
            "locality": [],
        }
        for rid in truths_by_regime
    }
    locality_difference = []
    regime_ids = list(truths_by_regime)
    for _ in range(replicates):
        locality_means = {}
        for rid, truths in truths_by_regime.items():
            draws = []
            for truth_index in rng.integers(0, len(truths), size=len(truths)):
                truth = truths[int(truth_index)]
                members = [truth[int(index)] for index in
                           rng.integers(0, len(truth), size=len(truth))]
                draws.append(members)
            fit = balanced_raw_fit(draws, "truth")
            equal_curve = binned_curve(draws, PRIMARY_BINS, "equal")
            pooled_curve = binned_curve(draws, PRIMARY_BINS, "pooled")
            locality = float(np.mean([
                np.mean([member["locality_fraction"] for member in members])
                for members in draws
            ]))
            regime_values[rid]["slope"].append(float(fit["slope"]))
            regime_values[rid]["binned_r2_equal_member"].append(
                float(equal_curve["fit"]["r2"]))
            regime_values[rid]["binned_r2_pooled_member"].append(
                float(pooled_curve["fit"]["r2"]))
            regime_values[rid]["locality"].append(locality)
            locality_means[rid] = locality
        locality_difference.append(locality_means[regime_ids[1]]
                                   - locality_means[regime_ids[0]])
    result = {}
    for rid, metrics in regime_values.items():
        result[rid] = {
            "slope": {
                "interval_2p5_50_97p5": percentile_interval(
                    np.asarray(metrics["slope"])),
                "probability_positive": float(
                    np.mean(np.asarray(metrics["slope"]) > 0)),
            },
            "binned_r2_equal_member": {
                "interval_2p5_50_97p5": percentile_interval(
                    np.asarray(metrics["binned_r2_equal_member"])),
                "note": "In-sample descriptive R2; no probability-positive interpretation.",
            },
            "binned_r2_pooled_member": {
                "interval_2p5_50_97p5": percentile_interval(
                    np.asarray(metrics["binned_r2_pooled_member"])),
                "note": "In-sample descriptive R2; no probability-positive interpretation.",
            },
            "locality": {
                "interval_2p5_50_97p5": percentile_interval(
                    np.asarray(metrics["locality"])),
            },
        }
    result["locality_difference_n512_minus_n256"] = {
        "interval_2p5_50_97p5": percentile_interval(np.asarray(locality_difference)),
        "probability_positive": float(np.mean(np.asarray(locality_difference) > 0)),
    }
    return result


def analyze_regime(truths: list[list[dict[str, Any]]]) -> dict[str, Any]:
    x = np.concatenate([member["x"] for truth in truths for member in truth])
    y = np.concatenate([member["y"] for truth in truths for member in truth])
    truth_fits, member_fits = [], []
    locality_by_truth = []
    for truth_index, truth in enumerate(truths):
        tx = np.concatenate([member["x"] for member in truth])
        ty = np.concatenate([member["y"] for member in truth])
        truth_fits.append({"truth_index": truth_index, **linear_fit(tx, ty)})
        locality_by_truth.append(float(np.mean(
            [member["locality_fraction"] for member in truth])))
        for member in truth:
            member_fits.append({
                "truth_index": truth_index,
                "member_index": member["member_index"],
                **linear_fit(member["x"], member["y"]),
            })
    sensitivity = {}
    for bins in BIN_COUNTS:
        sensitivity[str(bins)] = {
            "truth_balanced_equal_member": binned_curve(
                truths, bins, "equal"),
            "truth_balanced_pooled_member": binned_curve(
                truths, bins, "pooled"),
            "point_pooled": point_binned_curve(truths, bins),
        }
    return {
        "eligible_points": int(x.size),
        "x_range": [float(np.min(x)), float(np.max(x))],
        "point_pooled_fit": linear_fit(x, y),
        "truth_balanced_fit": balanced_raw_fit(truths, "truth"),
        "member_balanced_fit": balanced_raw_fit(truths, "member"),
        "response_component_audit": {
            "P_linear_only": component_audit(truths, "y_linear_only"),
            "P_tl_nonlinear_only": component_audit(
                truths, "y_tangent_nonlinear_only"),
            "certified_full_sum": component_audit(truths, "y"),
        },
        "truth_fits": truth_fits,
        "member_fits": member_fits,
        "bin_sensitivity": sensitivity,
        "leave_one_truth_out": {
            "equal_member": leave_one_truth_out(
                truths, PRIMARY_BINS, "equal"),
            "pooled_member": leave_one_truth_out(
                truths, PRIMARY_BINS, "pooled"),
        },
        "locality_fraction_by_truth": locality_by_truth,
        "locality_fraction_mean": float(np.mean(locality_by_truth)),
    }


def plot_results(result: dict[str, Any], truths_by_regime: dict[str, Any],
                 png: Path, pdf: Path) -> None:
    colors = {"n256_re470_steep": "#D55E00", "n512_re940_marginal": "#0072B2"}
    labels = {"n256_re470_steep": "N=256, Re≈470",
              "n512_re940_marginal": "N=512, Re≈940"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    for column, rid in enumerate(colors):
        axis = axes[0, column]
        truths = truths_by_regime[rid]
        for truth_index, truth in enumerate(truths):
            x = np.concatenate([member["x"] for member in truth])
            y = np.concatenate([member["y"] for member in truth])
            axis.scatter(x, y, s=9, alpha=0.12, color=colors[rid], linewidths=0)
        curve = result["regimes"][rid]["bin_sensitivity"][str(PRIMARY_BINS)][
            "truth_balanced_equal_member"]
        bx = np.asarray(curve["mean_x"])
        by = np.asarray(curve["mean_y"])
        sem = np.asarray(curve["sem_y"])
        axis.errorbar(bx, by, yerr=sem, fmt="o", ms=5, color="#161616",
                      ecolor="#555555", capsize=2.5, label="truth-balanced bins")
        fit = curve["fit"]
        line_x = np.linspace(0.15, 0.85, 100)
        axis.plot(line_x, fit["intercept"] + fit["slope"] * line_x,
                  color=colors[rid], lw=2.2, label=f"binned slope={fit['slope']:.2f}")
        axis.plot(line_x, line_x, color="#777777", lw=1, ls="--", label="y=x")
        axis.set_title(labels[rid])
        axis.set_xlabel(r"$x_C=2C/(E_t+E_f)$")
        axis.set_ylabel(
            r"$(P_{\rm linear}+P_{\rm TL,nonlinear})/(\lambda_b\epsilon)$")
        axis.set_xlim(0.12, 0.88)
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, fontsize=8)

    locality_axis = axes[1, 0]
    for index, rid in enumerate(colors):
        values = result["regimes"][rid]["locality_fraction_by_truth"]
        jitter = np.linspace(-0.08, 0.08, len(values))
        locality_axis.scatter(index + jitter, values, s=42, color=colors[rid],
                              edgecolor="white", linewidth=0.6)
        locality_axis.hlines(np.mean(values), index - 0.18, index + 0.18,
                             color="#111111", lw=2)
    locality_axis.set_xticks([0, 1], [labels[rid] for rid in colors])
    locality_axis.set_ylabel("octave-local tangent fraction")
    locality_axis.set_title("Measured locality contrast")
    locality_axis.grid(axis="y", alpha=0.18)

    cv_axis = axes[1, 1]
    positions, values, tick_labels = [], [], []
    position = 0
    for rid in colors:
        folds = result["regimes"][rid]["leave_one_truth_out"]["equal_member"]["folds"]
        values.extend([fold["binned_predictive_r2"] for fold in folds])
        positions.extend([position + offset for offset in np.linspace(-0.18, 0.18, 4)])
        tick_labels.append(labels[rid])
        position += 1
    cv_axis.scatter(positions[:4], values[:4], s=42, color=colors["n256_re470_steep"])
    cv_axis.scatter(positions[4:], values[4:], s=42, color=colors["n512_re940_marginal"])
    cv_axis.axhline(0, color="#555555", lw=1, ls="--")
    cv_axis.set_xticks([0, 1], tick_labels)
    cv_axis.set_ylabel("held-out truth binned predictive $R^2$")
    cv_axis.set_title("Leave-one-truth-out generalization")
    cv_axis.grid(axis="y", alpha=0.18)

    fig.suptitle("PRL v8 pilot: exact-recipe collapse diagnostics", fontsize=15)
    for target in (png, pdf):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite: {target}")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    config_path = args.config.resolve()
    output = args.output.resolve()
    evaluation_path = args.evaluation.resolve()
    report_dir = args.report_dir.resolve()
    spec_path = args.spec.resolve()
    if sha256(evaluation_path) != EVALUATION_SHA256:
        raise ValueError("authoritative 8-of-8 evaluation hash mismatch")
    evaluation = generator.load_json(evaluation_path)
    if not evaluation.get("pilot_green") or evaluation.get("production_authorized"):
        raise ValueError("analysis requires a green, nonproduction pilot evaluation")
    config = generator.load_json(config_path)
    generator.validate_config(config)
    if evaluation["config"]["sha256"] != sha256(config_path):
        raise ValueError("active config hash does not match the 8-of-8 evaluation")

    artifact_bindings = []
    for regime in config["design"]["regimes"]:
        rid = regime["id"]
        rows = evaluation["regimes"][rid]["truths"]
        for truth_index, row in enumerate(rows):
            path = output / rid / "clusters" / f"truth_{truth_index:02d}.npz"
            actual_hash = sha256(path)
            if actual_hash != row["sha256"]:
                raise ValueError(f"artifact hash mismatch: {path}")
            artifact_bindings.append({
                "regime": rid,
                "truth_index": truth_index,
                "path": str(path),
                "sha256": actual_hash,
                "bytes": path.stat().st_size,
            })

    truths_by_regime = {
        regime["id"]: load_regime(config, output, regime)
        for regime in config["design"]["regimes"]
    }
    regimes = {
        rid: analyze_regime(truths)
        for rid, truths in truths_by_regime.items()
    }
    bootstrap = hierarchical_bootstrap(
        truths_by_regime, args.bootstrap_replicates, BOOTSTRAP_SEED)
    result = {
        "schema": SCHEMA,
        "scope": "exploratory pilot decision analysis; not confirmatory evidence",
        "production_authorized": False,
        "exact_recipe": {
            "predictor": "2*C/(Et+Ef)",
            "response": "(P_linear+P_tl_nonlinear)/(lambda_b*epsilon)",
            "mask": "finite and 0.15 < rho < 0.85",
            "primary_band": [10, 20],
        },
        "script": {"path": str(SCRIPT), "sha256": sha256(SCRIPT)},
        "specification": {"path": str(spec_path), "sha256": sha256(spec_path)},
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "evaluation": {"path": str(evaluation_path), "sha256": sha256(evaluation_path)},
        "artifact_bindings": artifact_bindings,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
            "results": bootstrap,
        },
        "regimes": regimes,
        "null_hierarchy_evaluated": False,
        "null_hierarchy_note": (
            "Binning does not replace the frozen time-block, cross-truth reassignment, "
            "and phase-scramble null hierarchy."
        ),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    target = report_dir / "prl_v8_pilot_collapse_analysis_v4_20260717.json"
    png = report_dir / "prl_v8_pilot_collapse_analysis_v4_20260717.png"
    pdf = report_dir / "prl_v8_pilot_collapse_analysis_v4_20260717.pdf"
    generator.atomic_json(target, result)
    plot_results(result, truths_by_regime, png, pdf)
    print(target)


if __name__ == "__main__":
    main()
