#!/usr/bin/env python3
"""Exploratory diagnosis of the PRL v8 cross-truth null failure.

This analysis does not alter or reinterpret the frozen null-gate verdict. It
reuses the certified scalar observable, removes common lifecycle structure,
and tests whether paired member deviations retain a positive association.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import prl_v8_pilot_null_gate as null_engine
import prl_v8_two_regime as generator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "prl_v8_gate5_nonconfirmatory_config_20260715.json"
DEFAULT_OUTPUT = HERE / "out" / "prl_v8_gate5_nonconfirmatory_20260715"
DEFAULT_CERTIFIED = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_pilot_null_gate_v2_20260720.json"
)
DEFAULT_REPORT_DIR = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_cross_truth_residual_diagnostic_20260812"
)
SCHEMA = "prl-v8-cross-truth-residual-diagnostic-3"
RESULT_STEM = "prl_v8_cross_truth_residual_diagnostic_v3_20260812"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def weighted_fit(truths: list[list[dict[str, np.ndarray]]]) -> dict[str, float | int]:
    xs, ys, weights = [], [], []
    for members in truths:
        for member in members:
            mask = np.asarray(member["mask"], dtype=bool)
            x = np.asarray(member["x"], dtype=float)[mask]
            y = np.asarray(member["y"], dtype=float)[mask]
            if x.size < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError("invalid eligible trajectory in weighted fit")
            xs.append(x)
            ys.append(y)
            weights.append(np.full(x.size, 1.0 / (len(truths) * len(members) * x.size)))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    weight = np.concatenate(weights)
    design = np.column_stack([np.ones(x.size), x])
    root_weight = np.sqrt(weight)
    beta, *_ = np.linalg.lstsq(
        design * root_weight[:, None], y * root_weight, rcond=None
    )
    prediction = design @ beta
    mean_x = float(np.sum(weight * x) / np.sum(weight))
    mean_y = float(np.sum(weight * y) / np.sum(weight))
    sse = float(np.sum(weight * (y - prediction) ** 2))
    sst = float(np.sum(weight * (y - mean_y) ** 2))
    variance_x = float(np.sum(weight * (x - mean_x) ** 2) / np.sum(weight))
    variance_y = float(np.sum(weight * (y - mean_y) ** 2) / np.sum(weight))
    covariance = float(
        np.sum(weight * (x - mean_x) * (y - mean_y)) / np.sum(weight)
    )
    correlation = covariance / np.sqrt(max(variance_x * variance_y, 1e-300))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(1.0 - sse / sst) if sst > 0.0 else float("nan"),
        "weighted_correlation": float(correlation),
        "weighted_x_standard_deviation": float(np.sqrt(variance_x)),
        "weighted_y_standard_deviation": float(np.sqrt(variance_y)),
        "eligible_samples": int(x.size),
    }


def attach_rho(
    truths: list[list[dict[str, np.ndarray]]], config: dict[str, Any],
    output: Path, regime: dict[str, Any],
) -> None:
    bands = generator.expanded_bands(config, int(regime["n"]))
    primary = bands.index(tuple(config["design"]["primary_band"]))
    for truth_index, members in enumerate(truths):
        path = output / regime["id"] / "clusters" / f"truth_{truth_index:02d}.npz"
        with np.load(path, allow_pickle=False) as data:
            rho = np.asarray(data["rho"][:, :, primary], dtype=float)
        for member_index, member in enumerate(members):
            member["rho"] = rho[:, member_index]


def polynomial_lifecycle_fit(
    donor_truths: list[list[dict[str, np.ndarray]]], key: str, degree: int,
) -> np.ndarray:
    rows, values, weights = [], [], []
    for members in donor_truths:
        for member in members:
            mask = np.asarray(member["mask"], dtype=bool)
            rho = np.asarray(member["rho"], dtype=float)[mask]
            value = np.asarray(member[key], dtype=float)[mask]
            rows.append(np.vander(rho, N=degree + 1, increasing=True))
            values.append(value)
            weights.append(np.full(value.size, 1.0 / (
                len(donor_truths) * len(members) * value.size
            )))
    design = np.concatenate(rows)
    response = np.concatenate(values)
    weight = np.concatenate(weights)
    root_weight = np.sqrt(weight)
    beta, *_ = np.linalg.lstsq(
        design * root_weight[:, None], response * root_weight, rcond=None
    )
    return beta


def common_lifecycle_residuals(
    truths: list[list[dict[str, np.ndarray]]], degree: int,
) -> tuple[list[list[dict[str, np.ndarray]]], dict[str, Any]]:
    """Subtract leave-one-truth-out polynomial lifecycle curves in rho."""
    residual_truths: list[list[dict[str, np.ndarray]]] = []
    coverage_rows = []
    prediction_rows = {"x": [], "y": [], "x_hat": [], "y_hat": []}
    for target_truth, members in enumerate(truths):
        donors = [
            donor_truth for truth_index, donor_truth in enumerate(truths)
            if truth_index != target_truth
        ]
        beta_x = polynomial_lifecycle_fit(donors, "x", degree)
        beta_y = polynomial_lifecycle_fit(donors, "y", degree)
        residual_members = []
        for member in members:
            x = np.asarray(member["x"], dtype=float)
            y = np.asarray(member["y"], dtype=float)
            rho = np.asarray(member["rho"], dtype=float)
            design = np.vander(rho, N=degree + 1, increasing=True)
            x_hat = design @ beta_x
            y_hat = design @ beta_y
            original_mask = np.asarray(member["mask"], dtype=bool)
            mask = original_mask & np.isfinite(x_hat) & np.isfinite(y_hat)
            residual_members.append({
                "time": np.asarray(member["time"], dtype=float),
                "x": x - x_hat,
                "y": y - y_hat,
                "mask": mask,
            })
            prediction_rows["x"].append(x[mask])
            prediction_rows["y"].append(y[mask])
            prediction_rows["x_hat"].append(x_hat[mask])
            prediction_rows["y_hat"].append(y_hat[mask])
            coverage_rows.append({
                "truth_index": target_truth,
                "member_index": len(residual_members) - 1,
                "original_eligible": int(np.count_nonzero(original_mask)),
                "residual_eligible": int(np.count_nonzero(mask)),
            })
        residual_truths.append(residual_members)

    def predictive_r2(name: str) -> float:
        observed = np.concatenate(prediction_rows[name])
        predicted = np.concatenate(prediction_rows[f"{name}_hat"])
        denominator = float(np.sum((observed - np.mean(observed)) ** 2))
        return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)

    return residual_truths, {
        "definition": (
            "subtract degree-%d polynomial in rho fitted to the other three truths "
            "with equal-truth/equal-member/sample weights" % degree
        ),
        "polynomial_degree": degree,
        "coverage": coverage_rows,
        "point_pooled_predictive_r2": {
            "x": predictive_r2("x"),
            "y": predictive_r2("y"),
        },
    }


def paired_member_deviations(
    truths: list[list[dict[str, np.ndarray]]],
) -> list[list[dict[str, np.ndarray]]]:
    """Remove each truth's two-member mean at every jointly eligible time."""
    result = []
    for members in truths:
        if len(members) != 2:
            raise ValueError("paired-member diagnostic requires exactly two members")
        joint = np.asarray(members[0]["mask"], dtype=bool) & np.asarray(
            members[1]["mask"], dtype=bool
        )
        x_mean = 0.5 * (
            np.asarray(members[0]["x"], dtype=float)
            + np.asarray(members[1]["x"], dtype=float)
        )
        y_mean = 0.5 * (
            np.asarray(members[0]["y"], dtype=float)
            + np.asarray(members[1]["y"], dtype=float)
        )
        result.append([
            {
                "time": np.asarray(member["time"], dtype=float),
                "x": np.asarray(member["x"], dtype=float) - x_mean,
                "y": np.asarray(member["y"], dtype=float) - y_mean,
                "mask": joint,
            }
            for member in members
        ])
    return result


def member_swap_slopes(
    deviations: list[list[dict[str, np.ndarray]]],
) -> np.ndarray:
    values = []
    for bits in itertools.product((0, 1), repeat=len(deviations)):
        surrogate = []
        for truth_index, members in enumerate(deviations):
            surrogate_members = []
            for recipient_member, recipient in enumerate(members):
                donor = members[recipient_member ^ bits[truth_index]]
                surrogate_members.append({
                    "time": recipient["time"],
                    "x": recipient["x"],
                    "y": donor["y"],
                    "mask": recipient["mask"],
                })
            surrogate.append(surrogate_members)
        values.append(null_engine.weighted_slope(surrogate))
    return np.asarray(values, dtype=float)


def cross_truth_summary(
    observed: float, slopes: np.ndarray
) -> list[dict[str, Any]]:
    if slopes.shape != (384,):
        raise ValueError("certified cross-truth array must have 384 slopes")
    rows = []
    for index, truth_map in enumerate(itertools.permutations(range(4))):
        group = slopes[index * 16:(index + 1) * 16]
        rows.append({
            "truth_map": list(truth_map),
            "fixed_points": int(sum(i == value for i, value in enumerate(truth_map))),
            "minimum": float(np.min(group)),
            "median": float(np.median(group)),
            "mean": float(np.mean(group)),
            "maximum": float(np.max(group)),
            "at_least_observed": int(np.count_nonzero(group >= observed)),
        })
    return rows


def exact_cross_result(
    truths: list[list[dict[str, np.ndarray]]],
) -> dict[str, Any]:
    observed = null_engine.weighted_slope(truths)
    values = np.asarray([
        null_engine.weighted_slope(surrogate)
        for surrogate in null_engine.cross_truth_surrogates(truths)
    ])
    return {
        "observed_slope": observed,
        "surrogate_slopes": values.tolist(),
        "exploratory_exact_p": null_engine.exact_group_p(observed, values),
    }


def analyze_regime(
    truths: list[list[dict[str, np.ndarray]]], certified_scalar: dict[str, Any]
) -> dict[str, Any]:
    raw_fit = weighted_fit(truths)
    certified_slope = float(certified_scalar["observed_slope"])
    if not np.isclose(raw_fit["slope"], certified_slope, rtol=0.0, atol=1e-12):
        raise ValueError("raw diagnostic does not reproduce the certified slope")
    certified_cross = np.asarray(
        certified_scalar["cross_truth_reassignment"]["slopes"], dtype=float
    )

    lifecycle_truths, lifecycle_metadata = common_lifecycle_residuals(truths, 3)
    lifecycle_fit = weighted_fit(lifecycle_truths)
    lifecycle_sensitivity = {}
    for degree in (2, 3, 4, 5):
        sensitivity_truths, sensitivity_metadata = common_lifecycle_residuals(
            truths, degree
        )
        lifecycle_sensitivity[str(degree)] = {
            "fit": weighted_fit(sensitivity_truths),
            "point_pooled_predictive_r2": sensitivity_metadata[
                "point_pooled_predictive_r2"
            ],
        }

    member_truths = paired_member_deviations(truths)
    member_fit = weighted_fit(member_truths)
    member_truth_fits = [
        {"truth_index": truth_index, **weighted_fit([members])}
        for truth_index, members in enumerate(member_truths)
    ]
    swap_values = member_swap_slopes(member_truths)
    return {
        "certified_reproduction": {
            "fit": raw_fit,
            "certified_slope": certified_slope,
            "absolute_slope_error": abs(float(raw_fit["slope"]) - certified_slope),
        },
        "certified_truth_permutation_maps": cross_truth_summary(
            certified_slope, certified_cross
        ),
        "common_lifecycle_residual": {
            "metadata": lifecycle_metadata,
            "fit": lifecycle_fit,
            "degree_sensitivity": lifecycle_sensitivity,
        },
        "within_truth_member_deviation": {
            "definition": "subtract the two-member truth mean at each jointly eligible time",
            "fit": member_fit,
            "truth_fits": member_truth_fits,
            "member_swap_slopes": swap_values.tolist(),
            "exploratory_exact_p": float(
                np.count_nonzero(swap_values >= float(member_fit["slope"]))
                / swap_values.size
            ),
        },
    }


def plot_result(result: dict[str, Any], path: Path) -> None:
    regime_ids = ["n512_re940_marginal", "n256_re470_steep"]
    colors = {0: "#136f63", 1: "#e0a100", 2: "#d1495b", 4: "#244f73"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for column, regime_id in enumerate(regime_ids):
        regime = result["regimes"][regime_id]
        rows = sorted(
            regime["certified_truth_permutation_maps"],
            key=lambda row: row["mean"],
        )
        axis = axes[0, column]
        for y_index, row in enumerate(rows):
            color = colors[row["fixed_points"]]
            axis.plot([row["minimum"], row["maximum"]], [y_index, y_index], color=color)
            axis.plot(row["median"], y_index, marker="o", ms=3.5, color=color)
        observed = regime["certified_reproduction"]["certified_slope"]
        axis.axvline(observed, color="black", ls="--", lw=1.2, label="observed")
        axis.axvline(0.0, color="#777777", lw=0.8)
        axis.set_title(f"{regime_id}: 24 truth maps")
        axis.set_xlabel("slope; range over 16 member swaps")
        axis.set_ylabel("truth maps sorted by mean slope")
        axis.set_yticks([])
        handles = [Line2D([0], [0], color="black", ls="--", label="observed")]
        handles.extend(
            Line2D([0], [0], color=colors[count], marker="o", label=f"{count} fixed")
            for count in (0, 1, 2, 4)
        )
        axis.legend(handles=handles, frameon=False, ncol=2, fontsize=8)

        comparison = axes[1, column]
        labels = ["raw certified", "LOTO lifecycle residual", "within-truth member residual"]
        fits = [
            regime["certified_reproduction"]["fit"],
            regime["common_lifecycle_residual"]["fit"],
            regime["within_truth_member_deviation"]["fit"],
        ]
        comparison.bar(
            np.arange(3), [fit["weighted_correlation"] for fit in fits],
            color=["#244f73", "#d1495b", "#136f63"],
        )
        comparison.axhline(0.0, color="black", lw=0.8)
        comparison.set_xticks(np.arange(3), labels, rotation=18, ha="right")
        comparison.set_ylabel("equal-truth/equal-member weighted correlation")
        comparison.set_ylim(-0.5, 0.5)
        comparison.set_title(
            "Residual slopes; exploratory only\n"
            f"lifecycle R²(y)={regime['common_lifecycle_residual']['metadata']['point_pooled_predictive_r2']['y']:.3f}"
        )
    figure.suptitle(
        "PRL v8 cross-truth failure diagnosis (frozen verdict unchanged)",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--certified-result", type=Path, default=DEFAULT_CERTIFIED)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    config_path = args.config.resolve()
    output = args.output.resolve()
    certified_path = args.certified_result.resolve()
    report_dir = args.report_dir.resolve()
    config = generator.load_json(config_path)
    certified = generator.load_json(certified_path)
    if certified.get("schema") != "prl-v8-pilot-null-gate-result-2":
        raise ValueError("wrong certified null-result schema")

    regimes = {row["id"]: row for row in config["design"]["regimes"]}
    regime_results = {}
    artifact_bindings = []
    expected_artifacts = {
        (row["regime"], int(row["truth_index"])): row["sha256"]
        for row in certified["artifacts"]
    }
    for regime_id in ("n512_re940_marginal", "n256_re470_steep"):
        truths = null_engine.load_scalar_regime(config, output, regimes[regime_id])
        attach_rho(truths, config, output, regimes[regime_id])
        regime_results[regime_id] = analyze_regime(
            truths, certified["regimes"][regime_id]["scalar"]
        )
        for truth_index in range(len(truths)):
            artifact = output / regime_id / "clusters" / f"truth_{truth_index:02d}.npz"
            artifact_hash = sha256(artifact)
            if artifact_hash != expected_artifacts[(regime_id, truth_index)]:
                raise ValueError(f"artifact hash differs from certified result: {artifact}")
            artifact_bindings.append({
                "regime": regime_id,
                "truth_index": truth_index,
                "path": str(artifact),
                "bytes": artifact.stat().st_size,
                "sha256": artifact_hash,
            })

    result = {
        "schema": SCHEMA,
        "status": "exploratory; frozen FAIL and production hold unchanged",
        "production_authorized": False,
        "bindings": {
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "certified_result": {"path": str(certified_path), "sha256": sha256(certified_path)},
            "artifacts": artifact_bindings,
        },
        "regimes": regime_results,
        "safeguards": [
            "Residual p-values are exploratory and do not replace the frozen null gate.",
            "Time samples are not treated as independent inferential replicates.",
            "Only four truth clusters and two members per truth are available.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{RESULT_STEM}.json"
    png_path = report_dir / f"{RESULT_STEM}.png"
    atomic_json(json_path, result)
    plot_result(result, png_path)
    print(json.dumps({
        "json": str(json_path),
        "png": str(png_path),
        "regimes": {
            regime_id: {
                "raw": row["certified_reproduction"]["fit"]["slope"],
                "lifecycle_residual_correlation": row["common_lifecycle_residual"]["fit"]["weighted_correlation"],
                "within_truth_member_correlation": row["within_truth_member_deviation"]["fit"]["weighted_correlation"],
            }
            for regime_id, row in regime_results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
