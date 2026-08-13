#!/usr/bin/env python3
"""Build the PRL Figure 1 candidate from frozen SQG pilot artifacts.

The two-cluster partition uses only the prospectively frozen one-shape residual
and one-clock dispersion. It never uses agreement with the logistic curve or
the production-covariance response. The plotted logistic curve uses each
trajectory's independently measured tangent rate and a single time origin at
the 50-percent decorrelation crossing; no finite-error rate is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
DEFAULT_AUDIT = ROOT / "results/MTB_RESULT_2026-08-11.json"
DEFAULT_COMPACT = ROOT / "data/prl_figure1_data_v1.npz"
DEFAULT_FIGURE = ROOT / "figures/prl_figure1_data_v1"
DEFAULT_RESULT = ROOT / "results/prl_figure1_data_v1.json"

REGIMES = (
    ("n256_re470_steep", r"$N=256$", "o"),
    ("n512_re940_marginal", r"$N=512$", "s"),
)
PRIMARY_BAND = 2
LAMBDA_BURNIN = 5.0
RHO_INTERVAL = (0.15, 0.85)
R_ONE_SHAPE = 0.05
R_MOVING_FRONT = 0.25
LAMBDA_ONE_CLOCK = 0.10
CLUSTER_COLORS = ("#176B87", "#C05A3D")
INK = "#182126"
MUTED = "#69757A"
GOLD = "#C18B2F"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def repository_path(path: Path) -> str:
    """Use portable repository-relative paths when an artifact is packaged."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def frozen_features(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime, _, _ in REGIMES:
        for truth in audit["regimes"][regime]["truths"]:
            d1 = {int(row["member_index"]): row for row in truth["members"]["D1"]}
            d2 = {int(row["member_index"]): row for row in truth["members"]["D2"]}
            for member in sorted(d1):
                rows.append({
                    "regime": regime,
                    "truth": int(truth["truth_index"]),
                    "member": member,
                    "R": float(d1[member]["R_median"]),
                    "Lambda_disp": float(d2[member]["Lambda_disp_median"]),
                })
    return rows


def deterministic_kmeans(features: np.ndarray) -> np.ndarray:
    """Return a deterministic two-means partition in standardized feature space."""
    scale = np.std(features, axis=0, ddof=1)
    z = (features - np.mean(features, axis=0)) / np.where(scale > 0, scale, 1.0)
    _, _, vh = np.linalg.svd(z, full_matrices=False)
    score = z @ vh[0]
    centers = np.stack([z[np.argmin(score)], z[np.argmax(score)]])
    labels = np.zeros(len(z), dtype=int)
    for _ in range(100):
        new_labels = np.argmin(np.sum((z[:, None] - centers[None]) ** 2, axis=2), axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        centers = np.stack([np.mean(z[labels == index], axis=0) for index in range(2)])
    means = [np.mean(features[labels == index, 1]) for index in range(2)]
    low_clock = int(np.argmin(means))
    return np.where(labels == low_clock, 0, 1)


def crossing_time(time: np.ndarray, values: np.ndarray, level: float) -> float:
    eligible = np.flatnonzero((values[:-1] <= level) & (values[1:] > level))
    if not len(eligible):
        raise ValueError(f"trajectory does not cross level {level}")
    index = int(eligible[0])
    fraction = (level - values[index]) / (values[index + 1] - values[index])
    return float(time[index] + fraction * (time[index + 1] - time[index]))


def load_trajectories(cluster_root: Path, rows: list[dict[str, Any]]) -> None:
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    provenance: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["regime"], row["truth"])
        if key not in cache:
            path = cluster_root / key[0] / "clusters" / f"truth_{key[1]:02d}.npz"
            with np.load(path, allow_pickle=False) as data:
                cache[key] = {
                    name: np.asarray(data[name])
                    for name in (
                        "scalar_time", "lambda_time", "lambda_energy_rate", "Et", "Ef",
                        "C", "epsilon", "rho", "P_linear", "P_tl_nonlinear",
                        "P_pure_error",
                    )
                }
            provenance[key] = {
                "artifact": str(path.relative_to(cluster_root)),
                "artifact_sha256": sha256(path),
            }
        row.update(provenance[key])
        data = cache[key]
        member = row["member"]
        time = data["scalar_time"]
        et = data["Et"][:, member, PRIMARY_BAND]
        ef = data["Ef"][:, member, PRIMARY_BAND]
        cross = data["C"][:, member, PRIMARY_BAND]
        epsilon = data["epsilon"][:, member, PRIMARY_BAND]
        rho = data["rho"][:, member, PRIMARY_BAND]
        x_covariance = 2.0 * cross / np.maximum(et + ef, np.finfo(float).tiny)
        amplitude = 1.0 - x_covariance
        lambda_mask = data["lambda_time"] >= LAMBDA_BURNIN
        rate = float(np.mean(data["lambda_energy_rate"][lambda_mask, PRIMARY_BAND]))
        origin = crossing_time(time, amplitude, 0.5)
        tau = rate * (time - origin)
        mask = (
            np.isfinite(x_covariance)
            & np.isfinite(epsilon)
            & (epsilon > 0.0)
            & (rho > RHO_INTERVAL[0])
            & (rho < RHO_INTERVAL[1])
        )
        denominator = rate * epsilon
        linear = data["P_linear"][:, member, PRIMARY_BAND] / denominator
        tangent = data["P_tl_nonlinear"][:, member, PRIMARY_BAND] / denominator
        pure = data["P_pure_error"][:, member, PRIMARY_BAND] / denominator
        logistic = 1.0 / (1.0 + np.exp(-tau))
        row.update({
            "lambda_b": rate,
            "t_half": origin,
            "tau": tau[mask],
            "a": amplitude[mask],
            "x": x_covariance[mask],
            "y_linear": linear[mask],
            "y_tangent": tangent[mask],
            "y_sum": (linear + tangent)[mask],
            "y_pure": pure[mask],
            "logistic_rmse": float(np.sqrt(np.mean((amplitude[mask] - logistic[mask]) ** 2))),
        })


COMPACT_ARRAYS = ("tau", "a", "x", "y_linear", "y_tangent", "y_sum", "y_pure")


def write_compact(path: Path, rows: list[dict[str, Any]]) -> None:
    """Store only the derived vectors required to reproduce the published figure."""
    payload: dict[str, np.ndarray] = {}
    metadata = []
    for index, row in enumerate(rows):
        metadata.append(serializable_row(row))
        for name in COMPACT_ARRAYS:
            payload[f"row_{index:02d}_{name}"] = np.asarray(row[name], dtype=np.float64)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_compact(path: Path, rows: list[dict[str, Any]]) -> None:
    """Attach compact derived vectors while checking them against the frozen audit."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        indexed = {
            (row["regime"], int(row["truth"]), int(row["member"])): (index, row)
            for index, row in enumerate(metadata)
        }
        for row in rows:
            key = (row["regime"], row["truth"], row["member"])
            if key not in indexed:
                raise ValueError(f"compact data omit trajectory {key}")
            index, stored = indexed[key]
            for name in ("R", "Lambda_disp"):
                if not np.isclose(row[name], stored[name], rtol=0.0, atol=1e-14):
                    raise ValueError(f"compact {name} disagrees with frozen audit for {key}")
            row.update(stored)
            for name in COMPACT_ARRAYS:
                row[name] = np.asarray(data[f"row_{index:02d}_{name}"], dtype=np.float64)


def curve_summary(rows: list[dict[str, Any]], cluster: int) -> dict[str, np.ndarray]:
    grid = np.linspace(-3.0, 3.0, 121)
    curves = []
    for row in rows:
        if row["cluster"] != cluster:
            continue
        order = np.argsort(row["tau"])
        tau = row["tau"][order]
        amplitude = row["a"][order]
        values = np.interp(grid, tau, amplitude, left=np.nan, right=np.nan)
        values[(grid < np.min(tau)) | (grid > np.max(tau))] = np.nan
        curves.append(values)
    matrix = np.asarray(curves)
    count = np.sum(np.isfinite(matrix), axis=0)
    mean = np.full_like(grid, np.nan)
    lo = np.full_like(grid, np.nan)
    hi = np.full_like(grid, np.nan)
    valid = count > 0
    mean[valid] = np.nanmean(matrix[:, valid], axis=0)
    lo[valid] = np.nanquantile(matrix[:, valid], 0.16, axis=0)
    hi[valid] = np.nanquantile(matrix[:, valid], 0.84, axis=0)
    return {
        "grid": grid,
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "count": count,
    }


def binned_summary(
    rows: list[dict[str, Any]], cluster: int | None, field: str, bins: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(RHO_INTERVAL[0], RHO_INTERVAL[1], bins + 1)
    per_trajectory = []
    x_per_trajectory = []
    for row in rows:
        if cluster is not None and row["cluster"] != cluster:
            continue
        x = row["x"]
        y = row[field]
        y_bins = np.full(bins, np.nan)
        x_bins = np.full(bins, np.nan)
        for index in range(bins):
            use = (x >= edges[index]) & (x < edges[index + 1]) & np.isfinite(y)
            if index == bins - 1:
                use |= (x == edges[index + 1]) & np.isfinite(y)
            if np.any(use):
                x_bins[index] = np.mean(x[use])
                y_bins[index] = np.mean(y[use])
        per_trajectory.append(y_bins)
        x_per_trajectory.append(x_bins)
    y_matrix = np.asarray(per_trajectory)
    x_matrix = np.asarray(x_per_trajectory)
    return (
        np.nanmean(x_matrix, axis=0),
        np.nanmean(y_matrix, axis=0),
        np.nanstd(y_matrix, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(y_matrix), axis=0)),
    )


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 9.2,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.65), constrained_layout=True)

    axis = axes[0, 0]
    axis.axvspan(0, R_ONE_SHAPE, color="#DCEBDD", alpha=0.75, zorder=0)
    axis.axvspan(R_MOVING_FRONT, 0.34, color="#E5EEF2", alpha=0.8, zorder=0)
    axis.axhspan(0, LAMBDA_ONE_CLOCK, color="#DCEBDD", alpha=0.75, zorder=0)
    for regime, label, marker in REGIMES:
        for cluster in range(2):
            selected = [row for row in rows if row["regime"] == regime and row["cluster"] == cluster]
            axis.scatter(
                [row["R"] for row in selected],
                [row["Lambda_disp"] for row in selected],
                marker=marker, s=38, color=CLUSTER_COLORS[cluster], edgecolor="white",
                linewidth=0.55, label=f"{label}, C{cluster + 1}", zorder=3,
            )
    axis.axvline(R_ONE_SHAPE, color=MUTED, lw=0.8, ls="--")
    axis.axvline(R_MOVING_FRONT, color=MUTED, lw=0.8, ls=":")
    axis.axhline(LAMBDA_ONE_CLOCK, color=MUTED, lw=0.8, ls="--")
    axis.text(0.006, 0.025, "logistic\nadmission", fontsize=6.8, color="#2E6B3A")
    axis.text(0.258, 0.59, "similarity-front\ngeometry", fontsize=6.8, color="#315D70")
    axis.set_xlim(0, 0.34)
    axis.set_ylim(0, 0.67)
    axis.set_xlabel(r"one-shape residual $R$")
    axis.set_ylabel(r"one-clock dispersion $\Lambda_{\rm disp}$")
    axis.set_title("(a) Frozen spectral admission coordinates", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=1, handletextpad=0.3, labelspacing=0.25,
                loc="center right")

    axis = axes[0, 1]
    for row in rows:
        axis.plot(row["tau"], row["a"], color=CLUSTER_COLORS[row["cluster"]], alpha=0.18, lw=0.65)
    for cluster in range(2):
        summary = curve_summary(rows, cluster)
        valid = summary["count"] >= 2
        axis.fill_between(
            summary["grid"][valid], summary["lo"][valid], summary["hi"][valid],
            color=CLUSTER_COLORS[cluster], alpha=0.15, linewidth=0,
        )
        axis.plot(
            summary["grid"][valid], summary["mean"][valid],
            color=CLUSTER_COLORS[cluster], lw=2.0, label=f"C{cluster + 1} mean",
        )
    tau_grid = np.linspace(-3, 3, 300)
    axis.plot(tau_grid, 1.0 / (1.0 + np.exp(-tau_grid)), color=INK, lw=1.25,
              ls="--", label="independent-rate logistic")
    axis.set_xlim(-3, 3)
    axis.set_ylim(0.08, 0.92)
    axis.set_xlabel(r"independent clock $\tau=\lambda_b(t-t_{1/2})$")
    axis.set_ylabel(r"decorrelated fraction $a_b=1-x_C$")
    axis.set_title("(b) Clustered scalar lifecycles", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left")

    axis = axes[1, 0]
    xline = np.linspace(*RHO_INTERVAL, 100)
    axis.plot(xline, xline, color=INK, lw=1.0, ls="--", label=r"closure $y=x_C$")
    for cluster in range(2):
        x, y, sem = binned_summary(rows, cluster, "y_sum")
        axis.errorbar(
            x, y, yerr=sem, color=CLUSTER_COLORS[cluster], marker="o", ms=3.7,
            lw=1.6, capsize=1.8, label=f"C{cluster + 1}",
        )
    axis.set_xlim(*RHO_INTERVAL)
    axis.set_xlabel(r"remaining covariance $x_C=2C_b/(E_{t,b}+E_{f,b})$")
    axis.set_ylabel(r"$y=(P_{\rm linear}+P_{\rm TL,nl})/(\lambda_b\varepsilon_b)$")
    axis.set_title("(c) Certified production closure", loc="left", fontweight="bold")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    components = (
        ("y_linear", r"linear $P_{\rm linear}$", MUTED, "-"),
        ("y_tangent", r"state-error $P_{\rm TL,nl}$", "#2677A3", "-"),
        ("y_sum", r"certified sum $y$", INK, "-"),
        ("y_pure", r"error self-interaction $P_{\rm pure}$", GOLD, "--"),
    )
    for field, label, color, style in components:
        x, y, _ = binned_summary(rows, None, field)
        axis.plot(x, y, color=color, lw=1.7 if field == "y_sum" else 1.25,
                  ls=style, marker="o", ms=2.8, label=label)
    axis.axhline(0, color="#A0A7AA", lw=0.7)
    axis.set_xlim(*RHO_INTERVAL)
    axis.set_xlabel(r"remaining covariance $x_C$")
    axis.set_ylabel(r"component$/\,(\lambda_b\varepsilon_b)$")
    axis.set_title("(d) What enters the band budget", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="best")

    metadata = {
        "Title": "Frozen SQG trajectories and conditional logistic admission",
        "Author": "Logistic Error Growth Program",
        "Subject": "Actual SQG pilot data; clustering excludes closure response",
        "Creator": f"{SCRIPT.name}; sha256={sha256(SCRIPT)}",
    }
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    fig.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"tau", "a", "x", "y_linear", "y_tangent", "y_sum", "y_pure"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster-root", type=Path,
        help="optional raw SQG root; when omitted, use the packaged compact data",
    )
    parser.add_argument("--compact-data", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument(
        "--write-compact", action="store_true",
        help="write --compact-data from --cluster-root after verifying the raw archives",
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outputs = [args.figure.with_suffix(".pdf"), args.figure.with_suffix(".png"), args.result]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, existing)))

    audit = load_json(args.audit)
    rows = frozen_features(audit)
    features = np.asarray([[row["R"], row["Lambda_disp"]] for row in rows])
    labels = deterministic_kmeans(features)
    for row, label in zip(rows, labels):
        row["cluster"] = int(label)
    if args.cluster_root is None:
        if args.write_compact:
            parser.error("--write-compact requires --cluster-root")
        load_compact(args.compact_data, rows)
    else:
        load_trajectories(args.cluster_root, rows)
        if args.write_compact:
            write_compact(args.compact_data, rows)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, args.figure)

    result = {
        "schema": "prl-figure1-data-clusters-1",
        "evidence_status": "exploratory_visualization_of_frozen_artifacts",
        "clustering_predictors": ["R", "Lambda_disp"],
        "clustering_excludes": ["logistic agreement", "x_C", "production response"],
        "thresholds": {
            "one_shape_R_max": R_ONE_SHAPE,
            "moving_front_R_min": R_MOVING_FRONT,
            "one_clock_Lambda_disp_max": LAMBDA_ONE_CLOCK,
        },
        "audit": {"path": repository_path(args.audit), "sha256": sha256(args.audit)},
        "compact_data": {
            "path": repository_path(args.compact_data),
            "sha256": sha256(args.compact_data) if args.compact_data.exists() else None,
        },
        "script_sha256": sha256(SCRIPT),
        "trajectories": [serializable_row(row) for row in rows],
        "cluster_summary": {
            str(cluster + 1): {
                "count": int(sum(row["cluster"] == cluster for row in rows)),
                "R_mean": float(np.mean([row["R"] for row in rows if row["cluster"] == cluster])),
                "Lambda_disp_mean": float(np.mean([
                    row["Lambda_disp"] for row in rows if row["cluster"] == cluster
                ])),
                "logistic_rmse_mean": float(np.mean([
                    row["logistic_rmse"] for row in rows if row["cluster"] == cluster
                ])),
            }
            for cluster in range(2)
        },
        "admission_counts": {
            "logistic": int(np.sum(
                (features[:, 0] <= R_ONE_SHAPE) & (features[:, 1] <= LAMBDA_ONE_CLOCK)
            )),
            "moving_front": int(np.sum(features[:, 0] > R_MOVING_FRONT)),
        },
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["cluster_summary"], indent=2))
    print(json.dumps(result["admission_counts"], indent=2))
    print(args.figure.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
