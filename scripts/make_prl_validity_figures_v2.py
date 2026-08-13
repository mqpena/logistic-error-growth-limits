#!/usr/bin/env python3
"""Generate the bounded theory-map and stored-artifact audit figures for PRL.

This script performs no simulation and derives no new inferential statistic. It
reads two dated JSON records, draws their stored per-truth summaries, and emits
only the two requested PDF/PNG figure pairs. Existing outputs are protected
unless ``--force`` is supplied.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
INPUT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
MTB_PATH = INPUT_DIR / "MTB_RESULT_2026-08-11.json"
CLOCK_PATH = INPUT_DIR / "CLOCK_SHAPE_COVARIANCE_RESULT_2026-08-12.json"

THEORY_STEM = "prl_validity_theory_map_v2"
AUDIT_STEM = "prl_validity_audit_v2"
EXPECTED_MTB_SCHEMA = "prl-logistic-mtb-result-1"

NAVY = "#17324D"
BLUE = "#2C6E9B"
TEAL = "#287C71"
GOLD = "#D49B36"
RED = "#A64B3C"
INK = "#202A2E"
MUTED = "#667277"
PAPER = "#F8F5ED"
LIGHT_BLUE = "#E8F1F5"
LIGHT_TEAL = "#E7F1ED"
LIGHT_GOLD = "#F5ECD8"
REGIMES = (
    ("n256_re470_steep", r"$N=256$", BLUE, "o"),
    ("n512_re940_marginal", r"$N=512$", RED, "s"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required input not found: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def output_paths() -> list[Path]:
    return [
        FIGURE_DIR / f"{THEORY_STEM}.pdf",
        FIGURE_DIR / f"{THEORY_STEM}.png",
        FIGURE_DIR / f"{AUDIT_STEM}.pdf",
        FIGURE_DIR / f"{AUDIT_STEM}.png",
    ]


def protect_outputs(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "refusing to overwrite existing output(s); use --force explicitly:\n"
            + rendered
        )


def validate_inputs(mtb: dict[str, Any], clock: dict[str, Any]) -> None:
    if mtb.get("schema") != EXPECTED_MTB_SCHEMA:
        raise ValueError(f"unexpected MTB schema: {mtb.get('schema')!r}")
    if "regimes" not in clock or "schema" not in clock:
        raise ValueError("clock/covariance record is missing schema or regimes")

    expected = {regime for regime, _, _, _ in REGIMES}
    if set(mtb["regimes"]) != expected or set(clock["regimes"]) != expected:
        raise ValueError("input regime sets do not match the two expected regimes")
    for regime in expected:
        mtb_truths = {int(row["truth_index"]) for row in mtb["regimes"][regime]["truths"]}
        clock_truths = {
            int(row["truth_index"]) for row in clock["regimes"][regime]["truths"]
        }
        if mtb_truths != {0, 1, 2, 3} or clock_truths != mtb_truths:
            raise ValueError(f"truth-index mismatch in {regime}")
        for row in mtb["regimes"][regime]["truths"]:
            if not np.isfinite([row["R"], row["Lambda_disp"]]).all():
                raise ValueError(f"non-finite MTB value in {regime}")
        for row in clock["regimes"][regime]["truths"]:
            value = row["truth_summary"]["net_correction_fraction"]
            if not np.isfinite(value):
                raise ValueError(f"non-finite correction in {regime}")


def metadata(input_hashes: dict[str, str], title: str) -> dict[str, str]:
    provenance = "; ".join(
        [f"script={sha256(SCRIPT)}"]
        + [f"{name}={digest}" for name, digest in input_hashes.items()]
    )
    return {
        "Title": title,
        "Author": "Logistic Error Growth Program",
        "Subject": "Stored-artifact validity audit; no new simulation or inference",
        "Keywords": provenance,
        "Creator": f"{SCRIPT.name}; {provenance}",
    }


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_color: str = NAVY,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.15,
        transform=axis.transAxes, zorder=2,
    )
    axis.add_patch(box)
    axis.text(
        x + 0.025 * width, y + height - 0.20 * height, title,
        transform=axis.transAxes, color=title_color, fontsize=9.2,
        fontweight="bold", va="top", ha="left", zorder=3,
    )
    axis.text(
        x + 0.025 * width, y + height - 0.42 * height, body,
        transform=axis.transAxes, color=INK, fontsize=7.2,
        linespacing=1.20, va="top", ha="left", zorder=3,
    )


def add_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(FancyArrowPatch(
        start, end, transform=axis.transAxes, arrowstyle="-|>",
        mutation_scale=12, linewidth=1.25, color=MUTED, zorder=1,
        connectionstyle="arc3,rad=0.0",
    ))


def make_theory_map(path_pdf: Path, path_png: Path, meta: dict[str, str]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
    })
    fig, axis = plt.subplots(figsize=(7.15, 4.45))
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.text(
        0.04, 0.955, "Two-field closure: two admissible reductions",
        transform=axis.transAxes, fontsize=15.5, fontweight="bold",
        color=NAVY, va="top",
    )
    axis.text(
        0.04, 0.895,
        "Logistic growth is conditional; a moving decorrelation front follows a distinct similarity law.",
        transform=axis.transAxes, fontsize=8.8, color=MUTED, va="top",
    )

    add_box(
        axis, (0.29, 0.65), 0.42, 0.19,
        "EXACT TWO-FIELD BALANCE",
        "Truth–forecast interaction supplies\n"
        "full-field error variance. Conservative error\n"
        "self-interaction supplies no net source.",
        "#EDF0EE", NAVY,
    )
    axis.text(0.50, 0.615, "reduction requires structure", transform=axis.transAxes,
              ha="center", va="center", fontsize=7.4, color=MUTED, style="italic")

    add_arrow(axis, (0.43, 0.645), (0.245, 0.55))
    add_arrow(axis, (0.57, 0.645), (0.755, 0.55))

    add_box(
        axis, (0.045, 0.275), 0.405, 0.275,
        "INVARIANT-RAY REDUCTION",
        "Logistic amplitude law\n"
        "one shape  •  one clock\n"
        "residual control  •  coefficient matching",
        LIGHT_BLUE, BLUE,
    )

    add_box(
        axis, (0.55, 0.275), 0.405, 0.275,
        "MOVING-FRONT REDUCTION",
        "Spectrum-dependent power law\n"
        r"$p=(\beta+1-n)/(1-n)$"
        "\nfront propagation replaces\n"
        "one-shape evolution",
        LIGHT_TEAL, TEAL,
    )

    add_arrow(axis, (0.2475, 0.27), (0.2475, 0.22))
    add_arrow(axis, (0.7525, 0.27), (0.7525, 0.22))
    add_box(
        axis, (0.11, 0.045), 0.78, 0.17,
        "FINITE-BAND WARNING",
        "Band edges, forcing, dissipation, leakage, and unresolved transfer\n"
        "enter the residual. A scalar reduction requires that residual\n"
        "to be measured and controlled.",
        LIGHT_GOLD, GOLD, title_color=RED,
    )
    axis.text(
        0.965, 0.005, "Conceptual map • conditional statements, not an empirical fit",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=6.7, color=MUTED,
    )

    fig.savefig(path_pdf, bbox_inches="tight", metadata=meta)
    png_meta = {"Title": meta["Title"], "Description": meta["Creator"]}
    fig.savefig(path_png, dpi=450, bbox_inches="tight", metadata=png_meta)
    plt.close(fig)


def extract_audit(
    mtb: dict[str, Any], clock: dict[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for regime, _, _, _ in REGIMES:
        mtb_rows = sorted(mtb["regimes"][regime]["truths"], key=lambda row: row["truth_index"])
        clock_rows = sorted(
            clock["regimes"][regime]["truths"], key=lambda row: row["truth_index"]
        )
        result[regime] = {
            "R": np.asarray([row["R"] for row in mtb_rows], dtype=float),
            "clock": np.asarray([row["Lambda_disp"] for row in mtb_rows], dtype=float),
            "correction": 100.0 * np.asarray(
                [row["truth_summary"]["net_correction_fraction"] for row in clock_rows],
                dtype=float,
            ),
        }
    return result


def make_audit(
    path_pdf: Path,
    path_png: Path,
    mtb: dict[str, Any],
    clock: dict[str, Any],
    meta: dict[str, str],
) -> None:
    values = extract_audit(mtb, clock)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.75))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.21, top=0.72, wspace=0.36)
    x = np.arange(4, dtype=float)
    offsets = (-0.10, 0.10)
    panels = (
        ("R", r"shape residual $R$", "(a) One-shape audit"),
        ("clock", r"clock dispersion $\Lambda_{\rm disp}$", "(b) One-clock audit"),
        ("correction", "net correction (% of baseline)", "(c) Local-clock decomposition"),
    )

    for axis, (key, ylabel, title) in zip(axes, panels):
        for offset, (regime, label, color, marker) in zip(offsets, REGIMES):
            y = values[regime][key]
            axis.plot(
                x + offset, y, linestyle="none", marker=marker, markersize=6.2,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7,
                color=color, label=label, zorder=3,
            )
            for truth, value in enumerate(y):
                axis.annotate(
                    f"{value:.3f}" if key != "correction" else f"{value:.1f}",
                    (truth + offset, value), xytext=(0, 6 if value >= 0 else -10),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.1, color=color,
                )
        axis.axhline(0, color="#9AA1A3", lw=0.8, zorder=1)
        axis.set_xticks(x, [f"T{i}" for i in range(4)])
        axis.set_xlabel("truth cluster")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold", color=NAVY)
        axis.grid(axis="y", color="#D7DCDE", linewidth=0.65, alpha=0.8, zorder=0)
        axis.margins(x=0.13, y=0.24)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, loc="upper left",
        bbox_to_anchor=(0.075, 0.855), ncol=2,
        handletextpad=0.5, columnspacing=1.2,
    )
    fig.suptitle(
        "Stored-artifact validity audit by truth cluster",
        x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=NAVY,
    )
    fig.text(
        0.99, 0.035,
        "Panels (a,b): prospective audit; panel (c): exploratory post-exit diagnostic",
        ha="right", va="bottom", fontsize=6.8, color=MUTED,
    )

    fig.savefig(path_pdf, bbox_inches="tight", metadata=meta)
    png_meta = {"Title": meta["Title"], "Description": meta["Creator"]}
    fig.savefig(path_png, dpi=450, bbox_inches="tight", metadata=png_meta)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="replace the four versioned outputs if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = output_paths()
    protect_outputs(targets, args.force)

    mtb = load_json(MTB_PATH)
    clock = load_json(CLOCK_PATH)
    validate_inputs(mtb, clock)
    input_hashes = {MTB_PATH.name: sha256(MTB_PATH), CLOCK_PATH.name: sha256(CLOCK_PATH)}
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    make_theory_map(
        targets[0], targets[1],
        metadata(input_hashes, "Validity limits of logistic error growth: theory map"),
    )
    make_audit(
        targets[2], targets[3], mtb, clock,
        metadata(input_hashes, "Validity limits of logistic error growth: stored-artifact audit"),
    )

    print(f"script_sha256  {sha256(SCRIPT)}  {SCRIPT}")
    for name, digest in input_hashes.items():
        print(f"input_sha256   {digest}  {INPUT_DIR / name}")
    for target in targets:
        print(f"output_sha256  {sha256(target)}  {target}")


if __name__ == "__main__":
    main()
