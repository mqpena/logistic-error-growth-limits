#!/usr/bin/env python3
"""Measure descriptive truth shell-energy slopes over the PRL k=10--20 band."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REGIMES = ("n256_re470_steep", "n512_re940_marginal")
BAND = (10, 20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shell_labels(size: int) -> np.ndarray:
    modes = np.fft.fftfreq(size) * size
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    return np.rint(np.sqrt(kx**2 + ky**2)).astype(int)


def snapshot_slope(field_hat: np.ndarray, labels: np.ndarray) -> float:
    shells = np.arange(BAND[0], BAND[1] + 1)
    energy = np.asarray([
        0.5 * np.sum(np.abs(field_hat[labels == shell]) ** 2)
        for shell in shells
    ])
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("truth shell energy must be positive and finite")
    return float(np.polyfit(np.log(shells), np.log(energy), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    result = {
        "schema": "prl-finite-band-truth-slopes-1",
        "evidence_status": "post_audit_descriptive_context",
        "band": list(BAND),
        "definition": "OLS slope of log shell energy versus log integer shell k for each stored truth-field snapshot; median over snapshots, then median over four truths",
        "non_claim": "These finite-band slopes do not establish an asymptotic similarity interval or freeze an error-growth prediction.",
        "regimes": {},
    }
    for regime in REGIMES:
        truths = []
        for path in sorted((args.cluster_root / regime / "clusters").glob("truth_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                fields = np.asarray(data["field_truth_hat"])
            labels = shell_labels(fields.shape[-1])
            slopes = np.asarray([snapshot_slope(field, labels) for field in fields])
            truths.append({
                "truth_index": int(path.stem.split("_")[-1]),
                "artifact": str(path.relative_to(args.cluster_root)),
                "artifact_sha256": sha256(path),
                "snapshot_count": int(len(slopes)),
                "snapshot_slope_median": float(np.median(slopes)),
                "snapshot_slope_95pct_interval": [
                    float(np.quantile(slopes, 0.025)),
                    float(np.quantile(slopes, 0.975)),
                ],
            })
        medians = np.asarray([row["snapshot_slope_median"] for row in truths])
        result["regimes"][regime] = {
            "truths": truths,
            "regime_truth_median_slope": float(np.median(medians)),
            "truth_median_range": [float(np.min(medians)), float(np.max(medians))],
            "positive_exponent_n": float(-np.median(medians)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["regimes"], indent=2))


if __name__ == "__main__":
    main()
