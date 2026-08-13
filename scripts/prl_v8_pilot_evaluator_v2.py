#!/usr/bin/env python3
"""Versioned CPU evaluator for the PRL v8 nonconfirmatory pilot.

The v1 Metal generator remains immutable. This evaluator applies the PI-authorized
physical-time stationarity window and leaves every threshold and other formula unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import prl_v8_two_regime as generator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVALUATOR = Path(__file__).resolve()
DEFAULT_CONFIG = HERE / "prl_v8_gate5_nonconfirmatory_config_20260715.json"
DEFAULT_OUTPUT = HERE / "out" / "prl_v8_gate5_nonconfirmatory_20260715"
DEFAULT_REPORT = ROOT / "01_PRL" / "Codex_work" / "reports" / "v8_gate5_nonconfirmatory_20260715_per_cluster"
DEFAULT_RECORD_ROOT = ROOT / "01_PRL" / "Codex_work" / "reports" / "v8_gate5_nonconfirmatory_jobs_20260715"
DEFAULT_DECISION = ROOT / "01_PRL" / "Codex_work" / "reports" / "v8_pi_decision_stationarity_evaluator_v2_20260715.json"
DECISION_SCHEMA = "prl-v8-pi-decision-stationarity-evaluator-1"
REPORT_SCHEMA = "prl-v8-pilot-evaluation-2"


def physical_late_mask(times: np.ndarray, horizon: float) -> np.ndarray:
    """Select the physical second half of a forecast window."""
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("stationarity times must be a finite one-dimensional array")
    if np.any(np.diff(values) <= 0):
        raise ValueError("stationarity times must be strictly increasing")
    if not np.isclose(values[0], 0.0, rtol=0, atol=1e-12):
        raise ValueError("stationarity times must begin at zero")
    if not np.isclose(values[-1], horizon, rtol=0, atol=1e-12):
        raise ValueError("stationarity times must end at the forecast horizon")
    mask = values >= 0.5 * float(horizon)
    if np.count_nonzero(mask) < 2:
        raise ValueError("physical late-half stationarity window has fewer than two samples")
    return mask


def fractional_linear_drift(times: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    selected_times = np.asarray(times, dtype=float)[mask]
    selected_values = np.asarray(values, dtype=float)[mask]
    mean = float(np.mean(selected_values))
    if not np.isfinite(mean) or abs(mean) <= np.finfo(float).tiny:
        raise ValueError("stationarity-window mean is not finite and nonzero")
    slope = float(np.polyfit(selected_times, selected_values, 1)[0])
    return slope * float(selected_times[-1] - selected_times[0]) / mean


def observed_rows_green(rows: list[dict[str, Any]]) -> bool:
    observed = [row for row in rows if row.get("complete")]
    return bool(observed) and all(all(row["gates"].values()) for row in observed)


def validate_decision(decision: dict[str, Any], config: dict[str, Any], config_path: Path) -> None:
    if decision.get("schema") != DECISION_SCHEMA or decision.get("status") != "authorized":
        raise PermissionError("invalid or unauthorized stationarity evaluator decision")
    if decision.get("production_authorized") is not False:
        raise PermissionError("stationarity evaluator decision must not authorize production")
    scope = decision.get("scope", {})
    if scope.get("stage") != config.get("stage"):
        raise PermissionError("PI decision stage does not match the active configuration")
    if scope.get("config_sha256") != generator.sha256(config_path):
        raise PermissionError("PI decision does not cite the active configuration hash")
    correction = decision.get("correction", {})
    threshold = float(config["pilot_gates"]["stationarity_max_fractional_drift"])
    if correction.get("authorized_evaluator_version") != 2:
        raise PermissionError("PI decision does not authorize evaluator version 2")
    if correction.get("new_window") != "scalar_time >= forecast_horizon_tu / 2":
        raise PermissionError("PI decision does not authorize the physical late-half window")
    if correction.get("threshold_is_unchanged") is not True:
        raise PermissionError("PI decision must preserve the frozen threshold")
    if not np.isclose(correction.get("threshold_max_absolute_fractional_drift"), threshold,
                      rtol=0, atol=0):
        raise PermissionError("PI decision stationarity threshold does not match the config")
    ratified = decision.get("ratified_generator", {})
    if ratified.get("sha256") != generator.sha256(generator.RUNNER):
        raise PermissionError("immutable generator hash differs from the PI decision")


def verify_manifest(manifest_path: Path, config_path: Path, output: Path,
                    decision: dict[str, Any]) -> dict[str, Any]:
    manifest = generator.load_json(manifest_path)
    if manifest["config"]["sha256"] != generator.sha256(config_path):
        raise ValueError("dry-run manifest configuration hash mismatch")
    if manifest["solver"]["sha256"] != generator.sha256(generator.SOLVER):
        raise ValueError("dry-run manifest solver hash mismatch")
    if manifest["runner"]["sha256"] != decision["ratified_generator"]["sha256"]:
        raise ValueError("dry-run manifest generator hash mismatch")
    if Path(manifest["output_root"]).resolve() != output.resolve():
        raise ValueError("dry-run manifest output root mismatch")
    return manifest


def evaluate_cluster(config: dict[str, Any], config_path: Path, output: Path,
                     record_root: Path, regime: dict[str, Any], truth_index: int,
                     decision: dict[str, Any]) -> dict[str, Any]:
    rid = regime["id"]
    path = output / rid / "clusters" / f"truth_{truth_index:02d}.npz"
    completion_path = (record_root / "jobs" / "pilot" / rid /
                       f"truth_{truth_index:02d}" / "completed.json")
    if not path.exists() and not completion_path.exists():
        return {"truth_index": truth_index, "complete": False}
    if not path.exists() or not completion_path.exists():
        raise ValueError(f"artifact/completion-record mismatch for {rid} truth {truth_index}")

    completion = generator.load_json(completion_path)
    artifact_hash = generator.sha256(path)
    cluster_output = completion.get("cluster_output", {})
    if (cluster_output.get("sha256") != artifact_hash
            or cluster_output.get("bytes") != path.stat().st_size
            or Path(cluster_output.get("path", "")).resolve() != path.resolve()):
        raise ValueError(f"completion-record provenance mismatch: {path}")
    if completion.get("job") != {"regime": rid, "truth_index": truth_index}:
        raise ValueError(f"completion-record job mismatch: {path}")

    active_config_hash = generator.sha256(config_path)
    active_solver_hash = generator.sha256(generator.SOLVER)
    ratified_runner_hash = decision["ratified_generator"]["sha256"]
    bands = generator.expanded_bands(config, int(regime["n"]))
    primary = bands.index(tuple(config["design"]["primary_band"]))
    centers = np.sqrt(np.asarray([lo * hi for lo, hi in bands], dtype=float))
    local = np.abs(np.log2(centers / centers[primary])) <= 1.0
    gate_cfg = config["pilot_gates"]
    horizon = float(config["design"]["forecast_horizon_tu"])

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("schema") != generator.SCHEMA:
            raise ValueError(f"invalid cluster schema: {path}")
        if (metadata.get("config_sha256") != active_config_hash
                or metadata.get("solver_sha256") != active_solver_hash
                or metadata.get("runner_sha256") != ratified_runner_hash):
            raise ValueError(f"cluster provenance mismatch: {path}")
        truth_path = output / rid / "truths" / f"truth_{truth_index:02d}.npz"
        if metadata.get("truth_state_sha256") != generator.sha256(truth_path):
            raise ValueError(f"cluster truth-state provenance mismatch: {path}")

        expected_scalar = generator.cadence_times(config["output"]["scalar_cadence"], horizon)
        expected_locality = generator.cadence_times(
            config["output"]["spectral_locality_cadence"], horizon)
        expected_field = np.asarray(config["output"]["field_audit_times_tu"], dtype=float)
        if (not np.array_equal(data["scalar_time"], expected_scalar)
                or not np.array_equal(data["locality_time"], expected_locality)
                or not np.array_equal(data["field_time"], expected_field)):
            raise ValueError(f"cluster time coordinates mismatch: {path}")

        et = data["Et"][:, :, primary]
        ef = data["Ef"][:, :, primary]
        cross = data["C"][:, :, primary]
        eps = data["epsilon"][:, :, primary]
        rho = data["rho"][:, :, primary]
        identity = eps - et - ef + 2.0 * cross
        identity_rel = float(np.linalg.norm(identity) /
                             max(np.linalg.norm(eps), np.finfo(float).tiny))
        components = data["budget_components"][:, :, :, primary]
        observed = data["budget_observed"][:, :, primary]
        budget_rel = float(np.linalg.norm(observed - components.sum(axis=1)) /
                           max(np.linalg.norm(observed), np.finfo(float).tiny))
        forcing = float(np.max(data["shared_forcing_relative_energy_error"]))
        burn = float(config["lyapunov"]["burnin_tu"])
        lambda_mask = data["lambda_time"] >= burn
        lam = float(np.mean(data["lambda_energy_rate"][lambda_mask, primary]))

        # Preserve the v1 predictor formulas; only stationarity-window selection changes.
        ceiling = 2.0 * np.mean(et[len(et) // 2:], axis=0)
        predictors = {
            "two_C_over_Et_plus_Ef": 2.0 * cross / np.maximum(et + ef, 1e-30),
            "rho": rho,
            "one_minus_epsilon_over_L": 1.0 - eps / np.maximum(ceiling[None], 1e-30),
        }
        response = data["P_linear"][:, :, primary] + data["P_tl_nonlinear"][:, :, primary]
        response = response / (lam * eps) if lam > 0 else np.full_like(response, np.nan)
        lo, hi = gate_cfg["required_primary_rho_interval"]
        fit_mask = np.isfinite(response) & (rho > lo) & (rho < hi)
        fits = {name: generator._fit(value.ravel(), response.ravel(), fit_mask.ravel())
                for name, value in predictors.items()}

        source = data["locality_tl_source"][:, :, :, :, primary]
        local_matrix = local[:, None] & local[None, :]
        local_abs = np.sum(np.abs(source[:, :, local_matrix]), axis=-1)
        total_abs = np.sum(np.abs(source), axis=(-2, -1))
        locality_fraction = float(np.mean(local_abs / np.maximum(total_abs, 1e-30)))
        scalar_index = np.searchsorted(data["scalar_time"], data["locality_time"])
        tl_expected = data["P_tl_nonlinear"][scalar_index]
        phi_expected = data["P_pure_error"][scalar_index]
        tl_sources = data["locality_tl_source"].sum(axis=(2, 3))
        phi_sources = data["locality_phi_source"].sum(axis=(2, 3))
        residual = np.concatenate([
            (tl_sources - tl_expected).ravel(),
            (phi_sources - phi_expected).ravel(),
        ])
        scale = np.concatenate([tl_expected.ravel(), phi_expected.ravel()])
        source_rel = float(np.linalg.norm(residual) /
                           max(np.linalg.norm(scale), np.finfo(float).tiny))

        times = data["scalar_time"]
        truth_energy = np.mean(data["Et"][:, :, -1], axis=1)
        late_mask = physical_late_mask(times, horizon)
        drift = fractional_linear_drift(times, truth_energy, late_mask)
        legacy_mask = np.arange(len(times)) >= len(times) // 2
        legacy_drift = fractional_linear_drift(times, truth_energy, legacy_mask)

        required_finite = [
            et, ef, cross, eps, rho, response, data["budget_components"],
            data["locality_tl_source"], data["locality_phi_source"],
            data["lambda_energy_rate"],
        ]
        finite_fraction = float(
            sum(np.isfinite(value).sum() for value in required_finite)
            / sum(value.size for value in required_finite)
        )
        rho_span = np.nanmax(rho, axis=0) - np.nanmin(rho, axis=0)
        interval_points = fit_mask.sum(axis=0).astype(int)
        brackets = ((np.nanmin(rho, axis=0) <= lo) & (np.nanmax(rho, axis=0) >= hi))
        gates = {
            "finite": finite_fraction >= gate_cfg["minimum_finite_fraction"],
            "covariance_identity": identity_rel <= gate_cfg["maximum_covariance_identity_relative_error"],
            "discrete_budget": budget_rel <= gate_cfg["maximum_discrete_budget_relative_error"],
            "shared_forcing": forcing <= gate_cfg["maximum_shared_forcing_error_relative"],
            "locality_source_reconstruction": source_rel <= gate_cfg["maximum_locality_source_relative_error"],
            "positive_lyapunov": lam > gate_cfg["minimum_lambda_energy"],
            "stationarity": abs(drift) <= gate_cfg["stationarity_max_fractional_drift"],
            "rho_span": bool(np.all(rho_span >= gate_cfg["minimum_primary_rho_span"])),
            "rho_brackets_interval": bool(np.all(brackets)),
            "rho_interval_points": bool(np.all(
                interval_points >= gate_cfg["minimum_points_in_primary_interval"])),
        }
        return {
            "truth_index": truth_index,
            "complete": True,
            "sha256": artifact_hash,
            "bytes": path.stat().st_size,
            "lambda_energy": lam,
            "covariance_identity_relative_error": identity_rel,
            "discrete_budget_relative_error": budget_rel,
            "shared_forcing_relative_error": forcing,
            "locality_source_relative_error": source_rel,
            "stationary_truth_energy_drift": drift,
            "legacy_index_half_drift_diagnostic": legacy_drift,
            "stationarity_window": {
                "definition": "scalar_time >= forecast_horizon_tu / 2",
                "start_tu": float(times[late_mask][0]),
                "end_tu": float(times[late_mask][-1]),
                "samples": int(np.count_nonzero(late_mask)),
            },
            "finite_fraction": finite_fraction,
            "rho_span_by_member": rho_span.tolist(),
            "points_in_rho_interval_by_member": interval_points.tolist(),
            "rho_brackets_interval_by_member": brackets.tolist(),
            "octave_local_nonlinear_tl_fraction": locality_fraction,
            "fits": fits,
            "gates": gates,
        }


def evaluate(config_path: Path, output: Path, report: Path, record_root: Path,
             decision_path: Path) -> Path:
    config = generator.load_json(config_path)
    generator.validate_config(config)
    if config.get("stage") != "pilot_nonconfirmatory":
        raise PermissionError("evaluator v2 is limited to the nonconfirmatory pilot")
    decision = generator.load_json(decision_path)
    validate_decision(decision, config, config_path)
    verify_manifest(report / "prl_v8_pilot_manifest.json", config_path, output, decision)

    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "stage": config["stage"],
        "production_authorized": False,
        "evaluator": {"path": str(EVALUATOR), "sha256": generator.sha256(EVALUATOR)},
        "generator": {"path": str(generator.RUNNER), "sha256": generator.sha256(generator.RUNNER)},
        "config": {"path": str(config_path), "sha256": generator.sha256(config_path)},
        "pi_decision": {"path": str(decision_path), "sha256": generator.sha256(decision_path)},
        "stationarity_correction": decision["correction"],
        "regimes": {},
    }
    observed_green = True
    complete_count = 0
    total = len(config["design"]["regimes"]) * int(config["design"]["truths_per_regime"])
    for regime in config["design"]["regimes"]:
        rows = []
        for truth_index in range(config["design"]["truths_per_regime"]):
            row = evaluate_cluster(
                config, config_path, output, record_root, regime, truth_index, decision)
            rows.append(row)
            if row.get("complete"):
                complete_count += 1
                observed_green &= all(row["gates"].values())
        result["regimes"][regime["id"]] = {
            "truths": rows,
            "complete_truths": sum(bool(row.get("complete")) for row in rows),
            "observed_truth_gates_pass": observed_rows_green(rows),
        }
    result["complete_artifacts"] = complete_count
    result["expected_artifacts"] = total
    result["pilot_complete"] = complete_count == total
    result["observed_artifacts_green"] = bool(observed_green and complete_count > 0)
    result["pilot_green"] = bool(result["pilot_complete"] and result["observed_artifacts_green"])
    if not result["observed_artifacts_green"]:
        result["decision"] = "STOP: an observed pilot artifact fails at least one gate"
    elif result["pilot_complete"]:
        result["decision"] = "ready for independent Gate 5 evidence review"
    else:
        result["decision"] = "continue the authorized nonconfirmatory pilot"

    evaluator_hash = result["evaluator"]["sha256"]
    decision_hash = result["pi_decision"]["sha256"]
    target = report / (
        f"prl_v8_pilot_evaluation_v2_{complete_count}of{total}_"
        f"{evaluator_hash[:12]}_{decision_hash[:12]}.json"
    )
    generator.atomic_json(target, result)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--pi-decision", type=Path, default=DEFAULT_DECISION)
    args = parser.parse_args()
    print(evaluate(
        args.config.resolve(), args.output.resolve(), args.report_dir.resolve(),
        args.record_root.resolve(), args.pi_decision.resolve()))


if __name__ == "__main__":
    main()
