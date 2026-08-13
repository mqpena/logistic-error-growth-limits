#!/usr/bin/env python3
"""Prepare, run, and analyze the PRL v8 two-regime JFM-SQG pilot.

Dry-run and analysis use NumPy only. GPU commands import MLX lazily and refuse
to proceed when a real Metal allocation cannot be evaluated. Outputs are
immutable; reruns require a new output root rather than overwriting evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "prl_v8_two_regime_config.json"
DEFAULT_OUTPUT = HERE / "out" / "prl_v8_two_regime_pilot_20260715"
DEFAULT_REPORT = ROOT / "01_PRL" / "Codex_work" / "reports" / "v8_two_regime_pilot_frozen"
PYTHON = Path(sys.executable)
SOLVER = HERE / "jas_sqg_jfm.py"
RUNNER = Path(__file__).resolve()
PROTOCOL = ROOT / "01_PRL" / "Codex_work" / "reports" / "PRL_V8_TWO_REGIME_MECHANISM_PROTOCOL_v2_20260715.md"
GATE4_CLUSTER_LAUNCHER = HERE / "prl_v8_gate4_cluster_launcher.py"
SCHEMA = "prl-v8-jfm-two-regime-pilot-1"
PRODUCTION_AUTHORIZATION_SCHEMA = "prl-v8-confirmatory-production-authorization-1"
STAGE_DESIGNS = {
    "pilot_nonconfirmatory": (4, 2),
    "confirmatory_production": (16, 4),
}
LAUNCHER_MODES = {"sequential_legacy", "per_cluster"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    except FileExistsError as exc:
        raise FileExistsError(f"evidence path is already claimed: {path}") from exc
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    except FileExistsError as exc:
        raise FileExistsError(f"evidence path is already claimed: {path}") from exc
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.npz")
    try:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
        np.savez_compressed(temporary, **arrays)
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def cadence_times(cadence: Iterable[dict[str, float]], horizon: float) -> np.ndarray:
    """Build a deduplicated piecewise cadence including t=0 and the horizon."""
    values = [0.0]
    start = 0.0
    for segment in cadence:
        stop = min(float(segment["through_tu"]), horizon)
        step = float(segment["dt_tu"])
        if stop < start or step <= 0:
            raise ValueError("cadence segments must be ordered with positive dt")
        count = int(math.floor((stop - start) / step + 1e-10))
        values.extend(start + step * np.arange(1, count + 1))
        if not math.isclose(values[-1], stop, abs_tol=1e-10):
            values.append(stop)
        start = stop
        if start >= horizon:
            break
    if start < horizon:
        raise ValueError("cadence does not cover the forecast horizon")
    return np.unique(np.round(np.asarray(values, dtype=float), 12))


def step_lookup(times: np.ndarray, dt: float) -> dict[int, int]:
    steps = np.rint(times / dt).astype(int)
    if not np.allclose(steps * dt, times, rtol=0.0, atol=1e-10):
        raise ValueError("all output times must be integer multiples of dt")
    return {int(step): index for index, step in enumerate(steps)}


def expanded_bands(config: dict[str, Any], n: int) -> list[tuple[int, int]]:
    result = []
    # The square 2/3 mask retains diagonal corners through sqrt(2) N/3.  The
    # final label therefore denotes the full radial extent of analysis support.
    radial_label_max = int(math.ceil(math.sqrt(2.0) * (n // 3)))
    for lo, hi in config["design"]["analysis_bands"]:
        result.append((int(lo), radial_label_max if int(hi) < 0 else int(hi)))
    occupied: set[int] = set()
    for lo, hi in result:
        if lo < 1 or hi < lo or hi > radial_label_max:
            raise ValueError(f"invalid band {(lo, hi)} for radial label maximum={radial_label_max}")
        modes = set(range(lo, hi + 1))
        if occupied & modes:
            raise ValueError("analysis bands overlap")
        occupied |= modes
    if occupied != set(range(1, radial_label_max + 1)):
        raise ValueError("analysis bands must partition every retained radial shell")
    return result


def half_open_radial_masks(kmag: np.ndarray, square_mask: np.ndarray,
                           bands: list[tuple[int, int]]) -> np.ndarray:
    """Partition the complete square-dealiased support into radial bands.

    Configuration labels use conventional inclusive integer shell names, e.g.
    ``k1_5``.  Analysis masks instead use the contiguous half-open intervals
    ``[0, 5.5), [5.5, 9.5), ...``.  This prevents gaps for lattice radii such
    as ``sqrt(26)`` and assigns the square-mask corner modes to the final
    band.  The separate plotted shell spectra retain their historical
    center-shell convention in ``_shell_spectra``.
    """
    radius = np.asarray(kmag, dtype=float)
    retained = np.asarray(square_mask, dtype=bool)
    if radius.shape != retained.shape:
        raise ValueError("kmag and square_mask must have identical shapes")
    if not bands:
        raise ValueError("at least one analysis band is required")
    masks: list[np.ndarray] = []
    previous_upper = 0.0
    retained_max = float(np.max(radius[retained])) if np.any(retained) else 0.0
    for index, (lo, hi) in enumerate(bands):
        lower = 0.0 if index == 0 else float(lo) - 0.5
        upper = (np.nextafter(retained_max, np.inf) if index == len(bands) - 1
                 else float(hi) + 0.5)
        if not math.isclose(lower, previous_upper, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("analysis-band labels do not define contiguous half-open edges")
        if upper <= lower:
            raise ValueError("analysis-band edge is empty or reversed")
        masks.append(((radius >= lower) & (radius < upper) & retained).astype(np.float32))
        previous_upper = upper
    partition = np.asarray(masks, dtype=np.float32)
    if not np.array_equal(np.sum(partition, axis=0), retained.astype(np.float32)):
        raise ValueError("half-open analysis masks do not exactly partition the square-dealiased support")
    return partition


def regime(config: dict[str, Any], regime_id: str) -> dict[str, Any]:
    matches = [item for item in config["design"]["regimes"] if item["id"] == regime_id]
    if len(matches) != 1:
        raise ValueError(f"unknown regime: {regime_id}")
    return matches[0]


def validate_config(config: dict[str, Any]) -> None:
    stage = config.get("stage")
    if stage not in STAGE_DESIGNS:
        raise ValueError(f"unsupported stage: {stage!r}")
    design = config["design"]
    launcher_mode = config.get("execution", {}).get("launcher_mode", "sequential_legacy")
    if launcher_mode not in LAUNCHER_MODES:
        raise ValueError(f"unsupported execution.launcher_mode: {launcher_mode!r}")
    expected_truths, expected_members = STAGE_DESIGNS[stage]
    if (design["truths_per_regime"], design["perturbations_per_truth"]) != (expected_truths, expected_members):
        raise ValueError(
            f"{stage} design must be exactly {expected_truths} truths by {expected_members} perturbations"
        )
    force_hi = config["model"]["forcing"]["k_f"] + config["model"]["forcing"]["dk_f"]
    if force_hi >= design["primary_band"][0]:
        raise ValueError("forcing and primary test band must be disjoint")
    if not config["model"]["forcing"]["shared_within_truth_forecast_cluster"]:
        raise ValueError("shared truth-forecast forcing is mandatory")
    horizon = float(design["forecast_horizon_tu"])
    cadence_times(config["output"]["scalar_cadence"], horizon)
    cadence_times(config["output"]["spectral_locality_cadence"], horizon)
    for item in design["regimes"]:
        expanded_bands(config, int(item["n"]))
        if item["n"] not in (256, 512):
            raise ValueError("only the two stationary audited regimes are allowed")


def require_production_authorization(config: dict[str, Any], config_path: Path,
                                     authorization_path: Path | None) -> dict[str, Any] | None:
    """Fail closed unless a PI record binds this exact production run to protocol v2."""
    if config["stage"] != "confirmatory_production":
        return None
    if authorization_path is None:
        raise PermissionError("REFUSED: confirmatory production requires --authorization")
    if not authorization_path.is_file():
        raise PermissionError(f"REFUSED: missing PI authorization record: {authorization_path}")
    authorization = load_json(authorization_path)
    if authorization.get("schema") != PRODUCTION_AUTHORIZATION_SCHEMA:
        raise PermissionError("REFUSED: unrecognized production-authorization schema")
    if authorization.get("authorized") is not True:
        raise PermissionError("REFUSED: PI authorization does not declare authorized=true")
    if authorization.get("stage") != "confirmatory_production":
        raise PermissionError("REFUSED: authorization is not scoped to confirmatory production")
    try:
        dt.date.fromisoformat(str(authorization["issued_on"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PermissionError("REFUSED: authorization requires an ISO-8601 issued_on date") from exc
    protocol = authorization.get("protocol_v2", {})
    if protocol.get("path") != str(PROTOCOL.resolve()) or protocol.get("sha256") != sha256(PROTOCOL):
        raise PermissionError("REFUSED: authorization is not bound to the governing protocol v2")
    if authorization.get("config_sha256") != sha256(config_path):
        raise PermissionError("REFUSED: authorization is not bound to this exact configuration")
    evidence = authorization.get("evidence_records")
    if not isinstance(evidence, list) or len(evidence) != 5:
        raise PermissionError("REFUSED: authorization must cite exactly five evidence records")
    gate_ids: set[int] = set()
    for record in evidence:
        try:
            gate = int(record["gate"])
            path = Path(record["path"])
            digest = str(record["sha256"])
            dt.date.fromisoformat(str(record["dated_on"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PermissionError("REFUSED: each evidence record requires gate, path, sha256, and ISO dated_on") from exc
        if not path.is_file() or sha256(path) != digest:
            raise PermissionError(f"REFUSED: missing or hash-mismatched evidence for Gate {gate}")
        gate_ids.add(gate)
    if gate_ids != {1, 2, 3, 4, 5}:
        raise PermissionError("REFUSED: authorization must cite one record for each Gate 1 through Gate 5")
    return authorization


def covariance_observables(
    truth_hat: np.ndarray, forecast_hat: np.ndarray, masks: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy reference implementation used by tests and analysis audits."""
    truth = np.asarray(truth_hat)
    forecast = np.asarray(forecast_hat)
    if truth.ndim == 2:
        truth = truth[None, ...]
    if forecast.ndim == 2:
        forecast = forecast[None, ...]
    if truth.shape[0] == 1 and forecast.shape[0] > 1:
        truth = np.broadcast_to(truth, forecast.shape)
    weight = masks[None, ...]
    norm = float(n**4)
    et = 0.5 * np.sum(np.abs(truth[:, None]) ** 2 * weight, axis=(-2, -1)) / norm
    ef = 0.5 * np.sum(np.abs(forecast[:, None]) ** 2 * weight, axis=(-2, -1)) / norm
    cross = 0.5 * np.sum(
        np.real(np.conj(truth[:, None]) * forecast[:, None]) * weight,
        axis=(-2, -1),
    ) / norm
    epsilon = 0.5 * np.sum(
        np.abs((forecast - truth)[:, None]) ** 2 * weight, axis=(-2, -1)
    ) / norm
    rho = cross / np.sqrt(np.maximum(et * ef, np.finfo(float).tiny))
    return et, ef, cross, epsilon, rho


def allocated_energy_increments(
    old: np.ndarray, increments: np.ndarray, masks: np.ndarray, n: int
) -> np.ndarray:
    """Symmetrically allocate an exact quadratic increment among components.

    ``old`` has shape ``(member,n,n)`` and ``increments`` has shape
    ``(component,member,n,n)``. The result is ``(component,member,band)``.
    """
    total = np.sum(increments, axis=0)
    weight = masks[None, ...]
    norm = float(n**4)
    values = []
    for inc in increments:
        first = np.sum(
            np.real(np.conj(old[:, None]) * inc[:, None]) * weight,
            axis=(-2, -1),
        ) / norm
        second = 0.5 * np.sum(
            np.real(np.conj(inc[:, None]) * total[:, None]) * weight,
            axis=(-2, -1),
        ) / norm
        values.append(first + second)
    return np.asarray(values)


def _evidence_record(config: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    path = HERE / item["stationary_evidence"]
    if not path.exists():
        raise FileNotFoundError(f"missing stationary evidence: {path}")
    with np.load(path, allow_pickle=False) as data:
        source = json.loads(str(data["config_json"]))
        times = np.asarray(data["times"], dtype=float)
        spe = np.asarray(data["spe"], dtype=float)
    expected = {
        "n": item["n"], "dt": item["dt"], "nu": item["nu"],
        "mu": config["model"]["mu"],
        "k_f": config["model"]["forcing"]["k_f"],
        "dk_f": config["model"]["forcing"]["dk_f"],
        "eps_I": config["model"]["forcing"]["eps_I"],
    }
    if any(not math.isclose(float(source[key]), float(value), rel_tol=0, abs_tol=1e-12)
           for key, value in expected.items()):
        raise ValueError(f"stationary evidence config mismatch for {item['id']}")
    late = slice(len(times) // 2, None)
    drift = np.polyfit(times[late], spe[late], 1)[0]
    drift *= times[late][-1] - times[late][0]
    drift /= np.mean(spe[late])
    passed = abs(float(drift)) <= config["pilot_gates"]["stationarity_max_fractional_drift"]
    if not passed:
        raise ValueError(f"archived stationarity evidence fails the frozen gate for {item['id']}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "source_config": source,
        "n_snapshots": int(len(times)),
        "measured_late_fractional_spe_drift": float(drift),
        "stationarity_gate_passed": passed,
    }


def storage_estimate(config: dict[str, Any]) -> dict[str, Any]:
    horizon = float(config["design"]["forecast_horizon_tu"])
    scalar_n = len(cadence_times(config["output"]["scalar_cadence"], horizon))
    locality_n = len(cadence_times(config["output"]["spectral_locality_cadence"], horizon))
    field_n = len(config["output"]["field_audit_times_tu"])
    records = []
    total = 0
    for item in config["design"]["regimes"]:
        n = int(item["n"])
        truths = config["design"]["truths_per_regime"]
        members = config["design"]["perturbations_per_truth"]
        # One truth plus M forecasts at each sparse field-audit time.
        field_bytes = truths * field_n * (1 + members) * n * n * 8
        # Covariances, budgets, spectra, and 5x5 source matrices, float64 upper bound.
        kmax = n // 3
        diagnostic_values = truths * (
            scalar_n * members * (6 * 16 + 6 * 6)
            + locality_n * members * (kmax * 3 + 6 * 5 * 5 * 2)
        )
        diagnostic_bytes = diagnostic_values * 8
        regime_bytes = field_bytes + diagnostic_bytes
        total += regime_bytes
        records.append({
            "regime": item["id"],
            "scalar_times": scalar_n,
            "locality_times": locality_n,
            "field_audit_times": field_n,
            "uncompressed_upper_gib": regime_bytes / 2**30,
        })
    return {
        "regimes": records,
        "total_uncompressed_upper_gib": total / 2**30,
        "recommended_free_gib": max(5.0, 3.0 * total / 2**30),
        "note": "Field frames dominate; compressed NPZ should be smaller, but the gate uses the uncompressed upper bound.",
    }


def runtime_estimate(config: dict[str, Any]) -> dict[str, Any]:
    """Conservative planning range anchored to the audited N=512 nature run."""
    n512_nature_minutes = 91.0 * (
        config["design"]["truth_spinup_tu"]
        + (config["design"]["truths_per_regime"] - 1) * config["design"]["truth_separation_tu"]
    ) / 300.0
    # The cluster step advances three nonlinear states, a tangent state, and an
    # extra pure-error RHS. Batching helps, so retain a deliberately wide range.
    n512_cluster_base = 91.0 * config["design"]["forecast_horizon_tu"] / 300.0
    n512_cluster_range = [2.0 * n512_cluster_base, 4.0 * n512_cluster_base]
    total_range = [
        n512_nature_minutes + 4 * n512_cluster_range[0],
        n512_nature_minutes + 4 * n512_cluster_range[1],
    ]
    # N=256 is expected to add roughly 10--20% of the N=512 cost.
    total_range = [1.1 * total_range[0], 1.2 * total_range[1]]
    return {
        "audited_anchor": "N=512 300-TU nature run: approximately 91 minutes",
        "estimated_total_hours_range": [value / 60.0 for value in total_range],
        "warning": "Instrumented batching has not been benchmarked; the first N=256 cluster is the runtime calibration gate.",
    }


def launch_script(config_path: Path, output: Path, report: Path) -> str:
    config = load_json(config_path)
    launcher_mode = config.get("execution", {}).get("launcher_mode", "sequential_legacy")
    if config["stage"] == "confirmatory_production" or launcher_mode == "per_cluster":
        # Gate 4 owns confirmatory scheduling.  This artifact is deliberately
        # non-launching even when a valid PI authorization exists.
        return "\n".join([
            "#!/bin/zsh",
            "# REFUSED: this configuration requires one cluster per Gate 4 job.",
            f"print -u2 -- 'Use {GATE4_CLUSTER_LAUNCHER} for an allowlisted (regime, truth_index) cluster.'",
            "exit 64",
            "",
        ])
    commands = [
        "#!/bin/zsh",
        "set -euo pipefail",
        f"cd {HERE}",
        "export MPLCONFIGDIR=.mplconfig",
        f"PY={PYTHON}",
        f"CFG={config_path}",
        f"OUT={output}",
        f"REPORT={report}",
    ]
    for item in config["design"]["regimes"]:
        rid = item["id"]
        commands.append(f'"$PY" prl_v8_two_regime.py prepare-truths --config "$CFG" --output "$OUT" --regime {rid} --resume')
        for truth_index in range(int(config["design"]["truths_per_regime"])):
            commands.append(f'"$PY" prl_v8_two_regime.py run-truth --config "$CFG" --output "$OUT" --regime {rid} --truth-index {truth_index} --resume')
    commands.append('"$PY" prl_v8_two_regime.py analyze --config "$CFG" --output "$OUT" --report-dir "$REPORT"')
    return "\n".join(commands) + "\n"


def dry_run(config_path: Path, output: Path, report: Path) -> Path:
    config = load_json(config_path)
    validate_config(config)
    evidence = [_evidence_record(config, item) for item in config["design"]["regimes"]]
    scalar_times = cadence_times(config["output"]["scalar_cadence"], config["design"]["forecast_horizon_tu"])
    locality_times = cadence_times(config["output"]["spectral_locality_cadence"], config["design"]["forecast_horizon_tu"])
    launch = report / "launch_prl_v8_pilot.zsh"
    if launch.exists():
        raise FileExistsError(f"refusing to overwrite launch script: {launch}")
    launch.parent.mkdir(parents=True, exist_ok=True)
    launch.write_text(launch_script(config_path.resolve(), output.resolve(), report.resolve()), encoding="ascii")
    launch.chmod(0o755)
    launcher_mode = config.get("execution", {}).get("launcher_mode", "sequential_legacy")
    sleep_proof_command = None if launcher_mode == "per_cluster" else f"nohup caffeinate -i -m -s {launch} > {report / 'pilot.log'} 2>&1 &"
    manifest = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "stage": config["stage"],
        "execution": {"launcher_mode": launcher_mode,
                      "generic_sequential_launcher_disabled": launcher_mode == "per_cluster"},
        "production_authorized": False,
        "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
        "solver": {"path": str(SOLVER), "sha256": sha256(SOLVER)},
        "runner": {"path": str(RUNNER), "sha256": sha256(RUNNER)},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "stationary_evidence": evidence,
        "output_root": str(output.resolve()),
        "scalar_times": scalar_times.tolist(),
        "locality_times": locality_times.tolist(),
        "storage": storage_estimate(config),
        "runtime": runtime_estimate(config),
        "launch_script": {"path": str(launch), "sha256": sha256(launch)},
        "sleep_proof_command": sleep_proof_command,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "executable": sys.executable},
        "safety": {"overwrite_refused": True, "existing_nature_diagnostics_not_used_as_state": True},
        "clusters": [
            {
                "regime": item["id"], "truth_index": truth_index,
                "members": config["design"]["perturbations_per_truth"],
                "status": "pending",
                "output": str((output / item["id"] / "clusters" / f"truth_{truth_index:02d}.npz").resolve()),
            }
            for item in config["design"]["regimes"]
            for truth_index in range(config["design"]["truths_per_regime"])
        ],
    }
    target = report / "prl_v8_pilot_manifest.json"
    atomic_json(target, manifest)
    return target


def _require_metal() -> tuple[Any, Any, Any]:
    import mlx.core as mx
    from jas_sqg_jfm import SQGJFMConfig, SQGJFMSolver

    try:
        if not mx.metal.is_available():
            raise RuntimeError("MLX reports Metal unavailable")
        mx.set_default_device(mx.Device(mx.gpu, 0))
        probe = mx.sum(mx.ones((8, 8), dtype=mx.float32))
        mx.eval(probe)
        value = float(np.asarray(probe))
    except Exception as exc:
        raise RuntimeError(
            "REFUSED: no usable Metal device; run from the established Metal-visible session"
        ) from exc
    if not math.isclose(value, 64.0):
        raise RuntimeError("REFUSED: Metal allocation probe returned an invalid value")
    return mx, SQGJFMConfig, SQGJFMSolver


def _masks_np(solver: Any, bands: list[tuple[int, int]]) -> tuple[list[str], np.ndarray]:
    names = [f"k{lo}_{hi}" for lo, hi in bands]
    masks = list(half_open_radial_masks(solver.kmag_np, solver.mask_np, bands))
    names.append("full")
    masks.append(solver.mask_np)
    return names, np.asarray(masks, dtype=np.float32)


def prepare_truths(config: dict[str, Any], config_path: Path, output: Path, regime_id: str,
                   resume: bool = False, authorization_path: Path | None = None) -> None:
    require_production_authorization(config, config_path, authorization_path)
    mx, SQGJFMConfig, SQGJFMSolver = _require_metal()
    item = regime(config, regime_id)
    count = int(config["design"]["truths_per_regime"])
    truth_dir = output / regime_id / "truths"
    targets = [truth_dir / f"truth_{index:02d}.npz" for index in range(count)]
    if any(path.exists() for path in targets) and not resume:
        raise FileExistsError("refusing partial or repeated truth preparation; use a new output root")
    regime_index = [x["id"] for x in config["design"]["regimes"]].index(regime_id)
    seed = int(config["seeds"]["nature_seed_offset"]) + regime_index
    forcing = config["model"]["forcing"]
    cfg = SQGJFMConfig(
        n=int(item["n"]), dt=float(item["dt"]), nu=float(item["nu"]),
        mu=float(config["model"]["mu"]), k_f=float(forcing["k_f"]),
        dk_f=float(forcing["dk_f"]), eps_I=float(forcing["eps_I"]), seed=seed,
    )
    solver = SQGJFMSolver(cfg)
    rng = np.random.default_rng(seed + 1)
    q = rng.standard_normal((cfg.n, cfg.n)).astype(np.float32)
    state = solver.fft(mx.array(q)) * mx.array((solver.kmag_np <= 6).astype(np.float32)) * np.float32(1e-3)
    mx.eval(state)
    spin_steps = round(float(config["design"]["truth_spinup_tu"]) / cfg.dt)
    separation_steps = round(float(config["design"]["truth_separation_tu"]) / cfg.dt)
    target_steps = {spin_steps + index * separation_steps: index for index in range(count)}
    for step in range(max(target_steps) + 1):
        if step in target_steps:
            index = target_steps[step]
            mx.eval(state)
            metadata = {
                "schema": SCHEMA,
                "regime": regime_id,
                "truth_index": index,
                "nature_step": step,
                "nature_time_tu": step * cfg.dt,
                "rng_state": rng.bit_generator.state,
                "config_sha256": sha256(config_path),
                "solver_sha256": sha256(SOLVER),
                "runner_sha256": sha256(RUNNER),
            }
            state_np = np.asarray(state).astype(np.complex64)
            if targets[index].exists():
                with np.load(targets[index], allow_pickle=False) as existing:
                    previous = json.loads(str(existing["metadata_json"]))
                    if (previous.get("config_sha256") != metadata["config_sha256"]
                            or previous.get("solver_sha256") != metadata["solver_sha256"]
                            or previous.get("runner_sha256") != metadata["runner_sha256"]):
                        raise ValueError(f"existing truth cache has incompatible provenance: {targets[index]}")
                    if not np.array_equal(existing["theta_hat"], state_np):
                        raise ValueError(f"deterministic truth-cache replay mismatch: {targets[index]}")
            else:
                atomic_npz(targets[index], theta_hat=state_np,
                           metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
        if step < max(target_steps):
            state = solver.step(state, rng)
            if (step + 1) % 25 == 0:
                mx.eval(state)


def _bilinear_rhs(solver: Any, a_hat: Any, b_hat: Any) -> Any:
    psi_hat = solver.inv * a_hat
    psi_x = solver.ifft((1j * solver.kx) * psi_hat)
    psi_y = solver.ifft((1j * solver.ky) * psi_hat)
    b_x = solver.ifft((1j * solver.kx) * b_hat)
    b_y = solver.ifft((1j * solver.ky) * b_hat)
    return -solver.fft(psi_x * b_y - psi_y * b_x)


def _tangent_rhs(solver: Any, truth_hat: Any, tangent_hat: Any) -> Any:
    return _bilinear_rhs(solver, truth_hat, tangent_hat) + _bilinear_rhs(solver, tangent_hat, truth_hat)


def _pair_step(solver: Any, truth: Any, forecasts: Any, kick: Any) -> tuple[Any, Any, tuple[Any, ...], tuple[Any, ...], Any]:
    """Advance a shared-forcing cluster and return exact error increments."""
    dt, E1, E2 = np.float32(solver.cfg.dt), solver.E1, solver.E2

    def components(t: Any, f: Any) -> tuple[Any, Any, Any, Any]:
        nt = solver.jacobian_rhs(t)
        nf = solver.jacobian_rhs(f)
        phi = solver.jacobian_rhs(f - t)
        return nt, nf, nf - nt - phi, phi

    kt1, kf1, tl1, ph1 = components(truth, forecasts)
    truth2 = E2 * (truth + 0.5 * dt * kt1)
    fcst2 = E2 * (forecasts + 0.5 * dt * kf1)
    kt2, kf2, tl2, ph2 = components(truth2, fcst2)
    truth3 = E2 * truth + 0.5 * dt * kt2
    fcst3 = E2 * forecasts + 0.5 * dt * kf2
    kt3, kf3, tl3, ph3 = components(truth3, fcst3)
    truth4 = E1 * truth + dt * (E2 * kt3)
    fcst4 = E1 * forecasts + dt * (E2 * kf3)
    kt4, kf4, tl4, ph4 = components(truth4, fcst4)
    truth_det = E1 * truth + dt / 6.0 * (E1 * kt1 + 2.0 * E2 * (kt2 + kt3) + kt4)
    fcst_det = E1 * forecasts + dt / 6.0 * (E1 * kf1 + 2.0 * E2 * (kf2 + kf3) + kf4)
    old_delta = forecasts - truth
    linear = (E1 - 1.0) * old_delta
    tl = dt / 6.0 * (E1 * tl1 + 2.0 * E2 * (tl2 + tl3) + tl4)
    phi = dt / 6.0 * (E1 * ph1 + 2.0 * E2 * (ph2 + ph3) + ph4)
    root_dt = np.float32(np.sqrt(solver.cfg.dt))
    truth_next = (truth_det + root_dt * kick) * solver.mask
    forecast_next = (fcst_det + root_dt * kick) * solver.mask
    shared_forcing_residual = (forecast_next - truth_next) - (fcst_det - truth_det)
    numerical = (fcst_det - truth_det) - (old_delta + linear + tl + phi)
    stages = (truth, truth2, truth3, truth4)
    return truth_next, forecast_next, (linear, tl, phi, numerical), stages, shared_forcing_residual


def _tangent_step(solver: Any, tangent: Any, truth_stages: tuple[Any, ...]) -> Any:
    dt, E1, E2 = np.float32(solver.cfg.dt), solver.E1, solver.E2
    t1, t2, t3, t4 = truth_stages
    l1 = _tangent_rhs(solver, t1, tangent)
    e2 = E2 * (tangent + 0.5 * dt * l1)
    l2 = _tangent_rhs(solver, t2, e2)
    e3 = E2 * tangent + 0.5 * dt * l2
    l3 = _tangent_rhs(solver, t3, e3)
    e4 = E1 * tangent + dt * (E2 * l3)
    l4 = _tangent_rhs(solver, t4, e4)
    return (E1 * tangent + dt / 6.0 * (E1 * l1 + 2.0 * E2 * (l2 + l3) + l4)) * solver.mask


def _inner_mx(mx: Any, a: Any, b: Any, masks: Any, n: int) -> Any:
    return mx.sum(
        mx.real(mx.conj(a[:, None]) * b[:, None]) * masks[None], axis=(-2, -1)
    ) / np.float32(n**4)


def _energy_mx(mx: Any, a: Any, masks: Any, n: int) -> Any:
    return 0.5 * _inner_mx(mx, a, a, masks, n)


def _allocated_mx(mx: Any, old: Any, increments: tuple[Any, ...], masks: Any, n: int) -> Any:
    total = sum(increments)
    return mx.stack([
        _inner_mx(mx, old, inc, masks, n) + 0.5 * _inner_mx(mx, inc, total, masks, n)
        for inc in increments
    ])


def _continuous_diagnostics(mx: Any, solver: Any, truth: Any, forecasts: Any, masks: Any) -> dict[str, np.ndarray]:
    members = int(forecasts.shape[0])
    truth_batch = mx.broadcast_to(truth[None], forecasts.shape)
    delta = forecasts - truth
    et = mx.broadcast_to(_energy_mx(mx, truth[None], masks, solver.cfg.n), (members, masks.shape[0]))
    ef = _energy_mx(mx, forecasts, masks, solver.cfg.n)
    cross = 0.5 * _inner_mx(mx, truth_batch, forecasts, masks, solver.cfg.n)
    epsilon = _energy_mx(mx, delta, masks, solver.cfg.n)
    rho = cross / mx.sqrt(mx.maximum(et * ef, mx.array(1e-30, dtype=mx.float32)))
    linear_rhs = -mx.array(solver.rate_np.astype(np.float32)) * delta
    phi_rhs = solver.jacobian_rhs(delta)
    tl_rhs = solver.jacobian_rhs(forecasts) - solver.jacobian_rhs(truth) - phi_rhs
    linear = _inner_mx(mx, delta, linear_rhs, masks, solver.cfg.n)
    tl = _inner_mx(mx, delta, tl_rhs, masks, solver.cfg.n)
    phi = _inner_mx(mx, delta, phi_rhs, masks, solver.cfg.n)
    values = (et, ef, cross, epsilon, rho, linear, tl, phi)
    mx.eval(*values)
    names = ("Et", "Ef", "C", "epsilon", "rho", "P_linear", "P_tl_nonlinear", "P_pure_error")
    return {name: np.asarray(value, dtype=np.float64) for name, value in zip(names, values)}


def _locality(mx: Any, solver: Any, truth: Any, forecasts: Any, source_masks: Any, target_masks: Any) -> tuple[np.ndarray, np.ndarray]:
    delta = forecasts - truth
    sources = int(source_masks.shape[0])
    target_count = int(target_masks.shape[0])
    members = int(forecasts.shape[0])
    tl = np.empty((members, sources, sources, target_count), dtype=np.float64)
    phi = np.empty_like(tl)
    for truth_source in range(sources):
        tpart = truth * source_masks[truth_source]
        for error_source in range(sources):
            dpart = delta * source_masks[error_source]
            tl_rhs = _bilinear_rhs(solver, tpart, dpart) + _bilinear_rhs(solver, dpart, tpart)
            tl_value = _inner_mx(mx, delta, tl_rhs, target_masks, solver.cfg.n)
            phi_rhs = _bilinear_rhs(solver, delta * source_masks[truth_source], dpart)
            phi_value = _inner_mx(mx, delta, phi_rhs, target_masks, solver.cfg.n)
            mx.eval(tl_value, phi_value)
            tl[:, truth_source, error_source] = np.asarray(tl_value)
            phi[:, truth_source, error_source] = np.asarray(phi_value)
    return tl, phi


def _shell_spectra(solver: Any, truth: Any, forecasts: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Keep the historical plotted center-shell convention separate from the
    # complete half-open support partition used by covariance/locality masks.
    tn = np.asarray(truth).astype(np.complex64)
    fn = np.asarray(forecasts).astype(np.complex64)
    masks = np.asarray([
        (solver.kmag_np >= k - 0.5) & (solver.kmag_np < k + 0.5)
        for k in range(1, solver.cfg.k_max + 1)
    ])
    truth_e = 0.5 * np.sum(np.abs(tn)[None] ** 2 * masks, axis=(-2, -1)) / solver.cfg.n**4
    forecast_e = 0.5 * np.sum(np.abs(fn[:, None]) ** 2 * masks[None], axis=(-2, -1)) / solver.cfg.n**4
    error_e = 0.5 * np.sum(np.abs((fn - tn)[:, None]) ** 2 * masks[None], axis=(-2, -1)) / solver.cfg.n**4
    return truth_e, forecast_e, error_e


def run_truth(config: dict[str, Any], config_path: Path, output: Path, regime_id: str,
              truth_index: int, resume: bool = False,
              authorization_path: Path | None = None) -> Path:
    require_production_authorization(config, config_path, authorization_path)
    started = time.monotonic()
    mx, SQGJFMConfig, SQGJFMSolver = _require_metal()
    item = regime(config, regime_id)
    target = output / regime_id / "clusters" / f"truth_{truth_index:02d}.npz"
    if target.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite cluster: {target}")
        with np.load(target, allow_pickle=False) as existing:
            metadata = json.loads(str(existing["metadata_json"]))
        if (metadata.get("schema") != SCHEMA
                or metadata.get("config_sha256") != sha256(config_path)
                or metadata.get("solver_sha256") != sha256(SOLVER)
                or metadata.get("runner_sha256") != sha256(RUNNER)):
            raise ValueError(f"existing cluster has incompatible provenance: {target}")
        return target
    truth_path = output / regime_id / "truths" / f"truth_{truth_index:02d}.npz"
    if not truth_path.exists():
        raise FileNotFoundError(f"prepare truth states first: {truth_path}")
    with np.load(truth_path, allow_pickle=False) as saved:
        truth = mx.array(saved["theta_hat"])
        truth_metadata = json.loads(str(saved["metadata_json"]))
    forcing = config["model"]["forcing"]
    cfg = SQGJFMConfig(
        n=int(item["n"]), dt=float(item["dt"]), nu=float(item["nu"]),
        mu=float(config["model"]["mu"]), k_f=float(forcing["k_f"]),
        dk_f=float(forcing["dk_f"]), eps_I=float(forcing["eps_I"]),
        seed=int(config["seeds"]["nature_seed_offset"]),
    )
    solver = SQGJFMSolver(cfg)
    regime_index = [x["id"] for x in config["design"]["regimes"]].index(regime_id)
    forcing_seed = int(config["seeds"]["twin_forcing_seed_offset"]) + 100 * regime_index + truth_index
    rng = np.random.default_rng(forcing_seed)
    bands = expanded_bands(config, cfg.n)
    names, mask_np = _masks_np(solver, bands)
    masks = mx.array(mask_np)
    source_masks = masks[:-1]
    primary_index = bands.index(tuple(config["design"]["primary_band"]))
    active_config_hash = sha256(config_path)
    if truth_metadata.get("config_sha256") != active_config_hash:
        raise ValueError("truth cache was prepared with a different configuration")
    truth_state_hash = sha256(truth_path)

    perturbations = []
    perturbation_seeds = []
    base_seed = int(config["seeds"]["perturbation_seed_offset"]) + 10000 * regime_index + 100 * truth_index
    truth_primary = float(np.asarray(_energy_mx(mx, truth[None], masks[primary_index:primary_index + 1], cfg.n))[0, 0])
    target_error = float(config["design"]["initial_error_fraction_of_decorrelated_ceiling"]) * 2.0 * truth_primary
    for member in range(int(config["design"]["perturbations_per_truth"])):
        seed = base_seed + member
        perturbation_seeds.append(seed)
        prng = np.random.default_rng(seed)
        noise = prng.standard_normal((cfg.n, cfg.n)).astype(np.float32)
        delta = solver.fft(mx.array(noise)) * masks[primary_index]
        current = float(np.asarray(_energy_mx(mx, delta[None], masks[primary_index:primary_index + 1], cfg.n))[0, 0])
        delta = delta * np.float32(np.sqrt(target_error / max(current, 1e-30)))
        perturbations.append(delta)
    forecasts = mx.stack([truth + delta for delta in perturbations])

    lyap_seed = int(config["seeds"]["lyapunov_seed_offset"]) + 10000 * regime_index + truth_index
    lrng = np.random.default_rng(lyap_seed)
    tangent = solver.fft(mx.array(lrng.standard_normal((cfg.n, cfg.n)).astype(np.float32))) * masks[primary_index]
    tangent_target = float(config["lyapunov"]["initial_norm"])
    tangent_energy = float(np.asarray(_energy_mx(mx, tangent[None], masks[-1:], cfg.n))[0, 0])
    tangent = tangent * np.float32(tangent_target / np.sqrt(max(2.0 * tangent_energy, 1e-30)))
    mx.eval(truth, forecasts, tangent)
    tangent_interval_start = _energy_mx(mx, tangent[None], masks, cfg.n)[0]

    horizon = float(config["design"]["forecast_horizon_tu"])
    scalar_times = cadence_times(config["output"]["scalar_cadence"], horizon)
    locality_times = cadence_times(config["output"]["spectral_locality_cadence"], horizon)
    scalar_steps, locality_steps = step_lookup(scalar_times, cfg.dt), step_lookup(locality_times, cfg.dt)
    field_times = np.asarray(config["output"]["field_audit_times_tu"], dtype=float)
    field_steps = step_lookup(field_times, cfg.dt)
    renorm_steps = max(1, round(float(config["lyapunov"]["renormalization_dt_tu"]) / cfg.dt))
    n_steps = round(horizon / cfg.dt)
    budget_accum = mx.zeros((4, forecasts.shape[0], len(names)), dtype=mx.float32)
    forcing_error_accum = mx.zeros((forecasts.shape[0],), dtype=mx.float32)
    interval_start = _energy_mx(mx, forecasts - truth, masks, cfg.n)

    scalar_records: dict[str, list[np.ndarray]] = {key: [] for key in
        ("Et", "Ef", "C", "epsilon", "rho", "P_linear", "P_tl_nonlinear", "P_pure_error")}
    budget_components, budget_observed, forcing_errors = [], [], []
    lambda_times, lambda_rates = [], []
    locality_tl, locality_phi, truth_spectra, forecast_spectra, error_spectra = [], [], [], [], []
    field_truth, field_forecasts = [], []

    def save_scalar() -> None:
        nonlocal budget_accum, interval_start, forcing_error_accum
        values = _continuous_diagnostics(mx, solver, truth, forecasts, masks)
        for key, value in values.items():
            scalar_records[key].append(value)
        current = _energy_mx(mx, forecasts - truth, masks, cfg.n)
        mx.eval(current, budget_accum)
        budget_observed.append(np.asarray(current - interval_start, dtype=np.float64))
        budget_components.append(np.asarray(budget_accum, dtype=np.float64))
        forcing_errors.append(float(np.max(np.asarray(forcing_error_accum))))
        interval_start = current
        budget_accum = mx.zeros_like(budget_accum)
        forcing_error_accum = mx.zeros_like(forcing_error_accum)

    save_scalar()
    lt, lp = _locality(mx, solver, truth, forecasts, source_masks, masks)
    locality_tl.append(lt); locality_phi.append(lp)
    ts, fs, es = _shell_spectra(solver, truth, forecasts)
    truth_spectra.append(ts); forecast_spectra.append(fs); error_spectra.append(es)
    field_truth.append(np.asarray(truth).astype(np.complex64))
    field_forecasts.append(np.asarray(forecasts).astype(np.complex64))

    for step in range(1, n_steps + 1):
        old_delta = forecasts - truth
        kick = solver.forcing_kick(rng)
        truth, forecasts, increments, stages, shared_forcing_residual = _pair_step(solver, truth, forecasts, kick)
        tangent = _tangent_step(solver, tangent, stages)
        budget_accum = budget_accum + _allocated_mx(mx, old_delta, increments, masks, cfg.n)
        forcing_numerator = _energy_mx(mx, shared_forcing_residual, masks[-1:], cfg.n)[:, 0]
        forcing_denominator = _energy_mx(mx, forecasts - truth, masks[-1:], cfg.n)[:, 0]
        forcing_error_accum = mx.maximum(
            forcing_error_accum,
            forcing_numerator / mx.maximum(forcing_denominator, mx.array(1e-30, dtype=mx.float32)),
        )
        if step % renorm_steps == 0:
            tangent_band_energy = _energy_mx(mx, tangent[None], masks, cfg.n)[0]
            mx.eval(tangent_band_energy, tangent_interval_start)
            band_energy_np = np.asarray(tangent_band_energy, dtype=np.float64)
            start_np = np.asarray(tangent_interval_start, dtype=np.float64)
            amplitude = math.sqrt(max(2.0 * band_energy_np[-1], 1e-30))
            interval = renorm_steps * cfg.dt
            lambda_times.append(step * cfg.dt)
            lambda_rates.append(np.log(np.maximum(band_energy_np, 1e-30) /
                                       np.maximum(start_np, 1e-30)) / interval)
            scale = np.float32(tangent_target / amplitude)
            tangent = tangent * scale
            tangent_interval_start = tangent_band_energy * scale**2
        if step % 25 == 0:
            mx.eval(truth, forecasts, tangent, budget_accum, forcing_error_accum)
        if step in scalar_steps:
            save_scalar()
        if step in locality_steps:
            lt, lp = _locality(mx, solver, truth, forecasts, source_masks, masks)
            locality_tl.append(lt); locality_phi.append(lp)
            ts, fs, es = _shell_spectra(solver, truth, forecasts)
            truth_spectra.append(ts); forecast_spectra.append(fs); error_spectra.append(es)
        if step in field_steps:
            mx.eval(truth, forecasts)
            field_truth.append(np.asarray(truth).astype(np.complex64))
            field_forecasts.append(np.asarray(forecasts).astype(np.complex64))

    metadata = {
        "schema": SCHEMA, "regime": regime_id, "truth_index": truth_index,
        "band_names": names, "bands": bands, "primary_band_index": primary_index,
        "perturbation_seeds": perturbation_seeds, "lyapunov_seed": lyap_seed,
        "twin_forcing_seed": forcing_seed, "wall_seconds": time.monotonic() - started,
        "config_sha256": active_config_hash, "solver_sha256": sha256(SOLVER),
        "runner_sha256": sha256(RUNNER),
        "truth_state_sha256": truth_state_hash,
        "shared_forcing": True, "lambda_is_energy_growth_rate": True,
    }
    arrays: dict[str, Any] = {
        "scalar_time": scalar_times, "locality_time": locality_times, "field_time": field_times,
        "lambda_time": np.asarray(lambda_times), "lambda_energy_rate": np.asarray(lambda_rates),
        "budget_components": np.asarray(budget_components), "budget_observed": np.asarray(budget_observed),
        "shared_forcing_relative_energy_error": np.asarray(forcing_errors),
        "locality_tl_source": np.asarray(locality_tl), "locality_phi_source": np.asarray(locality_phi),
        "truth_spectrum": np.asarray(truth_spectra), "forecast_spectrum": np.asarray(forecast_spectra),
        "error_spectrum": np.asarray(error_spectra), "field_truth_hat": np.asarray(field_truth),
        "field_forecast_hat": np.asarray(field_forecasts),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    arrays.update({key: np.asarray(value) for key, value in scalar_records.items()})
    atomic_npz(target, **arrays)
    return target


def _fit(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    count = int(mask.sum())
    if count < 3:
        return {"n": count, "intercept": math.nan, "slope": math.nan, "r2": math.nan}
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    predicted = intercept + slope * x[mask]
    ss_res = float(np.sum((y[mask] - predicted) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return {"n": count, "intercept": float(intercept), "slope": float(slope),
            "r2": 1.0 - ss_res / max(ss_tot, np.finfo(float).tiny)}


def analyze(config: dict[str, Any], config_path: Path, output: Path, report: Path,
            authorization_path: Path | None = None) -> Path:
    validate_config(config)
    require_production_authorization(config, config_path, authorization_path)
    manifest_path = report / "prl_v8_pilot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("analysis requires the frozen dry-run manifest")
    manifest = load_json(manifest_path)
    active_config_hash, active_solver_hash = sha256(config_path), sha256(SOLVER)
    active_runner_hash = sha256(RUNNER)
    if (manifest["config"]["sha256"] != active_config_hash
            or manifest["solver"]["sha256"] != active_solver_hash
            or manifest["runner"]["sha256"] != active_runner_hash):
        raise ValueError("active configuration, solver, or runner does not match the dry-run manifest")
    if Path(manifest["output_root"]).resolve() != output.resolve():
        raise ValueError("analysis output root does not match the dry-run manifest")
    gate_cfg = config["pilot_gates"]
    result: dict[str, Any] = {"schema": SCHEMA, "experiment_id": config["experiment_id"], "regimes": {}}
    all_green = True
    for item in config["design"]["regimes"]:
        rid, rows = item["id"], []
        bands = expanded_bands(config, int(item["n"]))
        primary = bands.index(tuple(config["design"]["primary_band"]))
        centers = np.sqrt(np.asarray([lo * hi for lo, hi in bands], dtype=float))
        local = np.abs(np.log2(centers / centers[primary])) <= 1.0
        for truth_index in range(config["design"]["truths_per_regime"]):
            path = output / rid / "clusters" / f"truth_{truth_index:02d}.npz"
            if not path.exists():
                rows.append({"truth_index": truth_index, "complete": False})
                all_green = False
                continue
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"]))
                if metadata.get("schema") != SCHEMA:
                    raise ValueError(f"invalid cluster schema: {path}")
                if (metadata.get("config_sha256") != active_config_hash
                        or metadata.get("solver_sha256") != active_solver_hash
                        or metadata.get("runner_sha256") != active_runner_hash):
                    raise ValueError(f"cluster provenance mismatch: {path}")
                truth_path = output / rid / "truths" / f"truth_{truth_index:02d}.npz"
                if metadata.get("truth_state_sha256") != sha256(truth_path):
                    raise ValueError(f"cluster truth-state provenance mismatch: {path}")
                expected_scalar = cadence_times(config["output"]["scalar_cadence"], config["design"]["forecast_horizon_tu"])
                expected_locality = cadence_times(config["output"]["spectral_locality_cadence"], config["design"]["forecast_horizon_tu"])
                if not np.array_equal(data["scalar_time"], expected_scalar) or not np.array_equal(data["locality_time"], expected_locality):
                    raise ValueError(f"cluster time coordinates mismatch: {path}")
                et, ef, cross = data["Et"][:, :, primary], data["Ef"][:, :, primary], data["C"][:, :, primary]
                eps, rho = data["epsilon"][:, :, primary], data["rho"][:, :, primary]
                identity = eps - et - ef + 2.0 * cross
                identity_rel = float(np.linalg.norm(identity) / max(np.linalg.norm(eps), np.finfo(float).tiny))
                components = data["budget_components"][:, :, :, primary]
                observed = data["budget_observed"][:, :, primary]
                budget_rel = float(np.linalg.norm(observed - components.sum(axis=1)) /
                                   max(np.linalg.norm(observed), np.finfo(float).tiny))
                forcing = float(np.max(data["shared_forcing_relative_energy_error"]))
                burn = float(config["lyapunov"]["burnin_tu"])
                lmask = data["lambda_time"] >= burn
                lam = float(np.mean(data["lambda_energy_rate"][lmask, primary]))
                L = 2.0 * np.mean(et[len(et) // 2:], axis=0)
                predictors = {
                    "two_C_over_Et_plus_Ef": 2.0 * cross / np.maximum(et + ef, 1e-30),
                    "rho": rho,
                    "one_minus_epsilon_over_L": 1.0 - eps / np.maximum(L[None], 1e-30),
                }
                response = (data["P_linear"][:, :, primary] + data["P_tl_nonlinear"][:, :, primary])
                response = response / (lam * eps) if lam > 0 else np.full_like(response, np.nan)
                fit_mask = np.isfinite(response) & (rho > gate_cfg["required_primary_rho_interval"][0])
                fit_mask &= rho < gate_cfg["required_primary_rho_interval"][1]
                fits = {name: _fit(value.ravel(), response.ravel(), fit_mask.ravel())
                        for name, value in predictors.items()}
                source = data["locality_tl_source"][:, :, :, :, primary]
                local_matrix = local[:, None] & local[None, :]
                local_abs = np.sum(np.abs(source[:, :, local_matrix]), axis=-1)
                total_abs = np.sum(np.abs(source), axis=(-2, -1))
                locality_fraction = float(np.mean(local_abs / np.maximum(total_abs, 1e-30)))
                scalar_index = np.searchsorted(data["scalar_time"], data["locality_time"])
                tl_expected = data["P_tl_nonlinear"][scalar_index, :, :]
                phi_expected = data["P_pure_error"][scalar_index, :, :]
                tl_source_all = data["locality_tl_source"].sum(axis=(2, 3))
                phi_source_all = data["locality_phi_source"].sum(axis=(2, 3))
                source_residual = np.concatenate([
                    (tl_source_all - tl_expected).ravel(),
                    (phi_source_all - phi_expected).ravel(),
                ])
                source_scale = np.concatenate([tl_expected.ravel(), phi_expected.ravel()])
                source_rel = float(np.linalg.norm(source_residual) /
                                   max(np.linalg.norm(source_scale), np.finfo(float).tiny))
                t = data["scalar_time"]
                late = slice(len(t) // 2, None)
                truth_energy = np.mean(data["Et"][:, :, -1], axis=1)
                drift = np.polyfit(t[late], truth_energy[late], 1)[0] * (t[late][-1] - t[late][0])
                drift /= np.mean(truth_energy[late])
                required_finite = [et, ef, cross, eps, rho, response, data["budget_components"],
                                   data["locality_tl_source"], data["locality_phi_source"],
                                   data["lambda_energy_rate"]]
                finite_fraction = float(sum(np.isfinite(value).sum() for value in required_finite) /
                                        sum(value.size for value in required_finite))
                rho_span_by_member = np.nanmax(rho, axis=0) - np.nanmin(rho, axis=0)
                interval_points_by_member = fit_mask.sum(axis=0).astype(int)
                bracket_by_member = ((np.nanmin(rho, axis=0) <= gate_cfg["required_primary_rho_interval"][0]) &
                                     (np.nanmax(rho, axis=0) >= gate_cfg["required_primary_rho_interval"][1]))
                gates = {
                    "finite": finite_fraction >= gate_cfg["minimum_finite_fraction"],
                    "covariance_identity": identity_rel <= gate_cfg["maximum_covariance_identity_relative_error"],
                    "discrete_budget": budget_rel <= gate_cfg["maximum_discrete_budget_relative_error"],
                    "shared_forcing": forcing <= gate_cfg["maximum_shared_forcing_error_relative"],
                    "locality_source_reconstruction": source_rel <= gate_cfg["maximum_locality_source_relative_error"],
                    "positive_lyapunov": lam > gate_cfg["minimum_lambda_energy"],
                    "stationarity": abs(float(drift)) <= gate_cfg["stationarity_max_fractional_drift"],
                    "rho_span": bool(np.all(rho_span_by_member >= gate_cfg["minimum_primary_rho_span"])),
                    "rho_brackets_interval": bool(np.all(bracket_by_member)),
                    "rho_interval_points": bool(np.all(interval_points_by_member >= gate_cfg["minimum_points_in_primary_interval"])),
                }
                all_green &= all(gates.values())
                rows.append({
                    "truth_index": truth_index, "complete": True, "lambda_energy": lam,
                    "covariance_identity_relative_error": identity_rel,
                    "discrete_budget_relative_error": budget_rel,
                    "shared_forcing_relative_error": forcing,
                    "locality_source_relative_error": source_rel,
                    "stationary_truth_energy_drift": float(drift), "finite_fraction": finite_fraction,
                    "rho_span_by_member": rho_span_by_member.tolist(),
                    "points_in_rho_interval_by_member": interval_points_by_member.tolist(),
                    "rho_brackets_interval_by_member": bracket_by_member.tolist(),
                    "octave_local_nonlinear_tl_fraction": locality_fraction, "fits": fits, "gates": gates,
                    "sha256": sha256(path),
                })
        result["regimes"][rid] = {"truths": rows, "all_truth_gates_pass": all(
            row.get("complete") and all(row.get("gates", {}).values()) for row in rows
        )}
    result["pilot_green"] = bool(all_green)
    result["production_authorized"] = False
    result["decision"] = "review pilot before freezing production" if all_green else "production blocked"
    target = report / "prl_v8_pilot_analysis.json"
    atomic_json(target, result)
    return target


def submit(config: dict[str, Any], config_path: Path, output: Path, report: Path, launch: bool,
           authorization_path: Path | None = None) -> Path:
    launcher_mode = config.get("execution", {}).get("launcher_mode", "sequential_legacy")
    if launch and (config["stage"] == "confirmatory_production" or launcher_mode == "per_cluster"):
        raise PermissionError(
            f"REFUSED: this configuration must use the Gate 4 per-cluster launcher: {GATE4_CLUSTER_LAUNCHER}"
        )
    if launch:
        require_production_authorization(config, config_path, authorization_path)
    manifest_path = report / "prl_v8_pilot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("run dry-run before submission")
    script = report / "launch_prl_v8_pilot.zsh"
    status: dict[str, Any] = {
        "schema": SCHEMA, "manifest_sha256": sha256(manifest_path),
        "launch_script_sha256": sha256(script), "launched": False,
    }
    try:
        _require_metal()
        status["metal_visible"] = True
    except RuntimeError as exc:
        status["metal_visible"] = False
        status["blocked_reason"] = str(exc)
    if launch:
        if not status["metal_visible"]:
            raise RuntimeError(status["blocked_reason"])
        log = report / "pilot.log"
        if log.exists():
            raise FileExistsError(f"refusing to overwrite launch log: {log}")
        with log.open("xb") as stream:
            process = subprocess.Popen(
                ["caffeinate", "-i", "-m", "-s", str(script)],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
        status.update({"launched": True, "pid": process.pid, "log": str(log)})
    target = report / "prl_v8_pilot_submission.json"
    atomic_json(target, status)
    return target


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("dry-run", "prepare-truths", "run-truth", "analyze", "submit"))
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--regime")
    result.add_argument("--truth-index", type=int)
    result.add_argument("--launch", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--authorization", type=Path,
                        help="dated PI authorization JSON required for confirmatory production")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_json(args.config)
    validate_config(config)
    if args.command == "dry-run":
        print(dry_run(args.config, args.output, args.report_dir))
    elif args.command == "prepare-truths":
        if args.regime is None:
            raise SystemExit("--regime is required")
        prepare_truths(config, args.config, args.output, args.regime, args.resume, args.authorization)
    elif args.command == "run-truth":
        if args.regime is None or args.truth_index is None:
            raise SystemExit("--regime and --truth-index are required")
        print(run_truth(config, args.config, args.output, args.regime, args.truth_index,
                        args.resume, args.authorization))
    elif args.command == "analyze":
        print(analyze(config, args.config, args.output, args.report_dir, args.authorization))
    else:
        print(submit(config, args.config, args.output, args.report_dir, args.launch, args.authorization))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
