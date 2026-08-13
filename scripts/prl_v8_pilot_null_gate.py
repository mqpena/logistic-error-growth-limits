#!/usr/bin/env python3
"""Frozen pilot null hierarchy for the PRL v8 production decision."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

import prl_v8_two_regime as generator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPT = Path(__file__).resolve()
TEST_SCRIPT = HERE / "test_prl_v8_pilot_null_gate.py"
DEFAULT_FREEZE = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_null_gate_pass_criteria_freeze_proposal_20260719.json"
)
DEFAULT_CONFIG = HERE / "prl_v8_gate5_nonconfirmatory_config_20260715.json"
DEFAULT_OUTPUT = HERE / "out" / "prl_v8_gate5_nonconfirmatory_20260715"
DEFAULT_EVALUATION = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_gate5_nonconfirmatory_20260715_per_cluster"
    / "prl_v8_pilot_evaluation_v2_8of8_f5e91c6d608c_be29ceeaa19f.json"
)
DEFAULT_REPORT = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_pilot_null_gate_20260719.json"
)
SCHEMA = "prl-v8-pilot-null-gate-result-1"
RATIFICATION_SCHEMA = "prl-v8-pilot-null-gate-ratification-1"


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


def weighted_slope(truths: list[list[dict[str, np.ndarray]]]) -> float:
    """Equal-member/equal-truth WLS slope with an intercept."""
    xs, ys, weights = [], [], []
    truth_count = len(truths)
    if truth_count < 2:
        raise ValueError("at least two truths are required")
    for members in truths:
        member_count = len(members)
        if member_count < 1:
            raise ValueError("every truth requires at least one member")
        for member in members:
            mask = np.asarray(member["mask"], dtype=bool)
            x = np.asarray(member["x"], dtype=float)[mask]
            y = np.asarray(member["y"], dtype=float)[mask]
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError("nonfinite value on a frozen eligible mask")
            if x.size < 3:
                raise ValueError("every trajectory requires at least three eligible samples")
            xs.append(x)
            ys.append(y)
            weights.append(np.full(
                x.size, 1.0 / (truth_count * member_count * x.size)))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    weight = np.concatenate(weights)
    design = np.column_stack([np.ones(x.size), x])
    root_weight = np.sqrt(weight)
    beta, *_ = np.linalg.lstsq(
        design * root_weight[:, None], y * root_weight, rcond=None)
    slope = float(beta[1])
    if not np.isfinite(slope):
        raise ValueError("nonfinite test statistic")
    return slope


def _contiguous_runs(times: np.ndarray, maximum_gap: float) -> list[np.ndarray]:
    if times.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(times) > maximum_gap) + 1]
    stops = np.r_[starts[1:], times.size]
    return [np.arange(start, stop) for start, stop in zip(starts, stops)]


def permute_time_blocks(times: np.ndarray, values: np.ndarray,
                        block_length: int, maximum_gap: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Permute contiguous blocks within each dense eligible-time run."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape:
        raise ValueError("times and values must have identical shapes")
    if block_length < 1:
        raise ValueError("block length must be positive")
    result = values.copy()
    for run in _contiguous_runs(times, maximum_gap):
        blocks = [
            run[start:start + block_length]
            for start in range(0, run.size, block_length)
        ]
        order = rng.permutation(len(blocks))
        shuffled = np.concatenate([values[blocks[index]] for index in order])
        result[run] = shuffled
    return result


def time_block_surrogate(truths: list[list[dict[str, np.ndarray]]],
                         block_length: int, maximum_gap: float,
                         rng: np.random.Generator
                         ) -> list[list[dict[str, np.ndarray]]]:
    surrogate = []
    for members in truths:
        surrogate_members = []
        for member in members:
            mask = np.asarray(member["mask"], dtype=bool)
            eligible_times = np.asarray(member["time"], dtype=float)[mask]
            eligible_y = np.asarray(member["y"], dtype=float)[mask]
            permuted = permute_time_blocks(
                eligible_times, eligible_y, block_length, maximum_gap, rng)
            replacement = np.asarray(member["y"], dtype=float).copy()
            replacement[mask] = permuted
            surrogate_members.append({
                "time": np.asarray(member["time"]),
                "x": np.asarray(member["x"]),
                "y": replacement,
                "mask": mask,
            })
        surrogate.append(surrogate_members)
    return surrogate


def truth_permutations(count: int) -> list[tuple[int, ...]]:
    return list(itertools.permutations(range(count)))


def cross_truth_surrogates(
    truths: list[list[dict[str, np.ndarray]]],
) -> list[list[list[dict[str, np.ndarray]]]]:
    truth_count = len(truths)
    member_count = len(truths[0])
    if truth_count != 4 or member_count != 2:
        raise ValueError("frozen cross-truth enumeration requires four truths and two members")
    if any(len(members) != member_count for members in truths):
        raise ValueError("all truths must have two members")
    results = []
    for truth_map in truth_permutations(truth_count):
        for swap_bits in itertools.product((0, 1), repeat=truth_count):
            surrogate = []
            for recipient_truth in range(truth_count):
                donor_truth = truth_map[recipient_truth]
                members = []
                for recipient_member in range(member_count):
                    donor_member = recipient_member ^ swap_bits[recipient_truth]
                    recipient = truths[recipient_truth][recipient_member]
                    donor = truths[donor_truth][donor_member]
                    if not np.allclose(
                        recipient["time"], donor["time"], rtol=0.0, atol=1e-12
                    ):
                        raise ValueError("cross-truth scalar times do not match")
                    donor_y = np.asarray(donor["y"], dtype=float)
                    members.append({
                        "time": np.asarray(recipient["time"]),
                        "x": np.asarray(recipient["x"]),
                        "y": donor_y,
                        "mask": np.asarray(recipient["mask"], dtype=bool),
                    })
                surrogate.append(members)
            results.append(surrogate)
    if len(results) != 384:
        raise AssertionError("frozen cross-truth group must contain 384 transformations")
    return results


def monte_carlo_p(observed: float, null_values: np.ndarray) -> float:
    values = np.asarray(null_values, dtype=float)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("all frozen null replicates must be finite")
    return float((1 + np.count_nonzero(values >= observed)) / (values.size + 1))


def exact_group_p(observed: float, transformed_values: np.ndarray) -> float:
    values = np.asarray(transformed_values, dtype=float)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("all group-transformed statistics must be finite")
    return float(np.count_nonzero(values >= observed) / values.size)


def hermitian_phase(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.fft2(rng.standard_normal((n, n)))
    magnitude = np.abs(spectrum)
    phase = np.ones_like(spectrum, dtype=np.complex128)
    active = magnitude > np.finfo(float).tiny
    phase[active] = spectrum[active] / magnitude[active]
    return phase


def spectral_geometry(n: int, nu: float, mu: float,
                      bands: list[tuple[int, int]], primary: int
                      ) -> dict[str, np.ndarray]:
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky)
    square = ((np.abs(kx) <= n // 3) & (np.abs(ky) <= n // 3)).astype(float)
    safe = np.where(kmag > 0, kmag, 1.0)
    inv = np.where(kmag > 0, 1.0 / safe, 0.0) * square
    rate = (
        nu * kmag**2 + np.where(kmag > 0, mu / safe**2, 0.0)
    ) * square
    masks = generator.half_open_radial_masks(kmag, square, bands)
    return {
        "kx": kx,
        "ky": ky,
        "square": square,
        "inv": inv,
        "rate": rate,
        "primary": masks[primary].astype(float),
    }


def jacobian_rhs(field_hat: np.ndarray,
                 geometry: dict[str, np.ndarray]) -> np.ndarray:
    psi_hat = geometry["inv"] * field_hat
    psi_x = np.fft.ifft2(1j * geometry["kx"] * psi_hat).real
    psi_y = np.fft.ifft2(1j * geometry["ky"] * psi_hat).real
    field_x = np.fft.ifft2(1j * geometry["kx"] * field_hat).real
    field_y = np.fft.ifft2(1j * geometry["ky"] * field_hat).real
    return (
        -np.fft.fft2(psi_x * field_y - psi_y * field_x)
        * geometry["square"]
    )


def inner(a: np.ndarray, b: np.ndarray, mask: np.ndarray, n: int) -> float:
    return float(np.sum(np.real(np.conj(a) * b) * mask) / n**4)


def field_diagnostics(truth: np.ndarray, forecasts: np.ndarray,
                      geometry: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = truth.shape[-1]
    mask = geometry["primary"]
    truth_jacobian = jacobian_rhs(truth, geometry)
    et_value = 0.5 * inner(truth, truth, mask, n)
    rows = {key: [] for key in (
        "Et", "Ef", "C", "epsilon", "rho", "x_C",
        "P_linear", "P_tl_nonlinear",
    )}
    for forecast in forecasts:
        delta = forecast - truth
        ef = 0.5 * inner(forecast, forecast, mask, n)
        cross = 0.5 * inner(truth, forecast, mask, n)
        epsilon = 0.5 * inner(delta, delta, mask, n)
        rho = cross / np.sqrt(max(et_value * ef, 1e-30))
        x_c = 2.0 * cross / max(et_value + ef, 1e-30)
        linear = inner(delta, -geometry["rate"] * delta, mask, n)
        tangent_rhs = (
            jacobian_rhs(forecast, geometry)
            - truth_jacobian
            - jacobian_rhs(delta, geometry)
        )
        tangent = inner(delta, tangent_rhs, mask, n)
        values = (et_value, ef, cross, epsilon, rho, x_c, linear, tangent)
        for key, value in zip(rows, values):
            rows[key].append(value)
    return {key: np.asarray(value, dtype=float) for key, value in rows.items()}


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("nonfinite value in required relative-error check")
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1e-30)
    )


def validate_freeze(freeze: dict[str, Any]) -> None:
    if freeze.get("schema") != "prl-v8-pilot-null-gate-freeze-1":
        raise ValueError("wrong null-freeze schema")
    if freeze["alpha_and_combination"]["per_null_alpha"] != 0.05:
        raise ValueError("frozen per-null alpha must equal 0.05")
    expected = {
        "time_block": 9999,
        "phase_scramble": 999,
    }
    for name, replicates in expected.items():
        if freeze["nulls"][name]["replicates"] != replicates:
            raise ValueError(f"wrong frozen replicate count for {name}")
    if freeze["scope"]["primary_regime"] != "n512_re940_marginal":
        raise ValueError("wrong frozen primary regime")
    if freeze["nulls"]["cross_truth_reassignment"]["surrogates"] != 384:
        raise ValueError("cross-truth group must contain 384 transformations")


def verify_execution_bindings(freeze: dict[str, Any]) -> None:
    bindings = freeze["execution_bindings"]
    expected_paths = {
        "null_engine": SCRIPT,
        "structural_tests": TEST_SCRIPT,
    }
    for name, path in expected_paths.items():
        if bindings[name]["sha256"] != sha256(path):
            raise ValueError(f"{name} source hash does not match the freeze")
    interpreter = Path(bindings["interpreter"]["path"]).resolve()
    if interpreter != Path(platform.sys.executable).resolve():
        raise ValueError("wrong interpreter for frozen null evaluation")
    if bindings["interpreter"]["python_version"] != platform.python_version():
        raise ValueError("wrong Python version for frozen null evaluation")
    if bindings["interpreter"]["numpy_version"] != np.__version__:
        raise ValueError("wrong NumPy version for frozen null evaluation")


def validate_ratification(path: Path, freeze_path: Path) -> dict[str, Any]:
    ratification = generator.load_json(path)
    if ratification.get("schema") != RATIFICATION_SCHEMA:
        raise ValueError("wrong null-gate ratification schema")
    if ratification.get("ratified") is not True:
        raise ValueError("null-gate criteria are not PI-ratified")
    if ratification.get("production_authorized") is not False:
        raise ValueError("null-gate ratification cannot authorize production")
    binding = ratification.get("proposal", {})
    if binding.get("sha256") != sha256(freeze_path):
        raise ValueError("ratification does not bind the active freeze proposal")
    return ratification


def verify_bindings(freeze: dict[str, Any], config_path: Path,
                    evaluation_path: Path, output: Path) -> list[dict[str, Any]]:
    for name, path in (
        ("pilot_config", config_path),
        ("authoritative_pilot_evaluation", evaluation_path),
    ):
        expected = freeze["bindings"][name]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"{name} hash does not match the frozen binding")
    collapse_binding = freeze["bindings"]["collapse_analysis_v4"]
    collapse_path = (ROOT / collapse_binding["path"]).resolve()
    if sha256(collapse_path) != collapse_binding["sha256"]:
        raise ValueError("collapse-analysis hash does not match the frozen binding")
    evaluation = generator.load_json(evaluation_path)
    artifacts = []
    for regime_id, regime_result in evaluation["regimes"].items():
        for row in regime_result["truths"]:
            path = (
                output / regime_id / "clusters"
                / f"truth_{int(row['truth_index']):02d}.npz"
            )
            actual = sha256(path)
            if actual != row["sha256"]:
                raise ValueError(f"artifact hash mismatch: {path}")
            artifacts.append({
                "regime": regime_id,
                "truth_index": int(row["truth_index"]),
                "path": str(path),
                "sha256": actual,
                "bytes": path.stat().st_size,
            })
    if len(artifacts) != 8:
        raise ValueError("the frozen pilot requires exactly eight artifacts")
    return artifacts


def load_scalar_regime(config: dict[str, Any], output: Path,
                       regime: dict[str, Any]) -> list[list[dict[str, np.ndarray]]]:
    regime_id = regime["id"]
    bands = generator.expanded_bands(config, int(regime["n"]))
    primary = bands.index(tuple(config["design"]["primary_band"]))
    truths = []
    for truth_index in range(int(config["design"]["truths_per_regime"])):
        path = output / regime_id / "clusters" / f"truth_{truth_index:02d}.npz"
        with np.load(path, allow_pickle=False) as data:
            burn = data["lambda_time"] >= float(config["lyapunov"]["burnin_tu"])
            lam = float(np.mean(data["lambda_energy_rate"][burn, primary]))
            et = data["Et"][:, :, primary]
            ef = data["Ef"][:, :, primary]
            cross = data["C"][:, :, primary]
            epsilon = data["epsilon"][:, :, primary]
            rho = data["rho"][:, :, primary]
            x_c = 2.0 * cross / np.maximum(et + ef, 1e-30)
            response = (
                data["P_linear"][:, :, primary]
                + data["P_tl_nonlinear"][:, :, primary]
            ) / (lam * epsilon)
            members = []
            for member in range(x_c.shape[1]):
                mask = (
                    np.isfinite(x_c[:, member])
                    & np.isfinite(response[:, member])
                    & (rho[:, member] > 0.15)
                    & (rho[:, member] < 0.85)
                )
                members.append({
                    "time": np.asarray(data["scalar_time"], dtype=float),
                    "x": np.asarray(x_c[:, member], dtype=float),
                    "y": np.asarray(response[:, member], dtype=float),
                    "mask": mask,
                })
        truths.append(members)
    return truths


def scalar_nulls(truths: list[list[dict[str, np.ndarray]]],
                 freeze: dict[str, Any]) -> dict[str, Any]:
    observed = weighted_slope(truths)
    time_cfg = freeze["nulls"]["time_block"]
    rng = np.random.default_rng(int(time_cfg["seed"]))
    time_values = np.empty(int(time_cfg["replicates"]), dtype=float)
    for index in range(time_values.size):
        surrogate = time_block_surrogate(
            truths,
            int(time_cfg["dense_block_length_samples"]),
            0.0200001,
            rng,
        )
        time_values[index] = weighted_slope(surrogate)
    cross_values = np.asarray([
        weighted_slope(surrogate)
        for surrogate in cross_truth_surrogates(truths)
    ])
    return {
        "observed_slope": observed,
        "time_block": {
            "replicates": int(time_values.size),
            "slopes": time_values.tolist(),
            "p_value": monte_carlo_p(observed, time_values),
        },
        "cross_truth_reassignment": {
            "surrogates": int(cross_values.size),
            "slopes": cross_values.tolist(),
            "p_value": exact_group_p(observed, cross_values),
        },
    }


def matching_scalar_indices(scalar_time: np.ndarray,
                            field_time: np.ndarray) -> np.ndarray:
    lookup = {round(float(value), 12): index for index, value in enumerate(scalar_time)}
    try:
        return np.asarray([lookup[round(float(value), 12)] for value in field_time])
    except KeyError as exc:
        raise ValueError(f"field time lacks matching scalar record: {exc}") from exc


def load_field_regime(config: dict[str, Any], output: Path,
                      regime: dict[str, Any]) -> tuple[
                          list[dict[str, Any]], dict[str, np.ndarray], dict[str, float]
                      ]:
    regime_id = regime["id"]
    bands = generator.expanded_bands(config, int(regime["n"]))
    primary = bands.index(tuple(config["design"]["primary_band"]))
    geometry = spectral_geometry(
        int(regime["n"]), float(regime["nu"]), float(config["model"]["mu"]),
        bands, primary,
    )
    records = []
    check_values: dict[str, list[float]] = {
        key: [] for key in ("C", "epsilon", "P_linear", "P_tl_nonlinear")
    }
    check_reference: dict[str, list[float]] = {
        key: [] for key in check_values
    }
    for truth_index in range(int(config["design"]["truths_per_regime"])):
        path = output / regime_id / "clusters" / f"truth_{truth_index:02d}.npz"
        with np.load(path, allow_pickle=False) as data:
            burn = data["lambda_time"] >= float(config["lyapunov"]["burnin_tu"])
            lam = float(np.mean(data["lambda_energy_rate"][burn, primary]))
            indices = matching_scalar_indices(data["scalar_time"], data["field_time"])
            snapshots = []
            for field_index, scalar_index in enumerate(indices):
                diagnostics = field_diagnostics(
                    np.asarray(data["field_truth_hat"][field_index]),
                    np.asarray(data["field_forecast_hat"][field_index]),
                    geometry,
                )
                for key in check_values:
                    check_values[key].extend(diagnostics[key].tolist())
                    check_reference[key].extend(
                        np.asarray(data[key][scalar_index, :, primary], dtype=float).tolist()
                    )
                snapshots.append({
                    "time": float(data["field_time"][field_index]),
                    "truth": np.asarray(data["field_truth_hat"][field_index]),
                    "forecasts": np.asarray(data["field_forecast_hat"][field_index]),
                    "diagnostics": diagnostics,
                })
        records.append({"lambda": lam, "snapshots": snapshots})
    errors = {
        key: relative_l2(
            np.asarray(check_reference[key]), np.asarray(check_values[key]))
        for key in check_values
    }
    return records, geometry, errors


def field_truths(
    records: list[dict[str, Any]],
    key: str = "diagnostics",
    frozen_masks: list[list[np.ndarray]] | None = None,
) -> list[list[dict[str, np.ndarray]]]:
    truths = []
    for truth_index, record in enumerate(records):
        member_count = len(record["snapshots"][0][key]["x_C"])
        members = []
        for member in range(member_count):
            x = np.asarray([
                snapshot[key]["x_C"][member] for snapshot in record["snapshots"]
            ])
            epsilon = np.asarray([
                snapshot[key]["epsilon"][member] for snapshot in record["snapshots"]
            ])
            rho = np.asarray([
                snapshot[key]["rho"][member] for snapshot in record["snapshots"]
            ])
            production = np.asarray([
                snapshot[key]["P_linear"][member]
                + snapshot[key]["P_tl_nonlinear"][member]
                for snapshot in record["snapshots"]
            ])
            y = production / (float(record["lambda"]) * epsilon)
            if frozen_masks is None:
                mask = (
                    np.isfinite(x) & np.isfinite(y)
                    & (rho > 0.15) & (rho < 0.85)
                )
            else:
                mask = np.asarray(
                    frozen_masks[truth_index][member], dtype=bool).copy()
            members.append({
                "time": np.asarray([
                    snapshot["time"] for snapshot in record["snapshots"]
                ]),
                "x": x,
                "y": y,
                "mask": mask,
            })
        truths.append(members)
    return truths


def phase_nulls(records: list[dict[str, Any]],
                geometry: dict[str, np.ndarray],
                freeze: dict[str, Any]) -> dict[str, Any]:
    phase_cfg = freeze["nulls"]["phase_scramble"]
    observed_truths = field_truths(records)
    frozen_masks = [
        [np.asarray(member["mask"], dtype=bool).copy() for member in truth]
        for truth in observed_truths
    ]
    observed = weighted_slope(observed_truths)
    rng = np.random.default_rng(int(phase_cfg["seed"]))
    values = np.empty(int(phase_cfg["replicates"]), dtype=float)
    max_invariance = {key: 0.0 for key in (
        "Et", "Ef", "C", "epsilon", "x_C", "rho",
    )}
    n = geometry["square"].shape[0]
    for replicate in range(values.size):
        scrambled_records = []
        invariant_sums = {
            key: {"difference": 0.0, "reference": 0.0}
            for key in max_invariance
        }
        for record in records:
            snapshots = []
            for snapshot in record["snapshots"]:
                phase = hermitian_phase(n, rng)
                diagnostics = field_diagnostics(
                    snapshot["truth"] * phase,
                    snapshot["forecasts"] * phase[None],
                    geometry,
                )
                for key in max_invariance:
                    reference = np.asarray(snapshot["diagnostics"][key], dtype=float)
                    candidate = np.asarray(diagnostics[key], dtype=float)
                    if (
                        not np.all(np.isfinite(reference))
                        or not np.all(np.isfinite(candidate))
                    ):
                        raise ValueError(
                            "nonfinite value in phase-scramble invariance check")
                    invariant_sums[key]["difference"] += float(
                        np.sum((candidate - reference) ** 2))
                    invariant_sums[key]["reference"] += float(
                        np.sum(reference**2))
                snapshots.append({
                    "time": snapshot["time"],
                    "diagnostics": diagnostics,
                })
            scrambled_records.append({
                "lambda": record["lambda"],
                "snapshots": snapshots,
            })
        values[replicate] = weighted_slope(field_truths(
            scrambled_records, frozen_masks=frozen_masks))
        for key, sums in invariant_sums.items():
            error = np.sqrt(sums["difference"]) / max(
                np.sqrt(sums["reference"]), 1e-30)
            if not np.isfinite(error):
                raise ValueError(
                    "nonfinite aggregate phase-scramble invariance error")
            max_invariance[key] = max(max_invariance[key], float(error))
    return {
        "observed_field_slope": observed,
        "replicates": int(values.size),
        "slopes": values.tolist(),
        "p_value": monte_carlo_p(observed, values),
        "maximum_invariance_relative_error": max_invariance,
    }


def execute(freeze_path: Path, ratification_path: Path,
            config_path: Path, output: Path, evaluation_path: Path,
            report_path: Path) -> Path:
    freeze = generator.load_json(freeze_path)
    validate_freeze(freeze)
    ratification = validate_ratification(ratification_path, freeze_path)
    verify_execution_bindings(freeze)
    base_result = {
        "schema": SCHEMA,
        "production_authorized": False,
        "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
        "ratification": {
            "path": str(ratification_path),
            "sha256": sha256(ratification_path),
            "ratified_on": ratification["ratified_on"],
        },
        "script": {"path": str(SCRIPT), "sha256": sha256(SCRIPT)},
        "structural_tests": {
            "path": str(TEST_SCRIPT), "sha256": sha256(TEST_SCRIPT),
        },
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "evaluation": {
            "path": str(evaluation_path), "sha256": sha256(evaluation_path),
        },
        "runtime": {
            "executable": platform.sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    try:
        artifacts = verify_bindings(freeze, config_path, evaluation_path, output)
        config = generator.load_json(config_path)
        generator.validate_config(config)
    except Exception as exc:
        base_result.update({
            "artifacts": [],
            "regimes": {},
            "primary_conditions": {},
            "verdict": "FAIL",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "verdict_effect": (
                "Conditional null hold remains in force; production is unauthorized."
            ),
        })
        atomic_json(report_path, base_result)
        return report_path

    regimes = {}
    primary_id = freeze["scope"]["primary_regime"]
    ordered_regimes = sorted(
        config["design"]["regimes"],
        key=lambda item: item["id"] != primary_id,
    )
    for regime in ordered_regimes:
        try:
            scalar_truths = load_scalar_regime(config, output, regime)
            scalar_result = scalar_nulls(scalar_truths, freeze)
            field_records, geometry, recompute_errors = load_field_regime(
                config, output, regime)
            phase_result = phase_nulls(field_records, geometry, freeze)
            regimes[regime["id"]] = {
                "status": "COMPLETE",
                "scalar": scalar_result,
                "phase_scramble": phase_result,
                "original_field_recompute_relative_error": recompute_errors,
            }
        except Exception as exc:
            regimes[regime["id"]] = {
                "status": "FAIL" if regime["id"] == primary_id else "DIAGNOSTIC_ERROR",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    primary = regimes[primary_id]
    if primary["status"] != "COMPLETE":
        result = dict(base_result)
        result.update({
            "artifacts": artifacts,
            "regimes": regimes,
            "primary_conditions": {"primary_evaluation_complete": False},
            "verdict": "FAIL",
            "verdict_effect": (
                "Conditional null hold remains in force; production is unauthorized."
            ),
        })
        atomic_json(report_path, result)
        return report_path
    alpha = float(freeze["alpha_and_combination"]["per_null_alpha"])
    recompute_limit = float(
        freeze["nulls"]["phase_scramble"]
        ["original_field_recompute_check"]["maximum_relative_error"])
    invariance_limit = float(
        freeze["nulls"]["phase_scramble"]
        ["scramble_invariance_check"]["maximum_relative_error"])
    conditions = {
        "positive_scalar_slope": primary["scalar"]["observed_slope"] > 0.0,
        "positive_field_slope": (
            primary["phase_scramble"]["observed_field_slope"] > 0.0),
        "time_block_p": primary["scalar"]["time_block"]["p_value"] <= alpha,
        "cross_truth_p": (
            primary["scalar"]["cross_truth_reassignment"]["p_value"] <= alpha),
        "phase_scramble_p": primary["phase_scramble"]["p_value"] <= alpha,
        "field_recompute": max(
            primary["original_field_recompute_relative_error"].values()
        ) <= recompute_limit,
        "phase_invariance": max(
            primary["phase_scramble"]
            ["maximum_invariance_relative_error"].values()
        ) <= invariance_limit,
    }
    result = dict(base_result)
    result.update({
        "artifacts": artifacts,
        "regimes": regimes,
        "primary_conditions": conditions,
        "verdict": "PASS" if all(conditions.values()) else "FAIL",
        "verdict_effect": (
            "Clears only the conditional null hold; does not authorize production."
        ),
    })
    atomic_json(report_path, result)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--ratification", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = generator.load_json(freeze_path)
    validate_freeze(freeze)
    verify_execution_bindings(freeze)
    if not args.execute:
        print(json.dumps({
            "status": "READY_FOR_PI_RATIFICATION",
            "freeze": str(freeze_path),
            "freeze_sha256": sha256(freeze_path),
            "pilot_nulls_evaluated": False,
            "production_authorized": False,
        }, indent=2))
        return
    if args.ratification is None:
        raise ValueError("--execute requires --ratification")
    target = execute(
        freeze_path,
        args.ratification.resolve(),
        args.config.resolve(),
        args.output.resolve(),
        args.evaluation.resolve(),
        args.report.resolve(),
    )
    print(target)


if __name__ == "__main__":
    main()
