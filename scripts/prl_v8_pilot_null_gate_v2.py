#!/usr/bin/env python3
"""Memory-safe, versioned PRL v8 pilot null-gate evaluator."""

from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

import prl_v8_pilot_null_gate as v1
import prl_v8_two_regime as generator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPT = Path(__file__).resolve()
TEST_SCRIPT = HERE / "test_prl_v8_pilot_null_gate_v2.py"
DEFAULT_FREEZE = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_null_gate_pass_criteria_freeze_proposal_20260719.json"
)
DEFAULT_EXCEPTION = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_null_gate_memory_safe_exception_20260720.json"
)
DEFAULT_REPLACEMENT_FREEZE = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_null_gate_v2_execution_freeze_proposal_20260720.json"
)
DEFAULT_CONFIG = HERE / "prl_v8_gate5_nonconfirmatory_config_20260715.json"
DEFAULT_OUTPUT = HERE / "out" / "prl_v8_gate5_nonconfirmatory_20260715"
DEFAULT_EVALUATION = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_gate5_nonconfirmatory_20260715_per_cluster"
    / "prl_v8_pilot_evaluation_v2_8of8_f5e91c6d608c_be29ceeaa19f.json"
)
DEFAULT_CACHE = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_pilot_null_gate_field_cache_v2_20260720"
)
DEFAULT_REPORT = (
    ROOT / "01_PRL" / "Codex_work" / "reports"
    / "v8_pilot_null_gate_v2_20260720.json"
)
BASE_FREEZE_SHA256 = "7e780c587db824ddebbd4563ffa3affc08fe1ebebadb730a830df8753f59dc8d"
EXCEPTION_SHA256 = "1d64df5aa13fade093d772ff39e8c4bc895bfae3f5b4613a1a45945e3acf5a08"
SCHEMA = "prl-v8-pilot-null-gate-result-2"
REPLACEMENT_SCHEMA = "prl-v8-pilot-null-gate-execution-freeze-2"
RATIFICATION_SCHEMA = "prl-v8-pilot-null-gate-v2-ratification-1"


def atomic_npy(path: Path, array: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite cache array: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def preflight_result_path(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    occupied = [candidate for candidate in (path, temporary) if candidate.exists()]
    if occupied:
        raise FileExistsError(
            "refusing evaluation because result path is occupied: "
            + ", ".join(str(candidate) for candidate in occupied)
        )


def validate_cache_manifest(
    manifest: dict[str, Any],
    source: Path,
    target_dir: Path,
    expected_source_hash: str,
) -> None:
    if manifest.get("schema") != "prl-v8-null-field-cache-1":
        raise ValueError("wrong field-cache manifest schema")
    if Path(manifest.get("source", "")).resolve() != source.resolve():
        raise ValueError("field-cache manifest has the wrong source path")
    if manifest.get("source_sha256") != expected_source_hash:
        raise ValueError("existing field cache has the wrong source binding")
    rows = manifest.get("arrays")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("field-cache manifest must contain exactly two arrays")
    expected_keys = {"field_truth_hat", "field_forecast_hat"}
    if {row.get("key") for row in rows} != expected_keys:
        raise ValueError("field-cache manifest has the wrong array keys")
    paths = [Path(row.get("path", "")).resolve() for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("field-cache manifest paths must be unique")
    for row, path in zip(rows, paths):
        if path.parent != target_dir.resolve():
            raise ValueError("field-cache array escaped its artifact directory")
        if not path.is_file():
            raise FileNotFoundError(f"missing field-cache array: {path}")
        if path.stat().st_size != int(row.get("bytes", -1)):
            raise ValueError(f"field-cache byte-size mismatch: {path}")
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if list(mapped.shape) != row.get("shape"):
                raise ValueError(f"field-cache shape mismatch: {path}")
            if str(mapped.dtype) != row.get("dtype"):
                raise ValueError(f"field-cache dtype mismatch: {path}")
        finally:
            del mapped
        if v1.sha256(path) != row.get("sha256"):
            raise ValueError(f"field-cache hash mismatch: {path}")


def cache_artifact(source: Path, target_dir: Path,
                   expected_source_hash: str) -> dict[str, Any]:
    manifest_path = target_dir / "manifest.json"
    if manifest_path.exists():
        manifest = generator.load_json(manifest_path)
        validate_cache_manifest(
            manifest, source, target_dir, expected_source_hash)
        return manifest
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(
            f"incomplete field cache exists without a manifest: {target_dir}")
    if v1.sha256(source) != expected_source_hash:
        raise ValueError(f"source artifact hash mismatch: {source}")

    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with np.load(source, allow_pickle=False) as data:
        for key in ("field_truth_hat", "field_forecast_hat"):
            array = np.asarray(data[key])
            path = target_dir / f"{key}.npy"
            atomic_npy(path, array)
            rows.append({
                "key": key,
                "path": str(path.resolve()),
                "sha256": v1.sha256(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": path.stat().st_size,
            })
            del array
    manifest = {
        "schema": "prl-v8-null-field-cache-1",
        "source": str(source.resolve()),
        "source_sha256": expected_source_hash,
        "arrays": rows,
    }
    v1.atomic_json(manifest_path, manifest)
    validate_cache_manifest(
        manifest, source, target_dir, expected_source_hash)
    return manifest


def prepare_regime_field_cache(
    output: Path,
    evaluation: dict[str, Any],
    cache_root: Path,
    regime_id: str,
) -> list[dict[str, Any]]:
    manifests = []
    for row in evaluation["regimes"][regime_id]["truths"]:
        truth_index = int(row["truth_index"])
        source = (
            output / regime_id / "clusters"
            / f"truth_{truth_index:02d}.npz"
        )
        target = cache_root / regime_id / f"truth_{truth_index:02d}"
        print(
            f"CACHE {regime_id} truth={truth_index:02d}",
            flush=True,
        )
        manifests.append(
            cache_artifact(source, target, row["sha256"]))
    return manifests


def cache_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    return {row["key"]: Path(row["path"]) for row in manifest["arrays"]}


def observed_field_records(
    config: dict[str, Any],
    output: Path,
    regime: dict[str, Any],
    manifests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, float]]:
    regime_id = regime["id"]
    bands = generator.expanded_bands(config, int(regime["n"]))
    primary = bands.index(tuple(config["design"]["primary_band"]))
    geometry = v1.spectral_geometry(
        int(regime["n"]), float(regime["nu"]), float(config["model"]["mu"]),
        bands, primary,
    )
    records = []
    recomputed = {
        key: [] for key in ("C", "epsilon", "P_linear", "P_tl_nonlinear")
    }
    archived = {key: [] for key in recomputed}
    for truth_index, manifest in enumerate(manifests):
        source = (
            output / regime_id / "clusters" / f"truth_{truth_index:02d}.npz"
        )
        paths = cache_paths(manifest)
        truth_fields = np.load(paths["field_truth_hat"], mmap_mode="r")
        forecast_fields = np.load(paths["field_forecast_hat"], mmap_mode="r")
        with np.load(source, allow_pickle=False) as data:
            field_time = np.asarray(data["field_time"], dtype=float)
            scalar_indices = v1.matching_scalar_indices(
                data["scalar_time"], field_time)
            burn = data["lambda_time"] >= float(config["lyapunov"]["burnin_tu"])
            lam = float(np.mean(data["lambda_energy_rate"][burn, primary]))
            snapshots = []
            for field_index, scalar_index in enumerate(scalar_indices):
                diagnostics = v1.field_diagnostics(
                    np.asarray(truth_fields[field_index]),
                    np.asarray(forecast_fields[field_index]),
                    geometry,
                )
                for key in recomputed:
                    recomputed[key].extend(diagnostics[key].tolist())
                    archived[key].extend(
                        np.asarray(
                            data[key][scalar_index, :, primary],
                            dtype=float,
                        ).tolist()
                    )
                snapshots.append({
                    "time": float(field_time[field_index]),
                    "diagnostics": diagnostics,
                })
        records.append({
            "lambda": lam,
            "snapshots": snapshots,
            "cache_manifest": manifest,
        })
        del truth_fields, forecast_fields
    errors = {
        key: v1.relative_l2(np.asarray(archived[key]), np.asarray(recomputed[key]))
        for key in recomputed
    }
    return records, geometry, errors


def phase_nulls_streaming(
    records: list[dict[str, Any]],
    geometry: dict[str, np.ndarray],
    freeze: dict[str, Any],
    progress_every: int = 25,
) -> dict[str, Any]:
    phase_cfg = freeze["nulls"]["phase_scramble"]
    observed_truths = v1.field_truths(records)
    frozen_masks = [
        [np.asarray(member["mask"], dtype=bool).copy() for member in truth]
        for truth in observed_truths
    ]
    observed = v1.weighted_slope(observed_truths)
    rng = np.random.default_rng(int(phase_cfg["seed"]))
    values = np.empty(int(phase_cfg["replicates"]), dtype=float)
    max_invariance = {
        key: 0.0 for key in ("Et", "Ef", "C", "epsilon", "x_C", "rho")
    }
    n = geometry["square"].shape[0]
    for replicate in range(values.size):
        scrambled_records = []
        invariant_sums = {
            key: {"difference": 0.0, "reference": 0.0}
            for key in max_invariance
        }
        for record in records:
            paths = cache_paths(record["cache_manifest"])
            truth_fields = np.load(paths["field_truth_hat"], mmap_mode="r")
            forecast_fields = np.load(
                paths["field_forecast_hat"], mmap_mode="r")
            snapshots = []
            for field_index, snapshot in enumerate(record["snapshots"]):
                phase = v1.hermitian_phase(n, rng)
                diagnostics = v1.field_diagnostics(
                    np.asarray(truth_fields[field_index]) * phase,
                    np.asarray(forecast_fields[field_index]) * phase[None],
                    geometry,
                )
                for key in max_invariance:
                    reference = np.asarray(
                        snapshot["diagnostics"][key], dtype=float)
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
            del truth_fields, forecast_fields
        values[replicate] = v1.weighted_slope(v1.field_truths(
            scrambled_records, frozen_masks=frozen_masks))
        for key, sums in invariant_sums.items():
            error = np.sqrt(sums["difference"]) / max(
                np.sqrt(sums["reference"]), 1e-30)
            if not np.isfinite(error):
                raise ValueError(
                    "nonfinite aggregate phase-scramble invariance error")
            max_invariance[key] = max(max_invariance[key], float(error))
        if (
            progress_every > 0
            and (replicate + 1) % progress_every == 0
        ):
            print(
                f"PHASE progress={replicate + 1}/{values.size}",
                flush=True,
            )
    return {
        "observed_field_slope": observed,
        "replicates": int(values.size),
        "slopes": values.tolist(),
        "p_value": v1.monte_carlo_p(observed, values),
        "maximum_invariance_relative_error": max_invariance,
    }


def validate_replacement_freeze(
    replacement: dict[str, Any],
    base_freeze_path: Path,
    exception_path: Path,
) -> None:
    if replacement.get("schema") != REPLACEMENT_SCHEMA:
        raise ValueError("wrong v2 execution-freeze schema")
    if replacement["base_freeze"]["sha256"] != v1.sha256(base_freeze_path):
        raise ValueError("v2 freeze does not bind the original scientific freeze")
    if replacement["exception"]["sha256"] != v1.sha256(exception_path):
        raise ValueError("v2 freeze does not bind the PI executable exception")
    expected = {
        "evaluator": SCRIPT,
        "structural_tests": TEST_SCRIPT,
    }
    for name, path in expected.items():
        if replacement["execution_bindings"][name]["sha256"] != v1.sha256(path):
            raise ValueError(f"v2 {name} hash mismatch")
    runtime = replacement["execution_bindings"]["interpreter"]
    if Path(runtime["path"]).resolve() != Path(platform.sys.executable).resolve():
        raise ValueError("wrong v2 interpreter")
    if runtime["python_version"] != platform.python_version():
        raise ValueError("wrong v2 Python version")
    if runtime["numpy_version"] != np.__version__:
        raise ValueError("wrong v2 NumPy version")
    if replacement["scientific_changes"] != "none":
        raise ValueError("the v2 exception permits no scientific changes")


def verify_base_runtime(freeze: dict[str, Any]) -> None:
    v1.verify_execution_bindings(freeze)


def validate_v2_ratification(
    path: Path,
    replacement_path: Path,
) -> dict[str, Any]:
    record = generator.load_json(path)
    if record.get("schema") != RATIFICATION_SCHEMA:
        raise ValueError("wrong v2 ratification schema")
    if record.get("ratified") is not True:
        raise ValueError("v2 execution freeze is not PI-ratified")
    if record.get("production_authorized") is not False:
        raise ValueError("v2 ratification cannot authorize production")
    if record["replacement_freeze"]["sha256"] != v1.sha256(replacement_path):
        raise ValueError("v2 ratification does not bind the active replacement freeze")
    return record


def failed_result(base: dict[str, Any], report_path: Path,
                  exc: Exception, regimes: dict[str, Any] | None = None) -> Path:
    payload = copy.deepcopy(base)
    payload.update({
        "regimes": {} if regimes is None else regimes,
        "primary_conditions": {},
        "verdict": "FAIL",
        "failure": {"type": type(exc).__name__, "message": str(exc)},
        "verdict_effect": (
            "Conditional null hold remains in force; production is unauthorized."
        ),
    })
    v1.atomic_json(report_path, payload)
    return report_path


def execute(
    base_freeze_path: Path,
    exception_path: Path,
    replacement_path: Path,
    ratification_path: Path,
    config_path: Path,
    output: Path,
    evaluation_path: Path,
    cache_root: Path,
    report_path: Path,
) -> Path:
    if v1.sha256(base_freeze_path) != BASE_FREEZE_SHA256:
        raise ValueError("original scientific freeze hash changed")
    if v1.sha256(exception_path) != EXCEPTION_SHA256:
        raise ValueError("PI memory-safe exception hash changed")
    preflight_result_path(report_path)
    freeze = generator.load_json(base_freeze_path)
    v1.validate_freeze(freeze)
    verify_base_runtime(freeze)
    replacement = generator.load_json(replacement_path)
    validate_replacement_freeze(
        replacement, base_freeze_path, exception_path)
    ratification = validate_v2_ratification(
        ratification_path, replacement_path)
    base = {
        "schema": SCHEMA,
        "production_authorized": False,
        "base_freeze": {
            "path": str(base_freeze_path),
            "sha256": v1.sha256(base_freeze_path),
        },
        "exception": {
            "path": str(exception_path),
            "sha256": v1.sha256(exception_path),
        },
        "replacement_freeze": {
            "path": str(replacement_path),
            "sha256": v1.sha256(replacement_path),
        },
        "ratification": {
            "path": str(ratification_path),
            "sha256": v1.sha256(ratification_path),
            "ratified_on": ratification["ratified_on"],
        },
        "script": {"path": str(SCRIPT), "sha256": v1.sha256(SCRIPT)},
        "structural_tests": {
            "path": str(TEST_SCRIPT), "sha256": v1.sha256(TEST_SCRIPT),
        },
        "runtime": {
            "executable": platform.sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    try:
        artifacts = v1.verify_bindings(
            freeze, config_path, evaluation_path, output)
        config = generator.load_json(config_path)
        generator.validate_config(config)
        evaluation = generator.load_json(evaluation_path)
    except Exception as exc:
        return failed_result(base, report_path, exc)

    base["artifacts"] = artifacts
    base["field_cache"] = {
        "root": str(cache_root),
        "manifests": {},
    }
    regimes: dict[str, Any] = {}
    primary_id = freeze["scope"]["primary_regime"]
    ordered = sorted(
        config["design"]["regimes"],
        key=lambda item: item["id"] != primary_id,
    )
    for regime in ordered:
        regime_id = regime["id"]
        try:
            regime_manifests = prepare_regime_field_cache(
                output, evaluation, cache_root, regime_id)
            base["field_cache"]["manifests"][regime_id] = regime_manifests
            print(f"SCALAR {regime_id}", flush=True)
            scalar_truths = v1.load_scalar_regime(config, output, regime)
            scalar_result = v1.scalar_nulls(scalar_truths, freeze)
            print(f"FIELD observed {regime_id}", flush=True)
            records, geometry, recompute_errors = observed_field_records(
                config, output, regime, regime_manifests)
            print(f"PHASE start {regime_id}", flush=True)
            phase_result = phase_nulls_streaming(
                records, geometry, freeze)
            regimes[regime_id] = {
                "status": "COMPLETE",
                "scalar": scalar_result,
                "phase_scramble": phase_result,
                "original_field_recompute_relative_error": recompute_errors,
            }
        except Exception as exc:
            regimes[regime_id] = {
                "status": (
                    "FAIL" if regime_id == primary_id
                    else "DIAGNOSTIC_ERROR"
                ),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    primary = regimes[primary_id]
    if primary["status"] != "COMPLETE":
        return failed_result(
            base,
            report_path,
            RuntimeError(primary["error"]["message"]),
            regimes,
        )
    alpha = float(freeze["alpha_and_combination"]["per_null_alpha"])
    phase_cfg = freeze["nulls"]["phase_scramble"]
    recompute_limit = float(
        phase_cfg["original_field_recompute_check"]["maximum_relative_error"])
    invariance_limit = float(
        phase_cfg["scramble_invariance_check"]["maximum_relative_error"])
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
    result = copy.deepcopy(base)
    result.update({
        "regimes": regimes,
        "primary_conditions": conditions,
        "verdict": "PASS" if all(conditions.values()) else "FAIL",
        "verdict_effect": (
            "Clears only the conditional null hold; does not authorize production."
        ),
    })
    v1.atomic_json(report_path, result)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--exception", type=Path, default=DEFAULT_EXCEPTION)
    parser.add_argument(
        "--replacement-freeze", type=Path, default=DEFAULT_REPLACEMENT_FREEZE)
    parser.add_argument("--ratification", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    base_freeze_path = args.base_freeze.resolve()
    exception_path = args.exception.resolve()
    replacement_path = args.replacement_freeze.resolve()
    if v1.sha256(base_freeze_path) != BASE_FREEZE_SHA256:
        raise ValueError("original scientific freeze hash changed")
    if v1.sha256(exception_path) != EXCEPTION_SHA256:
        raise ValueError("PI memory-safe exception hash changed")
    freeze = generator.load_json(base_freeze_path)
    v1.validate_freeze(freeze)
    verify_base_runtime(freeze)
    replacement = generator.load_json(replacement_path)
    validate_replacement_freeze(
        replacement, base_freeze_path, exception_path)
    if not args.execute:
        print(json.dumps({
            "status": "READY_FOR_V2_PI_BINDING",
            "replacement_freeze": str(replacement_path),
            "replacement_freeze_sha256": v1.sha256(replacement_path),
            "pilot_nulls_evaluated": False,
            "production_authorized": False,
        }, indent=2))
        return
    if args.ratification is None:
        raise ValueError("--execute requires --ratification")
    target = execute(
        base_freeze_path,
        exception_path,
        replacement_path,
        args.ratification.resolve(),
        args.config.resolve(),
        args.output.resolve(),
        args.evaluation.resolve(),
        args.cache_root.resolve(),
        args.report.resolve(),
    )
    print(target)


if __name__ == "__main__":
    main()
