#!/usr/bin/env python3
"""Phase C: fixed-shell aggregation audit; no model integration or parameter fitting.

Prerequisite: the matching phase-A input/reconstruction gate passed. All inferred
shell clocks are descriptive. The optional a posteriori bound describes ONLY the
piecewise-linear interpolation of archived aggregate samples, not the unknown
continuous SQG trajectory or an independently predictive closure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from audit_dependencies import primary_scope
from audit_io import (
    iter_cluster_paths, read_config, sha256, shell_moments_from_path,
)

PRIMARY_BAND = (10, 20)
INFERRED_INTERIOR = (0.15, 0.85)
ARITHMETIC_RTOL = 1e-10
ARITHMETIC_ATOL_FACTOR = 1e-12
CADENCE_RTOL = 1e-8
CADENCE_ATOL = 1e-10  # archive time units, only an equality-of-cadence criterion
GL_ORDERS = (16, 32)
SPEC_VERSION = "npg-phase-c-fixed-shells-v1-20260911"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, obj: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False) + "\n")


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def stats(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0}
    return {"count": int(x.size), "mean": float(np.mean(x)),
            "median": float(np.median(x)), "min": float(np.min(x)),
            "max": float(np.max(x)), "rms": rms(x),
            "median_absolute": float(np.median(np.abs(x)))}


def centered_derivative(times: np.ndarray, values: np.ndarray,
                        span: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Unsmooth centered secant; omit endpoints and cadence transitions.

    span=2 is a wider centered secant on five equal-cadence support points,
    NOT a fourth-order finite difference and NOT a smoothing operation.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or values.shape[0] != len(times):
        raise ValueError("time/array shape mismatch")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    out = np.full_like(values, np.nan, dtype=float)
    support = np.zeros(len(times), dtype=bool)
    for i in range(span, len(times) - span):
        steps = np.diff(times[i-span:i+span+1])
        if np.allclose(steps, steps[0], rtol=CADENCE_RTOL, atol=CADENCE_ATOL):
            out[i] = (values[i+span] - values[i-span]) / (times[i+span] - times[i-span])
            support[i] = True
    return out, support


def cancellation_summary(terms: np.ndarray, characteristic: float) -> dict[str, Any]:
    terms = np.asarray(terms, dtype=float)
    good = np.all(np.isfinite(terms), axis=1)
    terms = terms[good]
    if not len(terms):
        return {"count": 0}
    signed = np.sum(terms, axis=1)
    absolute = np.sum(np.abs(terms), axis=1)
    noise = ARITHMETIC_ATOL_FACTOR * characteristic
    active = absolute > noise
    fraction = np.full(len(terms), np.nan)
    fraction[active] = 1.0 - np.abs(signed[active]) / absolute[active]
    opposed = (np.any(terms > noise, axis=1) & np.any(terms < -noise, axis=1))
    return {"count": len(terms), "signed_sum": stats(signed),
            "sum_absolute_terms": stats(absolute),
            "cancellation_fraction_1_minus_abs_sum_over_sumabs": stats(fraction),
            "fraction_with_opposing_signs_above_arithmetic_floor": float(np.mean(opposed)),
            "arithmetic_sign_floor": noise,
            "interpretation": "descriptive signed terms; arithmetic floor is not a physical significance threshold"}


def weighted_defect_integral(a0: float, a1: float, dt: float,
                             lam: float, order: int) -> float:
    """Integral exp(lam*(dt-s))*abs(A_PL' - lam*A_PL*(1-A_PL)).

    Split at exact quadratic sign-change roots. Gauss-Legendre convergence
    comparisons are numerical error estimates, not rigorous enclosures.
    """
    if dt <= 0:
        raise ValueError("nonpositive interval")
    slope = (a1 - a0) / dt
    if lam == 0:
        return abs(a1 - a0)
    coefficients = np.array([lam*slope*slope,
                             -lam*slope*(1-2*a0),
                             slope-lam*a0*(1-a0)])
    trimmed = np.trim_zeros(coefficients, trim="f")
    roots = np.roots(trimmed) if len(trimmed) > 1 else np.array([])
    cuts = [0.0, dt]
    for root in roots:
        if abs(float(np.imag(root))) <= 1e-10 * max(1.0, abs(float(np.real(root)))):
            r = float(np.real(root))
            if 0 < r < dt:
                cuts.append(r)
    cuts = sorted(set(cuts))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = 0.0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        s = (hi-lo)*0.5*nodes + (hi+lo)*0.5
        amplitude = a0 + slope*s
        defect = slope - lam*amplitude*(1-amplitude)
        with np.errstate(over="ignore", invalid="ignore"):
            result += (hi-lo)*0.5*float(np.sum(weights*np.exp(lam*(dt-s))*np.abs(defect)))
    return result


def sampled_path_bound(times: np.ndarray, amplitude: np.ndarray,
                       lam: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """A posteriori bound for the specified piecewise-linear sampled path only."""
    nt = len(times)
    names = ("pl_segment", "pl_reference", "pl_error", "pl_bound_32",
             "pl_bound_16", "pl_quadrature_difference", "pl_dt")
    arrays = {name: np.full(nt, np.nan) for name in names}
    if not np.isfinite(lam) or lam < 0:
        return arrays, {"status": "unavailable", "reason": "nonfinite or negative independent band reference rate"}
    eligible = np.isfinite(amplitude) & (amplitude >= 0) & (amplitude <= 1) & (times <= 12.0 + 1e-10)
    segments = []
    start = 0
    while start < nt:
        if not eligible[start]:
            start += 1
            continue
        stop = start + 1
        while stop < nt and eligible[stop] and np.isclose(times[stop]-times[stop-1], 0.1, rtol=CADENCE_RTOL, atol=CADENCE_ATOL):
            stop += 1
        if stop - start < 2:
            start = stop
            continue
        sid = len(segments)
        a0, t0 = float(amplitude[start]), float(times[start])
        b16 = b32 = 0.0
        overflow = False
        for i in range(start, stop):
            elapsed = times[i]-t0
            if a0 == 0 or a0 == 1:
                reference = a0
            else:
                reference = a0/(a0 + (1-a0)*math.exp(-lam*elapsed))
            if i > start:
                dt = float(times[i]-times[i-1])
                grow = math.exp(lam*dt) if lam*dt < 709 else float("inf")
                b16 = grow*b16 + weighted_defect_integral(float(amplitude[i-1]), float(amplitude[i]), dt, lam, GL_ORDERS[0])
                b32 = grow*b32 + weighted_defect_integral(float(amplitude[i-1]), float(amplitude[i]), dt, lam, GL_ORDERS[1])
                arrays["pl_dt"][i] = dt
            arrays["pl_segment"][i] = sid
            arrays["pl_reference"][i] = reference
            arrays["pl_error"][i] = abs(amplitude[i]-reference)
            arrays["pl_bound_16"][i] = b16
            arrays["pl_bound_32"][i] = b32
            arrays["pl_quadrature_difference"][i] = abs(b32-b16)
            overflow = overflow or not (np.isfinite(b16) and np.isfinite(b32))
        indices = slice(start, stop)
        roundoff = 128*np.finfo(float).eps*(1+np.abs(arrays["pl_bound_32"][indices]))
        numerical_allowance = arrays["pl_quadrature_difference"][indices] + roundoff
        violations = arrays["pl_error"][indices] > arrays["pl_bound_32"][indices] + numerical_allowance
        segments.append({"segment": sid, "start_index": start, "stop_index_exclusive": stop,
                         "start_time": t0, "end_time": float(times[stop-1]),
                         "overflow": overflow,
                         "sampled_max_error": float(np.max(arrays["pl_error"][indices])),
                         "sampled_max_bound": float(np.max(arrays["pl_bound_32"][indices])),
                         "max_quadrature_difference_estimate": float(np.max(arrays["pl_quadrature_difference"][indices])),
                         "sampled_bound_below_error_beyond_numerical_allowance": int(np.count_nonzero(violations))})
        start = stop
    return arrays, {"status": "auxiliary_sampled_path_only" if segments else "unavailable_no_contiguous_domain",
                    "eligible_amplitude_sample_count": int(np.sum(eligible)), "segments": segments,
                    "reference_initialization": "A at first point of each contiguous eligible interval; no half-height alignment",
                    "interpolation": "piecewise linear only on consecutive 0.1-TU samples through t<=12; sparse gaps excluded",
                    "quadrature": "sign-root split Gauss-Legendre 16 versus 32; difference is convergence estimate, not certified error",
                    "physical_continuous_trajectory_bound": "unavailable: interpolation and derivative error envelope unknown",
                    "predictive_closure_bound": "unavailable: independent shell clocks and shell closure residuals unmeasured"}


def operator_decomposition(times: np.ndarray, sk: np.ndarray, a: np.ndarray,
                           w: np.ndarray, A: np.ndarray, V: np.ndarray,
                           lam: float, operators: dict[str, np.ndarray],
                           table: dict[str, np.ndarray]) -> dict[str, Any]:
    """Snapshot operator tendencies in the unforced primary band.

    These are evaluations of the continuous spectral operators on archived
    rounded states, not finite differences, fitted rates, or new integration.
    Storage precision remains distinct from finite-sum arithmetic error.
    """
    needed = ("P_linear_shell", "P_tl_nonlinear_shell", "P_pure_error_shell", "Sdot_deterministic_shell")
    if any(key not in operators for key in needed):
        return {"status": "unavailable", "reason": "verified cache lacks reconstructed shell production or Sdot operators"}
    values = {key: np.asarray(operators[key], dtype=float) for key in needed}
    if any(value.shape != sk.shape for value in values.values()):
        raise ValueError("operator shell cache shape mismatch")
    ptotal = values["P_linear_shell"]+values["P_tl_nonlinear_shell"]+values["P_pure_error_shell"]
    sdot = values["Sdot_deterministic_shell"]
    sb = np.sum(sk, axis=1)
    sbdot = np.sum(sdot, axis=1)
    adot = (ptotal-a*sdot)/sk
    wdot = (sdot-w*sbdot[:,None])/sb[:,None]
    Adot = np.sum(ptotal, axis=1)/sb-A*sbdot/sb
    H = np.sum(wdot*(a-A[:,None]), axis=1)
    shape = -lam*V
    qref = adot-lam*a*(1-a)
    qmean = np.sum(w*qref, axis=1)
    qrms = np.sqrt(np.sum(w*qref*qref, axis=1))
    defect = Adot-lam*A*(1-A)
    reconstruction = shape+H+qmean
    error = defect-reconstruction
    product_error = Adot-np.sum(w*adot, axis=1)-H
    weight_sum_error = np.sum(wdot,axis=1)
    table.update(operator_A_derivative=Adot, operator_weight_term=H,
                 operator_reference_shape_term=shape,
                 operator_reference_shell_residual_mean=qmean,
                 operator_reference_shell_residual_rms=qrms,
                 operator_observed_total_defect=defect,
                 operator_reference_reconstructed_defect=reconstruction,
                 operator_arithmetic_reconstruction_error=error,
                 operator_arithmetic_product_rule_error=product_error,
                 operator_weight_derivative_sum_error=weight_sum_error,
                 operator_S_band_derivative=sbdot,
                 operator_normalization_term=-A*sbdot/sb,
                 operator_linear_over_S=np.sum(values["P_linear_shell"],axis=1)/sb,
                 operator_tl_nonlinear_over_S=np.sum(values["P_tl_nonlinear_shell"],axis=1)/sb,
                 operator_pure_error_over_S=np.sum(values["P_pure_error_shell"],axis=1)/sb)
    windows={"dense_t_le_12":times<=12.0+1e-10, "all_125_snapshot_records":np.ones(len(times),dtype=bool)}
    summaries={}
    for name,mask in windows.items():
        characteristic=max(rms(Adot[mask]),abs(lam)*rms((A*(1-A))[mask]),np.finfo(float).tiny)
        magnitude=np.abs(shape)+np.abs(H)+np.abs(qmean)
        tol=ARITHMETIC_ATOL_FACTOR*characteristic+ARITHMETIC_RTOL*magnitude
        good=mask & np.isfinite(error) & np.isfinite(tol)
        passed=bool(np.all(np.abs(error[good])<=tol[good])) if np.any(good) else None
        if passed is False:
            raise ArithmeticError("snapshot operator common-reference finite-sum identity failed")
        summaries[name]={"sample_count":int(np.sum(mask)),
            "characteristic_tendency":characteristic,
            "reference_shape_term":stats(shape[mask]), "weight_term":stats(H[mask]),
            "weighted_q_ref":stats(qmean[mask]), "weighted_rms_q_ref":stats(qrms[mask]),
            "operator_total_defect":stats(defect[mask]),
            "arithmetic_reconstruction_pass":passed,
            "arithmetic_reconstruction_error":stats(error[mask]),
            "arithmetic_product_rule_error":stats(product_error[mask]),
            "weight_derivative_sum_error":stats(weight_sum_error[mask]),
            "q_ref_per_shell":[stats(qref[mask,k]) for k in range(qref.shape[1])],
            "signed_cancellation":cancellation_summary(np.column_stack([shape,H,qmean])[mask],characteristic)}
    comparisons={}
    for span in (1,2):
        for label,op,fdname in (("A_derivative",Adot,"A_derivative"),
                                ("weight_term",H,"weight_term"),
                                ("total_defect",defect,"observed_total_defect"),
                                ("weighted_q_ref",qmean,"reference_shell_residual_mean")):
            diff=op-table[f"{fdname}_span{span}"]
            table[f"operator_minus_fd_{label}_span{span}"]=diff
            comparisons[f"{label}_span{span}"]=stats(diff[times<=12.0+1e-10])
    return {"status":"available independent-common-reference residual at archived primary-band states",
            "formula":"da_op=(P_linear+P_tl_nonlinear+P_pure_error-a*Sdot)/S; dw_op=(Sdot-w*Sdot_band)/S_band",
            "reference":"independent archived band lambda0, prescribed common local hypothesis; no inferred shell rate",
            "support":"fixed unforced shells10..20; no direct additive forcing or Ito term in these modal coefficients",
            "precision":"operators evaluated in complex128 on archived complex64 states; moments preserve historical float32 products, and field reconstruction tolerance applies",
            "windows":summaries,"operator_minus_finite_difference_dense":comparisons,
            "physical_continuous_trajectory_bound":"unavailable: intersnapshot derivative/interpolation error envelope unmeasured"}


def member_audit(times: np.ndarray, et: np.ndarray, ef: np.ndarray,
                 cross: np.ndarray, epsilon: np.ndarray,
                 lam: float, member: int, operators: dict[str, np.ndarray] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sk = et + ef
    valid = np.all(np.isfinite(sk) & (sk > 0), axis=1)
    sb = np.sum(sk, axis=1)
    a = np.divide(epsilon, sk, out=np.full_like(epsilon, np.nan), where=sk > 0)
    w = np.divide(sk, sb[:, None], out=np.full_like(sk, np.nan), where=sb[:, None] > 0)
    a[~valid] = np.nan
    w[~valid] = np.nan
    A = np.sum(w*a, axis=1)
    exact_A = np.sum(epsilon, axis=1)/sb
    V = np.sum(w*(a-A[:, None])**2, axis=1)
    rho = np.divide(cross, np.sqrt(et*ef), out=np.full_like(cross, np.nan), where=(et*ef) > 0)
    wt = et/np.sum(et, axis=1)[:, None]
    proxy = np.sum(wt*(1-rho), axis=1)
    table = {"time": times, "member": np.full(len(times), member), "A_exact": A,
             "S_band": sb, "shape_variance": V,
             "truth_weighted_correlation_proxy_fixed_support": proxy,
             "exact_minus_correlation_proxy": A-proxy,
             "coordinate_arithmetic_error": A-exact_A,
             "all_shells_01": np.all((a >= 0) & (a <= 1), axis=1).astype(float)}
    summaries = {}
    for span in (1, 2):
        da, support = centered_derivative(times, a, span)
        dw, _ = centered_derivative(times, w, span)
        dA, _ = centered_derivative(times, A, span)
        support &= times <= 12.0 + 1e-10
        da[~support] = np.nan
        dw[~support] = np.nan
        dA[~support] = np.nan
        direct = np.sum(w*da, axis=1)
        H = np.sum(dw*(a-A[:, None]), axis=1)
        product = dA-direct-H
        total = dA-lam*A*(1-A)
        interior = support & valid & np.all((a >= INFERRED_INTERIOR[0]) & (a <= INFERRED_INTERIOR[1]), axis=1)
        f = a*(1-a)
        q_reference = da-lam*f
        q_reference_mean = np.sum(w*q_reference, axis=1)
        q_reference_rms = np.sqrt(np.sum(w*q_reference*q_reference, axis=1))
        reference_shape = -lam*V
        reference_partial = reference_shape+H+q_reference_mean
        reference_complete = reference_partial+product
        reference_error = total-reference_complete
        inferred = np.divide(da, f, out=np.full_like(da, np.nan), where=(f != 0))
        inferred[~interior] = np.nan
        lbar = np.sum(w*inferred, axis=1)
        fbar = np.sum(w*f, axis=1)
        cov = np.sum(w*(inferred-lbar[:, None])*(f-fbar[:, None]), axis=1)
        shape = -lbar*V
        drift = (lbar-lam)*A*(1-A)
        partial = drift+shape+cov
        complete_sampled = partial+H+product
        local_rhs = lbar*A*(1-A)+shape+cov
        local_error = direct-local_rhs
        recon_error = total-complete_sampled
        scale_values = np.array([rms(dA), abs(lam)*rms(A*(1-A)), np.finfo(float).tiny])
        characteristic = float(np.max(scale_values[np.isfinite(scale_values)]))
        term_magnitude = np.abs(lbar*A*(1-A))+np.abs(shape)+np.abs(cov)
        tolerance = ARITHMETIC_ATOL_FACTOR*characteristic+ARITHMETIC_RTOL*term_magnitude
        eligible = interior & np.isfinite(local_error) & np.isfinite(tolerance)
        checks = np.abs(local_error[eligible]) <= tolerance[eligible]
        defect_terms = np.column_stack([drift, shape, cov, H, product])
        no_product_terms = np.column_stack([drift, shape, cov, H])
        suffix = f"_span{span}"
        reference_magnitude = np.abs(reference_shape)+np.abs(H)+np.abs(q_reference_mean)+np.abs(product)
        reference_tol = ARITHMETIC_ATOL_FACTOR*characteristic+ARITHMETIC_RTOL*reference_magnitude
        reference_good = np.isfinite(reference_error) & np.isfinite(reference_tol)
        reference_checks = np.abs(reference_error[reference_good]) <= reference_tol[reference_good]
        if (len(checks) and not np.all(checks)) or (len(reference_checks) and not np.all(reference_checks)):
            raise ArithmeticError("finite-sum aggregation identity exceeds frozen arithmetic tolerance")
        derived = {"reference_shape_term": reference_shape,
                   "reference_shell_residual_mean": q_reference_mean,
                   "reference_shell_residual_rms": q_reference_rms,
                   "reference_partial_defect_without_fd_residual": reference_partial,
                   "reference_complete_sampled_defect": reference_complete,
                   "reference_arithmetic_reconstruction_error": reference_error,
                   "derivative_support": support.astype(float), "inferred_interior_support": interior.astype(float),
                   "A_derivative": dA, "weighted_shell_derivative": direct, "weight_term": H,
                   "fd_product_rule_residual": product, "observed_total_defect": total,
                   "inferred_lambda_mean": lbar, "inferred_clock_drift": drift,
                   "inferred_shape_term": shape, "inferred_clock_shape_covariance": cov,
                   "inferred_partial_defect": partial, "sampled_complete_reconstruction": complete_sampled,
                   "arithmetic_local_identity_error": local_error, "arithmetic_defect_reconstruction_error": recon_error}
        table.update({key+suffix: value for key, value in derived.items()})
        summaries[f"span{span}"] = {
            "eligible_derivative_samples": int(np.sum(support)),
            "whole_row_inferred_interior_samples": int(np.sum(interior)),
            "excluded_inferred_rows": int(np.sum(support & ~interior)),
            "characteristic_tendency": characteristic,
            "arithmetic_local_identity": {"pass": bool(np.all(checks)) if len(checks) else None,
                                         "count": len(checks), "failed": int(np.count_nonzero(~checks)),
                                         "error": stats(local_error[eligible]),
                                         "rtol": ARITHMETIC_RTOL, "atol": ARITHMETIC_ATOL_FACTOR*characteristic},
            "observed_total_defect": stats(total), "weight_term": stats(H),
            "fd_product_rule_residual": stats(product),
            "fd_product_residual_over_characteristic_rms": rms(product)/characteristic,
            "inferred_partial_minus_observed_total": stats(partial-total),
            "inferred_plus_weights_minus_observed_total": stats(partial+H-total),
            "arithmetic_complete_reconstruction_error": stats(recon_error),
            "inferred_shape_clock_cancellation": cancellation_summary(np.column_stack([shape,cov]), characteristic),
            "inferred_terms_plus_weights_cancellation": cancellation_summary(no_product_terms, characteristic),
            "sampled_decomposition_including_fd_residual_cancellation": cancellation_summary(defect_terms, characteristic),
            "terms": {name: stats(derived[name]) for name in ("inferred_clock_drift", "inferred_shape_term", "inferred_clock_shape_covariance")},
            "common_reference": {
                "status": "available sampled residual against independently prescribed common band rate; not measured shell tangent clocks",
                "definition": "q_ref,k=D_t a_k-lambda0*a_k*(1-a_k)",
                "reference_shape_term": stats(reference_shape[support]),
                "weighted_q_ref": stats(q_reference_mean),
                "weighted_rms_q_ref": stats(q_reference_rms),
                "arithmetic_reconstruction_pass": bool(np.all(reference_checks)) if len(reference_checks) else None,
                "arithmetic_reconstruction_error": stats(reference_error),
                "physical_terms_cancellation_without_fd_residual": cancellation_summary(np.column_stack([reference_shape,H,q_reference_mean]), characteristic),
                "sampled_terms_cancellation_with_fd_residual": cancellation_summary(np.column_stack([reference_shape,H,q_reference_mean,product]), characteristic),
                "reference_partial_minus_observed_total": stats(reference_partial-total),
                "per_shell_q_ref": [stats(q_reference[:,k]) for k in range(q_reference.shape[1])],
                "interpretation": "quantifies sampled inadequacy of the common local logistic reference; no attribution to a specific physical mechanism, subject to derivative sensitivity"},
            "identity_status": "algebraic inferred-rate reconstruction only; q=0 by definition, not a physical closure test"}
    table["A_derivative_span2_minus_span1"] = table["A_derivative_span2"]-table["A_derivative_span1"]
    table["product_residual_span2_minus_span1"] = table["fd_product_rule_residual_span2"]-table["fd_product_rule_residual_span1"]
    op_summary = operator_decomposition(times, sk, a, w, A, V, lam, operators or {}, table)
    pl_arrays, pl_summary = sampled_path_bound(times, A, lam)
    table.update(pl_arrays)
    summary = {"member": member, "independent_band_reference_rate": lam,
               "sample_count": len(times), "all_positive_shell_energy_samples": int(np.sum(valid)),
               "exact_energy_coordinate_vs_fixed_support_correlation_proxy": stats(A-proxy),
               "coordinate_arithmetic_error": stats(A-exact_A),
               "all_shells_in_01_samples": int(np.sum(table["all_shells_01"])),
               "finite_difference": summaries,
               "derivative_span2_minus_span1": stats(table["A_derivative_span2_minus_span1"]),
               "auxiliary_sampled_path_bound": pl_summary,
               "snapshot_operator_common_reference": op_summary,
               "unavailable_independent_terms": ["independent shellwise lambda_k", "shell surrogate residual relative to independently measured heterogeneous shell clocks (common-reference q_ref is available)", "continuous-time derivative/interpolation error envelope"],
               "independent_mechanism_decomposition": "unavailable",
               "physical_continuous_trajectory_bound": "unavailable"}
    return summary, table


def write_csv(path: Path, tables: list[dict[str, np.ndarray]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    names = list(tables[0])
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for table in tables:
            for i in range(len(table["time"])):
                writer.writerow([json_safe(table[name][i]) for name in names])


def grouped_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    metrics = ("fd_product_residual_over_characteristic_rms",)
    for regime in sorted({r["regime"] for r in records}):
        selected = [r for r in records if r["regime"] == regime]
        output = {}
        for metric in metrics:
            truths = [float(np.mean([m["finite_difference"]["span1"][metric] for m in r["members"]])) for r in selected]
            output[metric] = {"truth_values_mean_of_members": truths,
                              "median_across_truths": float(np.median(truths)),
                              "leave_one_truth_out_medians": [float(np.median(truths[:i]+truths[i+1:])) for i in range(len(truths))] if len(truths)>1 else [],
                              "truth_count": len(truths)}
        result[regime] = output
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster", default=None)
    parser.add_argument("--physical-output", type=Path, required=True, help="phase A/B output root containing result.json satisfying primary dependencies and hashed shell_moments.npz files")
    args = parser.parse_args()
    root = args.model_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read_config(root)
    burn = float(config["lyapunov"]["burnin_tu"])
    records = []
    script_hash = sha256(Path(__file__))
    for path in iter_cluster_paths(root, args.cluster):
        phase_dir = args.physical_output / path.parent.parent.name / path.stem
        phase_record_path = phase_dir / "result.json"
        if not phase_record_path.is_file():
            raise FileNotFoundError(f"phase A/B record required before C: {phase_record_path}")
        phase_record = json.loads(phase_record_path.read_text())
        scope_gate = primary_scope(phase_record)
        if phase_record.get("phase_A_passed") is not True or not scope_gate["eligible"]:
            raise RuntimeError(f"primary-scope phase A/B prerequisite failed: {phase_record_path}; {scope_gate['reason']}")
        source_hash = sha256(path)
        if phase_record.get("source_sha256") != source_hash:
            raise ValueError(f"raw source hash differs from phase A/B record: {path}")
        source_relative = phase_record.get("source_path")
        if not source_relative or (root / source_relative).resolve() != path.resolve():
            raise ValueError("phase A/B source path does not match selected raw archive")
        cache = phase_dir / phase_record.get("shell_moments_path", "shell_moments.npz")
        if not cache.is_file() or sha256(cache) != phase_record.get("shell_moments_sha256"):
            raise ValueError(f"verified moment cache absent or hash mismatch: {cache}")
        with np.load(cache, allow_pickle=False) as saved:
            moments = {key: np.asarray(saved[key]) for key in ("Et", "Ef", "C", "epsilon", "shells", "lambda_time", "lambda_energy_rate")}
            moments["times"] = np.asarray(saved["field_time"])
            moments["metadata"] = json.loads(str(saved["metadata_json"].item()))
            for key in ("P_linear_shell", "P_tl_nonlinear_shell", "P_pure_error_shell", "Sdot_deterministic_shell"):
                if key in saved.files:
                    moments[key] = np.asarray(saved[key])
        times = np.asarray(moments["times"], dtype=float)
        metadata = moments["metadata"]
        shells = np.asarray(moments["shells"])
        include = (shells >= PRIMARY_BAND[0]) & (shells <= PRIMARY_BAND[1])
        if not np.array_equal(shells[include], np.arange(PRIMARY_BAND[0], PRIMARY_BAND[1]+1)):
            raise ValueError("primary fixed-shell support is incomplete")
        if metadata.get("lambda_is_energy_growth_rate") is not True:
            raise ValueError("archive does not certify the tangent clock as an energy-growth rate")
        pbi = int(metadata["primary_band_index"])
        if tuple(metadata["bands"][pbi]) != PRIMARY_BAND:
            raise ValueError("independent tangent reference band differs from primary shell partition")
        ltimes = np.asarray(moments["lambda_time"], dtype=float)
        lrates = np.asarray(moments["lambda_energy_rate"], dtype=float)
        lmask = (ltimes >= burn) & np.isfinite(lrates[:,pbi])
        lam = float(np.mean(lrates[lmask,pbi])) if np.any(lmask) else float("nan")
        summaries, tables = [], []
        for member in range(moments["Ef"].shape[1]):
            summary, table = member_audit(times, moments["Et"][:,include],
                                         moments["Ef"][:,member,include], moments["C"][:,member,include],
                                         moments["epsilon"][:,member,include], lam, member,
                                         {key:moments[key][:,member,include] for key in ("P_linear_shell", "P_tl_nonlinear_shell", "P_pure_error_shell", "Sdot_deterministic_shell") if key in moments})
            summaries.append(summary)
            tables.append(table)
        stem = f"{metadata['regime']}__truth_{int(metadata['truth_index']):02d}"
        record = {"schema": SPEC_VERSION, "regime": metadata["regime"],
                  "truth_index": int(metadata["truth_index"]), "input_path": str(path),
                  "input_sha256": source_hash, "script_sha256": script_hash,
                  "verified_physical_cache": str(cache),
                  "phase_A_B_record": str(phase_record_path),
                  "primary_source_dependency_gate": scope_gate,
                  "allowed_primary_source_status": "eligible under explicit primary-band dependencies; not validation of the entire numerical trajectory",
                  "all_band_implementation_checks_passed": bool(scope_gate["all_band_passed"]),
                  "failed_nonprimary_discrete_checks": scope_gate["failed_discrete_checks"],
                  "phase_A_B_record_sha256": sha256(phase_record_path),
                  "verified_physical_cache_sha256": phase_record["shell_moments_sha256"],
                  "primary_fixed_shells": list(range(PRIMARY_BAND[0],PRIMARY_BAND[1]+1)),
                  "independent_lambda_burnin_tu": burn,
                  "independent_lambda_sample_count": int(np.sum(lmask)),
                  "members": summaries, "timeseries_csv": stem+".csv",
                  "additional_bands": "not evaluated in phase C; physical band audit supplied separately"}
        write_csv(output/(stem+".csv"), tables)
        write_json(output/(stem+".json"), record)
        records.append(record)
        del moments, tables
        print(json.dumps({"completed": stem, "members": len(summaries)}), flush=True)
    write_json(output/"aggregation_summary.json", {"schema": SPEC_VERSION,
               "clusters": [{"regime":r["regime"],"truth_index":r["truth_index"],"input_sha256":r["input_sha256"]} for r in records],
               "grouped_sensitivity": grouped_summary(records),
               "all_input_sources_pass_all_band_implementation_checks": all(r["all_band_implementation_checks_passed"] for r in records),
               "source_scope": "primary-band dependent audit; upstream nonprimary failures retained in each cluster record",
               "status": "retrospective descriptive audit; independent shell mechanism and certified physical bound unavailable"})


if __name__ == "__main__":
    main()
