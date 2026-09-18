#!/usr/bin/env python3
"""Diagnose already-recorded non-primary allocated-budget failures, read-only.

No thresholds, raw data, or physical-audit outputs are changed. This script only
checks the arithmetic and provenance of failed archived increment balances.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
from audit_io import load_metadata, read_json, sha256, write_json


def rms(values):
    x=np.asarray(values,dtype=np.float64)
    if not np.all(np.isfinite(x)):raise ValueError('Nonfinite diagnostic input')
    return float(np.sqrt(np.mean(x*x)))


def inside(path,root):
    path=Path(path).resolve();root=Path(root).resolve()
    path.relative_to(root)
    return path


def generator_evidence(model_root,expected_hash):
    candidates=(model_root/'evidence/provenance/generator_original.py',
                model_root/'provenance/campaign_scripts/prl_v8_two_regime.py',
                model_root/'provenance/reference_scripts/prl_v8_two_regime.py',
                model_root/'scripts/prl_v8_two_regime.py')
    path=next((p for p in candidates if p.is_file() and sha256(p)==expected_hash),None)
    if path is None:raise ValueError('No copied generator matches the raw archive runner hash')
    lines=path.read_text().splitlines()
    patterns={'allocation_dtype':'budget_accum = mx.zeros(',
              'accumulation':'budget_accum = budget_accum + _allocated_mx',
              'stored_dtype':'budget_components.append(np.asarray(budget_accum, dtype=np.float64))',
              'observed_increment':'budget_observed.append(np.asarray(current - interval_start, dtype=np.float64))'}
    evidence={}
    for label,pattern in patterns.items():
        found=[dict(line=i+1,text=line.strip()) for i,line in enumerate(lines) if pattern in line]
        if len(found)!=1:raise ValueError(f'Expected exactly one source match for {label}, found {len(found)}')
        evidence[label]=found[0]
    return dict(path=str(path.relative_to(model_root)),sha256=expected_hash,lines=evidence)


def diagnose_member(components,observed,energy,member,band,recorded):
    terms=np.asarray(components[1:,:,member,band],dtype=np.float64)
    reference=np.asarray(observed[1:,member,band],dtype=np.float64)
    delta_energy=np.diff(np.asarray(energy[:,member,band],dtype=np.float64))
    if terms.ndim!=2 or terms.shape[1]!=4:raise ValueError(f'Expected four allocated components: {terms.shape}')
    if not np.all(np.isfinite(terms)):raise ValueError('Nonfinite allocated components')
    # Independent explicit summation, rather than the audit's reduction call.
    total64=terms[:,0]+terms[:,1]+terms[:,2]+terms[:,3]
    total_long=np.sum(terms.astype(np.longdouble),axis=1,dtype=np.longdouble)
    total_fsum=np.asarray([math.fsum(map(float,row)) for row in terms])
    exact_errors=[];exact_residuals=[];exact_equal=True
    for row,total,obs in zip(terms,total64,reference):
        exact_total=sum((Fraction.from_float(float(v)) for v in row),Fraction(0))
        error=Fraction.from_float(float(total))-exact_total
        exact_equal=exact_equal and error==0
        exact_errors.append(float(error))
        exact_residuals.append(float(Fraction.from_float(float(obs))-exact_total))
    residual=reference-total64
    reference_scale=rms(reference)
    if reference_scale<=0:raise ValueError('Recorded failed band has zero reference RMS; inspect original scaling')
    relative=rms(residual)/reference_scale
    expected=float(recorded['scaled_rms'])
    if not math.isclose(relative,expected,rel_tol=1e-12,abs_tol=1e-15):
        raise ValueError(f'Independent summation does not reproduce recorded result {relative} vs {expected}')
    sumabs=np.sum(np.abs(terms),axis=1)
    return dict(member=member,band_index=band,band=recorded['band'],
        recorded_threshold=recorded['rtol'],recorded_passed=recorded['passed'],
        recorded_scaled_rms=expected,recomputed_relative_rms=relative,
        reproduces_recorded_scaled_rms=True,absolute_residual_rms=rms(residual),
        observed_increment_rms=reference_scale,
        observed_vs_scalar_energy_difference_rms=rms(reference-delta_energy),
        observed_vs_scalar_energy_difference_max_abs=float(np.max(np.abs(reference-delta_energy))),
        sum_absolute_components_rms=rms(sumabs),
        cancellation_amplification_rms=rms(sumabs)/reference_scale,
        float64_vs_longdouble_sum_max_abs=float(np.max(np.abs(total64.astype(np.longdouble)-total_long))),
        float64_vs_math_fsum_max_abs=float(np.max(np.abs(total64-total_fsum))),
        float64_sum_exact_for_every_interval=bool(exact_equal),
        float64_vs_exact_rational_sum_max_abs=float(np.max(np.abs(exact_errors))),
        exact_rational_residual_rms=rms(exact_residuals),
        exact_rational_relative_rms=rms(exact_residuals)/reference_scale,
        component_dtype=str(components.dtype),observed_dtype=str(observed.dtype),
        stored_components_exactly_float32_representable=bool(np.array_equal(terms,terms.astype(np.float32).astype(np.float64))),
        intervals=int(len(reference)),units='energy per archived output interval',
        interpretation='The recorded tolerance failure is retained. Summation checks locate arithmetic error; they do not validate a failed allocated budget or uniquely identify its timestep-level cause.')


def run(model_root,physical_output):
    records=sorted(physical_output.glob('*/truth_*/result.json'))
    diagnoses=[];inputs=[];sources={};skipped=[];primary_failures=[]
    for path in records:
        result=read_json(path)
        if not result.get('phase_B_executed'):
            skipped.append(dict(path=str(path.relative_to(physical_output)),reason='No completed phase-B record'))
            continue
        failures=[x for x in result['budget']['discrete_budget_check']['per_band_member'] if not x['passed']]
        if not failures:continue
        raw=inside(model_root/result.get('source_path',result['raw_path']),model_root)
        expected=result.get('source_sha256',result['raw_sha256'])
        actual=sha256(raw)
        if actual!=expected:raise ValueError(f'Raw input hash mismatch: {raw}')
        with np.load(raw,allow_pickle=False) as data:
            meta=load_metadata(data)
            primary_name=meta['band_names'][int(meta['primary_band_index'])]
            eligible=[f for f in failures if f['band']!=primary_name]
            primary_failures.extend(dict(regime=result['regime'],truth=result['truth_index'],**f)
                                    for f in failures if f['band']==primary_name)
            if not eligible:continue
            components=data['budget_components'];observed=data['budget_observed'];energy=data['epsilon']
            source_hash=meta['runner_sha256']
            if source_hash not in sources:sources[source_hash]=generator_evidence(model_root,source_hash)
            record_info=dict(path=str(path.relative_to(physical_output)),sha256=sha256(path),
                phase_A_passed=result['phase_A_passed'],implementation_checks_passed=result['implementation_checks_passed'],
                audit_code_sha256=result.get('code_sha256'),source_path=str(raw.relative_to(model_root)),
                source_sha256=actual,generator_sha256=source_hash)
            inputs.append(record_info)
            for failure in eligible:
                bi=meta['band_names'].index(failure['band']);mi=int(failure['member'])
                diagnosis=diagnose_member(components,observed,energy,mi,bi,failure)
                diagnoses.append(dict(regime=result['regime'],truth_index=result['truth_index'],
                                      physical_record=str(path.relative_to(physical_output)),**diagnosis))
    fp64=np.finfo(np.float64);fpl=np.finfo(np.longdouble)
    return dict(schema='npg-archived-budget-diagnosis-v1',
        scope='Independent arithmetic diagnosis of recorded non-primary discrete-budget failures; no numerical integrations or threshold changes',
        script_sha256=sha256(Path(__file__)),io_sha256=sha256(Path(__file__).with_name('audit_io.py')),
        environment=dict(python=sys.version.split()[0],numpy=np.__version__,platform=platform.platform(),
                         float64_mantissa_bits=int(fp64.nmant),longdouble_mantissa_bits=int(fpl.nmant),
                         longdouble_has_extra_mantissa_bits=bool(fpl.nmant>fp64.nmant)),
        completed_records_considered=len(records),diagnosed_nonprimary_member_failures=len(diagnoses),
        input_records=inputs,generator_source_evidence=list(sources.values()),diagnoses=diagnoses,
        skipped_records=skipped,primary_failures_outside_this_diagnostic=primary_failures,
        interpretation_limits=[
          'The physical audit results, failure thresholds, and scope clarification remain unchanged.',
          'longdouble is only an independent precision comparison when it has more mantissa bits than float64.',
          'Fraction sums are exact for the stored binary component values and do not recover precision lost before storage.',
          'Source code establishes float32 stepwise accumulation followed by float64 storage. Cancellation amplifies any accumulated residual.',
          'Attributing all archived residual to a particular operation would require original timestep traces; this diagnosis does not make that stronger claim.'])


def methods_note(payload):
    lines=['# Archived allocated-budget diagnosis methods','',
           'This supplementary check reads completed physical-audit records and their hash-matched copied raw archives. It diagnoses only failed non-primary allocated-increment balances and preserves every original pass/fail decision and threshold.','',
           'For each failed band/member, it independently adds the four archived energy-increment components, computes the RMS discrepancy from the archived observed increment, and compares the observed increment with differences of archived scalar error energy. It reports the RMS of the sum of absolute component magnitudes divided by the observed-increment RMS as a cancellation-amplification diagnostic.','',
           'The final component sum is checked using float64, NumPy longdouble, Python math.fsum, and exact rational sums of the stored binary floating-point values. The output records mantissa precision: longdouble must not be described as extended precision when it has the same mantissa as float64. Exact rational sums locate loss in the final summation but cannot recover precision already lost during the original integration.','',
           'The copied generator is selected by the runner SHA-256 stored in each raw cache. It accumulates the four allocated budgets in float32 at each timestep and converts the accumulated values to float64 when writing output. This establishes a plausible source of the archived residual, particularly when large signed components cancel, without uniquely attributing every residual to a particular operation.','',
           '## Source evidence','']
    for source in payload['generator_source_evidence']:
        lines.extend([f"- Copied model path: `{source['path']}`; SHA-256 `{source['sha256']}`."])
        for kind,evidence in source['lines'].items():
            lines.append(f"- {kind}, line {evidence['line']}: `{evidence['text']}`")
    lines.extend(['','The source paths above are relative to the supplied CommonModels/SQG_Logistic/v1 model root. Raw inputs and physical outputs are never modified. This check does not change the separate dependency-scope clarification permitting independently verified primary-band analyses.',''])
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--physical-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New JSON output; adjacent .md methods note also written')
    args=parser.parse_args();model_root=args.model_root.resolve();physical=args.physical_output.resolve()
    output=args.output.resolve();note=output.with_suffix('.md')
    if output==note:raise ValueError('--output must have a JSON filename, not .md')
    if output.exists() or note.exists():raise FileExistsError('Refusing to overwrite diagnosis output or methods note')
    payload=run(model_root,physical)
    write_json(output,payload);note.write_text(methods_note(payload))
    print(json.dumps(dict(output=str(output),methods_note=str(note),
                          diagnosed_member_failures=payload['diagnosed_nonprimary_member_failures'],
                          longdouble_has_extra_mantissa_bits=payload['environment']['longdouble_has_extra_mantissa_bits'])),flush=True)

if __name__=='__main__':main()
