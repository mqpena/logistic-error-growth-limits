#!/usr/bin/env python3
"""Post-audit descriptive matching of existing operator diagnostics to legacy lifecycle."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,json
import numpy as np
from audit_dependencies import primary_scope
from audit_io import sha256,json_safe


def rms(v):
    v=np.asarray(v,dtype=float)
    return float(np.sqrt(np.mean(v*v))) if v.size and np.all(np.isfinite(v)) else None


def ratio(n,d):
    return float(n/d) if n is not None and d is not None and np.isfinite(d) and d>0 else None


def run(audit_root):
    physical=audit_root/'results/physical';aggregation=audit_root/'results/aggregation'
    rows=[];inputs=[]
    for path in sorted(aggregation.glob('*__truth_*.json')):
        record=json.loads(path.read_text());regime=record['regime'];truth=record['truth_index']
        directory=physical/regime/f'truth_{truth:02d}'
        ph=json.loads((directory/'result.json').read_text())
        if not primary_scope(ph)['eligible']:raise RuntimeError('Required primary dependency failed')
        cache=directory/'shell_moments.npz';curves=directory/'curves.npz';csvpath=aggregation/record['timeseries_csv']
        if sha256(cache)!=ph['shell_moments_sha256']:raise ValueError('Moment cache hash mismatch')
        table=np.genfromtxt(csvpath,delimiter=',',names=True,encoding='utf-8')
        with np.load(cache,allow_pickle=False) as saved:
            indices=np.asarray(saved['scalar_indices']);ft=np.asarray(saved['field_time'])
        with np.load(curves,allow_pickle=False) as saved:
            rho=np.asarray(saved['rho'])[indices,:,2]
            if not np.allclose(np.asarray(saved['scalar_time'])[indices],ft,rtol=0,atol=1e-10):raise ValueError('Time alignment failed')
        inputs.append({'aggregation_json':str(path.relative_to(audit_root)),'aggregation_json_sha256':sha256(path),
                       'aggregation_csv_sha256':sha256(csvpath),'curves_sha256':sha256(curves),'shell_cache_sha256':sha256(cache)})
        for member in range(2):
            data=table[table['member']==member]
            if not np.allclose(data['time'],ft,rtol=0,atol=1e-10):raise ValueError('CSV time alignment failed')
            lam=float(record['members'][member]['independent_band_reference_rate'])
            mask=(ft<=12+1e-10)&(rho[:,member]>.15)&(rho[:,member]<.85)&np.isfinite(rho[:,member])
            n=int(np.sum(mask));A=data['A_exact'];ad=data['operator_A_derivative']
            logistic=lam*A*(1-A)
            scale=max(rms(ad[mask]) or 0.,rms(logistic[mask]) or 0.) if n else None
            names={'q_ref_mean':'operator_reference_shell_residual_mean','q_ref_shell_rms':'operator_reference_shell_residual_rms',
                   'weight':'operator_weight_term','shape':'operator_reference_shape_term','total_defect':'operator_observed_total_defect',
                   'normalization':'operator_normalization_term','interband_transfer_over_S':'operator_pure_error_over_S',
                   'operator_derivative':'operator_A_derivative'}
            metrics={};bad=[]
            for label,column in names.items():
                value=rms(data[column][mask]);metrics[label+'_rms']=value
                metrics[label+'_rms_over_characteristic']=ratio(value,scale)
                if value is None:bad.append(column)
            netpower=data['operator_linear_over_S']+data['operator_tl_nonlinear_over_S']+data['operator_pure_error_over_S']
            power_rms=rms(netpower[mask]);metrics['net_production_over_S_rms']=power_rms
            metrics['normalization_rms_over_net_production_rms']=ratio(metrics['normalization_rms'],power_rms)
            metrics['interband_transfer_rms_over_net_production_rms']=ratio(metrics['interband_transfer_over_S_rms'],power_rms)
            common=mask&np.isfinite(data['A_derivative_span1'])&np.isfinite(data['operator_A_derivative'])
            ncommon=int(np.sum(common))
            cscale=max(rms(ad[common]) or 0.,rms(logistic[common]) or 0.) if ncommon else None
            metrics['operator_minus_FD_derivative_rms_shared_scale']=ratio(rms((ad-data['A_derivative_span1'])[common]),cscale)
            metrics['FD_product_residual_rms_shared_scale']=ratio(rms(data['fd_product_rule_residual_span1'][common]),cscale)
            terms=np.column_stack([data['operator_reference_shape_term'],data['operator_weight_term'],data['operator_reference_shell_residual_mean']])[mask]
            if n and np.all(np.isfinite(terms)) and scale is not None:
                absolute=np.sum(np.abs(terms),axis=1);signed=np.sum(terms,axis=1);active=absolute>1e-12*scale
                metrics['common_reference_opposite_sign_fraction']=float(np.mean(np.any(terms>1e-12*scale,axis=1)&np.any(terms < -1e-12*scale,axis=1)))
                metrics['common_reference_cancellation_fraction_median']=float(np.median(1-np.abs(signed[active])/absolute[active])) if np.any(active) else None
            else:
                metrics['common_reference_opposite_sign_fraction']=None;metrics['common_reference_cancellation_fraction_median']=None
            rows.append({'regime':regime,'truth_index':truth,'member':member,
                         'matched_lifecycle_sample_count':n,'operator_FD_common_stencil_count':ncommon,
                         'matched_times':ft[mask].tolist(),'operator_characteristic_on_matched_window':scale,
                         'operator_characteristic_on_common_FD_window':cscale,
                         'missing_operator_columns':bad,'metrics':metrics})
    if len(rows)!=16:raise ValueError('Expected exactly16 existing member outputs')
    regimes={}
    for regime in sorted({r['regime'] for r in rows}):
        local=[r for r in rows if r['regime']==regime];keys=list(local[0]['metrics']);grouped={}
        for key in keys:
            truths=[]
            for ti in range(4):
                members=[r['metrics'][key] for r in local if r['truth_index']==ti]
                value=float(np.mean(members)) if len(members)==2 and all(v is not None and np.isfinite(v) for v in members) else None
                truths.append({'truth_index':ti,'two_member_mean':value})
            complete=all(r['two_member_mean'] is not None for r in truths)
            grouped[key]={'four_truth_median_of_two_member_means':float(np.median([r['two_member_mean'] for r in truths])) if complete else None,
                          'all_four_complete_truth_groups':complete,'truth_groups':truths}
        regimes[regime]={'member_lifecycle_counts':[r['matched_lifecycle_sample_count'] for r in local],
                         'member_common_stencil_counts':[r['operator_FD_common_stencil_count'] for r in local],
                         'metrics':grouped}
    return {'status':'post_audit_descriptive_matched_window','created_utc':datetime.now(timezone.utc).isoformat(),
            'selection':'already specified A/B band rho in (0.15,0.85), matched field times<=12, same fixed shells10..20',
            'comparison_rule':'all numerator and denominator metrics recalculated on the stated matched window; operator/FD compare on common valid stencil support with shared operator-based characteristic',
            'power_denominator':'RMS of signed net normalized production (P_linear+P_tl_nonlinear+P_pure)/S on matched samples',
            'aggregation_rule':'two member metrics averaged within each truth, median of all four truth means; any missing member makes grouped metric unavailable',
            'interpretation':'descriptive post-audit comparison; no new fit, integration, resampling, threshold, frozen-output alteration or significance test',
            'inputs':inputs,'trajectories':rows,'regimes':regimes}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=run(args.audit_root.resolve());args.output.write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({regime:{k:v['four_truth_median_of_two_member_means'] for k,v in row['metrics'].items() if k in ('weight_rms_over_characteristic','total_defect_rms_over_characteristic','q_ref_mean_rms_over_characteristic','q_ref_shell_rms_rms_over_characteristic','normalization_rms_over_net_production_rms','interband_transfer_rms_over_net_production_rms','operator_minus_FD_derivative_rms_shared_scale')} for regime,row in result['regimes'].items()},indent=2))

if __name__=='__main__':main()
