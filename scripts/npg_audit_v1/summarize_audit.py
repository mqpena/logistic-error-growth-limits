#!/usr/bin/env python3
"""Descriptive tables and figures for the approved frozen-input audit."""
from pathlib import Path
import argparse,json,csv
from audit_dependencies import primary_scope
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=['#22677b','#bd583d','#737aa3','#7f924a']
REGIMES=['n256_re470_steep','n512_re940_marginal']
TITLES=['N = 256','N = 512']

def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def f(x):return 'unavailable' if x is None or not np.isfinite(x) else f'{x:.4g}'
def rms(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return float(np.sqrt(np.mean(x*x))) if len(x) else None

def finite_median(x):
 # Do not silently report fewer than the stated contributing truths.
 if not x or any(v is None or not np.isfinite(v) for v in x):return None
 return float(np.median(x))

def group_stat(records,fn):
 # Each truth contributes the mean of its two member statistics.
 values=[]
 for r in records:
  try: member_values=[fn(m) for m in r['members']]
  except (KeyError,TypeError,ZeroDivisionError):member_values=[None]
  values.append(float(np.mean(member_values)) if all(v is not None and np.isfinite(v) for v in member_values) else None)
 return finite_median(values)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-root',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True);args=p.parse_args()
 out=args.audit_root.resolve();phys=out/'results/physical';agg=out/'results/aggregation';fig=out/'figures';fig.mkdir(exist_ok=True)
 pr=[read(x) for x in sorted(phys.glob('*/truth_*/result.json'))]
 cr=[read(x) for x in sorted(agg.glob('*__truth_*.json'))]
 if len(pr)!=8 or len(cr)!=8:raise ValueError('Report requires eight completed physical and aggregation records')
 if not all(primary_scope(x)['eligible'] for x in pr):raise ValueError('Required primary dependency failed')
 failed_checks=[{'regime':r['regime'],'truth':r['truth_index'],**x} for r in pr for x in primary_scope(r)['failed_discrete_checks']]
 pa=read(phys/'aggregate.json')
 if not pa.get('complete_eight_clusters') or any(not c['passed'] for r in pa['regimes'].values() for c in r['frozen_regime_summary_checks']):raise ValueError('Frozen regime-summary check failed or incomplete')
 # Preserve primary/member rows rather than pretending times are independent.
 primary=[];bandrows=[]
 for r in pr:
  for x in r['budget']['trajectories']:
   row={'regime':r['regime'],'truth':r['truth_index'],**x};bandrows.append(row)
   if x['band']=='k10_20':primary.append(row)
 summary={'input_truth_groups':8,'member_trajectories':16,'phase_A_all_passed':all(r['phase_A_passed'] for r in pr),'all_band_checks_passed':all(r['implementation_checks_passed'] for r in pr),'failed_discrete_checks':failed_checks,'status':'completed_with_partial_failure' if failed_checks else 'completed_all_checks_passed','regimes':{},'interpretation':'Descriptive existing-data audit; no new integrations, bootstrap or null tests.'}
 for regime in REGIMES:
  prs=[r for r in pr if r['regime']==regime];crs=[r for r in cr if r['regime']==regime]
  rows=[r for r in primary if r['regime']==regime]
  def truthmed(key):
   vals=[[r[key] for r in rows if r['truth']==i] for i in range(4)]
   return finite_median([float(np.mean(v)) if len(v)==2 and all(x is not None and np.isfinite(x) for x in v) else None for v in vals])
  maxmoment=max(v['scaled_rms'] for r in prs for band in r['reconstruction'].values() for v in band.values())
  m={'max_moment_reconstruction_scaled_rms':maxmoment,'historical_R':pa['regimes'][regime]['regime_medians']['R'],'historical_clock_dispersion':pa['regimes'][regime]['regime_medians']['Lambda_disp'],'primary_S_record_cv':truthmed('S_record_cv'),'primary_coordinate_interior_rms':truthmed('coordinate_difference_interior_rms'),'primary_aligned_RMSE_truth_group_median':truthmed('aligned_rmse_energy'),'primary_individual_RMSE_range':([min(r['aligned_rmse_energy'] for r in rows),max(r['aligned_rmse_energy'] for r in rows)] if all(r['aligned_rmse_energy'] is not None for r in rows) else None)}
  m['finite_difference_product_residual_over_characteristic_rms']=group_stat(crs,lambda x:x['finite_difference']['span1']['fd_product_residual_over_characteristic_rms'])
  m['common_reference_shell_residual_rms_over_characteristic']=group_stat(crs,lambda x:x['finite_difference']['span1']['common_reference']['weighted_rms_q_ref']['rms']/x['finite_difference']['span1']['characteristic_tendency'])
  m['weight_term_rms_over_characteristic']=group_stat(crs,lambda x:x['finite_difference']['span1']['weight_term']['rms']/x['finite_difference']['span1']['characteristic_tendency'])
  counts=[m['finite_difference']['span1']['whole_row_inferred_interior_samples'] for r in crs for m in r['members']]
  m['inferred_decomposition_samples_per_member_range']=[min(counts),max(counts)]
  m['inferred_shape_covariance_opposite_sign_fraction']=group_stat(crs,lambda x:x['finite_difference']['span1']['inferred_shape_clock_cancellation'].get('fraction_with_opposing_signs_above_arithmetic_floor',float('nan')))
  m['common_reference_opposite_sign_fraction']=group_stat(crs,lambda x:x['finite_difference']['span1']['common_reference']['physical_terms_cancellation_without_fd_residual'].get('fraction_with_opposing_signs_above_arithmetic_floor',float('nan')))
  bound_maxima=[max(s['sampled_max_bound'] for s in x['auxiliary_sampled_path_bound'].get('segments',[])) for r in crs for x in r['members'] if x['auxiliary_sampled_path_bound'].get('segments')]
  m['operator_weight_rms_over_characteristic']=group_stat(crs,lambda x:x['snapshot_operator_common_reference']['windows']['dense_t_le_12']['weight_term']['rms']/x['snapshot_operator_common_reference']['windows']['dense_t_le_12']['characteristic_tendency'])
  m['operator_shell_residual_rms_over_characteristic']=group_stat(crs,lambda x:x['snapshot_operator_common_reference']['windows']['dense_t_le_12']['weighted_rms_q_ref']['rms']/x['snapshot_operator_common_reference']['windows']['dense_t_le_12']['characteristic_tendency'])
  m['operator_total_defect_rms_over_characteristic']=group_stat(crs,lambda x:x['snapshot_operator_common_reference']['windows']['dense_t_le_12']['operator_total_defect']['rms']/x['snapshot_operator_common_reference']['windows']['dense_t_le_12']['characteristic_tendency'])
  m['operator_minus_FD_derivative_rms_over_characteristic']=group_stat(crs,lambda x:x['snapshot_operator_common_reference']['operator_minus_finite_difference_dense']['A_derivative_span1']['rms']/x['snapshot_operator_common_reference']['windows']['dense_t_le_12']['characteristic_tendency'])
  m['sampled_path_max_bound_range']=[min(bound_maxima),max(bound_maxima)] if bound_maxima else None
  m['sampled_path_members_contributing']=len(bound_maxima)
  m['metric_contribution_rule']='A grouped statistic is unavailable if any of four complete two-member truth groups lacks the required quantity; no silent dropping.'
  m['sampled_path_bound_numerical_violations']=sum(s['sampled_bound_below_error_beyond_numerical_allowance'] for r in crs for x in r['members'] for s in x['auxiliary_sampled_path_bound'].get('segments',[]))
  summary['regimes'][regime]=m
 write(out/'results/audit_summary.json',summary)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
 # All individual primary lifecycles: no fitting or binned average.
 figure,axs=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
 for col,regime in enumerate(REGIMES):
  for r in [x for x in pr if x['regime']==regime]:
   ti=r['truth_index'];d=np.load(phys/regime/f'truth_{ti:02d}'/'curves.npz',allow_pickle=False)
   for mi in range(2):
    row=next(x for x in primary if x['regime']==regime and x['truth']==ti and x['member']==mi)
    t=d['scalar_time'];a=d['a_energy'][:,mi,2];tau=row['tangent_rate']*(t-row['t_half_energy']) if row['t_half_energy'] is not None and row['tangent_rate'] is not None else np.full_like(t,np.nan);mask=t<=12
    label=f'Truth {ti}' if mi==0 else None
    axs[0,col].plot(t[mask],a[mask],color=COLORS[ti],alpha=.8,lw=1.15,ls='-' if mi==0 else '--',label=label)
    mask=(tau>=-3)&(tau<=3);axs[1,col].plot(tau[mask],a[mask],color=COLORS[ti],alpha=.8,lw=1.15,ls='-' if mi==0 else '--')
  xx=np.linspace(-3,3,201);axs[1,col].plot(xx,1/(1+np.exp(-xx)),color='black',ls=':',lw=2,label='Logistic reference')
  axs[0,col].set(title=TITLES[col]+' | original time, first 12 units',xlabel='Model time',ylabel=r'Exact band error $\epsilon_b/S_b$')
  axs[1,col].set(title=TITLES[col]+' | retrospective alignment',xlabel=r'$\lambda_b(t-t_{1/2})$',ylabel=r'Exact band error $\epsilon_b/S_b$')
  axs[0,col].legend(ncol=2,fontsize=8);axs[1,col].legend(fontsize=8)
 figure.suptitle('Primary band 10–20: all 16 archived error trajectories',fontsize=14)
 figure.savefig(fig/'individual_lifecycles.png');plt.close(figure)
 # Across-band normalized budget: median of truth means of member-level quantities.
 bands=pr[0]['bands'];names=[x['name'] for x in bands];display_names=['1–5','6–9','10–20','21–32','33–edge','full']
 figure,axs=plt.subplots(1,3,figsize=(12.3,4.1),layout='constrained')
 metrics=[('S_record_cv','Normalizer variability\nCV over stored records'),('normalization_ratio','Normalization term / production\nRMS ratio, interior interval'),('residual_ratio','Sampled budget residual\nrelative RMS, interior interval')]
 for ax,(key,title) in zip(axs,metrics):
  for ci,regime in enumerate(REGIMES):
   vals=[]
   for bi in range(len(names)):
    truths=[]
    for r in [x for x in pr if x['regime']==regime]:
     b=r['bands'][bi]['name']
     if key=='S_record_cv':v=[x[key] for x in r['budget']['trajectories'] if x['band']==b]
     else:
      dx=[x for x in r['budget']['derivative_diagnostics'] if x['band']==b and x['stride']==1 and x['window']=='interior_rho']
      v=[x['normalization_rms']/x['production_over_S_rms'] if key=='normalization_ratio' else x['normalized_residual_scaled'] for x in dx if x['sample_count'] and x['production_over_S_rms']>0]
     truths.append(float(np.mean(v)) if v else None)
    vals.append(finite_median(truths))
   ax.plot(np.arange(len(names))+(ci-.5)*.1,vals,marker='o',lw=1.3,color=['#22677b','#bd583d'][ci],label=TITLES[ci])
  ax.set(title=title,xticks=np.arange(len(names)),xticklabels=display_names,xlabel='Shell band');ax.tick_params(axis='x',rotation=45);ax.set_yscale('log');ax.grid(axis='y',alpha=.2)
 axs[0].legend(fontsize=9);figure.suptitle('Physical bands: descriptive medians across four truth groups',fontsize=13)
 figure.savefig(fig/'band_budget_diagnostics.png');plt.close(figure)
 # Primary common-reference decomposition, scale as the predeclared characteristic tendency.
 figure,axs=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained')
 labels=['Shape','Weights','Mean shell\nresidual','Total\ndefect']
 for ci,regime in enumerate(REGIMES):
  ax=axs[ci]
  for r in [x for x in cr if x['regime']==regime]:
   values=[]
   for m in r['members']:
    z=m['snapshot_operator_common_reference']['windows']['dense_t_le_12'];c=z['characteristic_tendency']
    values.append([z['reference_shape_term']['rms']/c,z['weight_term']['rms']/c,z['weighted_q_ref']['rms']/c,z['operator_total_defect']['rms']/c])
   ax.plot(np.arange(4)+(r['truth_index']-1.5)*.06,np.mean(values,axis=0),marker='o',lw=.8,alpha=.8,color=COLORS[r['truth_index']],label=f'Truth {r["truth_index"]}')
  ax.set(title=TITLES[ci],xticks=np.arange(4),xticklabels=labels,ylabel='RMS / characteristic tendency');ax.set_yscale('log');ax.grid(axis='y',alpha=.2);ax.legend(ncol=2,fontsize=8)
 figure.suptitle('Snapshot operators: independent common-rate residuals (time 0–12)',fontsize=13)
 figure.savefig(fig/'common_reference_terms.png');plt.close(figure)
 # Show every sampled-path bound; it is an a posteriori interpolant calculation.
 figure,axs=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained')
 for ci,regime in enumerate(REGIMES):
  ax=axs[ci]
  for r in [x for x in cr if x['regime']==regime]:
   table=np.genfromtxt(agg/r['timeseries_csv'],delimiter=',',names=True,encoding='utf-8')
   for mi in range(2):
    good=(table['member']==mi)&(table['time']>0)&(table['time']<=12)
    t=table['time'][good];bound=table['pl_bound_32'][good];err=table['pl_error'][good]
    ax.plot(t,np.where(bound>0,bound,np.nan),color=COLORS[r['truth_index']],lw=1,alpha=.7)
    ax.plot(t,np.where(err>0,err,np.nan),color=COLORS[r['truth_index']],lw=1,alpha=.7,ls=':')
  ax.axhline(1,color='black',lw=.8,ls='--');ax.set(title=TITLES[ci],xlabel='Model time',ylabel='Absolute deviation and bound',yscale='log');ax.grid(axis='y',alpha=.2)
 figure.suptitle('Sampled paths only: solid = auxiliary bound; dotted = segment-reference deviation\nEach eligible segment restarts its reference; a bound above 1 is uninformative',fontsize=11)
 figure.savefig(fig/'sampled_path_bound.png');plt.close(figure)
 lines=['# Existing-data SQG audit — 11 September 2026','','The approved audit is complete for eight truth groups and sixteen member trajectories. All required primary-band dependencies pass; failed nonprimary discrete-budget checks remain explicitly failed. Model integrations, bootstrap samples and relational-null tests were not rerun. The audit uses locally copied data with access to the historical project blocked; this delivered run used the documented offline-restored environment.','','## Numerical summary','','Statistics below are medians of four truth-group values, each formed from its two members, except reconstruction maxima and stated ranges. Record-based summaries weight stored records equally; they do not imply uniform physical-time weighting or independent samples.','','| Diagnostic | N=256 | N=512 |','|---|---:|---:|']
 fields=[('Maximum scaled RMS moment reconstruction error','max_moment_reconstruction_scaled_rms'),('Reproduced shape statistic R','historical_R'),('Reproduced inferred clock dispersion','historical_clock_dispersion'),('Primary-band normalizer CV (stored records)','primary_S_record_cv'),('Energy versus correlation coordinate, interior RMS','primary_coordinate_interior_rms'),('Aligned individual RMSE, median truth-group mean','primary_aligned_RMSE_truth_group_median'),('Weight-term RMS / characteristic tendency','weight_term_rms_over_characteristic'),('Shell-reference residual RMS / characteristic tendency','common_reference_shell_residual_rms_over_characteristic'),('FD product residual RMS / characteristic tendency','finite_difference_product_residual_over_characteristic_rms'),('Inferred shape/covariance opposing-sign fraction','inferred_shape_covariance_opposite_sign_fraction'),('Snapshot operator weight RMS / characteristic','operator_weight_rms_over_characteristic'),('Snapshot shell residual RMS / characteristic','operator_shell_residual_rms_over_characteristic'),('Snapshot total defect RMS / characteristic','operator_total_defect_rms_over_characteristic'),('Operator minus 0.1-unit FD derivative RMS / characteristic','operator_minus_FD_derivative_rms_over_characteristic')]
 for label,key in fields:lines.append('| '+label+' | '+' | '.join(f(summary['regimes'][r][key]) for r in REGIMES)+' |')
 matched_path=out/'results/matched_lifecycle_summary.json'
 if matched_path.exists():
  matched=read(matched_path)
  lines.extend(['','## Matched lifecycle: post-audit descriptive comparison','','This additional comparison applies the original 0.15 < band correlation < 0.85 window at field times through12. It was specified after the full audit was inspected and is descriptive. Both numerator and scale are recomputed on the selected samples. It retains5–18 times per member at N256 and8–13 at N512, and all four two-member truth groups in each regime.','','| Matched-window diagnostic | N=256 | N=512 |','|---|---:|---:|'])
  for label,key in [('Operator defect RMS / characteristic','total_defect_rms_over_characteristic'),('Weighted-mean shell residual RMS / characteristic','q_ref_mean_rms_over_characteristic'),('Normalization RMS / net production RMS','normalization_rms_over_net_production_rms'),('Interband transfer RMS / net production RMS','interband_transfer_rms_over_net_production_rms'),('Operator minus finite-difference derivative RMS / shared characteristic','operator_minus_FD_derivative_rms_shared_scale')]:
   lines.append('| '+label+' | '+' | '.join(f(matched['regimes'][r]['metrics'][key]['four_truth_median_of_two_member_means']) for r in REGIMES)+' |')
  lines.extend(['','The discrepancy remains substantial in the matched interval. These single-realization instantaneous diagnostics do not falsify an ensemble-mean or stochastic logistic model, and do not isolate clock heterogeneity from normalization and transfer.'])
 lines.extend(['','## Preserved numerical failures','','| Regime | Truth | Band | Member | Relative residual | Frozen tolerance |','|---|---:|---|---:|---:|---:|'])
 for x in failed_checks:lines.append(f"| {x['regime']} | {x['truth']} | {x['band']} | {x['member']} | {x['scaled_rms']:.7g} | {x['threshold']:.7g} |")
 lines.extend(['','These failures concern archived allocated increments. Their tolerance was not relaxed. Primary-only analyses proceeded under the recorded dependency clarification because reconstructed fields/operators, primary-band incremental budgets, and conservative-transfer checks pass. The overall numerical audit has partial failures; it is not an all-band validation.','','## Figures','','![Individual lifecycle comparison](figures/individual_lifecycles.png)','','![Physical band diagnostics](figures/band_budget_diagnostics.png)','','![Common-reference terms](figures/common_reference_terms.png)','','![Auxiliary sampled-path bound](figures/sampled_path_bound.png)','','## Interpretation and limits','','Snapshot-operator metrics use all121 saved states through time12. Their characteristic tendency is the larger of the RMS normalized operator tendency and the RMS prescribed logistic tendency. Finite-difference metrics use their own declared characteristic tendency and eligible stencil support; ratios with different denominators should not be compared as identical scales.','The common-reference residual uses the independently archived band tangent rate as a prescribed common shell coefficient. It is a measured discrepancy against that hypothesis, not an independently measured shellwise tangent clock. Its physical cause is not identified by the algebraic decomposition. The inferred-rate decomposition remains descriptive because those rates are computed from the same derivatives.','','Finite-difference product-rule residuals are reported separately and are not expected to be zero at machine precision. Exact interval quotient identities retain realized changes in the denominator. In directly forced bands and full support, common forcing cancels from the direct error increment but leaves stochastic denominator increments; a smooth normalized tendency cannot be assumed without the appropriate stochastic calculus.','','The auxiliary bound applies only to a piecewise-linear path through consecutive 0.1-unit samples up to time12. Each eligible segment restarts its reference and bound at its first sample; a small later segment bound does not tighten the earlier bound. Quadrature-order agreement is a convergence estimate, not a certified error enclosure. This calculation does not provide a bound for the unknown continuous physical trajectory or independent closure predictions. A bound exceeding1 is uninformative for two amplitudes constrained to [0,1].','','The inferred fixed-shell decomposition retains only 2–10 times per member at N256 and 1–9 at N512. Opposing-sign fractions therefore have weak temporal support and should not be interpreted as a robust regime comparison.\n\nThe four truth states per regime originate from separated times along one nature trajectory. Leave-one-truth-out sensitivity is descriptive; no new significance or forecasting-skill claim follows. The original cross-truth null result remains unchanged.','','## Reproducibility','','Machine-readable checks and all individual series are in `results/physical` and `results/aggregation`; grouped sensitivity is in `results/physical/aggregate.json`. The pre-execution specification and addenda are in `freeze/`. A separate human-reviewed interpretation note records manuscript implications and any implementation corrections during verification.'])
 (out/'AUDIT_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
