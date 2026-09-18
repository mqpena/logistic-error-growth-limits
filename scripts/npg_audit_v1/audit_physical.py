#!/usr/bin/env python3
"""Approved SQG existing-data phases A/B. No model integration or random resampling.

Run only after the root analysis specification and input/code manifest are frozen.
Outputs preserve individual trajectories and distinguish numerical checks from
scientific effect sizes. Original evidence and raw archives are opened read-only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
from audit_io import (config_path, geometry, iter_cluster_paths, json_safe, load_metadata,
                      matching_indices, read_config, read_json, sha256, shell_moments, write_json)

RECON_RTOL = 1e-5
SCALE_FLOOR = 1e-10
FROZEN_RTOL = 1e-8
FROZEN_ATOL = 1e-10
DENSE_DT = .1
RHO_LO, RHO_HI = .15, .85
ENERGY_FLOOR = 1e-3
SCALAR_KEYS = ('Et','Ef','C','epsilon','rho','P_linear','P_tl_nonlinear','P_pure_error',
               'scalar_time','lambda_time','lambda_energy_rate','budget_components','budget_observed',
               'shared_forcing_relative_energy_error')

def rms(a):
    a=np.asarray(a,dtype=float)
    good=np.isfinite(a)
    return float(np.sqrt(np.mean(a[good]**2))) if np.any(good) else float('nan')

def stats(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    if not a.size:return {'count':0,'median':None,'rms':None,'p05':None,'p95':None,'max_abs':None}
    return dict(count=int(a.size),median=float(np.median(a)),rms=rms(a),
                p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)),max_abs=float(np.max(np.abs(a))))

def compare(reference,candidate,scale,rtol=RECON_RTOL):
    ref=np.asarray(reference,dtype=float);can=np.asarray(candidate,dtype=float)
    scl=np.broadcast_to(np.asarray(scale,dtype=float),ref.shape)
    if can.shape!=ref.shape:raise ValueError(f'Comparison shape mismatch {can.shape} != {ref.shape}')
    finite=bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(can)) and np.all(np.isfinite(scl)))
    denominator=max(rms(ref),SCALE_FLOOR*rms(scl),np.finfo(float).tiny)
    discrepancy=rms(can-ref)/denominator if finite else float('inf')
    point_denominator=np.maximum(np.abs(ref),SCALE_FLOOR*np.abs(scl))
    point=(can-ref)/np.maximum(point_denominator,np.finfo(float).tiny)
    return dict(passed=finite and discrepancy<=rtol,scaled_rms=discrepancy,rtol=rtol,
                reference_rms=rms(ref),scale_rms=rms(scl),absolute_floor=SCALE_FLOOR*rms(scl),
                pointwise_scaled=stats(point),scale_relative=stats((can-ref)/np.maximum(np.abs(scl),1e-30)))

def wm(v,w):return float(np.sum(v*w)/np.sum(w))

def compare_power(reference,candidate,characteristic,full_characteristic):
    """Absolute production error relative to resolved physical production scale."""
    ref=np.asarray(reference,dtype=float);can=np.asarray(candidate,dtype=float)
    char=np.asarray(characteristic,dtype=float);full=np.asarray(full_characteristic,dtype=float)
    finite=bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(can)))
    denominator=max(rms(char),SCALE_FLOOR*rms(full),np.finfo(float).tiny)
    point_denominator=np.maximum(np.abs(char),SCALE_FLOOR*np.abs(full))
    discrepancy=rms(can-ref)/denominator if finite else float('inf')
    return dict(passed=finite and discrepancy<=RECON_RTOL,scaled_rms=discrepancy,tolerance=RECON_RTOL,
                characteristic='sum absolute linear, TL nonlinear and pure-error production components',
                characteristic_rms=rms(char),full_support_scale_rms=rms(full),
                component_relative_rms=rms(can-ref)/max(rms(ref),np.finfo(float).tiny),
                absolute_floor=SCALE_FLOOR*rms(full),
                pointwise_scaled=stats((can-ref)/np.maximum(point_denominator,np.finfo(float).tiny)))

def field_productions(truthfields,forecastfields,geo,regime_config,model_config):
    """Recompute snapshot advection, not a model integration.

    NumPy complex128 transforms evaluate the same square-dealiased pseudospectral
    operators on the archived complex64 states. S tendencies below are deterministic:
    they exclude the additive forcing kicks and their quadratic energy contribution.
    """
    nt,nm,n=forecastfields.shape[0],forecastfields.shape[1],truthfields.shape[-1]
    labels=geo['labels'];support=geo['retained'];masklist=geo['masks']
    k=np.fft.fftfreq(n,d=1/n);kx,ky=np.meshgrid(k,k,indexing='ij');radius=np.hypot(kx,ky)
    safe=np.where(radius>0,radius,1.)
    inv=np.where(radius>0,1/safe,0.)*support
    rate=(float(regime_config['nu'])*radius**2+np.where(radius>0,float(model_config['mu'])/safe**2,0.))*support
    active_labels=labels[support];ns=int(active_labels.max())+1;nb=len(masklist)
    shell_index=[np.unique(labels[mask]) for mask in masklist]
    names=('P_linear','P_tl_nonlinear','P_pure_error','Sdot_deterministic')
    band={name:np.empty((nt,nm,nb),dtype=float) for name in names}
    shell={name+'_shell':np.empty((nt,nm,ns),dtype=float) for name in names}
    def jac(q):
        psi=inv*q
        px=np.fft.ifft2(1j*kx*psi).real;py=np.fft.ifft2(1j*ky*psi).real
        qx=np.fft.ifft2(1j*kx*q).real;qy=np.fft.ifft2(1j*ky*q).real
        return -np.fft.fft2(px*qy-py*qx)*support
    def inner_shell(a,b):
        weights=np.real(np.conj(a)*b)[support]/n**4
        return np.bincount(active_labels,weights=weights,minlength=ns)
    for ti in range(nt):
        truth=np.asarray(truthfields[ti],dtype=np.complex128)
        nt_rhs=jac(truth);etdot=inner_shell(truth,nt_rhs-rate*truth)
        for mi in range(nm):
            forecast=np.asarray(forecastfields[ti,mi],dtype=np.complex128)
            delta=forecast-truth;nf_rhs=jac(forecast);phi_rhs=jac(delta)
            values={'P_linear':inner_shell(delta,-rate*delta),
                    'P_tl_nonlinear':inner_shell(delta,nf_rhs-nt_rhs-phi_rhs),
                    'P_pure_error':inner_shell(delta,phi_rhs),
                    'Sdot_deterministic':etdot+inner_shell(forecast,nf_rhs-rate*forecast)}
            for name,v in values.items():
                shell[name+'_shell'][ti,mi]=v
                band[name][ti,mi]=[float(v[index].sum()) for index in shell_index]
    return band,shell

def historical_primary(moments,times):
    """Exact published R/clock masks and sequence; no bootstrap or clustering."""
    shells=np.arange(10,21)
    et=moments['Et'][:,shells];ef=moments['Ef'][:,:,shells];cross=moments['C'][:,:,shells]
    rho=moments['rho'][:,:,shells];eb=et.sum(axis=1)
    rb=cross.sum(axis=2)/np.sqrt(np.maximum(eb[:,None]*ef.sum(axis=2),1e-30))
    dense=np.isclose(np.diff(times),DENSE_DT,rtol=0,atol=1e-10)
    output=[]
    for member in range(ef.shape[1]):
        lifecycle=(times<=12+1e-12)&(rb[:,member]>=RHO_LO)&(rb[:,member]<=RHO_HI)
        values=[];excluded=total=0
        a=1-rho[:,member]
        for ti in np.flatnonzero(lifecycle):
            include=np.isfinite(rho[ti,member])&(et[ti]>=ENERGY_FLOOR*max(eb[ti],1e-30))
            total+=len(shells);excluded+=int((~include).sum())
            if include.sum()<2:continue
            v=a[ti,include];w=et[ti,include];abar=wm(v,w);denom=abar*(1-abar)
            if np.isfinite(denom) and denom>0:values.append(wm((v-abar)**2,w)/denom)
        smooth=np.full_like(a,np.nan);smooth[1:-1]=(a[:-2]+a[1:-1]+a[2:])/3
        adot=np.full_like(a,np.nan);valid_center=dense[:-1]&dense[1:]
        adot[1:-1][valid_center]=(smooth[2:][valid_center]-smooth[:-2][valid_center])/(2*DENSE_DT)
        dispersion=[];safe_dispersion=[];removed_times=[];nonpositive=endpoint=low=0
        for ti in range(2,len(times)-2):
            if not lifecycle[ti]:continue
            energy_ok=et[ti]>=ENERGY_FLOOR*max(eb[ti],1e-30)
            endpoint_ok=(a[ti]>=RHO_LO)&(a[ti]<=RHO_HI)
            with np.errstate(invalid='ignore',divide='ignore'):rate=adot[ti]/(a[ti]*(1-a[ti]))
            positive=np.isfinite(rate)&(rate>0)
            low+=int((~energy_ok).sum());endpoint+=int((energy_ok&~endpoint_ok).sum())
            nonpositive+=int((energy_ok&endpoint_ok&~positive).sum())
            include=energy_ok&endpoint_ok&positive
            if include.sum()>=2:
                v=np.log(rate[include]);w=et[ti,include];mu=wm(v,w)
                value=math.sqrt(wm((v-mu)**2,w));dispersion.append(value)
                footprint=np.diff(times[ti-2:ti+3])
                if len(footprint)==4 and np.allclose(footprint,DENSE_DT,rtol=0,atol=1e-10):
                    safe_dispersion.append(value)
                else:removed_times.append(float(times[ti]))
        output.append({'member_index':member,'D1':dict(R_median=float(np.median(values)) if values else None,
                       eligible_times=int(lifecycle.sum()),included_R_times=len(values),
                       excluded_shell_time_pairs=excluded,total_shell_time_pairs=total,
                       eligible=bool(lifecycle.sum()>=5 and values)),
                       'D2_cadence_safe':dict(Lambda_disp_median=float(np.median(safe_dispersion)) if safe_dispersion else None,
                       historical_included_times=len(dispersion),included_times=len(safe_dispersion),
                       removed_times=removed_times,removed_count=len(removed_times),
                       rule='All five derivative-of-smoothed-amplitude support times must be equally spaced by 0.1; denominator and all other masks unchanged'),
                       'D2':dict(Lambda_disp_median=float(np.median(dispersion)) if dispersion else None,
                       included_times=len(dispersion),excluded_nonpositive_lambda_eff_pairs=nonpositive,
                       excluded_endpoint_pairs=endpoint,excluded_low_energy_pairs=low)})
    return output

def frozen_primary_check(rows,reference):
    checks=[]
    for row in rows:
        mi=row['member_index']
        for label in ('D1','D2'):
            frozen=next(x for x in reference['members'][label] if x['member_index']==mi)
            for key,value in row[label].items():
                if key not in frozen:continue
                expected=frozen[key]
                if isinstance(expected,(bool,int)):
                    passed=value==expected
                    difference=None if isinstance(expected,bool) else value-expected
                else:
                    passed=value is not None and np.isclose(value,expected,rtol=FROZEN_RTOL,atol=FROZEN_ATOL)
                    difference=None if value is None else value-expected
                checks.append(dict(member=mi,statistic=label+'.'+key,actual=value,expected=expected,
                                   difference=difference,passed=bool(passed)))
    return {'passed':all(x['passed'] for x in checks),'checks':checks}

def centered(values,times,stride=1):
    """Centered differences with no stencil spanning a cadence change/gap.

    Stride 2/4 uses wider footprints at existing centers; it creates no new
    time samples. All adjacent steps within a footprint must be equal.
    """
    v=np.asarray(values,dtype=float);t=np.asarray(times,dtype=float)
    out=np.full_like(v,np.nan);valid=np.zeros(len(t),dtype=bool)
    for i in range(stride,len(t)-stride):
        steps=np.diff(t[i-stride:i+stride+1])
        if not np.allclose(steps,steps[0],rtol=1e-8,atol=1e-10):continue
        if steps[0]<=0:raise ValueError('Nonincreasing time coordinates')
        out[i]=(v[i+stride]-v[i-stride])/(t[i+stride]-t[i-stride]);valid[i]=True
    return out,valid

def crossing(t,a):
    eligible=np.flatnonzero((a[:-1]<=.5)&(a[1:]>.5)&np.isfinite(a[:-1])&np.isfinite(a[1:]))
    if not eligible.size:return float('nan')
    i=int(eligible[0]);return float(t[i]+(.5-a[i])/(a[i+1]-a[i])*(t[i+1]-t[i]))

def scalar_budget(scalars,metadata,config):
    t=scalars['scalar_time'];et=scalars['Et'];ef=scalars['Ef'];cross=scalars['C'];eps=scalars['epsilon']
    S=et+ef
    if np.any(~np.isfinite(S)) or np.any(S<=0):raise ValueError('Nonpositive/nonfinite band normalizer')
    rho=scalars['rho'];a=eps/S;xc=2*cross/S;ac=1-xc
    pl=scalars['P_linear'];pt=scalars['P_tl_nonlinear'];pp=scalars['P_pure_error'];power=pl+pt+pp
    clocks=scalars['lambda_energy_rate'];clock_times=scalars['lambda_time']
    rates=np.full(eps.shape[2],np.nan)
    clock_available=clocks.ndim==2 and clocks.shape[1]==eps.shape[2] and len(clock_times)==len(clocks)
    burnin=float(config['lyapunov']['burnin_tu'])
    if clock_available:rates=np.mean(clocks[clock_times>=burnin],axis=0)
    arrays={k:scalars[k] for k in ('scalar_time','Et','Ef','C','epsilon','rho','P_linear','P_tl_nonlinear','P_pure_error')}
    arrays.update(S=S,a_energy=a,a_covariance=ac,x_C=xc,a_correlation_proxy=1-rho,
                  coordinate_difference=a-(1-rho),energy_covariance_identity=eps-et-ef+2*cross,
                  twin_energy_balance=2*np.sqrt(np.maximum(et*ef,0))/S,
                  tangent_rate=rates,lambda_time=clock_times,lambda_energy_rate=clocks)
    summaries=[];derivative_tables=[]
    for stride in (1,2,4):
        de,valid=centered(eps,t,stride);ds,_=centered(S,t,stride);da,_=centered(a,t,stride)
        norm=-a*ds/S;production=power/S;closure=da-production-norm
        quotient_residual=da-de/S-norm
        arrays.update({f'depsilon_step{stride}':de,f'dS_step{stride}':ds,f'da_step{stride}':da,
                       f'normalization_step{stride}':norm,f'physical_residual_step{stride}':de-power,
                       f'normalized_residual_step{stride}':closure,
                       f'quotient_derivative_residual_step{stride}':quotient_residual,
                       f'derivative_valid_step{stride}':valid})
        for b,name in enumerate(metadata['band_names']):
            for member in range(eps.shape[1]):
                masks={'all_records':valid,'interior_rho':valid&(rho[:,member,b]>.15)&(rho[:,member,b]<.85),
                       'late_60_120':valid&(t>=60)&(t<=120)}
                for window,mask in masks.items():
                    powerscale=max(rms(power[mask,member,b]),rms(pl[mask,member,b])+rms(pt[mask,member,b])+rms(pp[mask,member,b]),1e-30)
                    asc=max(rms(production[mask,member,b]),rms(norm[mask,member,b]),1e-30)
                    derivative_tables.append(dict(band=name,member=member,stride=stride,window=window,
                        sample_count=int(mask.sum()),physical_residual_rms=rms((de-power)[mask,member,b]),
                        physical_residual_scaled=rms((de-power)[mask,member,b])/powerscale,
                        normalized_residual_rms=rms(closure[mask,member,b]),
                        normalized_residual_scaled=rms(closure[mask,member,b])/asc,
                        normalization_rms=rms(norm[mask,member,b]),production_over_S_rms=rms(production[mask,member,b]),
                        quotient_derivative_residual_rms=rms(quotient_residual[mask,member,b])))
    interval_eps=np.diff(eps,axis=0);interval_S=np.diff(S,axis=0);interval_a=np.diff(a,axis=0)
    q_energy=interval_eps/S[1:];q_normalization=-a[:-1]*interval_S/S[1:]
    components=scalars['budget_components'];observed=scalars['budget_observed']
    if components.shape!=(len(t),4,eps.shape[1],eps.shape[2]):raise ValueError(f'Unexpected budget shape {components.shape}')
    dt=np.diff(t)[:,None,None]
    sampled_integral=.5*(power[:-1]+power[1:])*dt
    arrays.update(interval_time=t[1:],interval_dt=np.diff(t),interval_epsilon=interval_eps,
                  interval_a=interval_a,interval_normalization=q_normalization,
                  interval_energy_over_S=q_energy,interval_quotient_residual=interval_a-q_energy-q_normalization,
                  interval_sampled_production_residual=interval_eps-sampled_integral,
                  interval_budget_observed=observed[1:],interval_budget_components=components[1:],
                  interval_discrete_budget_residual=observed[1:]-components[1:].sum(axis=1),
                  interval_observed_minus_scalar_difference=observed[1:]-interval_eps)
    discrete_checks=[]
    for bi,name in enumerate(metadata['band_names']):
        for mi in range(eps.shape[1]):
            check=compare(observed[1:,mi,bi],components[1:,:,mi,bi].sum(axis=1),np.abs(interval_eps[:,mi,bi]),rtol=1e-4)
            discrete_checks.append(dict(band=name,member=mi,**check))
    discrete_check=dict(passed=all(x['passed'] for x in discrete_checks),per_band_member=discrete_checks,rtol=1e-4)
    phi_den=rms(np.sum(np.abs(pp[:,:,:-1]),axis=2));phi_num=rms(pp[:,:,-1])
    power_char=rms(pl[:,:,-1])+rms(pt[:,:,-1])
    phi_resolved=phi_den>SCALE_FLOOR*max(power_char,1e-30)
    phi_check=dict(passed=bool(phi_resolved and phi_num/phi_den<=RECON_RTOL),
                   denominator_resolved=bool(phi_resolved),relative_l2=phi_num/phi_den if phi_den else None,
                   numerator_rms=phi_num,denominator_rms=phi_den,tolerance=RECON_RTOL)
    curvechecks=[]
    for b,name in enumerate(metadata['band_names']):
        for member in range(eps.shape[1]):
            sb=S[:,member,b];ab=a[:,member,b];cb=ac[:,member,b];rr=rho[:,member,b];rate=rates[b]
            mask=np.isfinite(xc[:,member,b])&np.isfinite(eps[:,member,b])&(eps[:,member,b]>0)&(rr>.15)&(rr<.85)
            origin=crossing(t,ab);oldorigin=crossing(t,cb)
            tau=rate*(t-origin);oldtau=rate*(t-oldorigin)
            ref=1/(1+np.exp(-np.clip(tau,-700,700)))
            oldref=1/(1+np.exp(-np.clip(oldtau,-700,700)))
            validscore=bool(np.isfinite(rate) and rate>0 and np.isfinite(origin) and mask.any())
            oldvalid=bool(np.isfinite(rate) and rate>0 and np.isfinite(oldorigin) and mask.any())
            rmse=rms((ab-ref)[mask]) if validscore else None
            oldrmse=rms((cb-oldref)[mask]) if oldvalid else None
            late=(t>=60)&(t<=120)
            slope=float(np.polyfit(t[late],sb[late],1)[0]) if late.sum()>=2 else float('nan')
            row=dict(band=name,member=member,direct_stochastic_S_forcing=bool(b==0 or name=='full'),tangent_clock_archived=bool(clock_available),
                     tangent_rate=float(rate),t_half_energy=origin,t_half_covariance=oldorigin,
                     aligned_rmse_energy=rmse,aligned_rmse_historical_covariance=oldrmse,
                     interior_samples=int(mask.sum()),S_mean=float(np.mean(sb)),S_record_cv=float(np.std(sb)/np.mean(sb)),
                     S_range_over_mean=float(np.ptp(sb)/np.mean(sb)),S_final_over_initial_minus_one=float(sb[-1]/sb[0]-1),
                     S_late_fractional_linear_drift=slope*(t[late][-1]-t[late][0])/float(np.mean(sb[late])),
                     coordinate_difference_record_median=float(np.median((ab-(1-rr)))),
                     coordinate_difference_record_rms=rms(ab-(1-rr)),
                     coordinate_difference_interior_rms=rms((ab-(1-rr))[mask]),
                     normalization_record_rms=rms(arrays['normalization_step1'][:,member,b]),
                     covariance_identity_scaled=stats((eps[:,member,b]-et[:,member,b]-ef[:,member,b]+2*cross[:,member,b])/sb))
            summaries.append(row)
    notes=['All summary medians/RMS named record use equal record weights, not uniform physical-time weights.',
           'Centered derivatives are sampled tendencies, not derivatives of white-noise paths.',
           'Full/forcing-band S contains common additive forcing although direct error forcing cancels.',
           'For forcing band k1_5 and full support, a smooth quotient derivative is not an exact pathwise Ito equation: quadratic-variation terms may enter; only the finite-increment quotient identity is certified algebra here.',
           'Primary k10_20 is disjoint from forcing; full support includes every square-mask corner mode and k=0.',
           'Exact interval quotient identity retains realized S changes without assuming differentiability.',
           'Sampled production trapezoids are temporal-resolution diagnostics; allocated RK increments are distinct.',
           'No new clocks, simulations, bootstrap, null replicates, or hypothesis thresholds were fitted.']
    return arrays,dict(tangent_burnin_tu=burnin,trajectories=summaries,derivative_diagnostics=derivative_tables,
                      discrete_budget_check=discrete_check,full_pure_transfer_cancellation=phi_check,
                      interval_quotient_absolute_error=stats(interval_a-q_energy-q_normalization),
                      archived_shared_forcing_error=stats(scalars['shared_forcing_relative_energy_error']),notes=notes)

def write_csv(path,rows):
    rows=json_safe(rows)
    if not rows:return
    flat=[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in rows]
    keys=list(dict.fromkeys(k for row in flat for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(flat)

def expected_artifacts(model_root):
    return read_json(Path(model_root)/'evidence/results/v8_pilot_null_gate_v2_20260720.json')['artifacts']

def process(path,model_root,output,expected,frozen,figure):
    started=time.monotonic();rid=path.parent.parent.name;truth=int(path.stem.split('_')[-1])
    destination=Path(output)/rid/path.stem
    if destination.exists():raise FileExistsError(f'Refusing overwrite: {destination}')
    exp=next(row for row in expected if row['regime']==rid and row['truth_index']==truth)
    actual=sha256(path)
    if actual!=exp['sha256']:raise ValueError(f'Raw archive hash mismatch: {path}')
    with np.load(path,allow_pickle=False) as data:
        metadata=load_metadata(data)
        if metadata['config_sha256']!=sha256(config_path(model_root)):raise ValueError('Original config hash mismatch')
        scalars={key:np.asarray(data[key],dtype=float) for key in SCALAR_KEYS}
        times=np.asarray(data['field_time'],dtype=float);indices=matching_indices(scalars['scalar_time'],times)
        truthfields=data['field_truth_hat'];forecastfields=data['field_forecast_hat'];n=truthfields.shape[-1]
        geo=geometry(n,metadata);moments=shell_moments(truthfields,forecastfields)
        config=read_config(model_root)
        regime_config=next(x for x in config['design']['regimes'] if x['id']==rid)
        field_power,shell_power=field_productions(truthfields,forecastfields,geo,regime_config,config['model'])
        del truthfields,forecastfields
    recon={}
    for bi,desc in enumerate(geo['bands']):
        if desc['full_support']:included=moments['shells']>=0
        else:
            labels=np.unique(geo['labels'][geo['masks'][bi]])
            included=np.isin(moments['shells'],labels)
        candidates={key:moments[key][...,included].sum(axis=-1) for key in ('Et','Ef','C','epsilon')}
        candidates['Et']=np.broadcast_to(candidates['Et'][:,None],candidates['Ef'].shape)
        fullS=scalars['Et'][indices,:,-1]+scalars['Ef'][indices,:,-1]
        recon[desc['name']]={key:compare(scalars[key][indices,:,bi],candidates[key],fullS) for key in candidates}
    powerkeys=('P_linear','P_tl_nonlinear','P_pure_error')
    powercharacteristic=sum(np.abs(scalars[k][indices]) for k in powerkeys)
    production_reconstruction={}
    for bi,desc in enumerate(geo['bands']):
        production_reconstruction[desc['name']]={}
        for key in powerkeys:
            member_checks=[dict(member=mi,**compare_power(scalars[key][indices,mi,bi],field_power[key][:,mi,bi],
                powercharacteristic[:,mi,bi],powercharacteristic[:,mi,-1])) for mi in range(scalars['epsilon'].shape[1])]
            production_reconstruction[desc['name']][key]=dict(passed=all(x['passed'] for x in member_checks),
                scaled_rms=max(x['scaled_rms'] for x in member_checks),tolerance=RECON_RTOL,per_member=member_checks)
    rawphi=field_power['P_pure_error']
    rawphi_den=rms(np.sum(np.abs(rawphi[:,:,:-1]),axis=2));rawphi_num=rms(rawphi[:,:,-1])
    rawphi_check=dict(passed=bool(rawphi_den>0 and rawphi_num/rawphi_den<=RECON_RTOL),
        relative_l2=rawphi_num/rawphi_den if rawphi_den>0 else None,rtol=RECON_RTOL,
        numerator_rms=rawphi_num,denominator_rms=rawphi_den)
    primary=historical_primary(moments,times)
    ref=next(row for row in frozen['regimes'][rid]['truths'] if row['truth_index']==truth)
    primary_check=frozen_primary_check(primary,ref)
    covidentity=(scalars['epsilon']-scalars['Et']-scalars['Ef']+2*scalars['C'])/(scalars['Et']+scalars['Ef'])
    covariance_check=dict(passed=bool(np.all(np.isfinite(covidentity)) and rms(covidentity)<=RECON_RTOL),
                          tolerance=RECON_RTOL,scaled_by='instantaneous local Et+Ef',diagnostic=stats(covidentity))
    phaseApass=all(v['passed'] for band in recon.values() for v in band.values()) and primary_check['passed'] and covariance_check['passed'] and all(v['passed'] for band in production_reconstruction.values() for v in band.values()) and rawphi_check['passed']
    destination.mkdir(parents=True)
    report=dict(schema='npg-existing-sqg-physical-audit-v1',regime=rid,truth_index=truth,
                raw_path=str(path.relative_to(Path(model_root))),raw_sha256=actual,bands=geo['bands'],
                reconstruction=recon,production_reconstruction=production_reconstruction,
                raw_full_pure_transfer_cancellation=rawphi_check,scalar_covariance_identity_check=covariance_check,
                source_sha256=actual,source_path=str(path.relative_to(Path(model_root))),
                code_sha256=sha256(Path(__file__)),io_sha256=sha256(Path(__file__).with_name('audit_io.py')),
                historical_primary=primary,historical_primary_check=primary_check,
                phase_A_passed=phaseApass,phase_B_executed=False)
    if not phaseApass:
        write_json(destination/'result.json',report)
        raise RuntimeError(f'Phase A failed for {rid}/{path.stem}; dependent analysis stopped')
    arrays,budget=scalar_budget(scalars,metadata,config)
    original_figure_checks=[]
    for row in budget['trajectories']:
        if row['band']!=metadata['band_names'][metadata['primary_band_index']]:continue
        old=next(x for x in figure['trajectories'] if x['regime']==rid and x['truth']==truth and x['member']==row['member'])
        val=row['aligned_rmse_historical_covariance'];passed=val is not None and np.isclose(val,old['logistic_rmse'],rtol=FROZEN_RTOL,atol=FROZEN_ATOL)
        original_figure_checks.append(dict(member=row['member'],actual=val,expected=old['logistic_rmse'],passed=bool(passed)))
    report.update(phase_B_executed=True,budget=budget,historical_logistic_rmse_checks=original_figure_checks,
                  implementation_checks_passed=bool(budget['discrete_budget_check']['passed'] and
                  budget['full_pure_transfer_cancellation']['passed'] and all(x['passed'] for x in original_figure_checks)))
    np.savez_compressed(destination/'curves.npz',**arrays,metadata_json=np.asarray(json.dumps(metadata)))
    np.savez_compressed(destination/'field_production.npz',**field_power,field_time=times,metadata_json=np.asarray(json.dumps(metadata)))
    np.savez_compressed(destination/'shell_moments.npz',**moments,**shell_power,field_time=times,scalar_indices=indices,
                        lambda_time=scalars['lambda_time'],lambda_energy_rate=scalars['lambda_energy_rate'],
                        metadata_json=np.asarray(json.dumps(metadata)))
    report['field_production_sha256']=sha256(destination/'field_production.npz')
    report['shell_moments_sha256']=sha256(destination/'shell_moments.npz')
    report['shell_moments_path']='shell_moments.npz'
    write_csv(destination/'trajectory_summary.csv',budget['trajectories'])
    write_csv(destination/'derivative_diagnostics.csv',budget['derivative_diagnostics'])
    report['resources']=dict(wall_seconds=time.monotonic()-started,
                             process_peak_rss_native=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                             peak_rss_units='bytes on macOS; KiB on Linux')
    write_json(destination/'result.json',report)
    if not report['implementation_checks_passed']:
        raise RuntimeError(f'Implementation check failed for {rid}/{path.stem}; outputs retained, no further clusters run')
    return report

def aggregate(output,frozen):
    """No resampling: reproduce group hierarchy and leave-one-group summaries."""
    reports=[read_json(p) for p in sorted(Path(output).glob('*/truth_*/result.json'))]
    reports=[r for r in reports if r.get('phase_A_passed') and r.get('phase_B_executed')]
    rows=[];regimes={}
    metrics=('aligned_rmse_energy','aligned_rmse_historical_covariance','S_record_cv',
             'S_range_over_mean','S_late_fractional_linear_drift','coordinate_difference_record_rms',
             'coordinate_difference_interior_rms','normalization_record_rms')
    for rid in sorted({r['regime'] for r in reports}):
        selected=[r for r in reports if r['regime']==rid]
        groups=[]
        for r in selected:
            group={'truth_index':r['truth_index'],
                   'R':float(np.mean([x['D1']['R_median'] for x in r['historical_primary']])),
                   'Lambda_disp':float(np.mean([x['D2']['Lambda_disp_median'] for x in r['historical_primary']]))}
            groups.append(group)
            for name in {x['band'] for x in r['budget']['trajectories']}:
                members=[x for x in r['budget']['trajectories'] if x['band']==name]
                row={'regime':rid,'truth_index':r['truth_index'],'band':name}
                for metric in metrics:
                    vals=[x[metric] for x in members]
                    row[metric]=float(np.mean(vals)) if all(v is not None and np.isfinite(v) for v in vals) else None
                rows.append(row)
        summary={k:float(np.median([g[k] for g in groups])) for k in ('R','Lambda_disp')}
        frozen_checks=[]
        if len(groups)==4:
            expected_values={'R':frozen['regimes'][rid]['D1']['regime_median_R'],
                             'Lambda_disp':frozen['regimes'][rid]['D2']['regime_median_Lambda_disp']}
            frozen_checks=[dict(statistic=k,actual=summary[k],expected=v,
                                passed=bool(np.isclose(summary[k],v,rtol=FROZEN_RTOL,atol=FROZEN_ATOL)))
                           for k,v in expected_values.items()]
        regimes[rid]={'frozen_regime_summary_checks':frozen_checks,'truth_groups':groups,'regime_medians':summary,'complete_four_truths':len(groups)==4,
                      'leave_one_truth_out':[{ 'excluded_truth':g['truth_index'],
                       **{k:float(np.median([o[k] for o in groups if o is not g])) for k in summary}}
                        for g in groups] if len(groups)>1 else []}
    sensitivity=[]
    for rid,band in sorted({(x['regime'],x['band']) for x in rows}):
        group=[x for x in rows if x['regime']==rid and x['band']==band]
        for metric in metrics:
            complete=[x for x in group if x[metric] is not None]
            if not complete:continue
            vals=[x[metric] for x in complete]
            sensitivity.append(dict(regime=rid,band=band,metric=metric,truth_count=len(vals),
                    median_truth_mean=float(np.median(vals)),mean_truth_mean=float(np.mean(vals)),
                    leave_one_out_medians=[dict(excluded_truth=x['truth_index'],
                      value=float(np.median([o[metric] for o in complete if o is not x]))) for x in complete] if len(vals)>1 else []))
    write_csv(Path(output)/'truth_group_summary.csv',rows)
    write_json(Path(output)/'aggregate.json',dict(regimes=regimes,band_sensitivity=sensitivity,
        evidence_status='descriptive existing-data audit; no new bootstrap/null replicates',
        complete_eight_clusters=len(reports)==8))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cluster',help='Optional regime/truth_00 selection for resource pilot')
    args=parser.parse_args();args.model_root=args.model_root.resolve();args.output=args.output.resolve()
    paths=iter_cluster_paths(args.model_root,args.cluster)
    expected=expected_artifacts(args.model_root)
    frozen=read_json(args.model_root/'evidence/results/MTB_RESULT_2026-08-11.json')
    figure=read_json(args.model_root/'evidence/results/prl_figure1_data_v1.json')
    args.output.mkdir(parents=True,exist_ok=True)
    for path in paths:
        report=process(path,args.model_root,args.output,expected,frozen,figure)
        print(json.dumps(dict(regime=report['regime'],truth=report['truth_index'],phase_A_passed=True,
                              implementation_checks_passed=report['implementation_checks_passed'],resources=report['resources'])),flush=True)
    aggregate(args.output,frozen)

if __name__=='__main__':main()
