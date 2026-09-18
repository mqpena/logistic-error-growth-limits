"""Explicit dependency scope; preserves every failed global numerical check."""

def primary_scope(report):
    budget=report.get('budget',{})
    checks=budget.get('discrete_budget_check',{}).get('per_band_member',[])
    primary=[x for x in checks if x.get('band')=='k10_20']
    failed=[x for x in checks if not x.get('passed',False)]
    historical=report.get('historical_logistic_rmse_checks',[])
    eligible=bool(report.get('phase_A_passed') and report.get('phase_B_executed')
        and len(primary)==2 and {x.get('member') for x in primary}=={0,1}
        and all(x.get('passed') for x in primary)
        and budget.get('full_pure_transfer_cancellation',{}).get('passed')
        and len(historical)==2 and all(x.get('passed') for x in historical))
    return {'eligible':eligible,'all_band_passed':bool(report.get('implementation_checks_passed')),
            'failed_discrete_checks':[{'band':x['band'],'member':x['member'],'scaled_rms':x['scaled_rms'],'threshold':x['rtol']} for x in failed],
            'reason':('Verified primary moments/operators and primary increment checks support primary-only C; failed nonprimary increment closures remain excluded.' if eligible else 'A required input or check for primary-band C failed or is missing.'),
            'policy':'Post-failure dependency clarification; no numerical threshold changed; overall audit retains partial failure if any all-band check fails.'}
