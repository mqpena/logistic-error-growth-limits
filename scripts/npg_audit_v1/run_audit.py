#!/usr/bin/env python3
"""Reproduce the approved existing-data audit using the local CommonModels bundle.

No numerical integrations, new bootstrap samples, or null replicates are launched.
By default the output directory must not exist. --resume-verified reuses completed
truth groups only when their input and implementation checks passed and all audit
source files match the original execution manifest.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,platform
import importlib.metadata
from audit_dependencies import primary_scope

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,default=here.parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-verified',action='store_true')
    args=p.parse_args();model=args.model_root.resolve();out=args.output.resolve()
    if not (model/'data/raw').is_dir():raise SystemExit('Invalid CommonModels bundle: no data/raw')
    sources={f.name:digest(f) for f in sorted(here.glob('*.py'))}
    record=out/'execution_sources.json'
    if out.exists():
        if not args.resume_verified:raise SystemExit(f'Refusing existing output: {out}')
        if not record.exists() or json.loads(record.read_text())['sources']!=sources:
            raise SystemExit('Resume requires identical execution source hashes')
    else:
        out.mkdir(parents=True)
        record.write_text(json.dumps({'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources},indent=2)+'\n')
    if (here/'spec').is_dir() and not (out/'freeze').exists():shutil.copytree(here/'spec',out/'freeze')
    actual_environment={'python':sys.version,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'legacy_project_access_blocked':True,'offline_restoration_claim':'See dependency restoration record; package presence alone does not establish install method.'}
    (out/'execution_environment_actual.json').write_text(json.dumps(actual_environment,indent=2)+'\n')
    logdir=out/'logs';logdir.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'runtime_cache/mpl'),XDG_CACHE_HOME=str(out/'runtime_cache/xdg'))
    guard=here/'with_archive_guard.py'
    def run(script,argv,log):
        cmd=[sys.executable,str(guard),str(here/script)]+argv
        with (logdir/log).open('x') as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode and script!='audit_physical.py':raise subprocess.CalledProcessError(result.returncode,cmd)
        return result.returncode
    physical=out/'results/physical'
    for raw in sorted((model/'data/raw').glob('*/clusters/truth_*.npz')):
        regime=raw.parent.parent.name;name=raw.stem
        result=physical/regime/name/'result.json'
        if result.exists():
            r=json.loads(result.read_text())
            if not (primary_scope(r)['eligible'] and r.get('raw_sha256')==digest(raw) and r.get('code_sha256')==sources['audit_physical.py'] and r.get('io_sha256')==sources['audit_io.py']):
                raise SystemExit(f'Cannot resume failed or mismatched group: {regime}/{name}')
        else:
            run('audit_physical.py',['--model-root',str(model),'--output',str(physical),'--cluster',regime+'/'+name],f'physical_{regime}_{name}.log')
        if not result.exists():raise SystemExit('Physical audit did not produce a record')
        r=json.loads(result.read_text());scope=primary_scope(r)
        (result.parent/'dependency_scope.json').write_text(json.dumps({'report_sha256':digest(result),'classification':scope},indent=2)+'\n')
        if not scope['eligible']:raise SystemExit('A required primary dependency failed; dependent analysis stopped')
        print('Physical group audited:',regime,name,'all-band checks:',scope['all_band_passed'],flush=True)
    from audit_physical import aggregate as summarize_physical
    frozen=json.loads((model/'evidence/results/MTB_RESULT_2026-08-11.json').read_text())
    summarize_physical(physical,frozen)
    aggregate=physical/'aggregate.json'
    if not aggregate.exists() or not json.loads(aggregate.read_text()).get('complete_eight_clusters'):
        raise SystemExit('Physical audit is incomplete')
    a=json.loads(aggregate.read_text())
    if len(a['regimes'])!=2 or any(len(r['frozen_regime_summary_checks'])!=2 or not all(c['passed'] for c in r['frozen_regime_summary_checks']) for r in a['regimes'].values()):
        raise SystemExit('Frozen regime-summary reproduction failed')
    c=out/'results/aggregation'
    if not (c/'aggregation_summary.json').exists():
        run('audit_aggregation.py',['--model-root',str(model),'--output',str(c),'--physical-output',str(physical)],'aggregation.log')
    csummary=json.loads((c/'aggregation_summary.json').read_text())
    expected={(r.parent.parent.name,int(r.stem.split('_')[-1])) for r in (model/'data/raw').glob('*/clusters/truth_*.npz')}
    found={(r['regime'],r['truth_index']) for r in csummary['clusters']}
    if found!=expected or len(csummary['clusters'])!=8:raise SystemExit('Aggregation is incomplete or duplicated')
    for regime,truth in sorted(expected):
        z=json.loads((c/f'{regime}__truth_{truth:02d}.json').read_text())
        raw=model/'data/raw'/regime/'clusters'/f'truth_{truth:02d}.npz'
        cache=physical/regime/f'truth_{truth:02d}'/'shell_moments.npz'
        if z['input_sha256']!=digest(raw) or z['script_sha256']!=sources['audit_aggregation.py']:
            raise SystemExit('Aggregation input or implementation mismatch')
        pr=json.loads((cache.parent/'result.json').read_text())
        if pr['shell_moments_sha256']!=digest(cache) or z['verified_physical_cache_sha256']!=digest(cache) or z['phase_A_B_record_sha256']!=digest(cache.parent/'result.json'):raise SystemExit('Aggregation cache or physical record changed')
    if (here/'diagnose_archived_budget.py').is_file() and not (out/'results/archived_budget_diagnosis.json').exists():
        run('diagnose_archived_budget.py',['--model-root',str(model),'--physical-output',str(physical),'--output',str(out/'results/archived_budget_diagnosis.json')],'budget_diagnosis.log')
    if (here/'matched_lifecycle_summary.py').is_file() and not (out/'results/matched_lifecycle_summary.json').exists():
        run('matched_lifecycle_summary.py',['--audit-root',str(out),'--output',str(out/'results/matched_lifecycle_summary.json')],'matched_lifecycle.log')
    if (here/'summarize_audit.py').is_file() and not (out/'AUDIT_RESULTS.md').exists():
        run('summarize_audit.py',['--audit-root',str(out),'--model-root',str(model)],'summary.log')
    print('Audit outputs:',out)
if __name__=='__main__':main()
