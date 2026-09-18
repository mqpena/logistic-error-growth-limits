"""Portable, read-only inputs for the approved existing-SQG-data audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

CONFIG_NAME = 'prl_v8_gate5_nonconfirmatory_config_20260715.json'
REGIMES = ('n256_re470_steep', 'n512_re940_marginal')

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')

def config_path(model_root):
    path = Path(model_root) / 'provenance' / 'configs' / CONFIG_NAME
    if not path.is_file(): raise FileNotFoundError(path)
    return path

def read_config(model_root):
    return read_json(config_path(model_root))

def iter_cluster_paths(model_root, cluster=None):
    root = Path(model_root).resolve()
    paths = sorted((root / 'data' / 'raw').glob('*/clusters/truth_*.npz'))
    if cluster:
        needle = str(cluster).removesuffix('.npz')
        chosen = []
        for path in paths:
            keys = (str(path), str(path.with_suffix('')), str(path.relative_to(root)),
                    str(path.relative_to(root / 'data' / 'raw')).removesuffix('.npz'),
                    path.parent.parent.name + '/' + path.stem)
            if needle in keys or str(cluster) in keys: chosen.append(path)
        paths = chosen
        if len(paths) != 1: raise ValueError(f'--cluster must select one archive; found {len(paths)} for {cluster}')
    elif len(paths) != 8:
        raise ValueError(f'Expected eight truth archives, found {len(paths)} under {root}')
    return paths

def load_metadata(data):
    return json.loads(str(data['metadata_json'].item()))

def matching_indices(scalar_times, field_times):
    lookup = {round(float(t), 12): i for i, t in enumerate(scalar_times)}
    result = np.asarray([lookup[round(float(t), 12)] for t in field_times], dtype=int)
    if not np.allclose(np.asarray(scalar_times)[result], field_times, rtol=0, atol=1e-10):
        raise ValueError('Field/scalar time alignment failed')
    return result

def geometry(n, metadata):
    coords = np.fft.fftfreq(n, d=1.0/n)
    kx, ky = np.meshgrid(coords, coords, indexing='ij')
    radius = np.hypot(kx, ky)
    retained = (np.abs(kx) <= n//3) & (np.abs(ky) <= n//3)
    labels = np.where(retained, np.floor(radius + .5).astype(int), -1)
    bands = metadata['bands']
    masks, descriptions = [], []
    previous = 0.0
    maximum = float(np.max(radius[retained]))
    for index, (lo, hi) in enumerate(bands):
        lower = 0.0 if index == 0 else float(lo)-.5
        upper = np.nextafter(maximum, np.inf) if index == len(bands)-1 else float(hi)+.5
        if not np.isclose(lower, previous, rtol=0, atol=1e-12): raise ValueError('Band gap')
        mask = retained & (radius >= lower) & (radius < upper)
        masks.append(mask)
        descriptions.append(dict(name=metadata['band_names'][index],lower=lower,upper_exclusive=float(upper),
                                 modes=int(mask.sum()),includes_zero=bool(mask[0,0]),full_support=False))
        previous=upper
    if not np.array_equal(np.sum(masks, axis=0), retained.astype(int)): raise ValueError('Band partition failed')
    masks.append(retained)
    descriptions.append(dict(name='full',lower=0.,upper_exclusive=float(np.nextafter(maximum,np.inf)),
                             modes=int(retained.sum()),includes_zero=True,full_support=True))
    if len(masks) != len(metadata['band_names']): raise ValueError('Band metadata mismatch')
    return {'labels':labels,'retained':retained,'masks':np.asarray(masks),'bands':descriptions}

def shell_moments(truth_hat, forecast_hat):
    """Historical float32 products, float64 shell sums; direct difference energy.

    Et has shape (time,shell). Ef/C/epsilon have (time,member,shell).
    Shell zero and all rounded-radius shells in square-dealiased support are kept.
    No fields are copied in full; caller loads each cluster only once.
    """
    truth = np.asarray(truth_hat)
    forecast = np.asarray(forecast_hat)
    nt, nm, n = forecast.shape[0], forecast.shape[1], truth.shape[-1]
    coords = np.fft.fftfreq(n, d=1.0/n)
    kx,ky=np.meshgrid(coords,coords,indexing='ij')
    valid=(np.abs(kx)<=n//3)&(np.abs(ky)<=n//3)
    labels=np.floor(np.hypot(kx,ky)+.5).astype(int)[valid]
    count=int(labels.max())+1
    et=np.empty((nt,count),dtype=float)
    ef=np.empty((nt,nm,count),dtype=float)
    cross=np.empty_like(ef); epsilon=np.empty_like(ef)
    def inner(a,b):
        weights=np.real(np.conj(a)*b)[valid]/n**4
        return np.bincount(labels,weights=weights,minlength=count)
    for ti in range(nt):
        et[ti]=.5*inner(truth[ti],truth[ti])
        for mi in range(nm):
            f=forecast[ti,mi]
            ef[ti,mi]=.5*inner(f,f)
            cross[ti,mi]=.5*inner(truth[ti],f)
            delta=f-truth[ti]
            epsilon[ti,mi]=.5*inner(delta,delta)
    return {'Et':et,'Ef':ef,'C':cross,'epsilon':epsilon,'shells':np.arange(count),
            'rho':cross/np.sqrt(np.maximum(et[:,None,:]*ef,1e-30))}

def shell_moments_from_path(path):
    with np.load(path,allow_pickle=False) as data:
        truth=data['field_truth_hat'];forecast=data['field_forecast_hat']
        result=shell_moments(truth,forecast)
        result.update(times=np.asarray(data['field_time'],dtype=float),metadata=load_metadata(data),
                      lambda_time=np.asarray(data['lambda_time'],dtype=float),
                      lambda_energy_rate=np.asarray(data['lambda_energy_rate'],dtype=float))
    return result
