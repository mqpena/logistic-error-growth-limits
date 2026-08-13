#!/usr/bin/env python3
"""JFM-faithful forced SQG (Valadao et al. 2024/2025) on Apple GPU (MLX).

Reproduces the documented protocol for the spectral-convergence campaign:

    d_t theta + v.grad theta = nu lap theta - mu lap^{-1} theta + f,
    psi_hat = theta_hat / k,  v = (-d_y psi, d_x psi),

on a 2pi-periodic box, 2/3 dealiasing (k_max = N/3). Dissipation is NORMAL
viscosity (nu lap) + inverse-Laplacian large-scale friction (mu lap^{-1}) ONLY
-- no hyperviscosity, no exponential filter. The linear dissipation is advanced
with an integrating-factor RK4 (exact on nu k^2 + mu/k^2); the nonlinear
Jacobian goes through the RK4 stages.

Forcing = Scheme A (audit 2026-06-19): fixed modulus F on the narrow shell,
random phases, white-in-time (one Ito kick per completed step, +sqrt(dt) f_hat).
The realized shell is exactly the 8 lattice modes at |k|=sqrt(13)~3.606 for
k_f=3.5, dk_f=0.5; F is derived from the actual Parseval normalization to give a
constant EXPECTED SPE injection eps_I and is verified by a zero-field kick test.
Then eps_I = (1/2)(1/N^4) sum_shell |f_hat|^2 and eta_I = eps_I/sqrt(13) exactly.

Diagnostics: SPE E=<theta^2>/2, VIE V=<psi theta>/2, eps_nu=nu<|grad theta|^2>,
eps_mu=mu<theta lap^{-1} theta>, SPE spectrum, SPE flux Pi(k), l_nu=(nu^3/eps_nu)^{1/4},
Re=eps_I^{1/3} l_f^{4/3}/nu, resolution gate k_max*l_nu>=1.5.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

K_F = 3.5
DK_F = 0.5
EPS_I = 24.0
L_F = 2.0 * np.pi / K_F


@dataclass(frozen=True)
class SQGJFMConfig:
    n: int = 256
    dt: float = 0.002
    nu: float = 0.0134          # set Re = eps_I^{1/3} l_f^{4/3} / nu
    mu: float = 1.0
    k_f: float = K_F
    dk_f: float = DK_F
    eps_I: float = EPS_I
    seed: int = 20260619

    @property
    def k_max(self) -> int:
        return self.n // 3

    @property
    def reynolds(self) -> float:
        return self.eps_I ** (1.0 / 3.0) * L_F ** (4.0 / 3.0) / self.nu


def _fftfreq_int(n: int) -> np.ndarray:
    return np.fft.fftfreq(n, d=1.0 / n).astype(np.float64)


class SQGJFMSolver:
    def __init__(self, cfg: SQGJFMConfig) -> None:
        if cfg.n % 2 or cfg.n < 16:
            raise ValueError("n must be even >= 16")
        self.cfg = cfg
        n = cfg.n
        k = _fftfreq_int(n)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        kmag = np.sqrt(kx * kx + ky * ky)
        self.kx_np, self.ky_np, self.kmag_np = kx, ky, kmag
        kc = cfg.k_max
        mask = ((np.abs(kx) <= kc) & (np.abs(ky) <= kc)).astype(np.float64)
        self.mask_np = mask
        ksafe = np.where(kmag > 0, kmag, 1.0)
        # psi_hat = theta_hat / k  (paper convention)
        self.inv_np = np.where(kmag > 0, 1.0 / ksafe, 0.0) * mask
        # linear dissipation rate r(k) = nu k^2 + mu / k^2  (k=0 -> 0)
        self.rate_np = (cfg.nu * kmag ** 2 + np.where(kmag > 0, cfg.mu / ksafe ** 2, 0.0)) * mask
        # integrating factors
        self.E1_np = np.exp(-self.rate_np * cfg.dt) * mask
        self.E2_np = np.exp(-self.rate_np * cfg.dt / 2.0) * mask

        # forcing shell: realized lattice modes nearest k_f within +/- dk_f/...
        shell = (kmag >= cfg.k_f - 0.25) & (kmag <= cfg.k_f + 0.25)
        self.shell_np = (shell & (mask > 0))
        self.n_shell = int(self.shell_np.sum())
        # fixed modulus F so expected SPE injection = eps_I:
        #   eps_I = (1/2)(1/N^4) sum_shell |f_hat|^2 = (1/2)(1/N^4) n_shell F^2
        self.F = np.sqrt(2.0 * cfg.eps_I * n ** 4 / max(self.n_shell, 1))
        # independent conjugate pairs in the shell (Hermitian forcing)
        self._pairs = self._conjugate_pairs(self.shell_np, n)
        self._fbuf = np.zeros((n, n), dtype=np.complex64)  # reused each kick

        # MLX arrays
        self.kx = mx.array(kx.astype(np.float32))
        self.ky = mx.array(ky.astype(np.float32))
        self.mask = mx.array(mask.astype(np.float32))
        self.inv = mx.array(self.inv_np.astype(np.float32))
        self.E1 = mx.array(self.E1_np.astype(np.complex64))
        self.E2 = mx.array(self.E2_np.astype(np.complex64))

    @staticmethod
    def _conjugate_pairs(shell: np.ndarray, n: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        idx = list(zip(*np.where(shell)))
        seen, pairs = set(), []
        for (i, j) in idx:
            if (i, j) in seen:
                continue
            ci, cj = (-i) % n, (-j) % n
            seen.add((i, j)); seen.add((ci, cj))
            pairs.append(((i, j), (ci, cj)))
        return pairs

    def fft(self, field: mx.array) -> mx.array:
        return mx.fft.fft2(field) * self.mask

    @staticmethod
    def ifft(field_hat: mx.array) -> mx.array:
        return mx.real(mx.fft.ifft2(field_hat))

    def jacobian_rhs(self, theta_hat: mx.array) -> mx.array:
        """-J(psi, theta), pseudospectral, dealiased."""
        psi_hat = self.inv * theta_hat
        psi_x = self.ifft((1j * self.kx) * psi_hat)
        psi_y = self.ifft((1j * self.ky) * psi_hat)
        th_x = self.ifft((1j * self.kx) * theta_hat)
        th_y = self.ifft((1j * self.ky) * theta_hat)
        jac = self.fft(psi_x * th_y - psi_y * th_x)
        return -jac

    def deterministic_step(self, theta_hat: mx.array) -> mx.array:
        """Integrating-factor RK4 on d theta/dt = -r theta + N(theta)."""
        E1, E2, dt = self.E1, self.E2, np.float32(self.cfg.dt)
        N = self.jacobian_rhs
        k1 = N(theta_hat)
        k2 = N(E2 * (theta_hat + (0.5 * dt) * k1))
        k3 = N(E2 * theta_hat + (0.5 * dt) * k2)
        k4 = N(E1 * theta_hat + dt * (E2 * k3))
        return (E1 * theta_hat
                + (dt / 6.0) * (E1 * k1 + 2.0 * E2 * (k2 + k3) + k4)) * self.mask

    def forcing_kick(self, rng: np.random.Generator) -> mx.array:
        """Fixed-modulus, random-phase, Hermitian f_hat on the shell.

        Only the 8 shell coefficients are written (buffer reused); the rest stay 0.
        """
        fhat = self._fbuf
        phis = rng.uniform(0.0, 2.0 * np.pi, size=len(self._pairs))
        for (a, b), phi in zip(self._pairs, phis):
            val = np.complex64(self.F * np.exp(1j * phi))
            if b != a:
                fhat[a] = val
                fhat[b] = np.conj(val)
            else:
                fhat[a] = np.complex64(self.F)  # self-conjugate mode: real
        return mx.array(fhat)

    def step(self, theta_hat: mx.array, rng: np.random.Generator,
             shared_kick: mx.array | None = None) -> mx.array:
        """One completed step: IF-RK4 deterministic, then ONE Ito forcing kick.

        shared_kick lets twins reuse an identical forcing realization (audit #5).
        """
        out = self.deterministic_step(theta_hat)
        kick = self.forcing_kick(rng) if shared_kick is None else shared_kick
        return (out + np.float32(np.sqrt(self.cfg.dt)) * kick) * self.mask

    # ---- diagnostics (CPU numpy at output times) ----
    def spe(self, thn: np.ndarray) -> float:
        return float(0.5 * np.sum(np.abs(thn) ** 2) / self.cfg.n ** 4)

    def vie(self, thn: np.ndarray) -> float:
        return float(0.5 * np.sum(self.inv_np * np.abs(thn) ** 2) / self.cfg.n ** 4)

    def dissipations(self, thn: np.ndarray) -> tuple[float, float]:
        modal = np.abs(thn) ** 2 / self.cfg.n ** 4
        eps_nu = float(self.cfg.nu * np.sum(self.kmag_np ** 2 * modal))
        ksafe = np.where(self.kmag_np > 0, self.kmag_np, 1.0)
        eps_mu = float(self.cfg.mu * np.sum(np.where(self.kmag_np > 0, modal / ksafe ** 2, 0.0)))
        return eps_nu, eps_mu

    def spe_spectrum(self, thn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ks = np.arange(1, self.cfg.k_max + 1)
        modal = 0.5 * np.abs(thn) ** 2 / self.cfg.n ** 4
        E = np.array([modal[(self.kmag_np >= j - .5) & (self.kmag_np < j + .5)].sum() for j in ks])
        return ks, E

    def spe_flux(self, theta_hat: mx.array) -> np.ndarray:
        """Pi(k) = - cumulative shell SPE transfer from the nonlinear term."""
        nl = self.jacobian_rhs(theta_hat)
        mx.eval(nl)
        th, dth = np.asarray(theta_hat), np.asarray(nl)
        transfer = np.real(np.conj(th) * dth) / self.cfg.n ** 4
        ks = np.arange(1, self.cfg.k_max + 1)
        shell_t = np.array([transfer[(self.kmag_np >= j - .5) & (self.kmag_np < j + .5)].sum() for j in ks])
        return -np.cumsum(shell_t)

    def l_nu(self, eps_nu: float) -> float:
        return float((self.cfg.nu ** 3 / max(eps_nu, 1e-30)) ** 0.25)


def verify_forcing(cfg: SQGJFMConfig) -> dict:
    """Audit #3: empirically verify F reproduces eps_I via a zero-field kick."""
    s = SQGJFMSolver(cfg)
    rng = np.random.default_rng(cfg.seed)
    th0 = mx.zeros((cfg.n, cfg.n), dtype=mx.complex64)
    # one Ito kick on the zero field: theta = sqrt(dt) f_hat
    kick = s.forcing_kick(rng)
    th1 = np.float32(np.sqrt(cfg.dt)) * kick
    mx.eval(th1)
    thn = np.asarray(th1)
    measured_eps_I = s.spe(thn) / cfg.dt
    measured_eta_I = s.vie(thn) / cfg.dt
    # average over many realizations (self-injection is exact; this confirms it)
    accum = []
    for _ in range(200):
        k = s.forcing_kick(rng)
        t = np.float32(np.sqrt(cfg.dt)) * k
        mx.eval(t)
        accum.append(s.spe(np.asarray(t)) / cfg.dt)
    return {
        "n_shell_modes": s.n_shell,
        "shell_is_sqrt13_only": bool(np.allclose(
            s.kmag_np[s.shell_np], np.sqrt(13.0))),
        "F_over_N2": float(s.F / cfg.n ** 2),
        "expected_eps_I": cfg.eps_I,
        "measured_eps_I_single": round(measured_eps_I, 6),
        "measured_eps_I_mean200": round(float(np.mean(accum)), 6),
        "measured_eps_I_std200": round(float(np.std(accum)), 6),
        "measured_eta_I_single": round(measured_eta_I, 6),
        "expected_eta_I_eps_over_sqrt13": round(cfg.eps_I / np.sqrt(13.0), 6),
        "reynolds": round(cfg.reynolds, 1),
    }


def _slope(k: np.ndarray, e: np.ndarray, lo: float, hi: float) -> float:
    use = (k >= lo) & (k <= hi) & (e > 0)
    return float(np.polyfit(np.log(k[use]), np.log(e[use]), 1)[0])


def integrator_accuracy_test(cfg: SQGJFMConfig, target_spe: float = 20.0) -> dict:
    """Deterministic (forcing-OFF) one-step(dt) vs two-step(dt/2) comparison from
    an identical realistic-amplitude field. Verifies IF-RK4 accuracy decoupled
    from the stochastic forcing (audit: separate from the statistical dt gate)."""
    s_full = SQGJFMSolver(cfg)
    s_half = SQGJFMSolver(replace(cfg, dt=cfg.dt / 2))
    rng = np.random.default_rng(cfg.seed)
    q = rng.standard_normal((cfg.n, cfg.n)).astype(np.float32)
    th0 = s_full.fft(mx.array(q)) * mx.array((s_full.kmag_np <= 20).astype(np.float32))
    mx.eval(th0)
    cur = s_full.spe(np.asarray(th0).astype(np.complex64))
    th0 = th0 * np.float32(np.sqrt(target_spe / max(cur, 1e-30)))
    mx.eval(th0)
    th_full = s_full.deterministic_step(th0)
    th_half = s_half.deterministic_step(s_half.deterministic_step(th0))
    mx.eval(th_full, th_half)
    a, b = np.asarray(th_full), np.asarray(th_half)
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    return {"test": "forcing-off one-step(dt) vs two-step(dt/2)", "dt": cfg.dt,
            "n": cfg.n, "target_spe": target_spe,
            "one_vs_two_half_relative_diff": rel,
            "integrator_accurate": bool(rel < 1e-3)}


def run_campaign(cfg: SQGJFMConfig, spin_tu: float, sample_tu: float,
                 save_dt_tu: float, *, npz_path=None, checkpoint_path=None,
                 checkpoint_every_tu: float = 25.0, init_from=None) -> dict:
    s = SQGJFMSolver(cfg)
    total = round((spin_tu + sample_tu) / cfg.dt)
    spin_steps = round(spin_tu / cfg.dt)
    stride = max(1, round(save_dt_tu / cfg.dt))
    ckpt_stride = max(1, round(checkpoint_every_tu / cfg.dt))
    rng = np.random.default_rng(cfg.seed + 1)
    spe, vie, eps_nu, eps_mu, spectra, fluxes, times = [], [], [], [], [], [], []
    start = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        ck = np.load(checkpoint_path, allow_pickle=False)
        th = mx.array(ck["th_hat"]); mx.eval(th)
        rng.bit_generator.state = json.loads(str(ck["rng_state"].item()))
        start = int(ck["step"])
        spe = list(ck["spe"]); vie = list(ck["vie"]); eps_nu = list(ck["eps_nu"])
        eps_mu = list(ck["eps_mu"]); times = list(ck["times"])
        spectra = [row for row in ck["spectra"]]; fluxes = [row for row in ck["fluxes"]]
        print(f"resumed from checkpoint at step {start}/{total}", flush=True)
    elif init_from is not None:
        # Fresh predeclared window initialized from an existing developed field
        # (e.g. the N=1024 checkpoint). spin/sample define the NEW window; the
        # forcing RNG stream continues for clean white-in-time statistics.
        ck = np.load(init_from, allow_pickle=False)
        th = mx.array(ck["th_hat"]); mx.eval(th)
        if "rng_state" in ck.files:
            rng.bit_generator.state = json.loads(str(ck["rng_state"].item()))
        print(f"init field from {init_from}; fresh window spin={spin_tu} sample={sample_tu} TU", flush=True)
    else:
        q = rng.standard_normal((cfg.n, cfg.n)).astype(np.float32)
        th = s.fft(mx.array(q)) * mx.array((s.kmag_np <= 6).astype(np.float32)) * np.float32(1e-3)
        mx.eval(th)
    for step in range(start, total):
        th = s.step(th, rng)
        if (step + 1) % 25 == 0:
            mx.eval(th)
        if step >= spin_steps and (step + 1 - spin_steps) % stride == 0:
            mx.eval(th)
            thn = np.asarray(th).astype(np.complex64)
            if not np.all(np.isfinite(thn.view(np.float32))):
                return {"status": "DIVERGED", "step": step, "config": asdict(cfg)}
            en, em = s.dissipations(thn)
            ks, E = s.spe_spectrum(thn)
            spe.append(s.spe(thn)); vie.append(s.vie(thn))
            eps_nu.append(en); eps_mu.append(em)
            spectra.append(E); fluxes.append(s.spe_flux(th))
            times.append((step + 1) * cfg.dt)
        if checkpoint_path and (step + 1) % ckpt_stride == 0:
            mx.eval(th)
            tmp = str(checkpoint_path) + ".tmp.npz"
            np.savez(tmp, th_hat=np.asarray(th).astype(np.complex64),
                     rng_state=np.array(json.dumps(rng.bit_generator.state)),
                     step=np.array(step + 1),
                     spe=np.array(spe), vie=np.array(vie),
                     eps_nu=np.array(eps_nu), eps_mu=np.array(eps_mu),
                     spectra=np.array(spectra) if spectra else np.zeros((0, cfg.k_max)),
                     fluxes=np.array(fluxes) if fluxes else np.zeros((0, cfg.k_max)),
                     times=np.array(times))
            Path(tmp).replace(checkpoint_path)
    t = np.asarray(times); spe = np.asarray(spe); vie = np.asarray(vie)
    eps_nu = np.asarray(eps_nu); eps_mu = np.asarray(eps_mu)
    spectra = np.asarray(spectra); fluxes = np.asarray(fluxes)
    mean_E = spectra.mean(axis=0); mean_flux = fluxes.mean(axis=0)
    eps_nu_m, eps_mu_m = float(eps_nu.mean()), float(eps_mu.mean())
    l_nu = s.l_nu(eps_nu_m)
    kf = cfg.k_f
    k0, k1 = 3 * kf, 9 * kf
    slope = _slope(ks, mean_E, k0, k1)
    xi = -slope - 5.0 / 3.0
    def trend(x):
        mid = len(t) // 2
        tt, xx = t[mid:], x[mid:]
        return float(np.polyfit(tt, xx, 1)[0] * (tt[-1] - tt[0]) / (np.mean(xx) + 1e-30))
    fit = (ks >= k0) & (ks <= k1)
    ff = mean_flux[fit]
    if npz_path is not None:
        # per-snapshot persistence for block-bootstrap uncertainty (audit)
        np.savez_compressed(npz_path, times=t, spe=spe, vie=vie, eps_nu=eps_nu,
                            eps_mu=eps_mu, spectra=spectra, fluxes=fluxes, k=ks,
                            config_json=np.array(json.dumps(asdict(cfg))))
    return {
        "status": "ok", "config": asdict(cfg),
        "spin_tu": spin_tu, "sample_tu": sample_tu, "n_snapshots": len(t),
        "snapshot_npz": str(npz_path) if npz_path else None,
        "reynolds": round(cfg.reynolds, 1),
        "SPE_mean": round(float(spe.mean()), 5), "VIE_mean": round(float(vie.mean()), 5),
        "eps_I_target": cfg.eps_I,
        "eps_nu_mean": round(eps_nu_m, 4), "eps_mu_mean": round(eps_mu_m, 4),
        "budget_eps_nu_plus_mu": round(eps_nu_m + eps_mu_m, 4),
        "budget_residual_frac": round((eps_nu_m + eps_mu_m - cfg.eps_I) / cfg.eps_I, 4),
        "eps_nu_over_eps_I": round(eps_nu_m / cfg.eps_I, 4),
        "l_nu": round(l_nu, 5), "k_max": cfg.k_max,
        "kmax_l_nu": round(cfg.k_max * l_nu, 3),
        "resolution_gate_pass": bool(cfg.k_max * l_nu >= 1.5),
        "spe_slope_fit": round(slope, 4), "xi_correction": round(xi, 4),
        "fit_window": [k0, k1],
        "spe_stationary": bool(abs(trend(spe)) <= 0.15),
        "spe_trend_frac": round(trend(spe), 4),
        # NOT a true plateau: at finite Re the forward flux is below eps_nu and
        # still varying; renamed per audit 2026-06-19.
        "positive_forward_flux_range_mean": round(float(ff.mean()), 4),
        "positive_forward_flux_range_cv": round(float(ff.std() / (abs(ff.mean()) + 1e-30)), 4),
        "positive_forward_flux_range_vs_eps_nu": round(float(ff.mean()) / max(eps_nu_m, 1e-30), 4),
        "flux_positive_direct_range": bool(np.all(ff > 0)),
        "spectrum_k": ks.tolist(), "spectrum_E": mean_E.tolist(),
        "flux_k": ks.tolist(), "flux_Pi": mean_flux.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-forcing", action="store_true")
    p.add_argument("--integrator-test", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--nu", type=float, default=0.0134)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--spin", type=float, default=20.0)
    p.add_argument("--sample", type=float, default=40.0)
    p.add_argument("--save-dt", type=float, default=0.5)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="checkpoint file for restartable long runs (e.g. N=1024)")
    p.add_argument("--checkpoint-every", type=float, default=25.0, help="TU between checkpoints")
    p.add_argument("--init-from", type=Path, default=None,
                   help="initialize the field from an existing checkpoint npz (fresh predeclared window)")
    args = p.parse_args()
    cfg = SQGJFMConfig(n=args.n, nu=args.nu, dt=args.dt)
    if args.verify_forcing:
        print(json.dumps({"config": asdict(cfg), **verify_forcing(cfg)}, indent=2))
        return
    if args.integrator_test:
        print(json.dumps({"config": asdict(cfg), **integrator_accuracy_test(cfg)}, indent=2))
        return
    if args.run:
        npz_path = args.output.with_suffix(".snapshots.npz") if args.output else None
        rep = run_campaign(cfg, args.spin, args.sample, args.save_dt, npz_path=npz_path,
                           checkpoint_path=args.checkpoint, checkpoint_every_tu=args.checkpoint_every,
                           init_from=args.init_from)
        if args.output:
            args.output.write_text(json.dumps(rep, indent=2) + "\n")
        print(json.dumps({k: v for k, v in rep.items()
                          if k not in ("spectrum_k", "spectrum_E", "flux_k", "flux_Pi")}, indent=2))
        return
    p.error("choose --verify-forcing, --integrator-test, or --run")


if __name__ == "__main__":
    main()
