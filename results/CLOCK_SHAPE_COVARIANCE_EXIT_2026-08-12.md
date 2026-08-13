# Exploratory finite-band clock-shape covariance — exit record

**Date:** 2026-08-12  
**Evidence:** exploratory descriptive magnitude on previously inspected stored data  
**Scope:** exact local-clock diagnostic identity; no closure validation, hypothesis test, T-code, E-code, or C0 change

## Result

All corrections below are fractions of `lambdabar*abar*(1-abar)`. Negative values reduce the baseline logistic tendency; positive values increase it.

| Regime | Shape correction | Clock-shape correction | Absolute clock-shape | Net correction | Direct/baseline |
|---|---:|---:|---:|---:|---:|
| `n256_re470_steep` | -0.0216 | -0.0095 | 0.0268 | -0.0318 | 0.9682 |
| `n512_re940_marginal` | -0.0304 | -0.0041 | 0.0177 | -0.0386 | 0.9614 |

Truth-bootstrap 95% intervals and all member/time values are retained in the machine record.

The median net departure from the local-clock logistic baseline is therefore
`-3.18%` in the steep regime and `-3.86%` in the marginal regime. The signed
clock-shape term is smaller than one percent in both regime medians; its median
absolute magnitude is `2.68%` and `1.77%`. Within this diagnostic, clock
heterogeneity invalidates GT1's one-clock theorem hypothesis but does not create
a large band-mean departure from logistic behaviour. This is an exploratory
description, not a confirmatory tolerance claim.

Across truths, the net correction ranges from `-4.96%` to `-1.78%` in the
steep regime and from `-5.25%` to `-3.07%` in the marginal regime. The signed
clock-shape correction ranges from `-1.12%` to `-0.05%` and from `-2.21%` to
`-0.17%`; the corresponding per-truth median absolute magnitudes range from
`0.47%` to `4.12%` and from `1.49%` to `2.63%`.

### Convention and cancellation caveats

Leith--Kraichnan's uncorrelated energy satisfies `epsilon=2*E_Delta` under the
manuscript half-energy convention. Under equal truth and forecast shell energy,
`a=1-rho=epsilon/L`; the factor two cancels from every normalized correction.
With unequal shell energies, `1-rho` is a correlation coordinate, so this
remains a local-clock surrogate rather than an exact finite-band energy closure.

A small signed covariance can reflect cancellation among shells, while a
regime median can conceal member or time variation. The absolute clock-shape
magnitude addresses sign variation in the aggregated covariance, but cannot
undo cancellation internal to the shell-weighted covariance. The result says
that the realised aggregate correction is modest in these lifecycles, not that
heterogeneous clocks cancel in general.

## Per-truth values

### `n256_re470_steep`

| Truth | Shape | Clock-shape | Absolute clock-shape | Net |
|---:|---:|---:|---:|---:|
| 0 | -0.0218 | -0.0080 | 0.0178 | -0.0299 |
| 1 | -0.0204 | -0.0110 | 0.0359 | -0.0336 |
| 2 | -0.0331 | -0.0112 | 0.0412 | -0.0496 |
| 3 | -0.0213 | -0.0005 | 0.0047 | -0.0178 |

### `n512_re940_marginal`

| Truth | Shape | Clock-shape | Absolute clock-shape | Net |
|---:|---:|---:|---:|---:|
| 0 | -0.0287 | -0.0221 | 0.0263 | -0.0525 |
| 1 | -0.0303 | -0.0038 | 0.0203 | -0.0349 |
| 2 | -0.0305 | -0.0017 | 0.0149 | -0.0307 |
| 3 | -0.0332 | -0.0045 | 0.0151 | -0.0423 |

## Interpretation boundary

Clock dispersion established that the GT1 eigencondition is not uniformly realised; it did not establish non-logistic band-mean behaviour. The covariance above is the relevant correction within the local-clock identity. It remains exploratory and does not evaluate the nonlocal I1, forcing, dissipation, or memory residuals required for a finite-band closure theorem.

Maximum algebraic identity error: `8.882e-16` (tolerance `1.0e-10`).
