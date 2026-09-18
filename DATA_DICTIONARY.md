# SQG error-growth dataset: data dictionary

This dictionary describes the archived, nonconfirmatory two-regime surface-quasigeostrophic (SQG) pilot and the compact data used for its original Figure 1. It covers **19 NPZ files, 1,688,539,326 bytes** (about 1.573 GiB): eight trajectory archives, eight corresponding initial truth states, two prior stationarity diagnostic archives, and one derived figure archive. No integrations were run to prepare this inventory. ZIP/NPY headers, small metadata and time vectors were inspected; spatial-field arrays were not loaded.

`FILE_INVENTORY.json` is the authoritative per-file listing of relative paths, exact byte counts, SHA-256 digests, every array key, shape, NumPy dtype and memory-order flag. It also records the initial-state/trajectory correspondence, band metadata and the known audit failures. Paths are relative to the dataset's `SQG_Logistic/v1` content root. The release separates the 19 NPZ files into a data ZIP and the code, configurations, dependencies and audit results into a reproduction ZIP. Extract both ZIPs into the same parent directory: data and code meet at `SQG_Logistic/v1/`, while the audit record is at `SQG_Logistic/audit_20260911/`. No private absolute paths are required to use the data.

## Model and sampling conventions

The domain is a doubly periodic square of side 2π. The archived scalar θ is advected by the incompressible velocity derived from ψ̂ = θ̂/|k| for nonzero Fourier wavenumber. Dissipation uses normal viscosity and inverse-Laplacian friction. The two regime identifiers are descriptive campaign labels; they should not be treated as independently replicated populations.

| Regime | Grid N×N | Integration step (model time) | Viscosity ν | Friction μ | Forecast members per truth |
|---|---:|---:|---:|---:|---:|
| `n256_re470_steep` | 256×256 | 0.0004 | 0.0134 | 1 | 2 |
| `n512_re940_marginal` | 512×512 | 0.0002 | 0.0067 | 1 | 2 |

There are four truth indices, 0–3, per regime. Their initial conditions were taken at nature-run times 100, 115, 130 and 145, respectively, from one nature trajectory in each regime. They are separated states, not four independent nature integrations. Within each truth group, the truth and both forecasts share the same future stochastic forcing. Different truth groups have different future forcing seeds. Forecast lead times run from 0 to 120.

**Units:** time is in model time units (TU); wavenumbers are integer Fourier indices in the 2π box. The source does not specify a conversion of θ, length or time to SI units. Accordingly, θ-based quantities below use model units; no kelvin, metre, second or joule interpretation is asserted. Energy means scalar half-variance, not kinetic energy: E = ⟨θ²⟩/2. Production and dissipation have units of scalar half-variance per TU. Ratios, correlations, normalized amplitudes and seeds/indices are dimensionless.

The full Fourier coefficients use the standard unnormalized forward 2-D transform. Inverse transformation uses the 1/N² factor. Parseval normalization is E = Σ|θ̂|²/(2N⁴). The retained support is the square |kx|,|ky| ≤ floor(N/3), not a circular truncation. Coefficients are stored on the full N×N complex grid in FFT ordering; these are not real-space maps or reduced `rfft` arrays.

The last band axis has six entries. The first five form a complete partition of the retained square support; the sixth repeats the full support as a diagnostic total.

| Band index | Label | Radial support, within the retained square |
|---:|---|---|
| 0 | `k1_5` | 0 ≤ |k| < 5.5, including the zero mode |
| 1 | `k6_9` | 5.5 ≤ |k| < 9.5 |
| 2 | `k10_20` | 9.5 ≤ |k| < 20.5; primary band |
| 3 | `k21_32` | 20.5 ≤ |k| < 32.5 |
| 4 | `k33_121` at N256; `k33_241` at N512 | |k| ≥ 32.5 through all retained square-corner modes |
| 5 | `full` | All retained square-support modes |

The high-band label is not a literal circular cutoff. Use the retained square mask and half-open radial boundaries when reconstructing diagnostics. The five source-band axes in locality arrays correspond to indices 0–4 above and exclude the duplicated `full` entry.

## Trajectory archives

Paths: `data/raw/<regime>/clusters/truth_00.npz` through `truth_03.npz`. Every file has the 24 keys below. Shapes use N = 256 or 512, M = 2 members, B = 6 target bands and J = floor(N/3) = 85 or 170 plotted shells. NumPy `<f8`, `<f4`, `<c8` mean little-endian float64, float32 and complex64. All arrays are in C order. Float64 storage does not imply that the historical solver or budget accumulation was computed in float64.

| Key | Shape; dtype | Definition and units |
|---|---|---|
| `scalar_time` | (733,); float64 | Lead times: 0–12 every 0.02 TU; then through 20 every 0.25 TU; then through 120 every 1 TU. Boundary times occur once. |
| `field_time` | (125,); float64 | Lead times 0, 0.1, …, 12, 20, 40, 80, 120 TU. |
| `locality_time` | (125,); float64 | Times for locality diagnostics and plotted spectra; equal to `field_time` in these archives. |
| `lambda_time` | (1200,); float64 | Ends of tangent renormalization intervals: 0.1, 0.2, …, 120 TU. |
| `lambda_energy_rate` | (1200, B); float64 | Log ratio of tangent **band energy** at interval end to interval start, divided by 0.1 TU. This is an energy-growth rate, not an amplitude-growth rate. One tangent trajectory follows each truth; there is no member axis. Units TU⁻¹. |
| `Et` | (733, M, B); float64 | Truth band half-variance, Σband |θ̂t|²/(2N⁴), duplicated across member axis. |
| `Ef` | (733, M, B); float64 | Forecast band half-variance, Σband |θ̂f|²/(2N⁴). |
| `C` | (733, M, B); float64 | Half cross moment, Re Σband θ̂t* θ̂f/(2N⁴). It is not the correlation coefficient. |
| `epsilon` | (733, M, B); float64 | Error half-variance from δ̂ = θ̂f − θ̂t: Σband |δ̂|²/(2N⁴). Algebraically Et + Ef − 2C, up to numerical precision. |
| `rho` | (733, M, B); float64 | C/√(Et Ef), with the historical small-denominator safeguard. Dimensionless. |
| `P_linear` | (733, M, B); float64 | Band inner product of δ with its viscous/frictional linear tendency. Units energy/TU. This component alone is not the complete tangent-linear production. |
| `P_tl_nonlinear` | (733, M, B); float64 | Band error production from the truth–error advection cross terms. `P_linear + P_tl_nonlinear` is the full linearized production used in the plotted response. Units energy/TU. |
| `P_pure_error` | (733, M, B); float64 | Band error production from error self-advection. It transfers error energy between bands; its full-support sum cancels within numerical tolerance. Units energy/TU. |
| `budget_components` | (733, 4, M, B); float64 | Accumulated **energy increments** over each scalar-output interval. Component order: linear, truth–error advection, error self-advection, numerical residual. These are allocated discrete integrator increments, not instantaneous production rates. Units energy. See the known failures below. |
| `budget_observed` | (733, M, B); float64 | Observed error-energy change from the previous scalar save to the current one. The first row corresponds to initialization. Units energy. |
| `shared_forcing_relative_energy_error` | (733,); float64 | Interval maximum, over steps and members, of full-support energy in the floating-point forcing-cancellation residual divided by current full-support error energy, with the historical denominator floor. Dimensionless. |
| `locality_tl_source` | (125, M, 5, 5, B); float64 | Truth–error production resolved by truth source band, error source band and target band, in that order. Units energy/TU. |
| `locality_phi_source` | (125, M, 5, 5, B); float64 | Error self-advection production resolved by its first error source band, second error source band and target band. Units energy/TU. |
| `truth_spectrum` | (125, J); float32 | Truth half-variance in shells centred at integer k = 1,…,J; radial shell [k−0.5,k+0.5). Units energy per stored shell. |
| `forecast_spectrum` | (125, M, J); float32 | Corresponding forecast half-variance shell sums. |
| `error_spectrum` | (125, M, J); float32 | Corresponding error half-variance shell sums. |
| `field_truth_hat` | (125, N, N); complex64 | Full Fourier coefficients of the truth scalar at `field_time`. Model θ Fourier units. |
| `field_forecast_hat` | (125, M, N, N); complex64 | Full Fourier coefficients of both forecast scalars at `field_time`. |
| `metadata_json` | scalar Unicode string | JSON with schema, regime/truth index, band definitions, primary-band index, seeds, shared-forcing and energy-rate flags, runtime duration and SHA-256 identifiers for config, solver, generator and initial truth file. Unicode string lengths vary by file. |

For an allocated step with old error δ and component increments Δδi, the source assigns Re⟨δ,Δδi⟩ + Re⟨Δδi,ΣjΔδj⟩/2 to each component, using the same band inner product. `budget_components` accumulates these allocations between scalar saves. Initial rows are zero. The historical accumulator is float32 before its conversion to the archived float64 array.

Summing the two source axes of a locality array reconstructs its corresponding instantaneous production at `locality_time`. In contrast, summing the short plotted spectra does **not** recover the full-support energy: shells above floor(N/3), including retained square-corner modes, and the zero mode are not represented there. Reconstruct full-support or alternative-band moments from the Fourier fields.

The tangent diagnostic is available for the five original bands plus the full support. Its historical analysis uses samples at `lambda_time >= 5` TU after burn-in. Archived band rates cannot supply independently measured tangent rates for arbitrary new shells or bands. Sparse field times after 12 TU also limit derivative reconstruction.

## Initial truth files

Paths: `data/raw/<regime>/truths/truth_00.npz` through `truth_03.npz`.

| Key | Shape; dtype | Definition |
|---|---|---|
| `theta_hat` | (N, N); complex64 | Initial truth Fourier state, in the same convention as `field_truth_hat`. |
| `metadata_json` | scalar Unicode string | JSON identifying regime/truth, nature step and time, saved PCG64 random-generator state, schema and original code/configuration hashes. |

The cluster metadata's `truth_state_sha256` identifies the matching initial file. The inventory records this link. These eight initial conditions must not be counted as an additional ensemble beyond the eight truth groups.

## Prior stationarity diagnostic files

Paths: `data/stationarity/sqg_jfm_n256_re470_dthalf.snapshots.npz` and `data/stationarity/sqg_jfm_n512_re940_snap.snapshots.npz`. Despite the filename suffix, these files store diagnostic series, not Fourier-field snapshots. Each contains 200 samples at absolute nature times 101–300 TU, at 1-TU spacing, from a separate stationarity run with seed 20260619.

| Key | Shape; dtype | Definition and units |
|---|---|---|
| `times` | (200,); float64 | Absolute model times in TU. |
| `spe` | (200,); float64 | Scalar half-variance Σ|θ̂|²/(2N⁴). |
| `vie` | (200,); float64 | Inverse-wavenumber-weighted quadratic quantity Σ|θ̂|²/(2N⁴|k|), with zero-mode inverse set to zero. Model scalar²×length units; no SI conversion supplied. |
| `eps_nu` | (200,); float64 | Viscous scalar-energy dissipation ν Σ|k|²|θ̂|²/N⁴. Energy/TU. |
| `eps_mu` | (200,); float64 | Large-scale frictional scalar-energy dissipation μ Σk≠0 |θ̂|²/(N⁴|k|²). Energy/TU. |
| `spectra` | (200, J); float32 | Scalar half-variance per integer-centred shell, with the same truncated plotted-shell convention as above. |
| `fluxes` | (200, J); float32 | Minus the cumulative shell nonlinear scalar-energy transfer, Π(k), as implemented in `spe_flux`. Energy/TU. |
| `k` | (J,); int64 | Integer shell centres 1,…,floor(N/3). |
| `config_json` | scalar Unicode string | JSON with N, step, ν, μ, forcing centre/width/injection and seed. |

## Compact original Figure 1 data

Path: `data/derived/prl_figure1_data_v1.npz`. There are 113 arrays: seven vectors for each of 16 member trajectories, plus `metadata_json`. Keys have prefixes `row_00_` through `row_15_`. Each row has its own length, 26–88 samples in this archive; the inventory records every length and maps rows to regime, truth and member.

Only the primary band (index 2) is included. Samples satisfy 0.15 < ρ < 0.85, positive error energy and the source's finite-data conditions. This is a filtered plotting dataset, not a substitute for full trajectories.

| Suffix | Meaning; all seven vectors are float64 |
|---|---|
| `tau` | λb(t−t1/2), with λb the mean archived band energy-growth rate after the 5-TU burn-in and t1/2 the retrospective half-height crossing. Dimensionless. |
| `a` | Historical plotted amplitude 1−2C/(Et+Ef), algebraically equal to ε/(Et+Ef) up to numerical precision. |
| `x` | Covariance coordinate 2C/(Et+Ef). This differs from ρ when truth and forecast energies differ. |
| `y_linear` | `P_linear`/(λb ε). |
| `y_tangent` | `P_tl_nonlinear`/(λb ε); despite this suffix, it is only the truth–error advection component. |
| `y_sum` | (`P_linear`+`P_tl_nonlinear`)/(λb ε). |
| `y_pure` | `P_pure_error`/(λb ε). |

All seven are dimensionless. `metadata_json` is a list mapping row prefixes to regime, truth and member; it includes source archive relative paths/hashes, λb, t1/2, diagnostic grouping and error summaries. Its `cluster` field is a plotting-group label, not an additional independent truth group.

## Known numerical limitations preserved in the release

The 11 September 2026 existing-data audit completed with **partial failure**, not an all-band validation. The unchanged scaled-RMS tolerance for allocated discrete budgets was 10⁻⁴. Five member checks failed, all in N512's `k33_241` band:

| Truth index | Member index | Scaled RMS discrepancy |
|---:|---:|---:|
| 0 | 0 | 0.00011215291920748338 |
| 0 | 1 | 0.00010422750631214429 |
| 1 | 0 | 0.00011263871729358597 |
| 2 | 1 | 0.00010160685954606063 |
| 3 | 1 | 0.00010417066326649019 |

The files and failed statuses remain unchanged. The audit's field/moment/operator reconstruction checks and required primary-band dependencies passed; primary-band analyses continued under an explicitly recorded dependency clarification. Passing those checks does not make the five failed allocations pass. Consult `SQG_Logistic/audit_20260911/results/audit_summary.json` and `SQG_Logistic/audit_20260911/results/archived_budget_diagnosis.json` in the reproduction ZIP for the full record. The sparse trajectory output does not retain every integrator-stage increment needed to reconstruct the original accumulation independently.

Additional limits: four separated truth states per regime do not establish independent sampling; finite-window stationarity does not imply constant band energies; inferred local clocks are distinct from the independently integrated tangent diagnostic. The existing pilot and original null-test evidence should not be promoted to a new confirmatory result by reuse of these data.

## Minimal code and loading

Reading NPZ arrays requires Python and NumPy; use `numpy.load(path, allow_pickle=False)`. Each key is decompressed when accessed. For memory control, load one truth archive at a time and avoid keeping multiple N512 field sets in memory. `metadata_json` and `config_json` can be parsed with `json.loads(str(archive[key].item()))`. All supplied array keys use numeric or Unicode dtypes, so object unpickling is unnecessary.

For existing-data audit reproduction, retain the complete `scripts/npg_audit_v1/` directory, its `spec/` and `methods/` documents, the original campaign configuration in `provenance/configs/`, and the required frozen results in `evidence/results/`. The runner uses the local data hierarchy and its sibling modules. Run it directly as `python scripts/npg_audit_v1/run_audit.py --output NEW_OUTPUT_DIRECTORY`. CPU audit computation uses NumPy, with Matplotlib for the final figures; the distributed pinned environment also includes SciPy for other supplied analyses. It does not need MLX or a new model integration.

For the original compact figure, preserve `scripts/make_prl_figure1_data_v1.py`, the compact NPZ, and the frozen `MTB_RESULT_2026-08-11.json`. To reconstruct other original figures, retain the associated script and its frozen inputs. Generating new SQG trajectories additionally requires `scripts/prl_v8_two_regime.py`, `scripts/jas_sqg_jfm.py`, the relevant configuration/launch helper and MLX on supported Apple hardware. Historical certification programs under `provenance/` preserve their original bindings; they are not the portable audit entrypoint.

Definitions were checked against the archived generator's `_continuous_diagnostics`, `_allocated_mx`, `_locality`, `_shell_spectra` and `run_truth`; the solver's `spe`, `vie`, `dissipations`, `spe_spectrum` and `spe_flux`; the compact-figure extraction code; and the manuscript methods. `FILE_INVENTORY.json` records the relative source filenames and SHA-256 identifiers used for this dictionary. Dataset DOI and code-release identifiers should be added to the release metadata after they are assigned; none is invented here.
