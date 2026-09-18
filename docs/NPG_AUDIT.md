# Existing-data audit for the NPG revision

The September 2026 audit reuses eight archived truth groups and sixteen member
trajectories. Its specification, dated addenda and mathematical checks are
included under `scripts/npg_audit_v1/spec/` and `scripts/npg_audit_v1/methods/`.
The published original outputs accompany
[dataset DOI 10.5281/zenodo.22822514](https://doi.org/10.5281/zenodo.22822514)
under `SQG_Logistic/audit_20260911/`.

Use the direct command in the root README with an explicit `--model-root`
pointing to the extracted `SQG_Logistic/v1` directory. The audit script's default
is suitable inside the original bundle; the GitHub layout requires this explicit
argument. New outputs are separate from the archived scientific evidence.

Expected outputs include `AUDIT_RESULTS.md`, `results/physical/aggregate.json`,
`results/aggregation/aggregation_summary.json`, the archived-budget diagnosis,
the matched-lifecycle summary, four figures and execution logs. The audit
processes raw groups sequentially. The original macOS run's largest recorded
process peak memory was about 1.20 GiB; this is not a guarantee for other systems.

All primary-band dependencies and original headline summaries reproduce.
**Five N512 highest-band allocated-budget checks fail the unchanged tolerance.**
Those failures remain in the physical result files. The separately documented
dependency scope permits independently passing primary-band calculations to
continue. A successful process exit denotes completion of that workflow,
not a scientific all-pass result. Review `AUDIT_RESULTS.md` and the underlying
records before interpreting a rerun.

The physical-operator and interpolation calculations have manufactured
mathematical checks. Matched-lifecycle diagnostics were added after the principal
audit and remain descriptive. Finite saved-field cadence and archived band
normalizers limit which spectral and continuous-time claims can be tested.

The audit runs no new turbulence integration, bootstrap or relational-null
replicate. Historical campaign and certification runners in the repository and
Zenodo provenance bind earlier source/configuration hashes and execution paths.
Certified reruns of those historical workflows require explicit migration of
their bindings and a new execution freeze.

Source hashes in `audit_source_provenance.json` map the GitHub copy to the
published sources. The sole packaging edit replaces a private workstation path
in the guard self-check with a synthetic forbidden path. The guard still blocks
access to any path naming the old project, and the numerical audit sources are
byte-identical.
