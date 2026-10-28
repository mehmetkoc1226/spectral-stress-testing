# Reproducibility status

This release provides implementations and a documented execution path for
original diagnostics, original calibration, input mean/contrast controls and
the full-cohort calibration replay. It distinguishes three checks:

1. CPU verification: implementation identities, frozen counts/splits, coefficients,
   source identity and reconciliation of saved point estimates.
2. GPU rerun: regenerate model outputs using recorded checkpoints, manifests,
   settings and the included runners. New GPU inference was not executed in
   this integration environment.
3. Exact historical confidence intervals: require the complete corresponding
   per-image outputs and analysis procedure. The Stage-1 runner analyzer supplied
   here is not the separate master-analysis v2 program used for every final audit.
   Stored paired analysis exports and subsequent record-audit scripts are included;
   absent raw caches are not claimed to be distributed.

The integration environment has no torch or torchvision. Six lightweight
repository tests and the additional NumPy checks are run here. The full original
unit/planning suite remains available and is configured in CI with its dependencies.
This distinction does not turn saved completion records into a fresh inference run.

The dataset split, original config and three fitted-parameter files agree
between original and follow-up evidence. Stage-1/Stage-2 core ASTs agree with
the supplied follow-up reference cores. Stage-1 analyzer differences include
report translation and output filenames; canonical original files are retained.
