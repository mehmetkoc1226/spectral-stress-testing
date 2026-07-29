# Data dictionary

## Stage-1 prediction schema

The full run writes indexed per-image records containing the stable dataset index, ImageNet-1K target, prediction, confidence, correctness, retained-energy diagnostics, reconstruction-range diagnostics, and output-protocol fields. Per-image chunks are intentionally excluded from Git history.

## Stage-2 record schema

Stage-2 records add calibration/evaluation partition membership, method identifiers, fitted or sample-dependent temperatures, NLL, Brier score, confidence, correctness, calibration-bin inputs, three FCTS spectral descriptors, and evaluation-family identifiers.

## Included frozen-run files

`artifacts/frozen_runs/` contains exact configurations, environments, dataset manifests, model metadata, completion markers, fitted calibration parameters, and compact aggregate analyses from the successful full runs.

## Summary tables

The CSV files in `artifacts/summary_tables/` are derived aggregate outputs. They support direct verification of headline comparisons but do not replace the omitted per-image records for a complete independent reanalysis.
