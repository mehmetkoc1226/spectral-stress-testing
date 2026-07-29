# Reproducibility guide

## Level 1: scientific unit tests

Run `pytest -q`. This validates the pure spectral and calibration utilities without downloading datasets or model weights.

## Level 2: sanity profiles

Use the sanity profile in a GPU-enabled Colab runtime. Sanity runs use deterministic subsets and verify the complete data-to-output path.

## Level 3: full profiles

Run all Stage-1 phases listed in `configs/stage1_paper.json`, freeze the resulting run directory, and run Stage 2 using the frozen Stage-1 identifier in `configs/stage2_paper.json`. The word `paper` in these configuration names denotes the full, non-subsampled experiment profile; manuscript files are not included in this repository.

## Required invariants

- Hard and Gaussian low/high-pass reconstructions are complementary before clipping.
- Per-image Gaussian noise is invariant to batch and chunk boundaries.
- Positive scalar and sample-dependent temperatures do not change top-1 predictions.
- Native and restricted output-space results are stored as separate protocols.
- Stage 2 never writes into the frozen Stage-1 directory.
- Dataset manifests preserve global dataset indices and exact WordNet mappings.

## Included validation evidence

The successful full-run configurations, environments, manifests, completion markers, aggregate summaries, fitted Stage-2 parameters, and analysis tables are under `artifacts/frozen_runs/`. Compact derived comparisons are under `artifacts/summary_tables/`.

## Not included in Git history

Raw datasets, pretrained model weights, per-image prediction chunks, selected logits, full logits, and cache directories are excluded. These can be regenerated or distributed separately as versioned release assets if required.

## Fresh-run validation

Before a public archival release:

1. execute the full profiles in a fresh GPU environment;
2. verify the generated checksums and completion markers;
3. compare regenerated aggregate outputs with the included frozen summaries;
4. attach any required large artifacts to a versioned release rather than Git history;
5. archive the release with a permanent DOI.
