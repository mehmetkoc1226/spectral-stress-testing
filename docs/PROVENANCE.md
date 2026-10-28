# Source provenance

## Canonical code inputs

| Component | Selected source | SHA-256 | Decision |
|---|---|---|---|
| Stage 1 base | `Spectral_Stage1_Colab (1).ipynb` | `d44e0a1f4090bcb2405e2cbcc49ad0d979ff14022c9ff90adf484d9c78443b14` | Executed full Stage-1 notebook and canonical scientific source. |
| Stage 1 patch | `Spectral_Stage1_GaussR20_Patch_Colab.ipynb` | `405fc8f84242dc06364b88bf2dcfcac711da20a5053bb5f048788376eb832614` | Adds the missing ImageNet-A Gaussian r=20 linear job-planning phase; used as a source-only orchestration patch. |
| Stage 2 | `Spectral_Stage2_Calibration_Pilot_Colab(4) (1).ipynb` | `c64bc4fb96be8436160c5e96f13d8a1215f7c17cdbd00e09c8ed5045a280b68f` | Latest v1.4 source; recorded outputs show 78/78 completed jobs and successful analysis. |
| Qualitative generation | `Spectral_Qualitative_Figures_Colab_v2_compact (1).ipynb` | `afd4452084806c573abf0e27a7f7ba3613594e603dd3bfa6c6365a3a47d0d2bb` | Latest compact qualitative generator. |

## Canonical frozen run inputs

| Stage | Frozen run | Included evidence |
|---|---|---|
| Stage 1 | `spectral_stage1_confirmatory_v1__7044005bc7a4` | Configuration, environment, condition/model/label metadata, dataset manifests, completion markers, summary table, and aggregate analyses. |
| Stage 2 | `spectral_stage2_calibration_pilot_v1__27bb9eee96fb` | Configuration, environment, dataset and split manifests, completion marker, fitted parameters, self-tests, summary table, and comparison analyses. |

## Integrated corrections

1. The Stage-1 patch adds only the ImageNet-A Gaussian r=20 linear jobs to the planner. Spectral masks, reconstruction, metrics, label mappings, and inference definitions remain unchanged.
2. Stage-2 v1.4 changes data transport and integrates the executed cache-path correction. The scientific calibration definitions remain unchanged.
3. Turkish markdown, comments, log messages, and narrative reports were removed or translated. Scientific constants, seeds, identifiers, conditions, equations, and numeric results were not altered.
4. The copied `FROZEN_DO_NOT_WRITE.json` record contains an English translation of its original Turkish rule sentence and an explicit language-normalization field.
5. Exploratory Stage 0 files and superseded Stage-2 copies are excluded from the canonical repository surface.
6. Manuscript files were used only to check consistency during preparation and are not distributed in this code-only package.

## Artifact boundary

The repository includes compact metadata and aggregate outputs. It excludes raw datasets, pretrained weights, per-image prediction chunks, selected logits, full logits, caches, manuscript sources, compiled articles, and submission materials.

## October submission integration

Added the two completed follow-up experiments without replacing scientific modules.
The exact historical spectral_stage2_pilot.py under strengthening_experiments/
retains its original log messages and recorded hash. The language check excludes
only that frozen upstream file. The two original Colab notebooks now reference
the existing requirements-gpu.txt and a separate reproduction Drive folder.
See REPRODUCIBILITY_STATUS.md for current verification scope.
