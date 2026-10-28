# Spectral Stress Tests Reveal Protocol-Dependent Robustness and Calibration in ImageNet Classifiers

Code, frozen experiment specifications and supporting results for the study by Mehmet Koc.
This merged submission snapshot retains the original Stage-1 and Stage-2 v4
implementation and adds the completed input-statistics controls and full-cohort
calibration replay. Existing source, configurations and frozen evidence are retained.

## Start here: verify saved results without a GPU

Use Python 3.12. From this repository root:

```bash
python -m pip install -r requirements-analysis.txt pytest
python tools/verify_reproducibility.py
```

The command runs six lightweight repository tests, twelve numerical checks,
checks the original/follow-up fitting inputs, and reconciles saved estimates.
It verifies 96 input-control effects, 30 balanced calibration scores and 108
calibration effects. A JSON record is written to validation/REPRODUCIBILITY_CHECK.json.
Reconciliation runs in a temporary directory and leaves frozen records intact.
The numerical check fixtures are implementation tests, not empirical paper data.

For the complete original PyTorch unit/planning suite:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The complete suite requires torch and torchvision. GitHub Actions runs that
suite and the additional saved-result verification on each push.

## Experiment-to-code map

| Experiment | Entry point | Included evidence |
|---|---|---|
| Six-checkpoint spectral diagnostics | notebooks/Stage1_Confirmatory_Colab.ipynb; src/stage1/run_stage1.py | artifacts/frozen_runs/stage1/: 432 cells, manifests and phase markers |
| Original calibration on the frozen split | notebooks/Stage2_Calibration_Colab.ipynb; src/stage2/run_stage2.py | artifacts/frozen_runs/stage2/: configuration, 1,963/1,962 split, fit coefficients and 780 summaries |
| Mean/contrast controls | notebooks/01_Mean_Contrast_Controls.ipynb | analysis_results/strengthening/controls/: 63 jobs, 126 summaries and 96 effects |
| Full-cohort calibration including pooled scalar | notebooks/02_Full_Calibration_Comparison.ipynb | analysis_results/strengthening/calibration/: 78 jobs, 780 summaries, 30 balanced scores and 108 effects |

## Rerun model inference on a GPU

1. Extract the complete repository ZIP to /content/spectral-stress-testing in
   Colab, or clone the published submission release there.
2. Select a GPU runtime and use the recorded environment in the relevant
   environment.json. Original runs used Python 3.12, timm 1.0.28,
   torch/torchvision 2.11.0/0.26.0 and a Tesla T4. CUDA variants differ between
   the original run and follow-up replay; both records are retained.
3. For the original diagnostic run, run all six phases listed in
   configs/stage1_paper.json. The notebook starts with checkpoint_replication;
   change STAGE1_PHASE and rerun its runner cell for each remaining phase.
4. Run the original calibration notebook. The original-run notebooks use the
   separate Drive folder CVIU_Original_Reproduction and preserve the required
   run-name identities. See docs/REPRODUCIBILITY.md for Stage-1 inputs.
5. Run the two additional notebooks in their stated order. They accept the
   complete repository ZIP or the prior supplementary ZIP, stage frozen fitting
   inputs in a separate project, and stream progress into the notebook cells.

For additional experiments, exact commands and inputs are in
[docs/ADDITIONAL_EXPERIMENTS.md](docs/ADDITIONAL_EXPERIMENTS.md).
The calibration preparation step can regenerate fitting-view logits without
refitting the saved clean-TS/FCTS/CCTS coefficients. The replay runner fits only
the pooled scalar anew and evaluates methods on identical within-cell logits.

## Data, coverage and interpretation

Imagenette and ImageNet-A are downloaded from their original providers.
Dataset images, weights and large original per-image/logit caches are not
redistributed. The repository includes manifests, frozen protocols and fitting
parameters, aggregate results and recorded paired effects. GPU reruns regenerate
required inference outputs. Full historical uncertainty reproduction also needs
the corresponding per-image records and original analysis procedure; see
[docs/REPRODUCIBILITY_STATUS.md](docs/REPRODUCIBILITY_STATUS.md).

The original v4 runner remains under src/. The follow-up reference modules
remain under strengthening_experiments/ with their recorded byte hashes. Core
Stage-1 and Stage-2 mathematical implementations match structurally; these
versions are not silently substituted for one another. One frozen upstream
module retains original non-English log messages to preserve file identity.

SHA256SUMS.txt covers the distributed source and scientific records. Re-running
analyses may change derived files; retain an unmodified copy for hash checks.
Citation metadata are in CITATION.cff. Existing licensing documentation is
retained; no new reuse license is asserted by this merge.
