# Code and supporting results

**Article:** Spectral Stress Tests Reveal Protocol-Dependent Robustness and Calibration in ImageNet Classifiers  
**Author:** Mehmet Koc, Eskisehir Technical University  
**Archive:** Supplementary_Code_and_Results.zip

This archive accompanies the manuscript as supplementary material. It contains
the supplied scientific implementations, frozen experiment specifications,
fitted calibration parameters, numerical summaries, paired analysis outputs,
and scripts for rebuilding selected manuscript figures and tables.

Start with the CPU checks below. Reading and validating the saved summaries
does not require a GPU, dataset download, or a new inference run.

## 1. Contents and entry points

| Path | Purpose |
|---|---|
| analysis_results/ | Published numerical summaries, manifests, model metadata, parameters and paired effects |
| analysis_results/provenance/stage1_summary.csv | All 432 diagnostic cells |
| analysis_results/provenance/stage2/ | Frozen calibration split, conditions and fitted coefficients |
| analysis_results/strengthening/controls/ | Completed 63-job mean/contrast experiment and 96 paired effects |
| analysis_results/strengthening/calibration/ | Completed 78-job full-cohort replay, 30 balanced scores and 108 paired effects |
| strengthening_experiments/ | Scientific modules and runners used for the additional controls and full-cohort replay |
| analysis_tools/ | Record-based energy, ranking and calibration analyses |
| assets/ | Saved-summary reconciliation and control-plot builder |
| notebooks/ | Colab entry points for the two additional experiments |
| prepare_replay_project.py | Stage frozen records in a separate project and optionally regenerate fitting logits |

Run commands from the top-level archive directory.

## 2. Environment

The completed runs used Python 3.12, timm 1.0.28 and a Tesla T4.
The original calibration environment records PyTorch 2.11.0+cu128 and
torchvision 0.26.0+cu128; the later replay records PyTorch 2.11.0+cu130 and
torchvision 0.26.0+cu130. Exact saved environment records are included beside
each run's summaries.

For CPU analysis and figure generation:

    python -m pip install -r requirements-analysis.txt

For GPU inference, use a CUDA-compatible PyTorch/torchvision installation and:

    python -m pip install timm==1.0.28 tqdm

The recorded torch/torchvision pair is 2.11.0/0.26.0. Select a wheel compatible
with the GPU environment using the official PyTorch installation instructions.
The Colab notebooks retain the runtime-provided CUDA installation.

## 3. CPU checks and saved-result reconciliation

    python strengthening_experiments/cpu_checks.py
    python assets/build_strengthening_results.py

The first command checks mean/contrast transformations, pooled-scalar fitting,
ECE resampling, frozen-protocol protection and the paired-analysis routines.
Its numerical fixtures test implementation behavior; they are not paper data.

The second command uses the actual saved summaries. It checks 96 control
contrasts against cell accuracies, independently recomputes all 30 balanced
calibration scores, reconciles all 108 calibration effects, checks run counts,
verifies the frozen experiment-source hashes and checks top-1 invariance.
It writes table snippets under analysis_results/strengthening/tables/,
a local validation record, and the control plot figures/Figure_16.*.
Figure_16.pdf is Figure 6 in the compiled article; source filenames are not
the same as the figure numbers assigned by LaTeX.

## 4. Data and checkpoints

Obtain images from the original providers:

- Imagenette v2-320: https://github.com/fastai/imagenette
- ImageNet-A: https://github.com/hendrycks/natural-adv-examples
- timm checkpoints: https://github.com/huggingface/pytorch-image-models

The recorded manifests identify images by global index, relative path and
ImageNet-1K target. The inference modules check class alignment and obtain
the pretrained checkpoints through timm. Dataset images and pretrained weights
are not redistributed here.

The checkpoint aliases used in the saved records are:
resnet50_a1, resnet50_a2, vitb16_in1k, vitb16_in21k, swin_t_in1k and swin_t_in22k.
Their full timm names, normalization and preprocessing are stored in
analysis_results/provenance/model_metadata/. The three additional-experiment
anchors are resnet50_a1, swin_t_in1k and vitb16_in1k.

## 5. Rerun the mean/contrast experiment

Use the Colab notebook notebooks/01_Mean_Contrast_Controls.ipynb, or a GPU
environment. A new writable PROJECT directory keeps regenerated outputs
separate from the supplied records.

    python prepare_replay_project.py --project PROJECT
    python strengthening_experiments/run_strengthening.py --project PROJECT --task controls --mode benchmark
    python strengthening_experiments/run_strengthening.py --project PROJECT --task controls --mode full

The runner obtains the public datasets and model checkpoints. The full control
experiment covers 3,925 Imagenette images, 21 conditions per anchor, 63 jobs,
247,275 image forwards and 126 method/protocol summaries. The runner then
computes the 96 paired accuracy contrasts.

Existing completed shards are reused only when their design identities agree.
The benchmark is a small timing/implementation check and is not a paper result.
The full-run analysis uses 1,000 bootstrap repetitions.

## 6. Rerun the full-cohort calibration comparison

Use notebooks/02_Full_Calibration_Comparison.ipynb, or:

    python prepare_replay_project.py --project PROJECT --regenerate-fit-logits
    python strengthening_experiments/run_strengthening.py --project PROJECT --task calibration --mode benchmark
    python strengthening_experiments/run_strengthening.py --project PROJECT --task calibration --mode full

The preparation command first stages the original configuration, split,
manifests and fitted coefficients. Its optional GPU step regenerates the nine
fitting-view logit files for the three anchors on clean/LP20/HP20. It does not
refit clean TS, FCTS or CCTS. Pooled TS is fitted by the replay runner using
only these calibration views.

Existing original calibration logits can instead be placed at:

    PROJECT/runs/spectral_stage2_calibration_pilot_v1__27bb9eee96fb/calibration_logits/MODEL/CONDITION.npz

The full replay covers 1,962 held-out Imagenette images and all 7,500
ImageNet-A images under their recorded condition grids. It produces 78 jobs,
335,790 image forwards, 780 summaries and 108 aggregate/family/clean effects.
ImageNet-A is evaluation-only. The split, descriptor definitions and saved
clean-TS/FCTS/CCTS coefficients remain fixed.

The omitted large caches are generated under the new PROJECT/runs/ directory:
raw_logits/ contains chunked logits and compact_predictions/ contains indexed
probabilities, correctness and loss contributions. Numerical differences from
the published run should be examined against the included environment and
REPLAY_AGREEMENT.csv records. No new outputs replace the published records.

## 7. Original implementation and source identity

The original Stage-1 and Stage-2 v4 implementation is available under src/.
The additional experiments retain their exact reference modules under
strengthening_experiments/, as recorded in UPSTREAM_MODULES.json. They are not
silently replaced by the original runner. The frozen split, configuration and
all three fitted calibration parameter files agree across both evidence trees;
tools/verify_reproducibility.py checks those correspondences.

The original master diagnostic-analysis v2 program is not the same program as
src/stage1/analyze_stage1.py. Historical aggregate outputs are included; exact
historical bootstrap intervals require that original analysis and per-image
records. Recomputing their point estimates and rerunning model inference are
separate verification tasks.

## 9. File identity and validation

SHA256SUMS.txt records the distributed files before optional reruns.
validation/ contains the local verification record.
Rebuilders can change derived files in your extracted copy; keep an unmodified
copy if checking original hashes.

The manuscript and this archive describe the same checkpoints, fitting split,
condition coverage and comparison weights. Source transcripts and frozen
experiment records are preserved to make the relationship between
implementations and published outputs explicit.

