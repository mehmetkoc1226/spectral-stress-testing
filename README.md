# Auditable Spectral Stress Testing of ImageNet Models

This repository contains audited experiment code, configuration files, compact derived results, frozen run metadata, and reproducibility documentation for the spectral stress-testing study.

## Scope

The study evaluates six public ImageNet checkpoints from ResNet-50, ViT-B/16, and Swin-T under complementary hard and Gaussian spectral interventions, clipping controls, natural corruptions, output-space sensitivity, checkpoint replication, and post-hoc calibration analyses.

The repository separates:

- **Stage 1:** spectral stress testing, per-image prediction records, paired comparisons, and protocol controls;
- **Stage 2:** preregistered frequency-conditioned temperature scaling and matched adaptive baselines;
- **Figure generation:** deterministic qualitative example selection from the frozen Stage-1 run;
- **Compact artifacts:** derived aggregate tables, exact dataset manifests, completion evidence, fitted calibration parameters, and deterministic qualitative-selection metadata.

## Source provenance

The canonical source selection and all notebook-level corrections are documented in [`docs/PROVENANCE.md`](docs/PROVENANCE.md). The repository integrates the Stage-1 Gaussian-r20 job-planning patch and the executed Stage-2 v1.4 cache hotfix. These changes affect orchestration and file transport only; the scientific intervention, metric, and calibration definitions are unchanged.

## Quick validation

```bash
python -m pip install -r requirements-test.txt
pytest -q
```

The unit tests cover spectral complementarity, deterministic corruption invariance, calibration top-1 invariance, configuration registries, planning smoke tests, and the English-only repository rule.

## Full Colab environment

The recorded GPU experiment environment used:

- Python 3.12.13
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0
- timm 1.0.28
- NVIDIA Tesla T4

Install the GPU experiment dependencies in a GPU-enabled Colab runtime:

```bash
python -m pip install -r requirements-gpu.txt
```

## Stage 1

Configure the run with environment variables or use the English Colab notebook in `notebooks/`.

```bash
export STAGE1_RUN_PROFILE=paper
export STAGE1_PHASE=checkpoint_replication
export STAGE1_RUN_NAME=spectral_stage1_confirmatory_v1
python src/stage1/run_stage1.py
```

Run the remaining phases in the order listed in `configs/stage1_paper.json`. The runner writes atomically and resumes completed chunks.

## Stage 2

Stage 2 is fail-closed with respect to the frozen Stage-1 run and writes to a configuration-hashed output directory.

```bash
export STAGE2_RUN_PROFILE=paper
export STAGE2_RUN_NAME=spectral_stage2_calibration_pilot_v1
python src/stage2/run_stage2.py
```

Then run the analysis and packaging scripts on the completed Stage-2 run directory:

```bash
python src/stage2/analyze_stage2.py /path/to/stage2_run --bootstrap-reps 1000
python src/stage2/package_stage2_results.py /path/to/stage2_run
```

## Data

Imagenette and ImageNet-A are downloaded from their original public providers by the runners. Dataset images and pretrained model weights are not redistributed in this repository.

Exact dataset and calibration-split manifests from the successful full runs are included under `artifacts/frozen_runs/`. Raw datasets, model weights, prediction chunks, and selected/full logits are not stored in Git history. See [`manifests/README.md`](manifests/README.md).

## Reproducibility status

Static compilation and scientific unit tests pass in the repository build environment. A complete end-to-end GPU rerun is not performed by the repository builder because it requires the paper's CUDA/timm environment, public dataset downloads, pretrained weights, and the frozen Stage-1 artifacts. Frozen completion markers and recorded run metadata document the successful full runs; the validation boundary is summarized in [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The canonical repository is `https://github.com/mehmetkoc1226/spectral-stress-testing`.

## License

No reuse license is asserted by this preparation package. Select and add an appropriate code license before making the repository public. See [`docs/LICENSING.md`](docs/LICENSING.md).

## Included frozen evidence

The repository includes the compact evidence needed to audit the reported experiment design and aggregate analyses:

- `artifacts/frozen_runs/stage1/`: full-run configuration, environments, manifests, phase markers, model metadata, summary table, and Stage-1 aggregate analyses;
- `artifacts/frozen_runs/stage2/`: full-run configuration, environments, manifests, fitted method parameters, completion marker, summary table, and calibration comparison analyses;
- `artifacts/summary_tables/`: compact cross-analysis tables used to check the principal reported results.

The manuscript source and compiled article are intentionally not included in this code repository.
