# Repository test report

Build date: 2026-07-29

## Source selection

- Stage 1 uses the executed full Stage-1 source plus the ImageNet-A Gaussian r=20 linear job-planning patch.
- Stage 2 uses the v1.4 shared-drive transport source with the executed cache-path hotfix integrated directly.
- Qualitative generation uses the v2 compact source.
- Frozen run metadata are taken from the successful full Stage-1 run `7044005bc7a4` and full Stage-2 run `27bb9eee96fb`.

## Tests performed

### Notebook and source validation

- All selected notebooks parsed as valid nbformat 4 JSON.
- Canonical embedded Python scripts were extracted and compared by SHA-256 during repository preparation.
- Duplicate Stage-2 notebook copies were identified and excluded.
- All repository Python files passed `compileall`.
- All cleaned Colab notebooks contain no saved execution counts or outputs.

### Scientific unit tests

Stage-1 pure utilities:

- hard-mask LP + HP maximum reconstruction error: `3.5762786865234375e-07`;
- Gaussian-mask LP + HP maximum reconstruction error: `3.5762786865234375e-07`;
- deterministic Gaussian-noise chunk invariance error: `0.0`;
- calibration metric regression checks passed.

Stage-2 built-in self-test:

- maximum spectral complementarity error: `3.5762786865234375e-07`;
- spectral descriptor shape: `[5, 3]`;
- positive-temperature top-1 invariance: passed;
- fitted temperatures remained within configured bounds.

Runner planning smoke tests:

- Stage-1 full design contains 6 checkpoints;
- Stage-1 `all_full` plan contains 204 jobs;
- Stage-1 ImageNet-A Gaussian r=20 linear patch contains 12 jobs;
- Stage-2 full design contains 3 anchor checkpoints and 78 jobs.

Frozen artifact checks:

- Stage-1 summary rows: 432;
- Stage-1 Imagenette manifest rows: 3,925;
- Stage-1 ImageNet-A manifest rows: 7,500;
- all six Stage-1 phase completion markers are present;
- Stage-2 completion marker reports 78/78 completed jobs;
- Stage-2 summary rows: 780;
- Stage-2 calibration split rows: 3,925;
- Stage-2 Imagenette manifest rows: 3,925;
- Stage-2 ImageNet-A manifest rows: 7,500;
- fitted parameters are present for all three anchor models.

Repository test suite result:

```text
11 passed
```

### Language and privacy checks

- No Turkish explanatory text or Turkish characters were detected in repository code, notebooks, documentation, configuration files, manifests, or compact CSV outputs.
- The original Turkish freeze-rule sentence was translated into English and marked as language-normalized in the copied public metadata record.
- No API keys, passwords, access tokens, private Drive shortcut identifiers, or private Drive URLs were included.
- Dataset source images, model weights, raw prediction chunks, selected logits, and full logits were excluded.

## End-to-end limitation

A fresh full-scale GPU rerun was not performed in the repository build environment. That execution requires the recorded CUDA/PyTorch environment, public datasets, public pretrained checkpoints, and persistent storage for the full run artifacts. The repository therefore distinguishes unit/integrity tests and frozen completion evidence from a new end-to-end rerun.

## Code-only policy

The manuscript source, compiled article, manuscript figure directory, and submission records are intentionally excluded. Compact experiment outputs and frozen reproducibility metadata remain because they are scientific artifacts rather than manuscript files.
