# Frozen run metadata

This directory contains compact, non-image artifacts from the successful full runs used for the study.

Included:

- exact configurations and recorded software environments;
- dataset and calibration-split manifests;
- model and label-space metadata;
- run and phase completion markers;
- aggregate summaries and paired/bootstrap analysis tables;
- fitted Stage-2 calibration parameters and scientific self-test output.

Excluded from Git history:

- raw datasets and pretrained model weights;
- per-image prediction chunks;
- selected or full logits;
- cache directories and temporary run files.

The excluded large artifacts may be attached to a versioned release when an independent Stage-2 rerun from frozen Stage-1 logits is required.
