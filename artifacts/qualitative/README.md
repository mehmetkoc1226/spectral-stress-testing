# Qualitative figure package

Frozen run: `spectral_stage1_confirmatory_v1__7044005bc7a4`

Outputs:
- `figure_condition_gallery.png/.pdf`
- `figure_class_contrast.png/.pdf`
- `figure_examples_manifest.csv`
- candidate rankings and exact source-image copies

Selection discipline:
- Gallery: LP/HP outcomes were not used. The image was selected among samples
  correctly classified on clean input by all six checkpoints, using a fixed
  SHA-256 ordering.
- French horn / tench: selected using the prespecified class-direction rule at
  hard r=10 across the six frozen checkpoints.
- Quantitative claims remain based on the complete paired analysis.

Display note:
Signed-linear reconstructions were clipped to [0,1] only for human-readable RGB
rendering. Frozen inference used signed tensors before model-specific normalization.
