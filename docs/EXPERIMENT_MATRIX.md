# Experiment matrix

## Models

The Stage-1 matrix contains two public checkpoints for each exact architecture:

- ResNet-50: `resnet50.a1_in1k`, `resnet50.a2_in1k`;
- ViT-B/16: `vit_base_patch16_224.augreg_in1k`, `vit_base_patch16_224.augreg_in21k_ft_in1k`;
- Swin-T: `swin_tiny_patch4_window7_224.ms_in1k`, `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`.

## Intervention families

- complementary hard low/high-pass masks at radii 10, 20, and 40;
- complementary Gaussian low/high-pass masks at radii 10 and 20;
- signed-linear and clipped inverse-transform controls at radius 20;
- Gaussian blur and additive Gaussian noise controls;
- clean native and common-geometry baselines.

## Output protocols

Native ImageNet-1K output and restricted dataset-subset output are retained as distinct decision problems. Restricted output is a sensitivity analysis and is not described as recalibration of the native classifier.

## Stage-2 calibration

Fitting conditions are clean common geometry and hard low/high-pass radius 20. Unseen evaluation includes held-out clean data, unseen radii, unseen Gaussian masks, natural corruptions, and ImageNet-A. FCTS is compared with clean scalar temperature scaling in the preregistered analysis and with confidence-conditioned scaling in a labeled post-hoc specificity test.
