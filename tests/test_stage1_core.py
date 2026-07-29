from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "stage1"))

from stage1_core import (  # noqa: E402
    Condition,
    adaptive_ece,
    apply_spectral_filter,
    deterministic_gaussian_noise,
    ece_equal_width,
)


def test_spectral_complementarity() -> None:
    torch.manual_seed(7)
    images = torch.rand(5, 3, 32, 32)
    for family in ("hard", "gaussian"):
        for radius in (3, 7, 11):
            lp = apply_spectral_filter(images, Condition("lp", family, "low_pass", radius, "linear"))
            hp = apply_spectral_filter(images, Condition("hp", family, "high_pass", radius, "linear"))
            assert float((lp + hp - images).abs().max()) < 2e-5


def test_noise_is_chunk_invariant() -> None:
    torch.manual_seed(9)
    images = torch.rand(5, 3, 16, 16)
    indices = list(range(5))
    whole = deterministic_gaussian_noise(images, indices, 0.1, 123, "test")
    chunked = torch.cat([
        deterministic_gaussian_noise(images[:2], indices[:2], 0.1, 123, "test"),
        deterministic_gaussian_noise(images[2:], indices[2:], 0.1, 123, "test"),
    ])
    assert torch.equal(whole, chunked)


def test_calibration_metric_regression() -> None:
    confidence = np.array([0.1, 0.3, 0.7, 0.9])
    correct = np.array([0, 1, 1, 1], dtype=bool)
    ece, mce = ece_equal_width(confidence, correct, n_bins=4)
    assert np.isclose(ece, 0.3)
    assert np.isclose(mce, 0.7)
    assert np.isclose(adaptive_ece(confidence, correct, n_bins=4), 0.3)
