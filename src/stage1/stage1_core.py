# -*- coding: utf-8 -*-
"""Pure scientific utilities for the Stage-1 spectral study.

This module intentionally has no timm or dataset download dependency so its
invariants can be unit-tested before a Colab run begins.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Condition:
    key: str
    family: str          # clean, hard, gaussian, blur, noise
    mode: str            # clean, low_pass, high_pass, blur, noise
    radius: int | None = None
    postprocess: str = "linear"  # none, linear, clip
    geometry: str = "common"     # common, native
    severity: float | None = None


def stable_int_seed(*parts: Any, modulo: int = 2**31 - 1) -> int:
    blob = "||".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:16], 16) % modulo


_MASK_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def radial_distance(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = ("distance", height, width, str(device), str(dtype))
    if key not in _MASK_CACHE:
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        dist = torch.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
        _MASK_CACHE[key] = dist.view(1, 1, height, width)
    return _MASK_CACHE[key]


def spectral_mask(
    height: int,
    width: int,
    radius: int,
    family: str,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a radial low/high-pass transfer function.

    Gaussian masks are parameterized so that the low-pass gain is 0.5 at the
    reported radius: sigma = r / sqrt(2 ln 2). The high-pass mask is exactly
    1 - low-pass, preserving complementarity.
    """
    key = ("mask", height, width, int(radius), family, mode, str(device), str(dtype))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    d = radial_distance(height, width, device, dtype)
    if family == "hard":
        low = (d <= float(radius)).to(dtype)
    elif family == "gaussian":
        sigma = float(radius) / math.sqrt(2.0 * math.log(2.0))
        low = torch.exp(-0.5 * (d / sigma) ** 2)
    else:
        raise ValueError(f"Unsupported spectral family: {family}")

    if mode == "low_pass":
        mask = low
    elif mode == "high_pass":
        mask = 1.0 - low
    else:
        raise ValueError(f"Unsupported spectral mode: {mode}")
    _MASK_CACHE[key] = mask
    return mask


def apply_spectral_filter(images: torch.Tensor, condition: Condition) -> torch.Tensor:
    if condition.family == "clean":
        return images
    if condition.family not in {"hard", "gaussian"}:
        raise ValueError("apply_spectral_filter only accepts hard/gaussian conditions")
    if condition.radius is None:
        raise ValueError("Spectral condition requires a radius")

    spectrum = torch.fft.fftshift(torch.fft.fftn(images, dim=(-2, -1)), dim=(-2, -1))
    mask = spectral_mask(
        images.shape[-2], images.shape[-1], condition.radius,
        condition.family, condition.mode, images.device, images.dtype,
    )
    recon = torch.fft.ifftn(
        torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), dim=(-2, -1)
    ).real
    if condition.postprocess == "linear":
        return recon
    if condition.postprocess == "clip":
        return recon.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported postprocess: {condition.postprocess}")


def gaussian_kernel2d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k1 = torch.exp(-(x**2) / (2.0 * sigma**2))
    k1 = k1 / k1.sum()
    return torch.outer(k1, k1)


def apply_spatial_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel2d(sigma, images.device, images.dtype)
    k = kernel.shape[0]
    weight = kernel.view(1, 1, k, k).repeat(images.shape[1], 1, 1, 1)
    return F.conv2d(images, weight, padding=k // 2, groups=images.shape[1])


def deterministic_gaussian_noise(
    images: torch.Tensor,
    global_indices: Iterable[int],
    std: float,
    seed: int,
    dataset_name: str,
) -> torch.Tensor:
    """Add per-image Gaussian noise invariant to batch/chunk ordering."""
    parts = []
    for image, idx in zip(images, global_indices):
        g = torch.Generator(device="cpu")
        g.manual_seed(stable_int_seed(seed, dataset_name, int(idx), "gaussian_noise"))
        noise = torch.randn(image.shape, generator=g, dtype=torch.float32)
        parts.append(noise)
    stacked = torch.stack(parts, dim=0).to(device=images.device, dtype=images.dtype)
    return (images + float(std) * stacked).clamp(0.0, 1.0)


def apply_condition(
    images: torch.Tensor,
    condition: Condition,
    global_indices: Iterable[int],
    seed: int,
    dataset_name: str,
) -> torch.Tensor:
    if condition.family == "clean":
        return images
    if condition.family in {"hard", "gaussian"}:
        return apply_spectral_filter(images, condition)
    if condition.family == "blur":
        if condition.severity is None:
            raise ValueError("Blur condition requires severity")
        return apply_spatial_blur(images, float(condition.severity)).clamp(0.0, 1.0)
    if condition.family == "noise":
        if condition.severity is None:
            raise ValueError("Noise condition requires severity")
        return deterministic_gaussian_noise(
            images, global_indices, float(condition.severity), seed, dataset_name
        )
    raise ValueError(f"Unsupported condition family: {condition.family}")


def spectral_diagnostics(original: torch.Tensor, transformed: torch.Tensor) -> dict[str, np.ndarray]:
    dims = (1, 2, 3)
    original_energy = original.square().sum(dim=dims).clamp_min(1e-12)
    transformed_energy = transformed.square().sum(dim=dims)
    flat = transformed.flatten(1)
    return {
        "energy_ratio": (transformed_energy / original_energy).detach().cpu().numpy().astype(np.float32),
        "filtered_min": flat.min(dim=1).values.detach().cpu().numpy().astype(np.float32),
        "filtered_max": flat.max(dim=1).values.detach().cpu().numpy().astype(np.float32),
        "fraction_below_zero": (flat < 0).float().mean(dim=1).detach().cpu().numpy().astype(np.float32),
        "fraction_above_one": (flat > 1).float().mean(dim=1).detach().cpu().numpy().astype(np.float32),
    }


def ece_equal_width(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if not np.any(mask):
            continue
        gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def adaptive_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    n = len(confidence)
    if n == 0:
        return float("nan")
    order = np.argsort(confidence, kind="mergesort")
    groups = np.array_split(order, min(n_bins, n))
    value = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        gap = abs(float(correct[group].mean()) - float(confidence[group].mean()))
        value += (len(group) / n) * gap
    return float(value)


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    if len(confidence) == 0:
        return float("nan")
    order = np.argsort(-confidence, kind="mergesort")
    errors = (~correct[order]).astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(cumulative_risk.mean())


def bootstrap_basic_ci(
    confidence: np.ndarray,
    correct: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if reps <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(confidence)
    values = {"accuracy": [], "average_confidence": [], "acms": []}
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        c = confidence[idx]
        ok = correct[idx]
        values["accuracy"].append(float(ok.mean()))
        values["average_confidence"].append(float(c.mean()))
        values["acms"].append(float(c[~ok].mean()) if np.any(~ok) else np.nan)
    result: dict[str, float] = {}
    for name, seq in values.items():
        arr = np.asarray(seq, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        result[f"{name}_ci_low"] = float(np.percentile(arr, 2.5)) if arr.size else float("nan")
        result[f"{name}_ci_high"] = float(np.percentile(arr, 97.5)) if arr.size else float("nan")
    return result


def compute_metrics_from_sufficient(
    confidence: np.ndarray,
    correct: np.ndarray,
    top5_correct: np.ndarray,
    nll_per_image: np.ndarray,
    brier_per_image: np.ndarray,
    n_bins: int,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    errors = ~correct
    ece, mce = ece_equal_width(confidence, correct, n_bins)
    accuracy = float(correct.mean())
    avg_conf = float(confidence.mean())
    acms = float(confidence[errors].mean()) if np.any(errors) else float("nan")
    conf_correct = float(confidence[correct].mean()) if np.any(correct) else float("nan")

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if np.unique(errors).size == 2:
            error_score = 1.0 - confidence
            error_auroc = float(roc_auc_score(errors.astype(np.int64), error_score))
            error_aupr = float(average_precision_score(errors.astype(np.int64), error_score))
        else:
            error_auroc = float("nan")
            error_aupr = float("nan")
    except Exception:
        error_auroc = float("nan")
        error_aupr = float("nan")

    if 0.0 < accuracy < 1.0 and np.isfinite(acms) and np.isfinite(conf_correct):
        identity_residual = avg_conf - (accuracy * conf_correct + (1.0 - accuracy) * acms)
    else:
        identity_residual = float("nan")

    result = {
        "n": int(len(correct)),
        "accuracy": accuracy,
        "top5_accuracy": float(np.asarray(top5_correct, dtype=np.float64).mean()),
        "average_confidence": avg_conf,
        "acms": acms,
        "confidence_correct": conf_correct,
        "ece": ece,
        "adaptive_ece": adaptive_ece(confidence, correct, n_bins),
        "mce": mce,
        "nll": float(np.asarray(nll_per_image, dtype=np.float64).mean()),
        "brier": float(np.asarray(brier_per_image, dtype=np.float64).mean()),
        "error_auroc": error_auroc,
        "error_aupr": error_aupr,
        "aurc": aurc(confidence, correct),
        "error_count": int(errors.sum()),
        "identity_residual": float(identity_residual),
    }
    result.update(bootstrap_basic_ci(confidence, correct, bootstrap_reps, seed))
    return result


def paired_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape")
    diff = a - b
    estimate = float(diff.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=np.float64)
    n = len(diff)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()
    p_two_sided = float(2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0)))
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": min(1.0, p_two_sided),
    }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=np.float64)
    out = np.full_like(p, np.nan)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out
    vals = p[finite_idx]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite_idx] = restored
    return out
