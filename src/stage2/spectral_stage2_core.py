# -*- coding: utf-8 -*-
"""Pure scientific utilities for the Stage-2 calibration pilot.

This module contains no dataset download or timm dependency.  It can be tested
on CPU before the Colab GPU run begins.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA_VERSION = 2
T_MIN = 0.05
T_MAX = 20.0
RADIAL_BINS = 64
EPS = 1e-12


@dataclass(frozen=True)
class Condition:
    key: str
    family: str
    mode: str
    radius: int | None = None
    postprocess: str = "linear"
    severity: float | None = None


def stable_int_seed(*parts: Any, modulo: int = 2**31 - 1) -> int:
    blob = "||".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:16], 16) % modulo


_MASK_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_RADIAL_CACHE: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}


def radial_distance(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (height, width, str(device), str(dtype))
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
    key = (height, width, int(radius), family, mode, str(device), str(dtype))
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
    kernel = torch.exp(-(x**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def apply_spatial_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel2d(sigma, images.device, images.dtype)
    size = kernel.shape[0]
    weight = kernel.view(1, 1, size, size).repeat(images.shape[1], 1, 1, 1)
    return F.conv2d(images, weight, padding=size // 2, groups=images.shape[1])


def deterministic_gaussian_noise(
    images: torch.Tensor,
    global_indices: Iterable[int],
    std: float,
    seed: int,
    dataset_name: str,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for image, idx in zip(images, global_indices):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_int_seed(seed, dataset_name, int(idx), "gaussian_noise"))
        noise = torch.randn(image.shape, generator=generator, dtype=torch.float32)
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


def _radial_geometry(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, str(device), str(dtype), int(n_bins))
    if key in _RADIAL_CACHE:
        return _RADIAL_CACHE[key]
    d = radial_distance(height, width, device, dtype).view(height, width)
    dmax = d.max().clamp_min(1.0)
    rho = d / dmax  # exactly [0, 1], including corners
    bin_idx = torch.clamp((rho * n_bins).long(), max=n_bins - 1)
    _RADIAL_CACHE[key] = (rho, bin_idx)
    return rho, bin_idx


def spectral_descriptors(images: torch.Tensor, cutoff_radius_pixels: float = 20.0) -> torch.Tensor:
    """Return [log high/low energy, centroid, radial entropy] per image.

    Descriptors are computed on the transformed, model-normalization-free image.
    Radius is normalized by the maximum corner radius; the high/low threshold is
    the corresponding normalized value for ``cutoff_radius_pixels``.
    """
    if images.ndim != 4:
        raise ValueError("Expected NCHW images")
    spectrum = torch.fft.fftshift(torch.fft.fftn(images.float(), dim=(-2, -1)), dim=(-2, -1))
    power = spectrum.abs().square().mean(dim=1)  # N,H,W
    n, h, w = power.shape
    rho, bin_idx = _radial_geometry(h, w, power.device, power.dtype, RADIAL_BINS)
    dmax_pixels = float(math.sqrt((h // 2) ** 2 + (w // 2) ** 2))
    rho0 = float(cutoff_radius_pixels) / max(dmax_pixels, 1.0)
    low_mask = rho <= rho0
    low_energy = power[:, low_mask].sum(dim=1)
    high_energy = power[:, ~low_mask].sum(dim=1)
    log_ratio = torch.log((high_energy + EPS) / (low_energy + EPS))
    total = power.sum(dim=(1, 2)).clamp_min(EPS)
    centroid = (power * rho.view(1, h, w)).sum(dim=(1, 2)) / total

    radial = torch.zeros((n, RADIAL_BINS), device=power.device, dtype=power.dtype)
    flat_idx = bin_idx.flatten().view(1, -1).expand(n, -1)
    radial.scatter_add_(1, flat_idx, power.flatten(1))
    q = radial / radial.sum(dim=1, keepdim=True).clamp_min(EPS)
    entropy = -(q * torch.log(q.clamp_min(EPS))).sum(dim=1) / math.log(RADIAL_BINS)
    return torch.stack([log_ratio, centroid, entropy], dim=1)


def protocol_view(
    logits: np.ndarray,
    targets1000: np.ndarray,
    restricted_indices: np.ndarray,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return protocol logits, target positions, and global-class lookup."""
    logits = np.asarray(logits)
    targets1000 = np.asarray(targets1000, dtype=np.int64)
    if protocol == "imagenet1000":
        return logits, targets1000, np.arange(1000, dtype=np.int64)
    if not protocol.startswith("restricted"):
        raise ValueError(f"Unknown protocol: {protocol}")
    restricted_indices = np.asarray(restricted_indices, dtype=np.int64)
    lookup = np.full(1000, -1, dtype=np.int64)
    lookup[restricted_indices] = np.arange(len(restricted_indices), dtype=np.int64)
    target_pos = lookup[targets1000]
    if np.any(target_pos < 0):
        raise RuntimeError("Target outside restricted class set")
    return logits[:, restricted_indices], target_pos, restricted_indices


def confidence_features(logits: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    probs = torch.softmax(tensor, dim=1)
    top2 = torch.topk(tensor, k=min(2, tensor.shape[1]), dim=1).values
    if tensor.shape[1] == 1:
        margin = torch.zeros(len(tensor), dtype=tensor.dtype)
    else:
        margin = top2[:, 0] - top2[:, 1]
    entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=1) / math.log(tensor.shape[1])
    max_prob = probs.max(dim=1).values
    return torch.stack([margin, entropy, max_prob], dim=1).cpu().numpy().astype(np.float64)


def fit_standardizer(features: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_standardizer(features: np.ndarray, stats: Mapping[str, Sequence[float]]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    return (np.asarray(features, dtype=np.float64) - mean) / std


def _inverse_bounded_temperature(temperature: float) -> float:
    fraction = (float(temperature) - T_MIN) / (T_MAX - T_MIN)
    fraction = min(max(fraction, 1e-6), 1.0 - 1e-6)
    return math.log(fraction / (1.0 - fraction))


def temperature_from_parameters(
    parameters: Sequence[float],
    features_standardized: np.ndarray | None,
) -> np.ndarray:
    params = np.asarray(parameters, dtype=np.float64)
    if features_standardized is None:
        linear = np.full(1, params[0], dtype=np.float64)
    else:
        x = np.asarray(features_standardized, dtype=np.float64)
        linear = params[0] + x @ params[1:]
    sig = 1.0 / (1.0 + np.exp(-np.clip(linear, -60.0, 60.0)))
    return T_MIN + (T_MAX - T_MIN) * sig


def fit_temperature_model(
    logits: np.ndarray,
    targets: np.ndarray,
    features_standardized: np.ndarray | None = None,
    regularization: float = 1e-3,
    max_iter: int = 200,
    init_temperature: float = 1.0,
) -> dict[str, Any]:
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    targets_t = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    if len(logits_t) != len(targets_t):
        raise ValueError("Logit and target lengths differ")
    if features_standardized is None:
        features_t = None
        parameter_count = 1
    else:
        features_t = torch.as_tensor(np.asarray(features_standardized), dtype=torch.float64)
        if len(features_t) != len(logits_t):
            raise ValueError("Feature and logit lengths differ")
        parameter_count = int(features_t.shape[1]) + 1

    params = torch.zeros(parameter_count, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        params[0] = _inverse_bounded_temperature(init_temperature)
    optimizer = torch.optim.LBFGS(
        [params], lr=0.5, max_iter=max_iter, tolerance_grad=1e-9,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def temperatures() -> torch.Tensor:
        if features_t is None:
            linear = params[0].expand(len(logits_t))
        else:
            linear = params[0] + features_t @ params[1:]
        return T_MIN + (T_MAX - T_MIN) * torch.sigmoid(linear)

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        temp = temperatures()
        loss = F.cross_entropy(logits_t / temp[:, None], targets_t)
        if parameter_count > 1 and regularization > 0:
            loss = loss + float(regularization) * params[1:].square().sum()
        loss.backward()
        evaluations += 1
        return loss

    initial_temp = np.full(len(logits_t), float(init_temperature), dtype=np.float64)
    initial_nll = float(
        F.cross_entropy(logits_t / torch.as_tensor(initial_temp)[:, None], targets_t).detach().cpu()
    )
    optimizer.step(closure)
    with torch.no_grad():
        fitted_temp = temperatures()
        fitted_nll = float(F.cross_entropy(logits_t / fitted_temp[:, None], targets_t).cpu())
        param_values = params.detach().cpu().numpy().astype(np.float64)
        temps_np = fitted_temp.cpu().numpy().astype(np.float64)
    return {
        "parameters": param_values.tolist(),
        "initial_temperature": float(init_temperature),
        "initial_nll": initial_nll,
        "fitted_nll": fitted_nll,
        "optimizer_evaluations": evaluations,
        "temperature_mean": float(temps_np.mean()),
        "temperature_std": float(temps_np.std(ddof=0)),
        "temperature_min": float(temps_np.min()),
        "temperature_max": float(temps_np.max()),
        "regularization": float(regularization),
        "max_iter": int(max_iter),
    }


def ece_equal_width(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if np.any(mask):
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-np.asarray(confidence), kind="mergesort")
    errors = (~np.asarray(correct, dtype=bool)[order]).astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(cumulative_risk.mean())


def method_outputs(
    logits: np.ndarray,
    target_positions: np.ndarray,
    global_classes: np.ndarray,
    temperatures: np.ndarray | float,
) -> dict[str, np.ndarray]:
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    targets_t = torch.as_tensor(np.asarray(target_positions), dtype=torch.long)
    if np.isscalar(temperatures):
        temp_t = torch.full((len(logits_t),), float(temperatures), dtype=torch.float64)
    else:
        temp_t = torch.as_tensor(np.asarray(temperatures), dtype=torch.float64)
    if len(temp_t) != len(logits_t):
        raise ValueError("Temperature and logit lengths differ")
    if torch.any(temp_t <= 0):
        raise RuntimeError("Temperature must remain positive")
    scaled = logits_t / temp_t[:, None]
    probs = torch.softmax(scaled, dim=1)
    conf, pred_pos = probs.max(dim=1)
    row = torch.arange(len(logits_t))
    true_prob = probs[row, targets_t].clamp_min(EPS)
    nll = -torch.log(true_prob)
    brier = probs.square().sum(dim=1) - 2.0 * true_prob + 1.0
    global_lookup = torch.as_tensor(np.asarray(global_classes), dtype=torch.long)
    pred_global = global_lookup[pred_pos]
    correct = pred_pos.eq(targets_t)
    return {
        "prediction": pred_global.cpu().numpy().astype(np.int16),
        "confidence": conf.cpu().numpy().astype(np.float32),
        "correct": correct.cpu().numpy().astype(np.bool_),
        "true_probability": true_prob.cpu().numpy().astype(np.float32),
        "nll": nll.cpu().numpy().astype(np.float32),
        "brier": brier.cpu().numpy().astype(np.float32),
        "temperature": temp_t.cpu().numpy().astype(np.float32),
    }


def metrics_from_outputs(outputs: Mapping[str, np.ndarray], n_bins: int = 15) -> dict[str, float]:
    confidence = np.asarray(outputs["confidence"], dtype=np.float64)
    correct = np.asarray(outputs["correct"], dtype=bool)
    errors = ~correct
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if errors.min() == errors.max():
            error_auroc = float("nan")
            aupr_error = float("nan")
        else:
            score = 1.0 - confidence
            error_auroc = float(roc_auc_score(errors.astype(np.int8), score))
            aupr_error = float(average_precision_score(errors.astype(np.int8), score))
    except Exception:
        error_auroc = float("nan")
        aupr_error = float("nan")
    return {
        "accuracy": float(correct.mean()),
        "average_confidence": float(confidence.mean()),
        "acms": float(confidence[errors].mean()) if np.any(errors) else float("nan"),
        "ece15": ece_equal_width(confidence, correct, n_bins),
        "nll": float(np.mean(outputs["nll"])),
        "brier": float(np.mean(outputs["brier"])),
        "error_auroc": error_auroc,
        "aupr_error": aupr_error,
        "aurc": aurc(confidence, correct),
        "mean_temperature": float(np.mean(outputs["temperature"])),
        "std_temperature": float(np.std(outputs["temperature"], ddof=0)),
        "min_temperature": float(np.min(outputs["temperature"])),
        "max_temperature": float(np.max(outputs["temperature"])),
    }


def self_test() -> dict[str, Any]:
    torch.manual_seed(7)
    x = torch.rand(5, 3, 32, 32)
    complementarity = []
    for family in ("hard", "gaussian"):
        for radius in (3, 7, 11):
            lp = apply_condition(x, Condition("lp", family, "low_pass", radius), range(5), 9, "test")
            hp = apply_condition(x, Condition("hp", family, "high_pass", radius), range(5), 9, "test")
            error = float((lp + hp - x).abs().max())
            complementarity.append(error)
            if error > 2e-5:
                raise AssertionError((family, radius, error))
    desc = spectral_descriptors(x, cutoff_radius_pixels=5)
    if desc.shape != (5, 3) or not torch.isfinite(desc).all():
        raise AssertionError("Invalid spectral descriptors")

    rng = np.random.default_rng(11)
    logits = rng.normal(size=(64, 10)).astype(np.float32)
    targets = rng.integers(0, 10, size=64)
    scalar = fit_temperature_model(logits, targets, max_iter=30)
    raw = method_outputs(logits, targets, np.arange(10), 1.0)
    cal = method_outputs(logits, targets, np.arange(10), temperature_from_parameters(scalar["parameters"], None)[0])
    if not np.array_equal(raw["prediction"], cal["prediction"]):
        raise AssertionError("Positive scalar temperature changed top-1 prediction")

    features = rng.normal(size=(64, 3))
    stats = fit_standardizer(features)
    standardized = apply_standardizer(features, stats)
    adaptive = fit_temperature_model(logits, targets, standardized, max_iter=30)
    temps = temperature_from_parameters(adaptive["parameters"], standardized)
    adaptive_out = method_outputs(logits, targets, np.arange(10), temps)
    if not np.array_equal(raw["prediction"], adaptive_out["prediction"]):
        raise AssertionError("Positive adaptive temperature changed top-1 prediction")
    if temps.min() < T_MIN - 1e-9 or temps.max() > T_MAX + 1e-9:
        raise AssertionError("Temperature outside bounds")
    return {
        "schema_version": SCHEMA_VERSION,
        "max_complementarity_error": max(complementarity),
        "descriptor_shape": list(desc.shape),
        "scalar_fitted_nll": scalar["fitted_nll"],
        "adaptive_fitted_nll": adaptive["fitted_nll"],
        "temperature_bounds": [float(temps.min()), float(temps.max())],
        "top1_invariance": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
