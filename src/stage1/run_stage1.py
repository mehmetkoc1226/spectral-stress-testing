# -*- coding: utf-8 -*-
"""Resumable Stage-1 confirmatory spectral experiment for Google Colab.

Scientific goals
----------------
S1A. Replicate Stage-0 findings across two public checkpoints of the same
     ResNet-50, ViT-B/16, and Swin-T architectures on Imagenette.
S1B. Test hard-mask findings with smooth complementary Gaussian masks.
S1C. Replicate output-space dependence on ImageNet-A, whose folder WNIDs map
     directly into ImageNet-1K without semantic relabeling.
S1D. Quantify clipping sensitivity and deterministic natural corruptions as
     optional extensions.

The notebook that embeds this file is the recommended entry point. Every chunk
is atomically written to Google Drive; rerunning after a Colab disconnect skips
valid completed chunks and jobs.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# -----------------------------------------------------------------------------
# USER CONFIGURATION (normally set by the first notebook cell)
# -----------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_json(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    return default if value is None else json.loads(value)


RUN_PROFILE = os.environ.get("STAGE1_RUN_PROFILE", "paper")  # sanity, paper
RUN_PHASE = os.environ.get("STAGE1_PHASE", "all_core")
RUN_NAME = os.environ.get("STAGE1_RUN_NAME", "spectral_stage1_confirmatory_v1")
DRIVE_PROJECT_FOLDER = os.environ.get("STAGE1_DRIVE_FOLDER", "Spectral_Sensitivity_Project")

BASE_BATCH_SIZE = int(os.environ.get("STAGE1_BATCH_SIZE", "24"))
CHUNK_SIZE = int(os.environ.get("STAGE1_CHUNK_SIZE", "128"))
NUM_WORKERS = int(os.environ.get("STAGE1_NUM_WORKERS", "2"))
USE_MIXED_PRECISION = _env_bool("STAGE1_MIXED_PRECISION", True)
REQUIRE_GPU = _env_bool("STAGE1_REQUIRE_GPU", True)
SAVE_SELECTED_FULL_LOGITS = _env_bool("STAGE1_SAVE_SELECTED_LOGITS", True)
ECE_BINS = int(os.environ.get("STAGE1_ECE_BINS", "15"))
BOOTSTRAP_REPS_OVERRIDE = os.environ.get("STAGE1_BOOTSTRAP_REPS")
SEED = int(os.environ.get("STAGE1_SEED", "20260724"))

# Optional override. Use exact timm names from MODEL_SPECS.
MODELS_OVERRIDE = _env_json("STAGE1_MODELS", None)

TIMM_VERSION = "1.0.28"
CODE_VERSION = "stage1-resumable-2026-07-24-v1"
COMMON_IMAGE_SIZE = 224
COMMON_RESIZE_SIZE = 256
PRIMARY_RADII = (10, 20, 40)
SMOOTH_RADII = (10, 20)
SELECTED_LOGIT_CONDITIONS = {"clean_common", "hard_hp_r020_linear"}

PROFILE_CONFIG = {
    "sanity": {
        "sample_limits": {"imagenette": 256, "imagenet_a": 400},
        "bootstrap_reps": 100,
    },
    "paper": {
        "sample_limits": {"imagenette": None, "imagenet_a": None},
        "bootstrap_reps": 500,
    },
}

# Two checkpoints per exact architecture.
MODEL_SPECS: dict[str, dict[str, str]] = {
    "resnet50_a1": {
        "timm_name": "resnet50.a1_in1k",
        "family": "cnn",
        "architecture": "resnet50",
        "pair": "resnet50",
        "role": "anchor",
        "training": "ImageNet-1K, timm A1 recipe",
    },
    "resnet50_a2": {
        "timm_name": "resnet50.a2_in1k",
        "family": "cnn",
        "architecture": "resnet50",
        "pair": "resnet50",
        "role": "replication",
        "training": "ImageNet-1K, timm A2 recipe",
    },
    "vitb16_in1k": {
        "timm_name": "vit_base_patch16_224.augreg_in1k",
        "family": "global_transformer",
        "architecture": "vit_base_patch16_224",
        "pair": "vit_b16",
        "role": "anchor",
        "training": "ImageNet-1K, AugReg",
    },
    "vitb16_in21k": {
        "timm_name": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "family": "global_transformer",
        "architecture": "vit_base_patch16_224",
        "pair": "vit_b16",
        "role": "replication",
        "training": "ImageNet-21K pretraining, ImageNet-1K fine-tuning, AugReg",
    },
    "swin_t_in1k": {
        "timm_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "family": "hierarchical_transformer",
        "architecture": "swin_tiny_patch4_window7_224",
        "pair": "swin_t",
        "role": "anchor",
        "training": "ImageNet-1K, Microsoft recipe",
    },
    "swin_t_in22k": {
        "timm_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "family": "hierarchical_transformer",
        "architecture": "swin_tiny_patch4_window7_224",
        "pair": "swin_t",
        "role": "replication",
        "training": "ImageNet-22K pretraining, ImageNet-1K fine-tuning",
    },
}

ANCHOR_MODELS = tuple(k for k, v in MODEL_SPECS.items() if v["role"] == "anchor")
REPLICATION_MODELS = tuple(k for k, v in MODEL_SPECS.items() if v["role"] == "replication")
ALL_MODELS = tuple(MODEL_SPECS)

IMAGENETTE_WNID_TO_IMAGENET_INDEX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}

IMAGENET_A_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
IMAGENET_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)

# -----------------------------------------------------------------------------
# Lightweight bootstrap: do not replace Colab's torch/torchvision installation.
# -----------------------------------------------------------------------------
def _ensure_packages() -> None:
    required_imports = {
        "numpy": "numpy",
        "pandas": "pandas",
        "tqdm": "tqdm",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "requests": "requests",
    }
    missing = []
    for import_name, pip_name in required_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    try:
        import timm  # type: ignore
        timm_ok = timm.__version__ == TIMM_VERSION
    except Exception:
        timm_ok = False
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    if not timm_ok:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", f"timm=={TIMM_VERSION}"]
        )


_ensure_packages()

import numpy as np
import pandas as pd
import requests
import torch
import timm
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# Local module is written beside this script by the notebook.
from stage1_core import (
    SCHEMA_VERSION,
    Condition,
    apply_condition,
    benjamini_hochberg,
    compute_metrics_from_sufficient,
    spectral_diagnostics,
    stable_int_seed,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (tuple, list, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(jsonable(payload), indent=2, ensure_ascii=False))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    local_root = Path("/content/stage1_tmp") if Path("/content").exists() else Path("/tmp/stage1_tmp")
    local_root.mkdir(parents=True, exist_ok=True)
    local_tmp = local_root / f"{path.name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(local_tmp, **arrays)
    drive_tmp = path.with_suffix(path.suffix + ".tmp")
    shutil.copy2(local_tmp, drive_tmp)
    os.replace(drive_tmp, path)
    local_tmp.unlink(missing_ok=True)


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("spectral_stage1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(jsonable(payload), ensure_ascii=False) + "\n")


def mount_google_drive() -> Path:
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        return Path("/content/drive/MyDrive")
    except ImportError:
        fallback = Path.cwd() / "drive_fallback"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Colab was not detected; using a local directory: {fallback}")
        return fallback


def config_digest(payload: Any, length: int = 12) -> str:
    blob = json.dumps(jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    dest_resolved = destination.resolve()
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"Unsafe tar member: {member.name}")
        tar.extractall(destination)


def download_with_wget(url: str, destination: Path, logger: logging.Logger) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_mb = destination.stat().st_size / 1e6 if destination.exists() else 0.0
    logger.info("Downloading/verifying with resume enabled: %s | existing %.1f MB", url, existing_mb)
    # Always invoke wget -c: a pre-existing file may be an interrupted partial download.
    subprocess.check_call(
        ["wget", "-c", "--tries=20", "--timeout=45", "--show-progress", "-O", str(destination), url]
    )


def fetch_json(url: str, cache_path: Path) -> Any:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    atomic_write_json(cache_path, payload)
    return payload

# -----------------------------------------------------------------------------
# Scientific design and phase planning
# -----------------------------------------------------------------------------
def condition_catalog() -> dict[str, Condition]:
    out: dict[str, Condition] = {
        "clean_native": Condition("clean_native", "clean", "clean", None, "none", "native"),
        "clean_common": Condition("clean_common", "clean", "clean", None, "none", "common"),
    }
    for radius in PRIMARY_RADII:
        for mode, short in (("low_pass", "lp"), ("high_pass", "hp")):
            c = Condition(f"hard_{short}_r{radius:03d}_linear", "hard", mode, radius, "linear", "common")
            out[c.key] = c
    for radius in SMOOTH_RADII:
        for mode, short in (("low_pass", "lp"), ("high_pass", "hp")):
            c = Condition(f"gauss_{short}_r{radius:03d}_linear", "gaussian", mode, radius, "linear", "common")
            out[c.key] = c
    # Optional post-processing ablations. Linear counterparts are in core phases.
    for family in ("hard", "gaussian"):
        prefix = "hard" if family == "hard" else "gauss"
        for mode, short in (("low_pass", "lp"), ("high_pass", "hp")):
            c = Condition(f"{prefix}_{short}_r020_clip", family, mode, 20, "clip", "common")
            out[c.key] = c
    # Optional deterministic natural corruptions.
    for sigma in (2.0, 4.0):
        key = f"blur_sigma{int(sigma)}"
        out[key] = Condition(key, "blur", "blur", None, "clip", "common", sigma)
    for std in (0.10, 0.20):
        key = f"noise_std{int(std * 100):03d}"
        out[key] = Condition(key, "noise", "noise", None, "clip", "common", std)
    return out


CATALOG = condition_catalog()
HARD_CORE = (
    "clean_native", "clean_common",
    "hard_lp_r010_linear", "hard_hp_r010_linear",
    "hard_lp_r020_linear", "hard_hp_r020_linear",
    "hard_lp_r040_linear", "hard_hp_r040_linear",
)
SMOOTH_CORE = (
    "gauss_lp_r010_linear", "gauss_hp_r010_linear",
    "gauss_lp_r020_linear", "gauss_hp_r020_linear",
)
IMAGENETA_CORE = (
    "clean_native", "clean_common",
    "hard_lp_r010_linear", "hard_hp_r010_linear",
    "hard_lp_r020_linear", "hard_hp_r020_linear",
    "gauss_lp_r010_linear", "gauss_hp_r010_linear",
)
IMAGENETA_GAUSS_R20_LINEAR_PATCH = (
    "gauss_lp_r020_linear", "gauss_hp_r020_linear",
)
ABLATION_CONDITIONS = (
    "hard_lp_r020_clip", "hard_hp_r020_clip",
    "gauss_lp_r020_clip", "gauss_hp_r020_clip",
)
NATURAL_CONDITIONS = ("blur_sigma2", "blur_sigma4", "noise_std010", "noise_std020")


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    condition: str
    phase: str


def phase_jobs(phase: str, model_names: Sequence[str]) -> list[Job]:
    all_set = tuple(model_names)
    anchors = tuple(m for m in ANCHOR_MODELS if m in all_set)
    replications = tuple(m for m in REPLICATION_MODELS if m in all_set)
    jobs: list[Job] = []

    def add(phase_name: str, dataset: str, models_: Sequence[str], conditions_: Sequence[str]) -> None:
        jobs.extend(Job(dataset, m, c, phase_name) for m in models_ for c in conditions_)

    if phase in {"checkpoint_replication", "all_core", "all_full"}:
        add("checkpoint_replication", "imagenette", all_set, HARD_CORE + SMOOTH_CORE)
    if phase in {"cross_dataset_core", "all_core", "all_full"}:
        add("cross_dataset_core", "imagenet_a", anchors, IMAGENETA_CORE)
    if phase in {"cross_dataset_replication", "all_full"}:
        add("cross_dataset_replication", "imagenet_a", replications, IMAGENETA_CORE)
    if phase in {"ablation", "all_full"}:
        add("ablation", "imagenette", all_set, ABLATION_CONDITIONS)
        add("ablation", "imagenet_a", all_set, ABLATION_CONDITIONS)
    if phase == "imagenet_a_gauss_r20_linear_patch":
        add("imagenet_a_gauss_r20_linear_patch", "imagenet_a", all_set, IMAGENETA_GAUSS_R20_LINEAR_PATCH)
    if phase in {"natural_corruptions", "all_full"}:
        add("natural_corruptions", "imagenette", all_set, NATURAL_CONDITIONS)
        add("natural_corruptions", "imagenet_a", anchors, NATURAL_CONDITIONS)
    if phase == "imagenette_hard":
        add("imagenette_hard", "imagenette", all_set, HARD_CORE)
    if phase == "imagenette_smooth":
        add("imagenette_smooth", "imagenette", all_set, SMOOTH_CORE)
    if not jobs:
        raise ValueError(
            f"Unknown or empty phase: {phase}. Valid examples: all_core, checkpoint_replication, "
            "cross_dataset_core, cross_dataset_replication, ablation, "
            "imagenet_a_gauss_r20_linear_patch, natural_corruptions, all_full"
        )
    # Stable de-duplication when combined phases overlap.
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for job in jobs:
        key = (job.dataset, job.model, job.condition)
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


@dataclass(frozen=True)
class DesignConfig:
    code_version: str
    timm_version: str
    run_name: str
    profile: str
    models: tuple[str, ...]
    common_image_size: int
    common_resize_size: int
    primary_radii: tuple[int, ...]
    smooth_radii: tuple[int, ...]
    sample_limits: dict[str, int | None]
    base_batch_size: int
    chunk_size: int
    num_workers: int
    mixed_precision: bool
    save_selected_full_logits: bool
    selected_logit_conditions: tuple[str, ...]
    ece_bins: int
    bootstrap_reps: int
    seed: int


def build_design() -> DesignConfig:
    if RUN_PROFILE not in PROFILE_CONFIG:
        raise ValueError(f"RUN_PROFILE must be one of {list(PROFILE_CONFIG)}")
    models_ = tuple(MODELS_OVERRIDE) if MODELS_OVERRIDE else ALL_MODELS
    unknown = sorted(set(models_) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown model aliases: {unknown}")
    profile = PROFILE_CONFIG[RUN_PROFILE]
    reps = int(BOOTSTRAP_REPS_OVERRIDE) if BOOTSTRAP_REPS_OVERRIDE else int(profile["bootstrap_reps"])
    return DesignConfig(
        code_version=CODE_VERSION,
        timm_version=TIMM_VERSION,
        run_name=RUN_NAME,
        profile=RUN_PROFILE,
        models=models_,
        common_image_size=COMMON_IMAGE_SIZE,
        common_resize_size=COMMON_RESIZE_SIZE,
        primary_radii=PRIMARY_RADII,
        smooth_radii=SMOOTH_RADII,
        sample_limits=dict(profile["sample_limits"]),
        base_batch_size=BASE_BATCH_SIZE,
        chunk_size=CHUNK_SIZE,
        num_workers=NUM_WORKERS,
        mixed_precision=USE_MIXED_PRECISION,
        save_selected_full_logits=SAVE_SELECTED_FULL_LOGITS,
        selected_logit_conditions=tuple(sorted(SELECTED_LOGIT_CONDITIONS)),
        ece_bins=ECE_BINS,
        bootstrap_reps=reps,
        seed=SEED,
    )

# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------
def common_geometry_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(COMMON_RESIZE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(COMMON_IMAGE_SIZE),
        transforms.ToTensor(),
    ])


class PathDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        targets1000: np.ndarray,
        active_indices: np.ndarray,
        transform: Any,
    ):
        self.paths = list(paths)
        self.targets1000 = np.asarray(targets1000, dtype=np.int64)
        self.active_indices = np.asarray(active_indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.active_indices))

    def __getitem__(self, local_index: int):
        global_index = int(self.active_indices[local_index])
        with Image.open(self.paths[global_index]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(self.targets1000[global_index]), global_index


@dataclass
class DatasetBundle:
    name: str
    paths: list[str]
    targets1000: np.ndarray
    wnids: list[str]
    restricted_indices: np.ndarray
    active_indices: np.ndarray
    manifest: pd.DataFrame
    source: str

    def dataset(self, transform: Any) -> PathDataset:
        return PathDataset(self.paths, self.targets1000, self.active_indices, transform)


def stratified_indices(targets: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    all_idx = np.arange(len(targets), dtype=np.int64)
    if limit is None or limit >= len(targets):
        return all_idx
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    per_class = max(1, limit // len(classes))
    selected: list[int] = []
    for cls in classes:
        candidates = np.flatnonzero(targets == cls)
        take = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(all_idx, np.asarray(selected, dtype=np.int64), assume_unique=False)
        extra = rng.choice(remaining, size=min(limit - len(selected), len(remaining)), replace=False)
        selected.extend(extra.tolist())
    return np.asarray(sorted(selected[:limit]), dtype=np.int64)


def copy_dataset_to_local(source: Path, local: Path, logger: logging.Logger) -> None:
    marker = local / ".COPY_COMPLETE"
    source_signature = f"{source.resolve()}::{sum(1 for _ in source.rglob('*'))}"
    if marker.exists() and marker.read_text(encoding="utf-8") == source_signature:
        logger.info("Local dataset cache is ready: %s", local)
        return
    logger.info("Copying dataset to the local Colab disk: %s", source.name)
    if local.exists():
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, local)
    marker.write_text(source_signature, encoding="utf-8")


def prepare_imagenette(
    drive_project: Path,
    local_root: Path,
    sample_limit: int | None,
    seed: int,
    logger: logging.Logger,
) -> DatasetBundle:
    persistent_root = drive_project / "datasets"
    datasets.Imagenette(
        root=str(persistent_root), split="val", size="320px", download=True, transform=None
    )
    source = persistent_root / "imagenette2-320"
    local = local_root / "imagenette2-320"
    copy_dataset_to_local(source, local, logger)
    base = datasets.Imagenette(root=str(local_root), split="val", size="320px", download=False)
    observed = list(base.wnids)
    expected = sorted(IMAGENETTE_WNID_TO_IMAGENET_INDEX)
    if observed != expected:
        raise RuntimeError(f"Imagenette WNID mismatch. Expected={expected}; observed={observed}")
    samples = list(getattr(base, "_samples"))
    paths = [str(path) for path, _ in samples]
    targets = np.asarray(
        [IMAGENETTE_WNID_TO_IMAGENET_INDEX[observed[int(ord_)]] for _, ord_ in samples],
        dtype=np.int64,
    )
    active = stratified_indices(targets, sample_limit, seed)
    restricted = np.asarray(sorted(set(targets.tolist())), dtype=np.int64)
    rows = []
    for pos, idx in enumerate(active):
        path, ord_ = samples[int(idx)]
        rows.append({
            "dataset": "imagenette", "active_position": pos, "global_dataset_index": int(idx),
            "relative_path": str(Path(path).relative_to(local_root)),
            "wnid": observed[int(ord_)], "target_imagenet1000": int(targets[int(idx)]),
        })
    return DatasetBundle(
        "imagenette", paths, targets, observed, restricted, active, pd.DataFrame(rows),
        "torchvision Imagenette v2-320 validation split",
    )


def prepare_imagenet_a(
    drive_project: Path,
    local_root: Path,
    sample_limit: int | None,
    seed: int,
    logger: logging.Logger,
) -> DatasetBundle:
    persistent_root = drive_project / "datasets" / "imagenet-a"
    persistent_root.mkdir(parents=True, exist_ok=True)
    tar_path = persistent_root / "imagenet-a.tar"
    extracted_root = persistent_root / "imagenet-a"
    marker = extracted_root / ".EXTRACT_COMPLETE"
    if not marker.exists():
        download_with_wget(IMAGENET_A_URL, tar_path, logger)
        tmp_extract = persistent_root / "_extract_tmp"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        safe_extract_tar(tar_path, tmp_extract)
        candidate = tmp_extract / "imagenet-a"
        if not candidate.exists():
            raise RuntimeError("ImageNet-A archive did not contain imagenet-a/ directory")
        if extracted_root.exists():
            shutil.rmtree(extracted_root)
        shutil.move(str(candidate), str(extracted_root))
        shutil.rmtree(tmp_extract, ignore_errors=True)
        marker.write_text("ok", encoding="utf-8")

    local = local_root / "imagenet-a"
    copy_dataset_to_local(extracted_root, local, logger)
    base = datasets.ImageFolder(root=str(local), transform=None)
    wnids = list(base.classes)
    if len(wnids) != 200:
        raise RuntimeError(f"ImageNet-A expected 200 class folders, observed {len(wnids)}")
    if len(base.samples) != 7500:
        raise RuntimeError(f"ImageNet-A expected 7500 images, observed {len(base.samples)}")

    class_index = fetch_json(IMAGENET_CLASS_INDEX_URL, persistent_root / "imagenet_class_index.json")
    if len(class_index) != 1000:
        raise RuntimeError("ImageNet class index JSON must contain exactly 1000 entries")
    wnid_to_index = {value[0]: int(key) for key, value in class_index.items()}
    if len(wnid_to_index) != 1000:
        raise RuntimeError("ImageNet class index JSON contains duplicate/missing WNIDs")
    # Known anchors make accidental alternate indexing fail loudly.
    for wnid, expected in {"n01440764": 0, "n02102040": 217, "n03888257": 701}.items():
        if wnid_to_index.get(wnid) != expected:
            raise RuntimeError(f"ImageNet class index validation failed for {wnid}")
    missing = sorted(set(wnids) - set(wnid_to_index))
    if missing:
        raise RuntimeError(f"ImageNet-A WNIDs missing from ImageNet-1K index: {missing[:10]}")

    paths = [str(path) for path, _ in base.samples]
    targets = np.asarray([wnid_to_index[wnids[int(ord_)]] for _, ord_ in base.samples], dtype=np.int64)
    active = stratified_indices(targets, sample_limit, seed + 17)
    restricted = np.asarray(sorted(set(targets.tolist())), dtype=np.int64)
    rows = []
    for pos, idx in enumerate(active):
        path, ord_ = base.samples[int(idx)]
        rows.append({
            "dataset": "imagenet_a", "active_position": pos, "global_dataset_index": int(idx),
            "relative_path": str(Path(path).relative_to(local_root)),
            "wnid": wnids[int(ord_)], "target_imagenet1000": int(targets[int(idx)]),
        })
    return DatasetBundle(
        "imagenet_a", paths, targets, wnids, restricted, active, pd.DataFrame(rows),
        "ImageNet-A official natural adversarial examples dataset",
    )

# -----------------------------------------------------------------------------
# Model loading and preprocessing
# -----------------------------------------------------------------------------
def validate_timm_names(model_aliases: Sequence[str]) -> None:
    available = set(timm.list_models(pretrained=True, include_tags=True))
    missing = [MODEL_SPECS[a]["timm_name"] for a in model_aliases if MODEL_SPECS[a]["timm_name"] not in available]
    if missing:
        raise RuntimeError(
            "Pinned timm model names are unavailable. Confirm timm version and internet access. Missing: "
            + ", ".join(missing)
        )


def load_model(alias: str, device: torch.device, cache_dir: Path):
    spec = MODEL_SPECS[alias]
    model = timm.create_model(spec["timm_name"], pretrained=True, cache_dir=str(cache_dir))
    model = model.to(device).eval()
    data_config = timm.data.resolve_model_data_config(model)
    input_size = tuple(int(v) for v in data_config.get("input_size", (3, 224, 224)))
    if input_size[0] != 3 or input_size[-2:] != (224, 224):
        raise RuntimeError(f"Stage-1 selected models must use 224x224 RGB input: {alias} -> {input_size}")
    if int(getattr(model, "num_classes", 1000)) != 1000:
        raise RuntimeError(f"Expected 1000 output classes for {alias}")
    native_transform = timm.data.create_transform(**data_config, is_training=False)
    mean = tuple(float(v) for v in data_config["mean"])
    std = tuple(float(v) for v in data_config["std"])
    return model, data_config, native_transform, mean, std


def normalize_model(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_t) / std_t

# -----------------------------------------------------------------------------
# Inference and sufficient statistics
# -----------------------------------------------------------------------------
def protocol_sufficient(
    logits: torch.Tensor,
    targets1000: torch.Tensor,
    restricted_indices: torch.Tensor,
) -> dict[str, np.ndarray]:
    logits = logits.float()
    batch = logits.shape[0]
    row = torch.arange(batch, device=logits.device)

    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets1000)
    top5 = logits.topk(k=5, dim=1).indices.eq(targets1000.view(-1, 1)).any(dim=1)
    p_true = probs[row, targets1000].clamp_min(1e-12)
    nll = -torch.log(p_true)
    brier = probs.square().sum(dim=1) - 2.0 * p_true + 1.0

    subset_logits = logits.index_select(1, restricted_indices)
    subset_probs = torch.softmax(subset_logits, dim=1)
    subset_conf, subset_pred_pos = subset_probs.max(dim=1)
    # Global target -> restricted position.
    target_pos_map = torch.full((1000,), -1, device=logits.device, dtype=torch.long)
    target_pos_map[restricted_indices] = torch.arange(len(restricted_indices), device=logits.device)
    target_pos = target_pos_map[targets1000]
    if torch.any(target_pos < 0):
        raise RuntimeError("Target is outside the restricted class set")
    subset_correct = subset_pred_pos.eq(target_pos)
    subset_top5 = subset_logits.topk(k=min(5, subset_logits.shape[1]), dim=1).indices.eq(
        target_pos.view(-1, 1)
    ).any(dim=1)
    subset_p_true = subset_probs[row, target_pos].clamp_min(1e-12)
    subset_nll = -torch.log(subset_p_true)
    subset_brier = subset_probs.square().sum(dim=1) - 2.0 * subset_p_true + 1.0
    subset_pred_global = restricted_indices[subset_pred_pos]

    in_subset = torch.zeros(1000, dtype=torch.bool, device=logits.device)
    in_subset[restricted_indices] = True
    outside = ~in_subset[pred]
    subset_mass = probs.index_select(1, restricted_indices).sum(dim=1)

    def cpu(name: str, value: torch.Tensor, dtype: np.dtype) -> tuple[str, np.ndarray]:
        return name, value.detach().cpu().numpy().astype(dtype)

    return dict([
        cpu("native_prediction", pred, np.int16),
        cpu("native_confidence", conf, np.float32),
        cpu("native_correct", correct, np.bool_),
        cpu("native_top5_correct", top5, np.bool_),
        cpu("native_true_probability", p_true, np.float32),
        cpu("native_nll", nll, np.float32),
        cpu("native_brier", brier, np.float32),
        cpu("restricted_prediction", subset_pred_global, np.int16),
        cpu("restricted_confidence", subset_conf, np.float32),
        cpu("restricted_correct", subset_correct, np.bool_),
        cpu("restricted_top5_correct", subset_top5, np.bool_),
        cpu("restricted_true_probability", subset_p_true, np.float32),
        cpu("restricted_nll", subset_nll, np.float32),
        cpu("restricted_brier", subset_brier, np.float32),
        cpu("native_prediction_outside_subset", outside, np.bool_),
        cpu("native_probability_mass_on_subset", subset_mass, np.float32),
    ])


def valid_shard(path: Path, expected_count: int, expected_digest: str, require_logits: bool) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema_version", "design_digest", "global_indices", "targets1000",
                "native_confidence", "native_correct", "restricted_confidence", "restricted_correct",
                "energy_ratio",
            }
            if require_logits:
                required.add("logits")
            if not required.issubset(data.files):
                return False
            if int(data["schema_version"][0]) != SCHEMA_VERSION:
                return False
            if str(data["design_digest"][0]) != expected_digest:
                return False
            return len(data["global_indices"]) == expected_count
    except Exception:
        return False


def infer_chunk(
    model: torch.nn.Module,
    dataset: Dataset,
    start: int,
    end: int,
    restricted_indices_np: np.ndarray,
    condition: Condition,
    dataset_name: str,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    initial_batch_size: int,
    num_workers: int,
    mixed_precision: bool,
    seed: int,
    save_logits: bool,
    design_digest: str,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    batch_size = initial_batch_size
    subset = Subset(dataset, list(range(start, end)))
    while batch_size >= 1:
        try:
            loader = DataLoader(
                subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=device.type == "cuda", persistent_workers=False,
            )
            collected: dict[str, list[np.ndarray]] = {}
            restricted_indices = torch.as_tensor(restricted_indices_np, device=device, dtype=torch.long)
            with torch.inference_mode():
                for images, targets1000, global_indices in loader:
                    images = images.to(device, non_blocking=True)
                    targets_t = torch.as_tensor(targets1000, device=device, dtype=torch.long)
                    global_np = np.asarray(global_indices, dtype=np.int64)

                    if condition.geometry == "native":
                        model_input = images  # timm native transform already normalized.
                        diagnostics = {
                            "energy_ratio": np.full(len(images), np.nan, np.float32),
                            "filtered_min": np.full(len(images), np.nan, np.float32),
                            "filtered_max": np.full(len(images), np.nan, np.float32),
                            "fraction_below_zero": np.full(len(images), np.nan, np.float32),
                            "fraction_above_one": np.full(len(images), np.nan, np.float32),
                        }
                    else:
                        transformed = apply_condition(
                            images, condition, global_np.tolist(), seed, dataset_name
                        )
                        diagnostics = spectral_diagnostics(images, transformed)
                        model_input = normalize_model(transformed, mean, std)

                    autocast_enabled = mixed_precision and device.type == "cuda"
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                        enabled=autocast_enabled,
                    ):
                        logits = model(model_input)
                    stats = protocol_sufficient(logits, targets_t, restricted_indices)
                    parts = {
                        "global_indices": global_np.astype(np.int32),
                        "targets1000": np.asarray(targets1000, dtype=np.int16),
                        **stats,
                        **diagnostics,
                    }
                    if save_logits:
                        parts["logits"] = logits.float().detach().cpu().numpy().astype(np.float16)
                    for key, value in parts.items():
                        collected.setdefault(key, []).append(value)

            result = {key: np.concatenate(parts) for key, parts in collected.items()}
            result["schema_version"] = np.asarray([SCHEMA_VERSION], dtype=np.int16)
            result["design_digest"] = np.asarray([design_digest], dtype=f"<U{len(design_digest)}")
            result["effective_batch_size"] = np.asarray([batch_size], dtype=np.int16)
            return result
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or batch_size == 1:
                raise
            new_size = max(1, batch_size // 2)
            logger.warning("CUDA OOM: batch %d -> %d; chunk yeniden deneniyor.", batch_size, new_size)
            batch_size = new_size
            torch.cuda.empty_cache()
            gc.collect()
    raise RuntimeError("No valid batch size remained")


def load_job_arrays(job_dir: Path) -> dict[str, np.ndarray]:
    shards = sorted(job_dir.glob("shard_*.npz"))
    if not shards:
        raise RuntimeError(f"No shards found in {job_dir}")
    collected: dict[str, list[np.ndarray]] = {}
    scalar_keys = {"schema_version", "design_digest", "effective_batch_size"}
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            for key in data.files:
                if key in scalar_keys:
                    continue
                collected.setdefault(key, []).append(data[key])
    arrays = {key: np.concatenate(parts) for key, parts in collected.items()}
    order = np.argsort(arrays["global_indices"], kind="mergesort")
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(np.unique(arrays["global_indices"])) != len(arrays["global_indices"]):
        raise RuntimeError(f"Duplicate indices in {job_dir}")
    return arrays


def upsert_summary(summary_path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    keys = ["dataset", "model", "condition", "protocol"]
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=keys, keep="last")
    else:
        combined = new
    combined = combined.sort_values(keys).reset_index(drop=True)
    atomic_write_csv(summary_path, combined)
    return combined


def finite_stat(arr: np.ndarray, fn: Any) -> float:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(fn(x)) if x.size else float("nan")


def aggregate_job(
    arrays: dict[str, np.ndarray],
    bundle: DatasetBundle,
    model_alias: str,
    condition: Condition,
    design: DesignConfig,
    run_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    pred_root = run_dir / "predictions" / bundle.name / model_alias
    pred_root.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[model_alias]
    common_fields = {
        "dataset": bundle.name,
        "model": model_alias,
        "timm_name": spec["timm_name"],
        "family": spec["family"],
        "architecture": spec["architecture"],
        "checkpoint_pair": spec["pair"],
        "checkpoint_role": spec["role"],
        "condition": condition.key,
        "filter_family": condition.family,
        "mode": condition.mode,
        "radius": condition.radius,
        "postprocess": condition.postprocess,
        "geometry": condition.geometry,
        "severity": condition.severity,
    }
    for protocol_index, (protocol, prefix) in enumerate(
        (("imagenet1000", "native"), (f"restricted{len(bundle.restricted_indices)}", "restricted"))
    ):
        metrics = compute_metrics_from_sufficient(
            arrays[f"{prefix}_confidence"], arrays[f"{prefix}_correct"],
            arrays[f"{prefix}_top5_correct"], arrays[f"{prefix}_nll"],
            arrays[f"{prefix}_brier"], design.ece_bins, design.bootstrap_reps,
            design.seed + protocol_index + stable_int_seed(bundle.name, model_alias, condition.key),
        )
        predictions = pd.DataFrame({
            "global_dataset_index": arrays["global_indices"].astype(np.int64),
            "target_imagenet1000": arrays["targets1000"].astype(np.int64),
            "prediction_imagenet1000": arrays[f"{prefix}_prediction"].astype(np.int64),
            "confidence": arrays[f"{prefix}_confidence"].astype(np.float32),
            "correct": arrays[f"{prefix}_correct"].astype(bool),
            "true_probability": arrays[f"{prefix}_true_probability"].astype(np.float32),
        })
        if prefix == "native":
            predictions["prediction_outside_restricted_set"] = arrays[
                "native_prediction_outside_subset"
            ].astype(bool)
            predictions["probability_mass_on_restricted_set"] = arrays[
                "native_probability_mass_on_subset"
            ].astype(np.float32)
        predictions.to_csv(
            pred_root / f"{condition.key}__{protocol}.csv.gz", index=False, compression="gzip"
        )
        row = {
            **common_fields,
            "protocol": protocol,
            **metrics,
            "mean_energy_ratio": finite_stat(arrays["energy_ratio"], np.mean),
            "std_energy_ratio": finite_stat(arrays["energy_ratio"], lambda x: np.std(x, ddof=1)),
            "mean_fraction_below_zero": finite_stat(arrays["fraction_below_zero"], np.mean),
            "mean_fraction_above_one": finite_stat(arrays["fraction_above_one"], np.mean),
            "filtered_global_min": finite_stat(arrays["filtered_min"], np.min),
            "filtered_global_max": finite_stat(arrays["filtered_max"], np.max),
            "native_prediction_outside_subset_rate": float(
                np.mean(arrays["native_prediction_outside_subset"])
            ),
            "native_probability_mass_on_subset": float(
                np.mean(arrays["native_probability_mass_on_subset"])
            ),
        }
        rows.append(row)

    if "logits" in arrays:
        out = run_dir / "selected_logits" / bundle.name / model_alias / f"{condition.key}.npz"
        atomic_save_npz(
            out,
            global_indices=arrays["global_indices"].astype(np.int32),
            targets1000=arrays["targets1000"].astype(np.int16),
            logits=arrays["logits"].astype(np.float16),
            restricted_indices=bundle.restricted_indices.astype(np.int16),
        )
    return rows

# -----------------------------------------------------------------------------
# Progress tracking and job execution
# -----------------------------------------------------------------------------
def job_key(job: Job) -> str:
    return f"{job.dataset}::{job.model}::{job.condition}"


def update_progress(
    run_dir: Path,
    plan: pd.DataFrame,
    current: dict[str, Any],
    started_monotonic: float,
) -> None:
    completed = int((plan["status"] == "done").sum())
    total = len(plan)
    elapsed = time.monotonic() - started_monotonic
    eta = (elapsed / completed * (total - completed)) if completed > 0 else None
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": RUN_PHASE,
        "profile": RUN_PROFILE,
        "completed_jobs": completed,
        "total_jobs": total,
        "fraction": completed / total if total else 1.0,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        **current,
    }
    atomic_write_json(run_dir / "LIVE_STATUS.json", payload)
    atomic_write_csv(run_dir / "job_plan.csv", plan)


def run_job(
    job: Job,
    model: torch.nn.Module,
    native_transform: Any,
    mean: Sequence[float],
    std: Sequence[float],
    bundle: DatasetBundle,
    design: DesignConfig,
    design_digest_value: str,
    run_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    plan: pd.DataFrame,
    started_monotonic: float,
) -> list[dict[str, Any]]:
    condition = CATALOG[job.condition]
    transform = native_transform if condition.geometry == "native" else common_geometry_transform()
    dataset = bundle.dataset(transform)
    job_dir = run_dir / "raw_shards" / bundle.name / job.model / condition.key
    job_dir.mkdir(parents=True, exist_ok=True)
    done_marker = job_dir / "DONE.json"
    save_logits = design.save_selected_full_logits and condition.key in design.selected_logit_conditions
    total = len(dataset)
    starts = list(range(0, total, design.chunk_size))

    progress = tqdm(starts, desc=f"{bundle.name} | {job.model} | {condition.key}", unit="chunk")
    for start in progress:
        end = min(start + design.chunk_size, total)
        shard = job_dir / f"shard_{start:06d}_{end:06d}.npz"
        if valid_shard(shard, end - start, design_digest_value, save_logits):
            progress.set_postfix_str("checkpoint mevcut")
            continue
        update_progress(
            run_dir, plan,
            {"dataset": bundle.name, "model": job.model, "condition": condition.key,
             "chunk_start": start, "chunk_end": end},
            started_monotonic,
        )
        chunk = infer_chunk(
            model, dataset, start, end, bundle.restricted_indices, condition, bundle.name,
            mean, std, device, design.base_batch_size, design.num_workers,
            design.mixed_precision, design.seed, save_logits, design_digest_value, logger,
        )
        atomic_save_npz(shard, **chunk)
        progress.set_postfix_str(f"Drive'a kaydedildi: {end}/{total}")
        logger.info(
            "CHUNK SAVED | %s | %s | %s | %d/%d",
            bundle.name, job.model, condition.key, end, total,
        )

    arrays = load_job_arrays(job_dir)
    if len(arrays["targets1000"]) != total:
        raise RuntimeError(f"Incomplete job {job_key(job)}: {len(arrays['targets1000'])}/{total}")
    rows = aggregate_job(arrays, bundle, job.model, condition, design, run_dir)
    atomic_write_json(done_marker, {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job": dataclasses.asdict(job), "n": total, "shards": len(starts),
        "design_digest": design_digest_value,
    })
    return rows

# -----------------------------------------------------------------------------
# Automated validation and plots
# -----------------------------------------------------------------------------
def run_self_tests(device: torch.device, run_dir: Path, logger: logging.Logger) -> None:
    logger.info("Running Stage-1 scientific invariant tests...")
    x = torch.rand(3, 3, 224, 224, device=device)
    rows = []
    for family in ("hard", "gaussian"):
        for radius in PRIMARY_RADII:
            lp = apply_condition(
                x, Condition("lp", family, "low_pass", radius), range(3), SEED, "selftest"
            )
            hp = apply_condition(
                x, Condition("hp", family, "high_pass", radius), range(3), SEED, "selftest"
            )
            max_err = float((lp + hp - x).abs().max().cpu())
            rows.append({"family": family, "radius": radius, "max_abs_error": max_err})
            if max_err > 2e-5:
                raise RuntimeError(f"Complementarity failed: {family} r={radius}: {max_err}")
    atomic_write_csv(run_dir / "spectral_invariant_tests.csv", pd.DataFrame(rows))
    logger.info("Invariant tests passed; maximum error=%.3e", max(r["max_abs_error"] for r in rows))


def make_plots(summary: pd.DataFrame, run_dir: Path) -> None:
    import matplotlib.pyplot as plt
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    primary = summary[summary["postprocess"].isin(["none", "linear"])]
    for dataset_name in primary["dataset"].dropna().unique():
        for protocol in primary["protocol"].dropna().unique():
            scoped = primary[(primary["dataset"] == dataset_name) & (primary["protocol"] == protocol)]
            for family in ("hard", "gaussian"):
                family_df = scoped[scoped["filter_family"] == family]
                for mode in ("low_pass", "high_pass"):
                    mode_df = family_df[family_df["mode"] == mode]
                    if mode_df.empty:
                        continue
                    for metric in ("accuracy", "acms", "ece"):
                        fig, ax = plt.subplots(figsize=(9, 5))
                        for model_name, group in mode_df.groupby("model"):
                            group = group.sort_values("radius")
                            ax.plot(group["radius"], group[metric], marker="o", label=model_name)
                        ax.set_xlabel("Cutoff radius r (224x224 frequency grid)")
                        ax.set_ylabel(metric)
                        ax.set_title(f"{dataset_name} | {protocol} | {family} {mode} | {metric}")
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)
                        fig.tight_layout()
                        fig.savefig(
                            plot_dir / f"{dataset_name}__{protocol}__{family}__{mode}__{metric}.png",
                            dpi=180, bbox_inches="tight",
                        )
                        plt.close(fig)


def make_completion_report(plan: pd.DataFrame, summary: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    summary_keys = set(
        summary[["dataset", "model", "condition"]].drop_duplicates().astype(str).agg("::".join, axis=1)
    ) if not summary.empty else set()
    report = plan.copy()
    report["has_summary"] = report.astype(str).apply(
        lambda r: f"{r['dataset']}::{r['model']}::{r['condition']}" in summary_keys, axis=1
    )
    atomic_write_csv(run_dir / "completion_report.csv", report)
    return report

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> Path:
    seed_everything(SEED)
    design = build_design()
    design_digest_value = config_digest(design)
    drive_home = mount_google_drive()
    drive_project = drive_home / DRIVE_PROJECT_FOLDER
    drive_project.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(drive_project / "model_cache" / "huggingface")
    os.environ["TORCH_HOME"] = str(drive_project / "model_cache" / "torch")

    run_dir = drive_project / "runs" / f"{design.run_name}__{design_digest_value}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run.log")
    errors_path = run_dir / "errors.jsonl"
    summary_path = run_dir / "summary.csv"
    started_monotonic = time.monotonic()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if REQUIRE_GPU and device.type != "cuda":
        raise RuntimeError("GPU was not found. In Colab, select Runtime > Change runtime type > T4 GPU.")

    logger.info("Stage-1 run directory: %s", run_dir)
    logger.info("Profile=%s | Phase=%s | Design=%s", RUN_PROFILE, RUN_PHASE, design_digest_value)
    logger.info("Device=%s | GPU=%s", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    atomic_write_json(run_dir / "config.json", design)
    atomic_write_json(run_dir / "active_phase.json", {"phase": RUN_PHASE, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    atomic_write_json(run_dir / "environment.json", {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    atomic_write_json(run_dir / "model_design.json", MODEL_SPECS)
    atomic_write_json(run_dir / "condition_catalog.json", [dataclasses.asdict(c) for c in CATALOG.values()])

    run_self_tests(device, run_dir, logger)
    validate_timm_names(design.models)

    jobs = phase_jobs(RUN_PHASE, design.models)
    if design.profile == "sanity":
        # Sanity runs all scientific branches on anchor models only, small stratified subsets.
        jobs = [j for j in jobs if j.model in ANCHOR_MODELS]
    plan = pd.DataFrame([{
        "job_id": i, "dataset": j.dataset, "model": j.model,
        "condition": j.condition, "phase": j.phase, "status": "pending",
    } for i, j in enumerate(jobs)])

    # A DONE marker alone is insufficient: a disconnect could occur after shard
    # aggregation but before summary.csv is updated. Mark a job complete only
    # when both output protocols are present; otherwise rerun_job will reuse all
    # valid shards and rebuild the summaries without repeating inference.
    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not existing_summary.empty:
        counts = existing_summary.groupby(["dataset", "model", "condition"])["protocol"].nunique()
        complete_keys = {"::".join(map(str, idx)) for idx, count in counts.items() if int(count) >= 2}
        for i, row in plan.iterrows():
            key = f"{row['dataset']}::{row['model']}::{row['condition']}"
            if key in complete_keys:
                plan.at[i, "status"] = "done"
    atomic_write_csv(run_dir / "job_plan.csv", plan)

    required_datasets = sorted(set(j.dataset for j in jobs))
    local_root = Path("/content/stage1_data") if Path("/content").exists() else Path.cwd() / "stage1_data"
    bundles: dict[str, DatasetBundle] = {}
    if "imagenette" in required_datasets:
        bundles["imagenette"] = prepare_imagenette(
            drive_project, local_root, design.sample_limits["imagenette"], design.seed, logger
        )
    if "imagenet_a" in required_datasets:
        bundles["imagenet_a"] = prepare_imagenet_a(
            drive_project, local_root, design.sample_limits["imagenet_a"], design.seed, logger
        )
    for name, bundle in bundles.items():
        atomic_write_csv(run_dir / "dataset_manifests" / f"{name}.csv", bundle.manifest)
        atomic_write_json(run_dir / "label_spaces" / f"{name}.json", {
            "dataset": name, "source": bundle.source,
            "n_active": len(bundle.active_indices), "n_total": len(bundle.targets1000),
            "restricted_indices": bundle.restricted_indices.tolist(), "class_count": len(bundle.restricted_indices),
            "wnids": bundle.wnids,
        })

    update_progress(run_dir, plan, {"state": "datasets_ready"}, started_monotonic)
    model_cache = drive_project / "model_cache" / "timm"
    model_cache.mkdir(parents=True, exist_ok=True)

    jobs_by_model: dict[str, list[tuple[int, Job]]] = {}
    for idx, job in enumerate(jobs):
        jobs_by_model.setdefault(job.model, []).append((idx, job))

    for model_alias in design.models:
        model_jobs = jobs_by_model.get(model_alias, [])
        if not model_jobs:
            continue
        logger.info("=" * 90)
        logger.info("MODEL START: %s (%s)", model_alias, MODEL_SPECS[model_alias]["timm_name"])
        model = None
        try:
            model, data_config, native_transform, mean, std = load_model(model_alias, device, model_cache)
            atomic_write_json(run_dir / "model_metadata" / f"{model_alias}.json", {
                **MODEL_SPECS[model_alias], "data_config": data_config,
                "normalization_mean": mean, "normalization_std": std,
                "parameters": sum(p.numel() for p in model.parameters()),
            })

            # Mandatory Imagenette native/common clean gates when present in plan.
            gate_results: dict[str, float] = {}
            ordered = sorted(model_jobs, key=lambda x: (0 if x[1].condition == "clean_native" else 1 if x[1].condition == "clean_common" else 2, x[0]))
            for plan_idx, job in ordered:
                if plan.at[plan_idx, "status"] == "done":
                    continue
                try:
                    plan.at[plan_idx, "status"] = "running"
                    update_progress(run_dir, plan, {"state": "job_start", "job": dataclasses.asdict(job)}, started_monotonic)
                    rows = run_job(
                        job, model, native_transform, mean, std, bundles[job.dataset], design,
                        design_digest_value, run_dir, device, logger, plan, started_monotonic,
                    )
                    summary = upsert_summary(summary_path, rows)
                    plan.at[plan_idx, "status"] = "done"
                    update_progress(run_dir, plan, {"state": "job_done", "job": dataclasses.asdict(job)}, started_monotonic)
                    primary = next(r for r in rows if r["protocol"] == "imagenet1000")
                    logger.info(
                        "JOB DONE | %s | %s | %s | Acc=%.4f | ACMS=%.4f",
                        job.dataset, model_alias, job.condition, primary["accuracy"], primary["acms"],
                    )
                    display(summary.tail(10))
                    if job.dataset == "imagenette" and job.condition in {"clean_native", "clean_common"}:
                        gate_results[job.condition] = float(primary["accuracy"])
                        if job.condition == "clean_native" and gate_results[job.condition] < 0.40:
                            raise RuntimeError(
                                f"Clean native accuracy gate failed for {model_alias}: {gate_results[job.condition]:.4f}"
                            )
                        if job.condition == "clean_common" and gate_results[job.condition] < 0.35:
                            raise RuntimeError(
                                f"Clean common accuracy gate failed for {model_alias}: {gate_results[job.condition]:.4f}"
                            )
                        if {"clean_native", "clean_common"}.issubset(gate_results):
                            drop = gate_results["clean_native"] - gate_results["clean_common"]
                            if drop > 0.08:
                                message = f"Common geometry accuracy drop is large for {model_alias}: {drop:.4f}"
                                logger.warning(message)
                                append_jsonl(errors_path, {"type": "preprocessing_warning", "message": message})
                except Exception as exc:
                    plan.at[plan_idx, "status"] = "failed"
                    logger.exception("JOB FAILED: %s", job_key(job))
                    append_jsonl(errors_path, {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "job": dataclasses.asdict(job), "error": repr(exc), "traceback": traceback.format_exc(),
                    })
                    update_progress(run_dir, plan, {"state": "job_failed", "job": dataclasses.asdict(job)}, started_monotonic)
                    # A failed clean gate skips remaining jobs for the model in this invocation.
                    if job.dataset == "imagenette" and job.condition in {"clean_native", "clean_common"}:
                        logger.error("Skipping the remaining jobs for this model in the current session: %s", model_alias)
                        break
                    continue
        except Exception as exc:
            logger.exception("MODEL FAILED: %s", model_alias)
            append_jsonl(errors_path, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model_alias,
                "error": repr(exc), "traceback": traceback.format_exc(),
            })
        finally:
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("MODEL END: %s", model_alias)

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    completion = make_completion_report(plan, summary, run_dir)
    if not summary.empty:
        make_plots(summary, run_dir)
    all_done = bool((plan["status"] == "done").all())
    phase_marker = run_dir / f"PHASE_COMPLETE__{RUN_PHASE}.json"
    atomic_write_json(phase_marker, {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": RUN_PHASE, "all_jobs_done": all_done,
        "completed_jobs": int((plan["status"] == "done").sum()), "total_jobs": len(plan),
        "run_dir": str(run_dir),
    })
    update_progress(run_dir, plan, {"state": "phase_finished", "all_jobs_done": all_done}, started_monotonic)
    logger.info("PHASE FINISHED | %s | %d/%d done", RUN_PHASE, int((plan["status"] == "done").sum()), len(plan))
    print("\n" + "=" * 90)
    print("STAGE-1 OUTPUT DIRECTORY:")
    print(run_dir)
    print("Completed jobs:", int((plan["status"] == "done").sum()), "/", len(plan))
    if not all_done:
        print("Some jobs are missing or failed. Running all cells again with the same configuration retries only incomplete jobs.")
    return run_dir


if __name__ == "__main__":
    main()
