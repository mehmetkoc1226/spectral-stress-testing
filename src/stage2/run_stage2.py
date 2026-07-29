# -*- coding: utf-8 -*-
"""Resumable Stage-2 frequency-conditioned calibration pilot for Google Colab.

The runner is fail-closed: it never writes to the frozen Stage-1 run family,
uses a config-derived output directory, writes atomically, and resumes valid
completed calibration/evaluation chunks after disconnects.
"""
from __future__ import annotations

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
from typing import Any, Iterable, Mapping, Sequence


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_json(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    return default if value is None else json.loads(value)


RUN_PROFILE = os.environ.get("STAGE2_RUN_PROFILE", "sanity")
RUN_NAME = os.environ.get("STAGE2_RUN_NAME", "spectral_stage2_calibration_pilot_v1")
DRIVE_PROJECT_FOLDER = os.environ.get("STAGE2_DRIVE_FOLDER", "Spectral_Sensitivity_Project")
BASE_BATCH_SIZE = int(os.environ.get("STAGE2_BATCH_SIZE", "24"))
CHUNK_SIZE = int(os.environ.get("STAGE2_CHUNK_SIZE", "128"))
NUM_WORKERS = int(os.environ.get("STAGE2_NUM_WORKERS", "2"))
USE_MIXED_PRECISION = _env_bool("STAGE2_MIXED_PRECISION", True)
REQUIRE_GPU = _env_bool("STAGE2_REQUIRE_GPU", True)
SEED = int(os.environ.get("STAGE2_SEED", "20260727"))
MODELS_OVERRIDE = _env_json("STAGE2_MODELS", None)

TIMM_VERSION = "1.0.28"
CODE_VERSION = "stage2-calibration-pilot-2026-07-27-v4"
PROTOCOL_VERSION = "stage2-preregistration-v2"
SOURCE_ANALYSIS_ID = "spectral_master_analysis_v2__ecd5548085d9"
FROZEN_STAGE1_RUN = "spectral_stage1_confirmatory_v1__7044005bc7a4"
FORBIDDEN_RUN_PREFIXES = ("spectral_stage1", "stage1")
COMMON_IMAGE_SIZE = 224
COMMON_RESIZE_SIZE = 256
ECE_BINS = 15
CALIBRATION_FRACTION = 0.5
CALIBRATION_CONDITIONS = (
    "clean_common", "hard_lp_r020_linear", "hard_hp_r020_linear",
)
METHODS = ("raw", "clean_ts", "confidence_ts", "fcts", "oracle_ts")

MODEL_SPECS: dict[str, dict[str, str]] = {
    "resnet50_a1": {
        "timm_name": "resnet50.a1_in1k",
        "family": "cnn",
        "architecture": "resnet50",
        "training": "ImageNet-1K, timm A1 recipe",
    },
    "vitb16_in1k": {
        "timm_name": "vit_base_patch16_224.augreg_in1k",
        "family": "global_transformer",
        "architecture": "vit_base_patch16_224",
        "training": "ImageNet-1K, AugReg",
    },
    "swin_t_in1k": {
        "timm_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "family": "hierarchical_transformer",
        "architecture": "swin_tiny_patch4_window7_224",
        "training": "ImageNet-1K, Microsoft recipe",
    },
}
ANCHOR_MODELS = tuple(MODEL_SPECS)

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

PAPER_IMAGENETTE_CONDITIONS = (
    "clean_common",
    "hard_lp_r010_linear", "hard_hp_r010_linear",
    "hard_lp_r020_linear", "hard_hp_r020_linear",
    "hard_lp_r040_linear", "hard_hp_r040_linear",
    "gauss_lp_r010_linear", "gauss_hp_r010_linear",
    "gauss_lp_r020_linear", "gauss_hp_r020_linear",
    "blur_sigma2", "blur_sigma4", "noise_std010", "noise_std020",
)
PAPER_IMAGENETA_CONDITIONS = (
    "clean_common",
    "hard_lp_r010_linear", "hard_hp_r010_linear",
    "hard_lp_r020_linear", "hard_hp_r020_linear",
    "gauss_lp_r010_linear", "gauss_hp_r010_linear",
    "gauss_lp_r020_linear", "gauss_hp_r020_linear",
    "blur_sigma4", "noise_std020",
)
SANITY_IMAGENETTE_CONDITIONS = (
    "clean_common", "hard_lp_r020_linear", "hard_hp_r020_linear",
    "hard_hp_r010_linear", "hard_lp_r040_linear", "gauss_hp_r010_linear",
    "blur_sigma4", "noise_std020",
)
SANITY_IMAGENETA_CONDITIONS = (
    "clean_common", "hard_hp_r020_linear", "gauss_hp_r010_linear",
    "blur_sigma4", "noise_std020",
)
PROFILE_CONFIG = {
    "sanity": {
        "sample_limits": {"imagenette": 96, "imagenet_a": 120},
        "imagenette_conditions": SANITY_IMAGENETTE_CONDITIONS,
        "imagenet_a_conditions": SANITY_IMAGENETA_CONDITIONS,
    },
    "paper": {
        "sample_limits": {"imagenette": None, "imagenet_a": None},
        "imagenette_conditions": PAPER_IMAGENETTE_CONDITIONS,
        "imagenet_a_conditions": PAPER_IMAGENETA_CONDITIONS,
    },
}


def _ensure_packages() -> None:
    required = {
        "numpy": "numpy", "pandas": "pandas", "requests": "requests",
        "sklearn": "scikit-learn", "tqdm": "tqdm",
    }
    missing = []
    for import_name, pip_name in required.items():
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

from spectral_stage2_core import (
    SCHEMA_VERSION,
    Condition,
    apply_condition,
    apply_standardizer,
    confidence_features,
    fit_standardizer,
    fit_temperature_model,
    method_outputs,
    metrics_from_outputs,
    protocol_view,
    self_test as core_self_test,
    spectral_descriptors,
    stable_int_seed,
    temperature_from_parameters,
)


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


def atomic_write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    local_root = Path("/content/stage2_tmp") if Path("/content").exists() else Path("/tmp/stage2_tmp")
    local_root.mkdir(parents=True, exist_ok=True)
    local_tmp = local_root / f"{path.stem}.{os.getpid()}.csv.gz"
    frame.to_csv(local_tmp, index=False, compression="gzip")
    drive_tmp = path.with_suffix(path.suffix + ".tmp")
    shutil.copy2(local_tmp, drive_tmp)
    os.replace(drive_tmp, path)
    local_tmp.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    local_root = Path("/content/stage2_tmp") if Path("/content").exists() else Path("/tmp/stage2_tmp")
    local_root.mkdir(parents=True, exist_ok=True)
    local_tmp = local_root / f"{path.name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(local_tmp, **arrays)
    drive_tmp = path.with_suffix(path.suffix + ".tmp")
    shutil.copy2(local_tmp, drive_tmp)
    os.replace(drive_tmp, path)
    local_tmp.unlink(missing_ok=True)


def config_digest(payload: Any, length: int = 12) -> str:
    blob = json.dumps(jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("spectral_stage2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def mount_google_drive() -> Path:
    """Return the already-mounted Drive in Colab, or mount only from a notebook kernel.

    The Stage-2 runner is launched as a child Python process. Calling
    google.colab.drive.mount() from that child process fails because it has no
    IPython kernel. Therefore an existing mount is always reused first.
    """
    mounted_root = Path("/content/drive/MyDrive")
    if mounted_root.exists():
        return mounted_root

    try:
        from google.colab import drive  # type: ignore
        from IPython import get_ipython  # type: ignore

        ipython = get_ipython()
        if ipython is None or not hasattr(ipython, "kernel"):
            raise RuntimeError(
                "Google Drive is not mounted. In the notebook, first run "
                "drive.mount('/content/drive')."
            )

        drive.mount("/content/drive", force_remount=False)
        if not mounted_root.exists():
            raise RuntimeError("Drive mounting completed, but MyDrive was not found.")
        return mounted_root
    except ImportError:
        fallback = Path.cwd() / "drive_fallback"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Colab was not detected; using a local directory: {fallback}")
        return fallback


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe tar member: {member.name}")
        archive.extractall(destination)


def download_with_wget(url: str, destination: Path, logger: logging.Logger) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading/verifying: %s", url)
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


def condition_catalog() -> dict[str, Condition]:
    out: dict[str, Condition] = {"clean_common": Condition("clean_common", "clean", "clean")}
    for radius in (10, 20, 40):
        for mode, short in (("low_pass", "lp"), ("high_pass", "hp")):
            key = f"hard_{short}_r{radius:03d}_linear"
            out[key] = Condition(key, "hard", mode, radius, "linear")
    for radius in (10, 20):
        for mode, short in (("low_pass", "lp"), ("high_pass", "hp")):
            key = f"gauss_{short}_r{radius:03d}_linear"
            out[key] = Condition(key, "gaussian", mode, radius, "linear")
    for sigma in (2.0, 4.0):
        key = f"blur_sigma{int(sigma)}"
        out[key] = Condition(key, "blur", "blur", severity=sigma)
    for std in (0.10, 0.20):
        key = f"noise_std{int(std * 100):03d}"
        out[key] = Condition(key, "noise", "noise", severity=std)
    return out


CATALOG = condition_catalog()


@dataclass(frozen=True)
class DesignConfig:
    code_version: str
    protocol_version: str
    source_analysis_id: str
    run_name: str
    profile: str
    models: tuple[str, ...]
    sample_limits: dict[str, int | None]
    imagenette_conditions: tuple[str, ...]
    imagenet_a_conditions: tuple[str, ...]
    calibration_conditions: tuple[str, ...]
    calibration_fraction: float
    batch_size: int
    chunk_size: int
    num_workers: int
    mixed_precision: bool
    seed: int
    timm_version: str


def build_design() -> DesignConfig:
    if RUN_PROFILE not in PROFILE_CONFIG:
        raise ValueError(f"RUN_PROFILE must be one of {list(PROFILE_CONFIG)}")
    lowered = RUN_NAME.lower()
    if RUN_NAME == FROZEN_STAGE1_RUN or any(lowered.startswith(p) for p in FORBIDDEN_RUN_PREFIXES):
        raise RuntimeError(f"Safety stop: a Stage-1 run name cannot be used: {RUN_NAME}")
    models = tuple(MODELS_OVERRIDE) if MODELS_OVERRIDE else ANCHOR_MODELS
    unknown = sorted(set(models) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown model aliases: {unknown}")
    profile = PROFILE_CONFIG[RUN_PROFILE]
    return DesignConfig(
        code_version=CODE_VERSION,
        protocol_version=PROTOCOL_VERSION,
        source_analysis_id=SOURCE_ANALYSIS_ID,
        run_name=RUN_NAME,
        profile=RUN_PROFILE,
        models=models,
        sample_limits=dict(profile["sample_limits"]),
        imagenette_conditions=tuple(profile["imagenette_conditions"]),
        imagenet_a_conditions=tuple(profile["imagenet_a_conditions"]),
        calibration_conditions=CALIBRATION_CONDITIONS,
        calibration_fraction=CALIBRATION_FRACTION,
        batch_size=BASE_BATCH_SIZE,
        chunk_size=CHUNK_SIZE,
        num_workers=NUM_WORKERS,
        mixed_precision=USE_MIXED_PRECISION,
        seed=SEED,
        timm_version=TIMM_VERSION,
    )


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
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.targets1000[global_index]), global_index


@dataclass
class DatasetBundle:
    name: str
    paths: list[str]
    targets1000: np.ndarray
    restricted_indices: np.ndarray
    active_indices: np.ndarray
    manifest: pd.DataFrame
    source: str

    def dataset(self, indices: np.ndarray) -> PathDataset:
        return PathDataset(self.paths, self.targets1000, indices, common_geometry_transform())


def stratified_indices(targets: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    all_indices = np.arange(len(targets), dtype=np.int64)
    if limit is None or limit >= len(targets):
        return all_indices
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    selected: list[int] = []
    base_take = max(1, limit // len(classes))
    for cls in classes:
        candidates = np.flatnonzero(targets == cls)
        take = min(base_take, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(all_indices, np.asarray(selected, dtype=np.int64))
        extra = rng.choice(remaining, size=min(limit - len(selected), len(remaining)), replace=False)
        selected.extend(extra.tolist())
    return np.asarray(sorted(selected[:limit]), dtype=np.int64)


def stratified_split(
    targets: np.ndarray,
    active_indices: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    calibration: list[int] = []
    evaluation: list[int] = []
    active_targets = targets[active_indices]
    for cls in np.unique(active_targets):
        candidates = active_indices[active_targets == cls].copy()
        rng.shuffle(candidates)
        n_cal = max(1, min(len(candidates) - 1, int(round(len(candidates) * fraction))))
        calibration.extend(candidates[:n_cal].tolist())
        evaluation.extend(candidates[n_cal:].tolist())
    return np.asarray(sorted(calibration), dtype=np.int64), np.asarray(sorted(evaluation), dtype=np.int64)


def copy_dataset_to_local(source: Path, local: Path, logger: logging.Logger) -> None:
    marker = local / ".COPY_COMPLETE"
    signature = f"{source.resolve()}::{sum(1 for _ in source.rglob('*'))}"
    if marker.exists() and marker.read_text(encoding="utf-8") == signature:
        logger.info("Local dataset cache is ready: %s", local)
        return
    if local.exists():
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying dataset to local storage: %s", source)
    shutil.copytree(source, local)
    marker.write_text(signature, encoding="utf-8")


def prepare_imagenette(
    drive_project: Path,
    local_root: Path,
    sample_limit: int | None,
    seed: int,
    logger: logging.Logger,
) -> DatasetBundle:
    # Direct-to-local download avoids recursively traversing thousands of
    # small files through a shared-Drive shortcut. The scientific dataset is
    # unchanged; only the transport/cache path differs.
    local_root.mkdir(parents=True, exist_ok=True)
    logger.info("DATASET PHASE | Direct local Imagenette preparation started")
    base = datasets.Imagenette(
        root=str(local_root),
        split="val",
        size="320px",
        download=True,
    )
    logger.info("DATASET PHASE | Local Imagenette is ready | n=%d", len(base))
    observed = list(base.wnids)
    expected = sorted(IMAGENETTE_WNID_TO_IMAGENET_INDEX)
    if observed != expected:
        raise RuntimeError(f"Imagenette WNID mismatch: {observed}")
    samples = list(getattr(base, "_samples"))
    paths = [str(path) for path, _ in samples]
    targets = np.asarray(
        [IMAGENETTE_WNID_TO_IMAGENET_INDEX[observed[int(ord_)]] for _, ord_ in samples],
        dtype=np.int64,
    )
    active = stratified_indices(targets, sample_limit, seed)
    restricted = np.asarray(sorted(set(targets.tolist())), dtype=np.int64)
    manifest = pd.DataFrame({
        "dataset": "imagenette",
        "global_dataset_index": active,
        "target_imagenet1000": targets[active],
        "relative_path": [str(Path(paths[i]).relative_to(local_root)) for i in active],
    })
    return DatasetBundle(
        "imagenette", paths, targets, restricted, active, manifest,
        "torchvision Imagenette v2-320 validation split",
    )


def prepare_imagenet_a(
    drive_project: Path,
    local_root: Path,
    sample_limit: int | None,
    seed: int,
    logger: logging.Logger,
) -> DatasetBundle:
    # Direct-to-local download avoids shared-Drive small-file traversal.
    # wget -c resumes a partial archive within the same Colab runtime.
    local_root.mkdir(parents=True, exist_ok=True)
    tar_path = local_root / "imagenet-a.tar"
    extracted = local_root / "imagenet-a"
    marker = extracted / ".EXTRACT_COMPLETE"

    logger.info("DATASET PHASE | Direct local ImageNet-A preparation started")
    if not marker.exists():
        download_with_wget(IMAGENET_A_URL, tar_path, logger)
        temp = local_root / "_imagenet_a_extract_tmp"
        shutil.rmtree(temp, ignore_errors=True)
        safe_extract_tar(tar_path, temp)
        candidate = temp / "imagenet-a"
        if not candidate.exists():
            raise RuntimeError("ImageNet-A archive missing imagenet-a/")
        shutil.rmtree(extracted, ignore_errors=True)
        shutil.move(str(candidate), str(extracted))
        shutil.rmtree(temp, ignore_errors=True)
        marker.write_text("ok", encoding="utf-8")

    base = datasets.ImageFolder(root=str(extracted))
    logger.info(
        "DATASET PHASE | Local ImageNet-A is ready | classes=%d | n=%d",
        len(base.classes),
        len(base.samples),
    )
    if len(base.classes) != 200 or len(base.samples) != 7500:
        raise RuntimeError(f"ImageNet-A shape mismatch: {len(base.classes)} classes, {len(base.samples)} images")
    class_index = fetch_json(IMAGENET_CLASS_INDEX_URL, local_root / "imagenet_class_index.json")
    wnid_to_index = {value[0]: int(key) for key, value in class_index.items()}
    for wnid, expected in {"n01440764": 0, "n02102040": 217, "n03888257": 701}.items():
        if wnid_to_index.get(wnid) != expected:
            raise RuntimeError(f"ImageNet class index validation failed: {wnid}")
    paths = [str(path) for path, _ in base.samples]
    targets = np.asarray(
        [wnid_to_index[base.classes[int(ord_)]] for _, ord_ in base.samples], dtype=np.int64
    )
    active = stratified_indices(targets, sample_limit, seed + 17)
    restricted = np.asarray(sorted(set(targets.tolist())), dtype=np.int64)
    manifest = pd.DataFrame({
        "dataset": "imagenet_a",
        "global_dataset_index": active,
        "target_imagenet1000": targets[active],
        "relative_path": [str(Path(paths[i]).relative_to(local_root)) for i in active],
    })
    return DatasetBundle(
        "imagenet_a", paths, targets, restricted, active, manifest,
        "ImageNet-A official natural adversarial examples dataset",
    )


def validate_timm_names(model_aliases: Sequence[str]) -> None:
    available = set(timm.list_models(pretrained=True, include_tags=True))
    missing = [MODEL_SPECS[a]["timm_name"] for a in model_aliases if MODEL_SPECS[a]["timm_name"] not in available]
    if missing:
        raise RuntimeError("Pinned timm model names unavailable: " + ", ".join(missing))


def load_model(alias: str, device: torch.device, cache_dir: Path):
    spec = MODEL_SPECS[alias]
    model = timm.create_model(spec["timm_name"], pretrained=True, cache_dir=str(cache_dir))
    model = model.to(device).eval()
    config = timm.data.resolve_model_data_config(model)
    input_size = tuple(int(v) for v in config.get("input_size", (3, 224, 224)))
    if input_size != (3, 224, 224) or int(getattr(model, "num_classes", 1000)) != 1000:
        raise RuntimeError(f"Unexpected model I/O: {alias}, {input_size}, {getattr(model, 'num_classes', None)}")
    mean = tuple(float(v) for v in config["mean"])
    std = tuple(float(v) for v in config["std"])
    return model, mean, std


def normalize_model(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_t) / std_t


def valid_raw_shard(path: Path, expected_count: int, digest: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"schema_version", "design_digest", "global_indices", "targets1000", "logits", "descriptors"}
            return (
                required.issubset(data.files)
                and int(data["schema_version"][0]) == SCHEMA_VERSION
                and str(data["design_digest"][0]) == digest
                and len(data["global_indices"]) == expected_count
            )
    except Exception:
        return False


def infer_raw_chunk(
    model: torch.nn.Module,
    dataset: Dataset,
    start: int,
    end: int,
    condition: Condition,
    dataset_name: str,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    design: DesignConfig,
    digest: str,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    batch_size = design.batch_size
    subset = Subset(dataset, list(range(start, end)))
    while batch_size >= 1:
        try:
            loader = DataLoader(
                subset, batch_size=batch_size, shuffle=False, num_workers=design.num_workers,
                pin_memory=device.type == "cuda", persistent_workers=False,
            )
            collected: dict[str, list[np.ndarray]] = {}
            with torch.inference_mode():
                for images, targets, global_indices in loader:
                    images = images.to(device, non_blocking=True)
                    global_np = np.asarray(global_indices, dtype=np.int64)
                    transformed = apply_condition(images, condition, global_np.tolist(), design.seed, dataset_name)
                    descriptors = spectral_descriptors(transformed)
                    model_input = normalize_model(transformed, mean, std)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                        enabled=design.mixed_precision and device.type == "cuda",
                    ):
                        logits = model(model_input)
                    parts = {
                        "global_indices": global_np.astype(np.int32),
                        "targets1000": np.asarray(targets, dtype=np.int16),
                        "logits": logits.float().cpu().numpy().astype(np.float16),
                        "descriptors": descriptors.cpu().numpy().astype(np.float32),
                    }
                    for key, value in parts.items():
                        collected.setdefault(key, []).append(value)
            result = {key: np.concatenate(value) for key, value in collected.items()}
            result["schema_version"] = np.asarray([SCHEMA_VERSION], dtype=np.int16)
            result["design_digest"] = np.asarray([digest], dtype=f"<U{len(digest)}")
            result["effective_batch_size"] = np.asarray([batch_size], dtype=np.int16)
            return result
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or batch_size == 1:
                raise
            new_size = max(1, batch_size // 2)
            logger.warning("CUDA OOM: batch %d -> %d", batch_size, new_size)
            batch_size = new_size
            torch.cuda.empty_cache()
            gc.collect()
    raise RuntimeError("No valid batch size")


def load_shards(job_dir: Path) -> dict[str, np.ndarray]:
    shards = sorted(job_dir.glob("shard_*.npz"))
    if not shards:
        raise RuntimeError(f"No shards: {job_dir}")
    collected: dict[str, list[np.ndarray]] = {}
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            for key in ("global_indices", "targets1000", "logits", "descriptors"):
                collected.setdefault(key, []).append(data[key])
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    order = np.argsort(arrays["global_indices"], kind="mergesort")
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(np.unique(arrays["global_indices"])) != len(arrays["global_indices"]):
        raise RuntimeError(f"Duplicate indices: {job_dir}")
    return arrays


def collect_raw_arrays(
    model: torch.nn.Module,
    mean: Sequence[float],
    std: Sequence[float],
    bundle: DatasetBundle,
    active_indices: np.ndarray,
    condition_key: str,
    raw_dir: Path,
    design: DesignConfig,
    digest: str,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    condition = CATALOG[condition_key]
    dataset = bundle.dataset(active_indices)
    raw_dir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(dataset), design.chunk_size))
    progress = tqdm(starts, desc=f"{bundle.name}|{condition_key}", unit="chunk")
    for start in progress:
        end = min(start + design.chunk_size, len(dataset))
        shard = raw_dir / f"shard_{start:06d}_{end:06d}.npz"
        if valid_raw_shard(shard, end - start, digest):
            progress.set_postfix_str("resume")
            continue
        arrays = infer_raw_chunk(
            model, dataset, start, end, condition, bundle.name, mean, std,
            device, design, digest, logger,
        )
        atomic_save_npz(shard, **arrays)
        progress.set_postfix_str(f"saved {end}/{len(dataset)}")
    arrays = load_shards(raw_dir)
    if len(arrays["global_indices"]) != len(dataset):
        raise RuntimeError(f"Incomplete raw arrays: {raw_dir}")
    return arrays


def calibration_file(run_dir: Path, model_alias: str, condition: str) -> Path:
    return run_dir / "calibration_logits" / model_alias / f"{condition}.npz"


def valid_calibration_file(path: Path, expected_count: int, digest: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"global_indices", "targets1000", "logits", "descriptors", "design_digest"}
            return required.issubset(data.files) and len(data["global_indices"]) == expected_count and str(data["design_digest"][0]) == digest
    except Exception:
        return False


def ensure_calibration_data(
    model: torch.nn.Module,
    mean: Sequence[float],
    std: Sequence[float],
    model_alias: str,
    imagenette: DatasetBundle,
    calibration_indices: np.ndarray,
    design: DesignConfig,
    digest: str,
    run_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for condition in design.calibration_conditions:
        final_path = calibration_file(run_dir, model_alias, condition)
        if valid_calibration_file(final_path, len(calibration_indices), digest):
            with np.load(final_path, allow_pickle=False) as data:
                output[condition] = {key: data[key] for key in ("global_indices", "targets1000", "logits", "descriptors")}
            continue
        temp_dir = run_dir / "calibration_shards" / model_alias / condition
        arrays = collect_raw_arrays(
            model, mean, std, imagenette, calibration_indices, condition,
            temp_dir, design, digest, device, logger,
        )
        atomic_save_npz(
            final_path,
            global_indices=arrays["global_indices"].astype(np.int32),
            targets1000=arrays["targets1000"].astype(np.int16),
            logits=arrays["logits"].astype(np.float16),
            descriptors=arrays["descriptors"].astype(np.float32),
            restricted_indices=imagenette.restricted_indices.astype(np.int16),
            design_digest=np.asarray([digest], dtype=f"<U{len(digest)}"),
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        output[condition] = arrays
    return output


def fit_path(run_dir: Path, model_alias: str) -> Path:
    return run_dir / "fit_parameters" / f"{model_alias}.json"


def fit_calibrators(
    calibration: Mapping[str, Mapping[str, np.ndarray]],
    restricted_indices: np.ndarray,
    model_alias: str,
    design: DesignConfig,
    digest: str,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    path = fit_path(run_dir, model_alias)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("design_digest") != digest:
            raise RuntimeError(f"Fit file digest mismatch: {path}")
        return payload

    condition_order = list(design.calibration_conditions)
    all_logits = np.concatenate([calibration[c]["logits"].astype(np.float32) for c in condition_order])
    all_targets = np.concatenate([calibration[c]["targets1000"].astype(np.int64) for c in condition_order])
    all_descriptors = np.concatenate([calibration[c]["descriptors"].astype(np.float64) for c in condition_order])
    clean_logits = calibration["clean_common"]["logits"].astype(np.float32)
    clean_targets = calibration["clean_common"]["targets1000"].astype(np.int64)
    descriptor_stats = fit_standardizer(all_descriptors)
    descriptor_standardized = apply_standardizer(all_descriptors, descriptor_stats)

    result: dict[str, Any] = {
        "model": model_alias,
        "design_digest": digest,
        "protocol_version": design.protocol_version,
        "calibration_conditions": condition_order,
        "n_observations_per_condition": {c: int(len(calibration[c]["targets1000"])) for c in condition_order},
        "descriptor_standardization": descriptor_stats,
        "protocols": {},
    }
    protocols = ("imagenet1000", f"restricted{len(restricted_indices)}")
    for protocol in protocols:
        clean_view, clean_target_pos, _ = protocol_view(clean_logits, clean_targets, restricted_indices, protocol)
        clean_fit = fit_temperature_model(clean_view, clean_target_pos, regularization=0.0)
        clean_temperature = float(temperature_from_parameters(clean_fit["parameters"], None)[0])

        all_view, all_target_pos, _ = protocol_view(all_logits, all_targets, restricted_indices, protocol)
        conf_features = confidence_features(all_view)
        conf_stats = fit_standardizer(conf_features)
        conf_standardized = apply_standardizer(conf_features, conf_stats)
        ccts_fit = fit_temperature_model(
            all_view, all_target_pos, conf_standardized,
            regularization=1e-3, init_temperature=clean_temperature,
        )
        fcts_fit = fit_temperature_model(
            all_view, all_target_pos, descriptor_standardized,
            regularization=1e-3, init_temperature=clean_temperature,
        )
        result["protocols"][protocol] = {
            "clean_scalar": clean_fit,
            "clean_scalar_temperature": clean_temperature,
            "confidence_standardization": conf_stats,
            "confidence_conditioned": ccts_fit,
            "frequency_conditioned": fcts_fit,
        }
        logger.info(
            "FIT | %s | %s | clean T=%.4f | CCTS NLL %.4f | FCTS NLL %.4f",
            model_alias, protocol, clean_temperature, ccts_fit["fitted_nll"], fcts_fit["fitted_nll"],
        )
    atomic_write_json(path, result)
    return result


def compact_path(run_dir: Path, dataset: str, model: str, condition: str, protocol: str) -> Path:
    return run_dir / "compact_predictions" / dataset / model / f"{condition}__{protocol}.csv.gz"


def valid_compact(path: Path, expected_count: int) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        frame = pd.read_csv(path, usecols=["global_dataset_index"])
        return len(frame) == expected_count and frame["global_dataset_index"].is_unique
    except Exception:
        return False


def upsert_summary(summary_path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    keys = ["dataset", "model", "condition", "protocol", "method"]
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last")
    else:
        combined = new
    combined = combined.sort_values(keys).reset_index(drop=True)
    atomic_write_csv(summary_path, combined)
    return combined


def evaluate_job(
    arrays: Mapping[str, np.ndarray],
    bundle: DatasetBundle,
    model_alias: str,
    condition_key: str,
    fits: Mapping[str, Any],
    design: DesignConfig,
    digest: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    logits_full = arrays["logits"].astype(np.float32)
    targets = arrays["targets1000"].astype(np.int64)
    descriptors = arrays["descriptors"].astype(np.float64)
    descriptor_standardized = apply_standardizer(descriptors, fits["descriptor_standardization"])
    protocols = ("imagenet1000", f"restricted{len(bundle.restricted_indices)}")
    summary_rows: list[dict[str, Any]] = []
    spec = MODEL_SPECS[model_alias]

    for protocol in protocols:
        protocol_logits, target_pos, global_classes = protocol_view(
            logits_full, targets, bundle.restricted_indices, protocol
        )
        # Calibration is learned on Imagenette. For an unseen restricted
        # label space such as ImageNet-A restricted200, transfer the single
        # restricted-space calibrator learned on Imagenette restricted10.
        # Native and restricted calibrators remain separate; ImageNet-A is
        # never used for fitting or model selection.
        protocol_fit_key = protocol
        if protocol_fit_key not in fits["protocols"]:
            if protocol.startswith("restricted"):
                restricted_fit_keys = sorted(
                    key for key in fits["protocols"] if key.startswith("restricted")
                )
                if len(restricted_fit_keys) != 1:
                    raise RuntimeError(
                        f"Expected exactly one transferable restricted fit, "
                        f"found {restricted_fit_keys}"
                    )
                protocol_fit_key = restricted_fit_keys[0]
            else:
                raise KeyError(protocol)
        protocol_fit = fits["protocols"][protocol_fit_key]
        clean_t = float(protocol_fit["clean_scalar_temperature"])
        conf = confidence_features(protocol_logits)
        conf_standardized = apply_standardizer(conf, protocol_fit["confidence_standardization"])
        ccts_t = temperature_from_parameters(
            protocol_fit["confidence_conditioned"]["parameters"], conf_standardized
        )
        fcts_t = temperature_from_parameters(
            protocol_fit["frequency_conditioned"]["parameters"], descriptor_standardized
        )
        oracle_fit = fit_temperature_model(
            protocol_logits, target_pos, regularization=0.0, init_temperature=clean_t
        )
        oracle_t = float(temperature_from_parameters(oracle_fit["parameters"], None)[0])
        temperatures: dict[str, np.ndarray | float] = {
            "raw": 1.0,
            "clean_ts": clean_t,
            "confidence_ts": ccts_t,
            "fcts": fcts_t,
            "oracle_ts": oracle_t,
        }
        method_data: dict[str, dict[str, np.ndarray]] = {}
        for method, temp in temperatures.items():
            method_data[method] = method_outputs(protocol_logits, target_pos, global_classes, temp)
        raw_prediction = method_data["raw"]["prediction"]
        for method in METHODS[1:]:
            if not np.array_equal(raw_prediction, method_data[method]["prediction"]):
                raise RuntimeError(f"Top-1 invariance failed: {model_alias}, {condition_key}, {protocol}, {method}")

        compact = pd.DataFrame({
            "dataset": bundle.name,
            "model": model_alias,
            "condition": condition_key,
            "protocol": protocol,
            "global_dataset_index": arrays["global_indices"].astype(np.int64),
            "target_imagenet1000": targets,
            "spectral_log_hl_ratio": descriptors[:, 0],
            "spectral_centroid": descriptors[:, 1],
            "spectral_entropy": descriptors[:, 2],
        })
        for method in METHODS:
            outputs = method_data[method]
            for field in ("prediction", "confidence", "correct", "true_probability", "nll", "brier", "temperature"):
                compact[f"{method}_{field}"] = outputs[field]
            metrics = metrics_from_outputs(outputs, ECE_BINS)
            condition = CATALOG[condition_key]
            summary_rows.append({
                "dataset": bundle.name,
                "model": model_alias,
                "timm_name": spec["timm_name"],
                "family": spec["family"],
                "architecture": spec["architecture"],
                "condition": condition_key,
                "filter_family": condition.family,
                "mode": condition.mode,
                "radius": condition.radius,
                "severity": condition.severity,
                "protocol": protocol,
                "method": method,
                "n": len(targets),
                **metrics,
            })
        out = compact_path(run_dir, bundle.name, model_alias, condition_key, protocol)
        atomic_write_csv_gzip(out, compact)

    audit_n = min(256, len(targets))
    rng = np.random.default_rng(stable_int_seed(design.seed, bundle.name, model_alias, condition_key, "audit"))
    audit_idx = np.sort(rng.choice(len(targets), size=audit_n, replace=False))
    audit_path = run_dir / "audit_logits" / bundle.name / model_alias / f"{condition_key}.npz"
    atomic_save_npz(
        audit_path,
        global_indices=arrays["global_indices"][audit_idx].astype(np.int32),
        targets1000=targets[audit_idx].astype(np.int16),
        logits=arrays["logits"][audit_idx].astype(np.float16),
        descriptors=arrays["descriptors"][audit_idx].astype(np.float32),
        restricted_indices=bundle.restricted_indices.astype(np.int16),
        design_digest=np.asarray([digest], dtype=f"<U{len(digest)}"),
    )
    return summary_rows


def job_plan(design: DesignConfig) -> pd.DataFrame:
    rows = []
    for model in design.models:
        for dataset, conditions in (
            ("imagenette", design.imagenette_conditions),
            ("imagenet_a", design.imagenet_a_conditions),
        ):
            for condition in conditions:
                rows.append({
                    "dataset": dataset, "model": model, "condition": condition,
                    "status": "pending", "started_at": "", "completed_at": "", "error": "",
                })
    return pd.DataFrame(rows)


def update_status(run_dir: Path, plan: pd.DataFrame, current: Mapping[str, Any], start_time: float) -> None:
    completed = int((plan["status"] == "done").sum())
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": RUN_PROFILE,
        "completed_jobs": completed,
        "total_jobs": len(plan),
        "elapsed_seconds": time.monotonic() - start_time,
        **current,
    }
    atomic_write_csv(run_dir / "job_plan.csv", plan)
    atomic_write_json(run_dir / "LIVE_STATUS.json", payload)


def run_self_tests(run_dir: Path) -> None:
    result = core_self_test()
    atomic_write_json(run_dir / "scientific_self_tests.json", result)


def prepare_run_directory(
    drive_project: Path,
    design: DesignConfig,
    digest: str,
) -> Path:
    runs_root = drive_project / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    frozen_source = (runs_root / FROZEN_STAGE1_RUN).resolve()
    run_dir = (runs_root / f"{design.run_name}__{digest}").resolve()
    if run_dir == frozen_source or frozen_source in run_dir.parents:
        raise RuntimeError("Safety stop: output was directed into the frozen Stage-1 source directory")
    config_path = run_dir / "config.json"
    if run_dir.exists():
        if not config_path.exists():
            if any(run_dir.iterdir()):
                raise RuntimeError(f"The existing directory has no configuration and will not be overwritten: {run_dir}")
        else:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing.get("design_digest") != digest:
                raise RuntimeError(f"Configuration hash mismatch; the directory will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config_path, {"design_digest": digest, **dataclasses.asdict(design)})
    atomic_write_json(run_dir / "PROTOCOL_LOCK.json", {
        "protocol_version": PROTOCOL_VERSION,
        "source_analysis_id": SOURCE_ANALYSIS_ID,
        "primary_success_protocol": "imagenet1000",
        "primary_unseen_families": ["unseen_radius", "unseen_filter", "natural_corruption", "unseen_dataset"],
        "note": "Restricted protocols are secondary sensitivity analyses.",
    })
    return run_dir


def main() -> Path:
    seed_everything(SEED)
    design = build_design()
    digest = config_digest(design)
    drive_root = mount_google_drive()
    drive_project = drive_root / DRIVE_PROJECT_FOLDER
    run_dir = prepare_run_directory(drive_project, design, digest)
    logger = setup_logger(run_dir / "run.log")
    start_time = time.monotonic()
    logger.info("Stage-2 started | profile=%s | digest=%s | run_dir=%s", RUN_PROFILE, digest, run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if REQUIRE_GPU and device.type != "cuda":
        raise RuntimeError("A GPU is required. In Colab, select Runtime > Change runtime type > GPU.")
    validate_timm_names(design.models)
    run_self_tests(run_dir)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    atomic_write_json(run_dir / "environment.json", environment)

    # Create an observable plan and heartbeat before dataset preparation.
    # This lets a second notebook verify that the run is alive even before
    # the first inference job starts.
    plan_path = run_dir / "job_plan.csv"
    if plan_path.exists():
        plan = pd.read_csv(plan_path, dtype=str).fillna("")
    else:
        plan = job_plan(design)
        atomic_write_csv(plan_path, plan)
    update_status(
        run_dir,
        plan,
        {"phase": "dataset_preparation", "detail": "starting"},
        start_time,
    )

    local_root = Path("/content/stage2_data") if Path("/content").exists() else Path("/tmp/stage2_data")
    imagenette = prepare_imagenette(
        drive_project, local_root, design.sample_limits["imagenette"], design.seed, logger
    )
    imagenet_a = prepare_imagenet_a(
        drive_project, local_root, design.sample_limits["imagenet_a"], design.seed, logger
    )
    calibration_indices, evaluation_indices = stratified_split(
        imagenette.targets1000, imagenette.active_indices,
        design.calibration_fraction, design.seed + 101,
    )
    split_rows = []
    calibration_set = set(calibration_indices.tolist())
    for idx in imagenette.active_indices:
        split_rows.append({
            "global_dataset_index": int(idx),
            "target_imagenet1000": int(imagenette.targets1000[idx]),
            "split": "calibration" if int(idx) in calibration_set else "evaluation",
        })
    atomic_write_csv(run_dir / "calibration_split_manifest.csv", pd.DataFrame(split_rows))
    atomic_write_csv(run_dir / "dataset_manifest_imagenette.csv", imagenette.manifest)
    atomic_write_csv(run_dir / "dataset_manifest_imagenet_a.csv", imagenet_a.manifest)

    # Re-read the early plan so resume state is preserved.
    plan = pd.read_csv(plan_path, dtype=str).fillna("")
    update_status(
        run_dir,
        plan,
        {"phase": "model_evaluation", "detail": "datasets_ready"},
        start_time,
    )
    summary_path = run_dir / "summary.csv"
    model_cache = drive_project / "model_cache"

    for model_alias in design.models:
        logger.info("MODEL LOAD | %s", model_alias)
        model, mean, std = load_model(model_alias, device, model_cache)
        try:
            calibration = ensure_calibration_data(
                model, mean, std, model_alias, imagenette, calibration_indices,
                design, digest, run_dir, device, logger,
            )
            fits = fit_calibrators(
                calibration, imagenette.restricted_indices, model_alias,
                design, digest, run_dir, logger,
            )
            model_jobs = plan[plan["model"] == model_alias].index.tolist()
            for row_index in model_jobs:
                row = plan.loc[row_index]
                dataset_name = str(row["dataset"])
                condition_key = str(row["condition"])
                bundle = imagenette if dataset_name == "imagenette" else imagenet_a
                eval_indices = evaluation_indices if dataset_name == "imagenette" else bundle.active_indices
                protocols = ("imagenet1000", f"restricted{len(bundle.restricted_indices)}")
                done_marker = run_dir / "job_markers" / dataset_name / model_alias / f"{condition_key}.json"
                compact_ok = all(
                    valid_compact(compact_path(run_dir, dataset_name, model_alias, condition_key, p), len(eval_indices))
                    for p in protocols
                )
                if done_marker.exists() and compact_ok:
                    plan.at[row_index, "status"] = "done"
                    continue
                plan.at[row_index, "status"] = "running"
                plan.at[row_index, "started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                update_status(
                    run_dir, plan,
                    {"dataset": dataset_name, "model": model_alias, "condition": condition_key},
                    start_time,
                )
                temp_dir = run_dir / "eval_temp" / dataset_name / model_alias / condition_key
                try:
                    arrays = collect_raw_arrays(
                        model, mean, std, bundle, eval_indices, condition_key,
                        temp_dir, design, digest, device, logger,
                    )
                    rows = evaluate_job(
                        arrays, bundle, model_alias, condition_key, fits,
                        design, digest, run_dir,
                    )
                    upsert_summary(summary_path, rows)
                    atomic_write_json(done_marker, {
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "dataset": dataset_name,
                        "model": model_alias,
                        "condition": condition_key,
                        "n": len(eval_indices),
                        "design_digest": digest,
                    })
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    plan.at[row_index, "status"] = "done"
                    plan.at[row_index, "completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    plan.at[row_index, "error"] = ""
                except Exception as exc:
                    plan.at[row_index, "status"] = "error"
                    plan.at[row_index, "error"] = repr(exc)
                    update_status(run_dir, plan, {"error": repr(exc)}, start_time)
                    logger.error("JOB ERROR | %s", traceback.format_exc())
                    raise
                update_status(run_dir, plan, {}, start_time)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    expected_summary_rows = len(plan) * 2 * len(METHODS)
    all_done = bool((plan["status"] == "done").all()) and len(summary) == expected_summary_rows
    completion = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": design.profile,
        "all_jobs_done": all_done,
        "completed_jobs": int((plan["status"] == "done").sum()),
        "total_jobs": len(plan),
        "summary_rows": len(summary),
        "expected_summary_rows": expected_summary_rows,
        "run_dir": str(run_dir),
        "design_digest": digest,
    }
    atomic_write_json(run_dir / f"RUN_COMPLETE__{design.profile}.json", completion)
    if not all_done:
        raise RuntimeError(f"Completion check failed: {completion}")
    logger.info("Stage-2 completed: %s", completion)
    return run_dir


if __name__ == "__main__":
    try:
        print(main())
    except Exception:
        traceback.print_exc()
        raise
