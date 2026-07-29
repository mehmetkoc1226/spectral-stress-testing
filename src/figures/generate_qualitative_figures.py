from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms

PROJECT_FOLDER = "Spectral_Sensitivity_Project"
FROZEN_STAGE1_RUN = "spectral_stage1_confirmatory_v1__7044005bc7a4"
OUTPUT_FOLDER = "paper_assets/qualitative_figures_v2_compact"
GALLERY_HASH_SALT = "spectral-gallery-v1-20260728"
GALLERY_GLOBAL_INDEX: int | None = None

MODELS = [
    "resnet50_a1",
    "resnet50_a2",
    "vitb16_in1k",
    "vitb16_in21k",
    "swin_t_in1k",
    "swin_t_in22k",
]

IMAGENETTE_CLASSES = {
    0: ("n01440764", "tench"),
    217: ("n02102040", "English springer"),
    482: ("n02979186", "cassette player"),
    491: ("n03000684", "chain saw"),
    497: ("n03028079", "church"),
    566: ("n03394916", "French horn"),
    569: ("n03417042", "garbage truck"),
    571: ("n03425413", "gas pump"),
    574: ("n03445777", "golf ball"),
    701: ("n03888257", "parachute"),
}

COMMON_RESIZE = 256
COMMON_SIZE = 224


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(relative_path: str) -> str:
    return hashlib.sha256(
        f"{GALLERY_HASH_SALT}::{relative_path}".encode("utf-8")
    ).hexdigest()


def radial_distance(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy = torch.arange(h, device=device, dtype=dtype) - h // 2
    xx = torch.arange(w, device=device, dtype=dtype) - w // 2
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.sqrt(x.square() + y.square())


def spectral_mask(
    h: int,
    w: int,
    radius: int,
    family: str,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    d = radial_distance(h, w, device, dtype)
    if family == "hard":
        low = (d <= float(radius)).to(dtype)
    elif family == "gaussian":
        sigma = float(radius) / math.sqrt(2.0 * math.log(2.0))
        low = torch.exp(-0.5 * (d / sigma) ** 2)
    else:
        raise ValueError(f"Unsupported family: {family}")
    if mode == "low_pass":
        return low
    if mode == "high_pass":
        return 1.0 - low
    raise ValueError(f"Unsupported mode: {mode}")


def apply_spectral(
    x: torch.Tensor,
    radius: int,
    family: str,
    mode: str,
    postprocess: str = "linear",
) -> torch.Tensor:
    batched = x.unsqueeze(0)
    spectrum = torch.fft.fftshift(
        torch.fft.fftn(batched, dim=(-2, -1)), dim=(-2, -1)
    )
    mask = spectral_mask(
        x.shape[-2], x.shape[-1], radius, family, mode, x.device, x.dtype
    )
    recon = torch.fft.ifftn(
        torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), dim=(-2, -1)
    ).real.squeeze(0)
    if postprocess == "linear":
        return recon
    if postprocess == "clip":
        return recon.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported postprocess: {postprocess}")


def energy_ratio(original: torch.Tensor, transformed: torch.Tensor) -> float:
    denom = original.square().sum().clamp_min(1e-12)
    return float(transformed.square().sum() / denom)


def display_rgb(x: torch.Tensor) -> np.ndarray:
    # Display only. Frozen inference used the signed-linear tensor.
    return x.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def main() -> Path:
    drive_root = Path("/content/drive/MyDrive")
    project_dir = drive_root / PROJECT_FOLDER
    run_dir = project_dir / "runs" / FROZEN_STAGE1_RUN
    output_dir = project_dir / OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Project:", project_dir)
    print("Frozen run:", run_dir)
    print("Output:", output_dir)

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Frozen Stage-1 run was not found: {run_dir}. "
            "Check the Google account and the MyDrive project shortcut."
        )

    manifest_path = run_dir / "dataset_manifests" / "imagenette.csv"
    pred_root = run_dir / "predictions" / "imagenette"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not pred_root.exists():
        raise FileNotFoundError(pred_root)

    manifest = pd.read_csv(manifest_path).sort_values("global_dataset_index").reset_index(drop=True)
    required_cols = {
        "dataset", "active_position", "global_dataset_index",
        "relative_path", "wnid", "target_imagenet1000",
    }
    missing = required_cols - set(manifest.columns)
    if missing:
        raise RuntimeError(f"Manifest columns are missing: {sorted(missing)}")
    if len(manifest) != 3925 or not manifest["global_dataset_index"].is_unique:
        raise RuntimeError("The frozen Imagenette manifest failed the integrity check.")
    if set(manifest["target_imagenet1000"].unique()) != set(IMAGENETTE_CLASSES):
        raise RuntimeError("Frozen Imagenette target set mismatch.")

    for target, (wnid, _) in IMAGENETTE_CLASSES.items():
        rows = manifest[manifest["target_imagenet1000"] == target]
        if set(rows["wnid"]) != {wnid}:
            raise RuntimeError(f"WNID/target mismatch: {target}")

    local_root = Path("/content/qualitative_imagenette")
    base = datasets.Imagenette(
        root=str(local_root), split="val", size="320px", download=True, transform=None
    )
    samples = list(getattr(base, "_samples"))
    observed_wnids = list(base.wnids)
    if len(samples) != 3925:
        raise RuntimeError(f"Beklenmeyen Imagenette boyutu: {len(samples)}")

    dataset_paths = [Path(path) for path, _ in samples]
    wnid_to_target = {wnid: target for target, (wnid, _) in IMAGENETTE_CLASSES.items()}
    dataset_targets = np.asarray(
        [wnid_to_target[observed_wnids[int(ord_)]] for _, ord_ in samples], dtype=np.int64
    )

    for row in manifest.itertuples(index=False):
        idx = int(row.global_dataset_index)
        if dataset_targets[idx] != int(row.target_imagenet1000):
            raise RuntimeError(f"Dataset target mismatch: global index {idx}")
        if dataset_paths[idx].name != Path(row.relative_path).name:
            raise RuntimeError(f"Dataset filename mismatch: global index {idx}")

    common_geometry = transforms.Compose([
        transforms.Resize(COMMON_RESIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(COMMON_SIZE),
        transforms.ToTensor(),
    ])

    def load_common_tensor(global_index: int) -> torch.Tensor:
        with Image.open(dataset_paths[int(global_index)]) as image:
            return common_geometry(image.convert("RGB"))

    test_x = load_common_tensor(int(manifest.iloc[0]["global_dataset_index"]))
    for family in ("hard", "gaussian"):
        for radius in (10, 20):
            lp = apply_spectral(test_x, radius, family, "low_pass")
            hp = apply_spectral(test_x, radius, family, "high_pass")
            err = float((lp + hp - test_x).abs().max())
            if err > 2e-5:
                raise RuntimeError(
                    f"Complementarity failed: {family}, r={radius}, err={err}"
                )

    def prediction_path(model: str, condition: str) -> Path:
        return pred_root / model / f"{condition}__imagenet1000.csv.gz"

    def load_prediction(model: str, condition: str) -> pd.DataFrame:
        path = prediction_path(model, condition)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {
            "global_dataset_index", "target_imagenet1000",
            "prediction_imagenet1000", "confidence", "correct", "true_probability",
        }
        miss = required - set(frame.columns)
        if miss:
            raise RuntimeError(f"{path.name}: eksik kolonlar {sorted(miss)}")
        frame["correct"] = frame["correct"].astype(bool)
        return frame.set_index("global_dataset_index").sort_index()

    needed_conditions = [
        "clean_common",
        "hard_lp_r010_linear", "hard_hp_r010_linear",
        "hard_lp_r020_linear", "hard_hp_r020_linear",
        "gauss_lp_r010_linear", "gauss_hp_r010_linear",
        "gauss_hp_r020_linear", "gauss_hp_r020_clip",
    ]
    pred = {
        (model, condition): load_prediction(model, condition)
        for model in MODELS for condition in needed_conditions
    }
    expected_indices = set(manifest["global_dataset_index"].astype(int))
    for key, frame in pred.items():
        if set(frame.index.astype(int)) != expected_indices:
            raise RuntimeError(f"Prediction index mismatch: {key}")

    def clean_correct_count(global_index: int) -> int:
        return sum(
            bool(pred[(m, "clean_common")].loc[global_index, "correct"])
            for m in MODELS
        )

    if GALLERY_GLOBAL_INDEX is None:
        candidates = manifest.copy()
        candidates["clean_correct_count"] = candidates["global_dataset_index"].map(
            clean_correct_count
        )
        candidates = candidates[candidates["clean_correct_count"] == len(MODELS)].copy()
        if candidates.empty:
            raise RuntimeError("No gallery candidate was correctly classified by all 6 checkpoints on clean input.")
        candidates["selection_hash"] = candidates["relative_path"].map(stable_hash)
        gallery_row = candidates.sort_values("selection_hash").iloc[0]
        gallery_index = int(gallery_row["global_dataset_index"])
    else:
        gallery_index = int(GALLERY_GLOBAL_INDEX)

    def class_direction_table(target: int, prefer: str) -> pd.DataFrame:
        rows = manifest[manifest["target_imagenet1000"] == target]
        records = []
        for row in rows.itertuples(index=False):
            idx = int(row.global_dataset_index)
            hp_only = 0
            lp_only = 0
            diffs = []
            for model in MODELS:
                lp = pred[(model, "hard_lp_r010_linear")].loc[idx]
                hp = pred[(model, "hard_hp_r010_linear")].loc[idx]
                hp_only += int(bool(hp["correct"]) and not bool(lp["correct"]))
                lp_only += int(bool(lp["correct"]) and not bool(hp["correct"]))
                diffs.append(float(hp["true_probability"]) - float(lp["true_probability"]))
            records.append({
                "global_dataset_index": idx,
                "relative_path": row.relative_path,
                "wnid": row.wnid,
                "target_imagenet1000": target,
                "clean_correct_count": clean_correct_count(idx),
                "hp_only_votes": hp_only,
                "lp_only_votes": lp_only,
                "direction_score_hp_minus_lp": hp_only - lp_only,
                "mean_true_probability_hp_minus_lp": float(np.mean(diffs)),
            })
        out = pd.DataFrame(records)
        if prefer == "hp":
            return out.sort_values(
                ["clean_correct_count", "direction_score_hp_minus_lp",
                 "mean_true_probability_hp_minus_lp", "global_dataset_index"],
                ascending=[False, False, False, True],
            )
        return out.sort_values(
            ["clean_correct_count", "direction_score_hp_minus_lp",
             "mean_true_probability_hp_minus_lp", "global_dataset_index"],
            ascending=[False, True, True, True],
        )

    french_candidates = class_direction_table(566, "hp")
    tench_candidates = class_direction_table(0, "lp")
    french_index = int(french_candidates.iloc[0]["global_dataset_index"])
    tench_index = int(tench_candidates.iloc[0]["global_dataset_index"])

    print("Gallery index:", gallery_index)
    print("French horn index:", french_index)
    print("Tench index:", tench_index)

    # Figure A: compact condition gallery plus diagnostic companion.
    x = load_common_tensor(gallery_index)
    hard_lp10 = apply_spectral(x, 10, "hard", "low_pass")
    hard_hp10 = apply_spectral(x, 10, "hard", "high_pass")
    hard_lp20 = apply_spectral(x, 20, "hard", "low_pass")
    hard_hp20 = apply_spectral(x, 20, "hard", "high_pass")
    gauss_lp10 = apply_spectral(x, 10, "gaussian", "low_pass")
    gauss_hp10 = apply_spectral(x, 10, "gaussian", "high_pass")
    gauss_hp20_linear = apply_spectral(
        x, 20, "gaussian", "high_pass", "linear"
    )
    gauss_hp20_clip = apply_spectral(
        x, 20, "gaussian", "high_pass", "clip"
    )
    clip_delta = (
        gauss_hp20_linear - gauss_hp20_clip
    ).abs().mean(dim=0).cpu().numpy()

    manifest_indexed = manifest.set_index("global_dataset_index")
    row = manifest_indexed.loc[gallery_index]
    class_name = IMAGENETTE_CLASSES[
        int(row["target_imagenet1000"])
    ][1]

    gallery_energy = {
        "clean": 100.0 * energy_ratio(x, x),
        "hard_lp10": 100.0 * energy_ratio(x, hard_lp10),
        "hard_hp10": 100.0 * energy_ratio(x, hard_hp10),
        "hard_lp20": 100.0 * energy_ratio(x, hard_lp20),
        "hard_hp20": 100.0 * energy_ratio(x, hard_hp20),
        "gauss_lp10": 100.0 * energy_ratio(x, gauss_lp10),
        "gauss_hp10": 100.0 * energy_ratio(x, gauss_hp10),
        "gauss_hp20_linear": 100.0 * energy_ratio(
            x, gauss_hp20_linear
        ),
        "gauss_hp20_clip": 100.0 * energy_ratio(
            x, gauss_hp20_clip
        ),
    }
    outside = float(
        ((gauss_hp20_linear < 0) | (gauss_hp20_linear > 1))
        .float()
        .mean()
    )
    mean_clip_delta = float(
        (gauss_hp20_linear - gauss_hp20_clip).abs().mean()
    )

    # No large title/subtitle. Explanatory text and numerical details are
    # written to generated caption files.
    fig = plt.figure(
        figsize=(15.5, 10.2),
        constrained_layout=False,
    )
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 0.84],
        hspace=0.075,
        left=0.012,
        right=0.992,
        top=0.992,
        bottom=0.055,
    )

    row_hard = outer[0].subgridspec(
        1, 5, wspace=0.012
    )
    hard_panels = [
        ("(a) Clean", x),
        ("(b) Hard LP, r=10", hard_lp10),
        ("(c) Hard HP, r=10", hard_hp10),
        ("(d) Hard LP, r=20", hard_lp20),
        ("(e) Hard HP, r=20", hard_hp20),
    ]

    row_gauss = outer[1].subgridspec(
        1, 4, wspace=0.012
    )
    gauss_panels = [
        ("(f) Gaussian LP, r=10", gauss_lp10),
        ("(g) Gaussian HP, r=10", gauss_hp10),
        (
            "(h) Gaussian HP, r=20 — signed",
            gauss_hp20_linear,
        ),
        (
            "(i) Gaussian HP, r=20 — clipped",
            gauss_hp20_clip,
        ),
    ]

    def show_gallery_panel(
        ax,
        label: str,
        image: torch.Tensor,
    ) -> None:
        ax.imshow(display_rgb(image))
        ax.axis("off")
        ax.set_anchor("C")
        ax.text(
            0.015,
            0.985,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.4,
            },
        )

    for cell, (label, image) in zip(
        row_hard,
        hard_panels,
    ):
        show_gallery_panel(
            fig.add_subplot(cell),
            label,
            image,
        )

    for cell, (label, image) in zip(
        row_gauss,
        gauss_panels,
    ):
        show_gallery_panel(
            fig.add_subplot(cell),
            label,
            image,
        )

    row_diag = outer[2].subgridspec(
        1,
        3,
        width_ratios=[1.0, 1.05, 1.2],
        wspace=0.22,
    )

    ax = fig.add_subplot(row_diag[0])
    im = ax.imshow(clip_delta)
    ax.text(
        0.015,
        0.985,
        "(j) |signed − clipped|",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 1.4,
        },
    )
    ax.axis("off")
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.043,
        pad=0.025,
    )

    ax = fig.add_subplot(row_diag[1])
    gray = x.mean(dim=0)
    power = torch.fft.fftshift(
        torch.fft.fftn(gray)
    ).abs().square()
    d = radial_distance(
        COMMON_SIZE,
        COMMON_SIZE,
        power.device,
        power.dtype,
    )
    r_int = d.round().long()
    max_r = int(r_int.max())
    profile = torch.zeros(max_r + 1)
    counts = torch.zeros(max_r + 1)
    profile.scatter_add_(
        0,
        r_int.flatten().cpu(),
        power.flatten().cpu(),
    )
    counts.scatter_add_(
        0,
        r_int.flatten().cpu(),
        torch.ones_like(power).flatten().cpu(),
    )
    profile = profile / counts.clamp_min(1)
    ax.plot(
        np.arange(len(profile)),
        np.log10(profile.numpy() + 1e-12),
    )
    ax.axvline(10, linestyle="--", linewidth=1)
    ax.axvline(20, linestyle="--", linewidth=1)
    ax.set_title(
        "(k) Radial log-power profile",
        fontsize=9,
    )
    ax.set_xlabel("Radius")
    ax.set_ylabel("log10 mean power")
    ax.margins(x=0.01)

    ax = fig.add_subplot(row_diag[2])
    radius_axis = np.linspace(0, 60, 500)
    for radius in (10, 20):
        hard = (
            radius_axis <= radius
        ).astype(float)
        sigma = (
            radius / np.sqrt(2 * np.log(2))
        )
        gauss = np.exp(
            -0.5
            * (radius_axis / sigma) ** 2
        )
        ax.plot(
            radius_axis,
            hard,
            label=f"Hard LP, r={radius}",
        )
        ax.plot(
            radius_axis,
            gauss,
            linestyle="--",
            label=f"Gaussian LP, r={radius}",
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Radius")
    ax.set_ylabel("Gain")
    ax.set_title(
        "(l) Low-pass transfer functions",
        fontsize=9,
    )
    ax.legend(
        fontsize=7,
        frameon=False,
    )
    ax.margins(x=0.01)

    from matplotlib.lines import Line2D
    fig.add_artist(
        Line2D(
            [0.015, 0.99],
            [0.325, 0.325],
            transform=fig.transFigure,
            linewidth=0.7,
        )
    )

    fig_a_png = (
        output_dir
        / "figure_condition_gallery.png"
    )
    fig_a_pdf = (
        output_dir
        / "figure_condition_gallery.pdf"
    )
    fig.savefig(
        fig_a_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    fig.savefig(
        fig_a_pdf,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)

    gallery_caption = (
        f"Illustrative condition gallery for a {class_name} image "
        f"(global dataset index {gallery_index}). "
        "(a) Clean input; (b–e) hard low- and high-pass complements at "
        "r=10 and r=20; (f–g) Gaussian low- and high-pass complements at "
        "r=10; (h–i) Gaussian HP r=20 before and after actual clipping; "
        "(j) absolute pixelwise change induced by clipping; "
        "(k) radial log-power profile with r=10 and r=20 marked; and "
        "(l) hard and Gaussian low-pass transfer functions. "
        "The retained input-energy ratios for panels (a–i) are "
        f"{gallery_energy['clean']:.1f}%, "
        f"{gallery_energy['hard_lp10']:.1f}%, "
        f"{gallery_energy['hard_hp10']:.1f}%, "
        f"{gallery_energy['hard_lp20']:.1f}%, "
        f"{gallery_energy['hard_hp20']:.1f}%, "
        f"{gallery_energy['gauss_lp10']:.1f}%, "
        f"{gallery_energy['gauss_hp10']:.1f}%, "
        f"{gallery_energy['gauss_hp20_linear']:.1f}%, and "
        f"{gallery_energy['gauss_hp20_clip']:.1f}%, respectively. "
        f"For Gaussian HP r=20, {100.0 * outside:.1f}% of channel values "
        "fall outside [0,1] before clipping and the mean absolute clipping "
        f"change is {mean_clip_delta:.4f}. Signed-linear RGB panels are "
        "clipped to [0,1] only for human-readable display; frozen model "
        "inference used the original signed tensors. The example was selected "
        "among images classified correctly on clean input by all six "
        "checkpoints using a fixed SHA-256 ordering. It is illustrative only; "
        "quantitative claims use the complete paired analysis."
    )
    (
        output_dir
        / "figure_condition_gallery_caption.txt"
    ).write_text(
        gallery_caption,
        encoding="utf-8",
    )
    (
        output_dir
        / "figure_condition_gallery_caption.tex"
    ).write_text(
        gallery_caption.replace("%", r"\%"),
        encoding="utf-8",
    )

    def vote_summary(global_index: int) -> tuple[int, int, float]:
        hp_only = 0
        lp_only = 0
        diffs = []
        for model in MODELS:
            lp = pred[(model, "hard_lp_r010_linear")].loc[global_index]
            hp = pred[(model, "hard_hp_r010_linear")].loc[global_index]
            hp_only += int(bool(hp["correct"]) and not bool(lp["correct"]))
            lp_only += int(bool(lp["correct"]) and not bool(hp["correct"]))
            diffs.append(float(hp["true_probability"]) - float(lp["true_probability"]))
        return hp_only, lp_only, float(np.mean(diffs))

    # Figure B: French horn and tench class contrast.
    # Large title and explanatory subtitle are moved to caption files.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.2, 7.0),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.985,
        bottom=0.025,
        wspace=0.022,
        hspace=0.045,
    )

    class_caption_values = []
    for row_i, (idx, expected) in enumerate([
        (
            french_index,
            "HP-favoring",
        ),
        (
            tench_index,
            "LP-favoring",
        ),
    ]):
        clean = load_common_tensor(idx)
        lp = apply_spectral(
            clean,
            10,
            "hard",
            "low_pass",
        )
        hp = apply_spectral(
            clean,
            10,
            "hard",
            "high_pass",
        )
        target = int(
            manifest_indexed.loc[
                idx,
                "target_imagenet1000",
            ]
        )
        class_name = (
            IMAGENETTE_CLASSES[target][1]
        )
        hp_only, lp_only, tprob_diff = (
            vote_summary(idx)
        )

        row_energy = [
            100.0 * energy_ratio(
                clean,
                clean,
            ),
            100.0 * energy_ratio(
                clean,
                lp,
            ),
            100.0 * energy_ratio(
                clean,
                hp,
            ),
        ]
        class_caption_values.append({
            "class_name": class_name,
            "global_index": idx,
            "expected": expected,
            "energies": row_energy,
            "hp_only": hp_only,
            "lp_only": lp_only,
            "mean_delta_true_probability": (
                tprob_diff
            ),
        })

        for col_i, (name, image) in enumerate([
            ("Clean", clean),
            ("Hard LP, r=10", lp),
            ("Hard HP, r=10", hp),
        ]):
            ax = axes[row_i, col_i]
            ax.imshow(display_rgb(image))
            ax.axis("off")
            panel_letter = chr(
                ord("a")
                + row_i * 3
                + col_i
            )
            ax.text(
                0.015,
                0.985,
                f"({panel_letter}) {name}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.4,
                },
            )

        axes[row_i, 0].text(
            -0.055,
            0.5,
            f"{class_name}\n{expected}",
            transform=axes[
                row_i,
                0,
            ].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.2,
        )

    fig_b_png = (
        output_dir
        / "figure_class_contrast.png"
    )
    fig_b_pdf = (
        output_dir
        / "figure_class_contrast.pdf"
    )
    fig.savefig(
        fig_b_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    fig.savefig(
        fig_b_pdf,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)

    french_meta, tench_meta = (
        class_caption_values
    )
    class_caption = (
        "Illustrative class-level contrast under the hard r=10 intervention. "
        f"Panels (a–c) show a {french_meta['class_name']} image "
        f"(global dataset index {french_meta['global_index']}), a class with "
        "a consistent aggregate HP advantage; panels (d–f) show a "
        f"{tench_meta['class_name']} image "
        f"(global dataset index {tench_meta['global_index']}), a class with "
        "a consistent aggregate LP advantage. Within each row, the panels are "
        "clean, hard LP r=10, and hard HP r=10. Retained energies are "
        f"{french_meta['energies'][0]:.1f}%, "
        f"{french_meta['energies'][1]:.1f}%, and "
        f"{french_meta['energies'][2]:.1f}% for French horn, and "
        f"{tench_meta['energies'][0]:.1f}%, "
        f"{tench_meta['energies'][1]:.1f}%, and "
        f"{tench_meta['energies'][2]:.1f}% for tench. "
        "The frozen six-checkpoint selection rule yielded "
        f"{french_meta['hp_only']} HP-only versus "
        f"{french_meta['lp_only']} LP-only correctness votes for the French "
        f"horn example, and {tench_meta['hp_only']} HP-only versus "
        f"{tench_meta['lp_only']} LP-only votes for the tench example. "
        "Signed-linear panels are clipped to [0,1] only for display; model "
        "inference used signed tensors. These examples are illustrative; "
        "all class-level and aggregate claims use the full paired analysis."
    )
    (
        output_dir
        / "figure_class_contrast_caption.txt"
    ).write_text(
        class_caption,
        encoding="utf-8",
    )
    (
        output_dir
        / "figure_class_contrast_caption.tex"
    ).write_text(
        class_caption.replace("%", r"\%"),
        encoding="utf-8",
    )

    source_dir = output_dir / "source_images"
    source_dir.mkdir(exist_ok=True)
    selected = [
        ("condition_gallery", gallery_index, "manifest hash among clean 6/6 correct"),
        ("class_contrast_french_horn", french_index, "max HP-favoring six-checkpoint vote"),
        ("class_contrast_tench", tench_index, "max LP-favoring six-checkpoint vote"),
    ]
    provenance_rows = []
    for role, idx, rule in selected:
        mrow = manifest_indexed.loc[idx]
        source_path = dataset_paths[idx]
        copied = source_dir / f"{role}__gidx_{idx}__{source_path.name}"
        shutil.copy2(source_path, copied)
        target = int(mrow["target_imagenet1000"])
        hp_only, lp_only, tprob_diff = vote_summary(idx)
        provenance_rows.append({
            "role": role,
            "selection_rule": rule,
            "global_dataset_index": idx,
            "relative_path_frozen_manifest": mrow["relative_path"],
            "wnid": mrow["wnid"],
            "target_imagenet1000": target,
            "class_name": IMAGENETTE_CLASSES[target][1],
            "source_filename": source_path.name,
            "source_sha256": sha256_file(source_path),
            "clean_correct_count_of_6": clean_correct_count(idx),
            "hp_only_votes_at_hard_r10": hp_only,
            "lp_only_votes_at_hard_r10": lp_only,
            "mean_true_probability_hp_minus_lp": tprob_diff,
        })

    provenance = pd.DataFrame(provenance_rows)
    provenance.to_csv(output_dir / "figure_examples_manifest.csv", index=False)
    french_candidates.head(20).to_csv(
        output_dir / "french_horn_candidate_ranking.csv", index=False
    )
    tench_candidates.head(20).to_csv(
        output_dir / "tench_candidate_ranking.csv", index=False
    )

    config = {
        "project_folder": PROJECT_FOLDER,
        "frozen_stage1_run": FROZEN_STAGE1_RUN,
        "gallery_hash_salt": GALLERY_HASH_SALT,
        "gallery_global_index": gallery_index,
        "french_horn_global_index": french_index,
        "tench_global_index": tench_index,
        "models": MODELS,
        "geometry": {
            "resize": COMMON_RESIZE,
            "center_crop": COMMON_SIZE,
            "interpolation": "bilinear",
            "filtering_space": "unnormalized pixel tensor",
        },
        "visualization_note": (
            "Signed-linear RGB panels are clipped to [0,1] for display only. "
            "Inference used the original signed-linear tensors."
        ),
    }
    (output_dir / "figure_generation_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = f"""# Qualitative figure package

Frozen run: `{FROZEN_STAGE1_RUN}`

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
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    zip_path = output_dir / "spectral_qualitative_figures_v2_compact.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, arcname=path.relative_to(output_dir))

    print("Figure A:", fig_a_png)
    print("Figure B:", fig_b_png)
    print("Package:", zip_path)
    print(provenance.to_string(index=False))
    return zip_path


if __name__ == "__main__":
    main()
