from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "artifacts" / "frozen_runs" / "stage1" / "spectral_stage1_confirmatory_v1__7044005bc7a4"
S2 = ROOT / "artifacts" / "frozen_runs" / "stage2" / "spectral_stage2_calibration_pilot_v1__27bb9eee96fb"


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def test_stage1_frozen_evidence() -> None:
    config = json.loads((S1 / "config.json").read_text(encoding="utf-8"))
    assert config["profile"] == "paper"
    assert config["sample_limits"] == {"imagenette": None, "imagenet_a": None}
    assert len(config["models"]) == 6
    assert csv_rows(S1 / "summary.csv") == 432
    assert csv_rows(S1 / "dataset_manifests" / "imagenette.csv") == 3925
    assert csv_rows(S1 / "dataset_manifests" / "imagenet_a.csv") == 7500
    for name in (
        "PHASE_COMPLETE__checkpoint_replication.json",
        "PHASE_COMPLETE__cross_dataset_core.json",
        "PHASE_COMPLETE__cross_dataset_replication.json",
        "PHASE_COMPLETE__ablation.json",
        "PHASE_COMPLETE__imagenet_a_gauss_r20_linear_patch.json",
        "PHASE_COMPLETE__natural_corruptions.json",
    ):
        assert (S1 / name).is_file()


def test_stage2_frozen_evidence() -> None:
    config = json.loads((S2 / "config.json").read_text(encoding="utf-8"))
    complete = json.loads((S2 / "RUN_COMPLETE__paper.json").read_text(encoding="utf-8"))
    assert config["profile"] == "paper"
    assert config["sample_limits"] == {"imagenette": None, "imagenet_a": None}
    assert len(config["models"]) == 3
    assert complete["all_jobs_done"] is True
    assert complete["completed_jobs"] == complete["total_jobs"] == 78
    assert csv_rows(S2 / "summary.csv") == 780
    assert csv_rows(S2 / "calibration_split_manifest.csv") == 3925
    assert csv_rows(S2 / "dataset_manifest_imagenette.csv") == 3925
    assert csv_rows(S2 / "dataset_manifest_imagenet_a.csv") == 7500
    for model in ("resnet50_a1", "vitb16_in1k", "swin_t_in1k"):
        assert (S2 / "fit_parameters" / f"{model}.json").is_file()
