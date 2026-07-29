from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_model_registry() -> None:
    models = load("models.json")
    assert len(models) == 6
    assert {spec["role"] for spec in models.values()} == {"anchor", "replication"}
    assert len({spec["timm_name"] for spec in models.values()}) == 6


def test_condition_registry() -> None:
    conditions = load("conditions.json")
    keys = {row["key"] for row in conditions}
    required = {
        "clean_native", "clean_common",
        "hard_lp_r010_linear", "hard_hp_r010_linear",
        "gauss_lp_r020_linear", "gauss_hp_r020_linear",
        "hard_hp_r020_clip", "gauss_hp_r020_clip",
        "blur_sigma4", "noise_std020",
    }
    assert required <= keys
    assert len(keys) == len(conditions)
