from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_fake_timm() -> None:
    module = types.ModuleType("timm")
    module.__version__ = "1.0.28"
    sys.modules["timm"] = module


def test_stage1_and_stage2_job_planning(monkeypatch) -> None:
    install_fake_timm()

    monkeypatch.setenv("STAGE1_RUN_PROFILE", "paper")
    monkeypatch.setenv("STAGE1_PHASE", "all_full")
    sys.path.insert(0, str(ROOT / "src" / "stage1"))
    stage1 = load_module("stage1_runner_smoke", ROOT / "src" / "stage1" / "run_stage1.py")
    design1 = stage1.build_design()
    assert len(design1.models) == 6
    assert len(stage1.phase_jobs("all_full", design1.models)) == 204
    patch_jobs = stage1.phase_jobs("imagenet_a_gauss_r20_linear_patch", design1.models)
    assert len(patch_jobs) == 12

    monkeypatch.setenv("STAGE2_RUN_PROFILE", "paper")
    sys.path.insert(0, str(ROOT / "src" / "stage2"))
    stage2 = load_module("stage2_runner_smoke", ROOT / "src" / "stage2" / "run_stage2.py")
    design2 = stage2.build_design()
    assert len(design2.models) == 3
    assert len(stage2.job_plan(design2)) == 78
