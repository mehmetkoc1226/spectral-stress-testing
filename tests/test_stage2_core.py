from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "stage2"))

from spectral_stage2_core import self_test  # noqa: E402


def test_stage2_scientific_self_test() -> None:
    result = self_test()
    assert result["max_complementarity_error"] < 2e-5
    assert result["descriptor_shape"] == [5, 3]
    assert result["top1_invariance"] is True
