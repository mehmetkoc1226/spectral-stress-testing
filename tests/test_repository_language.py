from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".cff", ".tex", ".csv", ".ipynb"}
TURKISH_CHARS = re.compile("[\u00e7\u011f\u0131\u0130\u00f6\u015f\u00fc\u00c7\u011e\u00d6\u015e\u00dc]")

def iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"CITATION.cff"}:
            yield path


def test_repository_contains_no_turkish_explanatory_text() -> None:
    failures = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8")
        if TURKISH_CHARS.search(text):
            failures.append(str(path.relative_to(ROOT)))
    assert not failures, failures


def test_notebooks_have_no_saved_execution_state() -> None:
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs", []) == []
