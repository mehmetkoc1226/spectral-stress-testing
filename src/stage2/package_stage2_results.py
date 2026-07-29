# -*- coding: utf-8 -*-
"""Create shareable Stage-2 result bundles without duplicating full logits."""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path


def atomic_zip(output: Path, roots: list[tuple[Path, Path]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, arc_root in roots:
            if source.is_file():
                archive.write(source, arcname=arc_root / source.name)
            elif source.is_dir():
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        archive.write(path, arcname=arc_root / path.relative_to(source))
    os.replace(tmp, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    profile = config["profile"]
    package_dir = run_dir / "share_packages"

    report_items = []
    for name in (
        "config.json", "PROTOCOL_LOCK.json", "environment.json",
        "scientific_self_tests.json", "calibration_split_manifest.csv",
        "dataset_manifest_imagenette.csv", "dataset_manifest_imagenet_a.csv",
        "job_plan.csv", "summary.csv", "run.log",
        f"RUN_COMPLETE__{profile}.json",
    ):
        path = run_dir / name
        if path.exists():
            report_items.append((path, Path(run_dir.name)))
    for dirname in ("fit_parameters", "analysis", "job_markers"):
        path = run_dir / dirname
        if path.exists():
            report_items.append((path, Path(run_dir.name) / dirname))

    report_zip = package_dir / f"stage2_{profile}_report_bundle.zip"
    atomic_zip(report_zip, report_items)

    compact_items = [
        (run_dir / "compact_predictions", Path(run_dir.name) / "compact_predictions"),
        (run_dir / "summary.csv", Path(run_dir.name)),
        (run_dir / "config.json", Path(run_dir.name)),
        (run_dir / "calibration_split_manifest.csv", Path(run_dir.name)),
    ]
    compact_items = [(p, a) for p, a in compact_items if p.exists()]
    compact_zip = package_dir / f"stage2_{profile}_compact_predictions.zip"
    atomic_zip(compact_zip, compact_items)

    print(report_zip)
    print(compact_zip)


if __name__ == "__main__":
    main()
