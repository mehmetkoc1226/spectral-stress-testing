# -*- coding: utf-8 -*-
"""Post-hoc analysis for the preregistered Stage-2 calibration pilot."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METHOD_A = "fcts"
METHOD_B = "clean_ts"
METRICS = ("nll", "ece15", "brier")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def evaluation_family(row: pd.Series) -> str:
    if row["dataset"] == "imagenet_a":
        return "unseen_dataset"
    condition = str(row["condition"])
    family = str(row["filter_family"])
    radius = row.get("radius")
    if condition == "clean_common":
        return "heldout_clean"
    if family == "hard" and pd.notna(radius) and int(float(radius)) in {10, 40}:
        return "unseen_radius"
    if family == "hard" and pd.notna(radius) and int(float(radius)) == 20:
        return "seen_spectral"
    if family == "gaussian":
        return "unseen_filter"
    if family in {"blur", "noise"}:
        return "natural_corruption"
    return "other"


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return out
    order_local = np.argsort(p[finite])
    ordered = finite[order_local]
    m = len(ordered)
    running = 0.0
    for rank, idx in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


def weighted_ece_replicates(
    weights: np.ndarray,
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int = 15,
) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(confidence)
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = np.zeros(weights.shape[0], dtype=np.float64)
    for i in range(bins):
        if i == 0:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if not np.any(mask):
            continue
        mask_float = mask.astype(np.float64)
        count = weights @ mask_float
        correct_sum = weights @ (mask_float * correct)
        conf_sum = weights @ (mask_float * confidence)
        valid = count > 0
        gap = np.zeros_like(count, dtype=np.float64)
        gap[valid] = np.abs(correct_sum[valid] / count[valid] - conf_sum[valid] / count[valid])
        values += (count / n) * gap
    return values


def bootstrap_file(path: Path, reps: int, seed: int) -> list[dict[str, Any]]:
    usecols = [
        "dataset", "model", "condition", "protocol",
        f"{METHOD_A}_confidence", f"{METHOD_A}_correct", f"{METHOD_A}_nll", f"{METHOD_A}_brier",
        f"{METHOD_B}_confidence", f"{METHOD_B}_correct", f"{METHOD_B}_nll", f"{METHOD_B}_brier",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    n = len(frame)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=reps).astype(np.float64)
    rows: list[dict[str, Any]] = []

    nll_diff = frame[f"{METHOD_A}_nll"].to_numpy(np.float64) - frame[f"{METHOD_B}_nll"].to_numpy(np.float64)
    brier_diff = frame[f"{METHOD_A}_brier"].to_numpy(np.float64) - frame[f"{METHOD_B}_brier"].to_numpy(np.float64)
    rep_values = {
        "nll": weights @ nll_diff / n,
        "brier": weights @ brier_diff / n,
    }
    correct_a = frame[f"{METHOD_A}_correct"].to_numpy(bool)
    correct_b = frame[f"{METHOD_B}_correct"].to_numpy(bool)
    if not np.array_equal(correct_a, correct_b):
        raise RuntimeError(f"Correctness invariance failed: {path}")
    ece_a = weighted_ece_replicates(
        weights, frame[f"{METHOD_A}_confidence"].to_numpy(np.float64), correct_a
    )
    ece_b = weighted_ece_replicates(
        weights, frame[f"{METHOD_B}_confidence"].to_numpy(np.float64), correct_b
    )
    rep_values["ece15"] = ece_a - ece_b

    first = frame.iloc[0]
    family = evaluation_family(pd.Series({
        "dataset": first["dataset"],
        "condition": first["condition"],
        "filter_family": (
            "gaussian" if str(first["condition"]).startswith("gauss_") else
            "hard" if str(first["condition"]).startswith("hard_") else
            "blur" if str(first["condition"]).startswith("blur_") else
            "noise" if str(first["condition"]).startswith("noise_") else "clean"
        ),
        "radius": (
            int(str(first["condition"]).split("_r")[1][:3])
            if "_r" in str(first["condition"]) else np.nan
        ),
    }))
    exact = {
        "nll": float(nll_diff.mean()),
        "brier": float(brier_diff.mean()),
        "ece15": float(
            weighted_ece_replicates(
                np.ones((1, n), dtype=np.float64),
                frame[f"{METHOD_A}_confidence"].to_numpy(np.float64), correct_a,
            )[0]
            - weighted_ece_replicates(
                np.ones((1, n), dtype=np.float64),
                frame[f"{METHOD_B}_confidence"].to_numpy(np.float64), correct_b,
            )[0]
        ),
    }
    for metric, values in rep_values.items():
        se = float(values.std(ddof=1))
        z = exact[metric] / se if se > 0 else float("inf") if exact[metric] != 0 else 0.0
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
        rows.append({
            "dataset": first["dataset"],
            "model": first["model"],
            "condition": first["condition"],
            "protocol": first["protocol"],
            "evaluation_family": family,
            "metric": metric,
            "delta_fcts_minus_clean_ts": exact[metric],
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "bootstrap_se": se,
            "p_value_normal": p_value,
            "n": n,
            "bootstrap_reps": reps,
        })
    return rows


def scan_top1_invariance(run_dir: Path) -> dict[str, Any]:
    checked = 0
    mismatches = 0
    details = []
    methods = ("clean_ts", "confidence_ts", "fcts", "oracle_ts")
    for path in sorted((run_dir / "compact_predictions").rglob("*.csv.gz")):
        usecols = ["raw_prediction"] + [f"{method}_prediction" for method in methods]
        frame = pd.read_csv(path, usecols=usecols)
        raw = frame["raw_prediction"].to_numpy()
        for method in methods:
            count = int(np.sum(raw != frame[f"{method}_prediction"].to_numpy()))
            checked += len(frame)
            mismatches += count
            if count:
                details.append({"file": str(path), "method": method, "mismatches": count})
    return {"comparisons_checked": checked, "mismatches": mismatches, "details": details[:20]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    run_dir = args.run_dir
    summary = pd.read_csv(run_dir / "summary.csv")
    summary["evaluation_family"] = summary.apply(evaluation_family, axis=1)
    keys = [
        "dataset", "model", "condition", "protocol", "filter_family", "mode",
        "radius", "severity", "evaluation_family", "n",
    ]
    left = summary[summary["method"] == METHOD_A][keys + [
        "accuracy", "average_confidence", "acms", "ece15", "nll", "brier",
        "error_auroc", "aupr_error", "aurc",
    ]].copy()
    right = summary[summary["method"] == METHOD_B][keys + [
        "accuracy", "average_confidence", "acms", "ece15", "nll", "brier",
        "error_auroc", "aupr_error", "aurc",
    ]].copy()
    comparisons = left.merge(right, on=keys, suffixes=("_fcts", "_clean_ts"), validate="one_to_one")
    for metric in ("accuracy", "average_confidence", "acms", "ece15", "nll", "brier", "error_auroc", "aupr_error", "aurc"):
        comparisons[f"delta_{metric}"] = comparisons[f"{metric}_fcts"] - comparisons[f"{metric}_clean_ts"]
    analysis_dir = run_dir / "analysis"
    atomic_write_csv(analysis_dir / "METHOD_COMPARISONS.csv", comparisons)

    bootstrap_rows = []
    for path in sorted((run_dir / "compact_predictions").rglob("*.csv.gz")):
        seed = int.from_bytes(path.name.encode("utf-8"), "little", signed=False) % (2**31 - 1)
        bootstrap_rows.extend(bootstrap_file(path, args.bootstrap_reps, seed))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap["p_holm"] = np.nan
    for (_, _), indices in bootstrap.groupby(["evaluation_family", "metric"]).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        bootstrap.loc[idx, "p_holm"] = holm_adjust(bootstrap.loc[idx, "p_value_normal"].to_numpy())
    bootstrap["significant_holm_005"] = bootstrap["p_holm"] < 0.05
    atomic_write_csv(analysis_dir / "BOOTSTRAP_FCTS_VS_CLEAN_TS.csv", bootstrap)

    primary = comparisons[
        (comparisons["protocol"] == "imagenet1000")
        & comparisons["evaluation_family"].isin(
            ["unseen_radius", "unseen_filter", "natural_corruption", "unseen_dataset"]
        )
    ].copy()
    family_model = primary.groupby(["model", "evaluation_family"], as_index=False)[
        ["delta_nll", "delta_ece15", "delta_brier"]
    ].mean()
    model_balanced = family_model.groupby("model", as_index=False)[
        ["delta_nll", "delta_ece15", "delta_brier"]
    ].mean()
    overall = model_balanced[["delta_nll", "delta_ece15", "delta_brier"]].mean()
    clean = comparisons[
        (comparisons["protocol"] == "imagenet1000")
        & (comparisons["dataset"] == "imagenette")
        & (comparisons["condition"] == "clean_common")
    ]
    top1 = scan_top1_invariance(run_dir)
    accuracy_invariant = bool(np.allclose(comparisons["delta_accuracy"], 0.0, atol=1e-12, rtol=0.0))
    model_improvements = int((model_balanced["delta_nll"] < 0).sum())
    clean_max_worsening = float(clean["delta_nll"].max())
    supporting = bool(overall["delta_ece15"] < 0 or overall["delta_brier"] < 0)
    criteria = {
        "unseen_family_model_balanced_delta_nll_lt_0": bool(overall["delta_nll"] < 0),
        "at_least_2_of_3_models_improve_unseen_nll": model_improvements >= 2,
        "ece_or_brier_supports": supporting,
        "each_model_heldout_clean_delta_nll_le_0_02": bool((clean["delta_nll"] <= 0.02).all()),
        "accuracy_invariant": accuracy_invariant,
        "top1_invariant": top1["mismatches"] == 0,
        "imagenet_a_not_used_for_selection": True,
    }
    success = all(criteria.values())
    decision = {
        "success": success,
        "criteria": criteria,
        "overall_family_model_balanced_deltas": overall.to_dict(),
        "models_with_unseen_nll_improvement": model_improvements,
        "heldout_clean_max_delta_nll": clean_max_worsening,
        "per_model_unseen_deltas": model_balanced.to_dict(orient="records"),
        "top1_invariance": top1,
        "primary_protocol": "imagenet1000",
        "restricted_protocol_role": "secondary sensitivity analysis",
    }
    atomic_write_json(analysis_dir / "SUCCESS_DECISION.json", decision)
    atomic_write_csv(analysis_dir / "PRIMARY_FAMILY_MODEL_DELTAS.csv", family_model)
    atomic_write_csv(analysis_dir / "PRIMARY_MODEL_BALANCED_DELTAS.csv", model_balanced)

    report = f"""# Stage-2 pilot analysis report

## Preregistered decision

**Pilot success status:** {'SUCCESS' if success else 'FAILURE'}

Primary protocol: native ImageNet-1000. Restricted protocols are secondary sensitivity analyses.

## Family- and model-balanced unseen differences (FCTS - clean scalar TS)

- ΔNLL: {overall['delta_nll']:.6f}
- ΔECE-15: {overall['delta_ece15']:.6f}
- ΔBrier: {overall['delta_brier']:.6f}

Negative values favor FCTS.

## Criteria

"""
    for name, value in criteria.items():
        report += f"- {name}: **{value}**\n"
    report += f"""

## Additional information

- Anchor models with improved unseen NLL: {model_improvements}/3
- Largest held-out clean delta NLL: {clean_max_worsening:.6f}
- Top-1 mismatch: {top1['mismatches']}
- Bootstrap replicates: {args.bootstrap_reps}

If the pilot fails, results must not be deleted and successful checkpoints must not be selectively reported.
"""
    path = analysis_dir / "STAGE2_ANALYSIS_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(report, encoding="utf-8")
    os.replace(tmp, path)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
