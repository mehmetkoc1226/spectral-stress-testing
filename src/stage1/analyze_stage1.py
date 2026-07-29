# -*- coding: utf-8 -*-
"""Post-hoc confirmatory analysis for Stage-1 outputs.

Run after any phase. Missing jobs are skipped and recorded; completed phases
produce paired-bootstrap tables, rank summaries, and an English status report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage1_core import benjamini_hochberg, paired_bootstrap_difference, stable_int_seed

PAIR_MAP = {
    "resnet50": ("resnet50_a1", "resnet50_a2"),
    "vit_b16": ("vitb16_in1k", "vitb16_in21k"),
    "swin_t": ("swin_t_in1k", "swin_t_in22k"),
}


def pred_path(run_dir: Path, dataset: str, model: str, condition: str, protocol: str) -> Path:
    return run_dir / "predictions" / dataset / model / f"{condition}__{protocol}.csv.gz"


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"global_dataset_index", "confidence", "correct"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Prediction file missing columns: {path}")
    return frame.sort_values("global_dataset_index").reset_index(drop=True)


def paired_from_files(
    path_a: Path,
    path_b: Path,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    a = read_prediction(path_a)
    b = read_prediction(path_b)
    merged = a.merge(b, on="global_dataset_index", suffixes=("_a", "_b"), validate="one_to_one")
    acc = paired_bootstrap_difference(
        merged["correct_a"].astype(float).to_numpy(),
        merged["correct_b"].astype(float).to_numpy(), reps, seed,
    )
    avg_conf = paired_bootstrap_difference(
        merged["confidence_a"].to_numpy(), merged["confidence_b"].to_numpy(), reps, seed + 1,
    )
    both_wrong = (~merged["correct_a"].astype(bool)) & (~merged["correct_b"].astype(bool))
    if both_wrong.sum() >= 20:
        common_error = paired_bootstrap_difference(
            merged.loc[both_wrong, "confidence_a"].to_numpy(),
            merged.loc[both_wrong, "confidence_b"].to_numpy(), reps, seed + 2,
        )
    else:
        common_error = {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    return {
        "n": len(merged), "n_common_errors": int(both_wrong.sum()),
        **{f"accuracy_{k}": v for k, v in acc.items()},
        **{f"avg_confidence_{k}": v for k, v in avg_conf.items()},
        **{f"common_error_confidence_{k}": v for k, v in common_error.items()},
    }


def checkpoint_pair_tests(run_dir: Path, summary: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    available = set(summary[["dataset", "model", "condition", "protocol"]].astype(str).agg("::".join, axis=1))
    for pair, (a, b) in PAIR_MAP.items():
        for dataset in sorted(summary["dataset"].unique()):
            for condition in sorted(summary["condition"].unique()):
                for protocol in sorted(summary["protocol"].unique()):
                    ka = f"{dataset}::{a}::{condition}::{protocol}"
                    kb = f"{dataset}::{b}::{condition}::{protocol}"
                    if ka not in available or kb not in available:
                        continue
                    pa = pred_path(run_dir, dataset, a, condition, protocol)
                    pb = pred_path(run_dir, dataset, b, condition, protocol)
                    if not pa.exists() or not pb.exists():
                        continue
                    result = paired_from_files(pa, pb, reps, seed + stable_int_seed(pair, dataset, condition, protocol))
                    rows.append({
                        "comparison": "checkpoint_pair", "pair": pair,
                        "model_a": a, "model_b": b, "dataset": dataset,
                        "condition": condition, "protocol": protocol, **result,
                    })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for col in ["accuracy_p_value", "avg_confidence_p_value", "common_error_confidence_p_value"]:
            frame[col.replace("p_value", "q_value_bh")] = benjamini_hochberg(frame[col].to_numpy())
    return frame


def hp_lp_tests(run_dir: Path, summary: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for dataset in sorted(summary["dataset"].unique()):
        for model in sorted(summary["model"].unique()):
            for family_prefix in ("hard", "gauss"):
                hp = f"{family_prefix}_hp_r010_linear"
                lp = f"{family_prefix}_lp_r010_linear"
                protocol = "imagenet1000"
                ph = pred_path(run_dir, dataset, model, hp, protocol)
                pl = pred_path(run_dir, dataset, model, lp, protocol)
                if not ph.exists() or not pl.exists():
                    continue
                result = paired_from_files(ph, pl, reps, seed + stable_int_seed(dataset, model, family_prefix))
                rows.append({
                    "comparison": "hp_minus_lp_r10", "dataset": dataset, "model": model,
                    "filter_family": family_prefix, "protocol": protocol,
                    "condition": f"{hp} minus {lp}", **result,
                })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["accuracy_q_value_bh"] = benjamini_hochberg(frame["accuracy_p_value"])
    return frame


def protocol_rank_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, condition), group in summary.groupby(["dataset", "condition"]):
        protocols = list(group["protocol"].unique())
        native = "imagenet1000"
        restricted = next((p for p in protocols if p.startswith("restricted")), None)
        if native not in protocols or restricted is None:
            continue
        a = group[group["protocol"] == native].set_index("model")
        b = group[group["protocol"] == restricted].set_index("model")
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 2:
            continue
        for metric in ("accuracy", "acms", "average_confidence", "ece"):
            rank_a = a.loc[common, metric].rank(ascending=False, method="average")
            rank_b = b.loc[common, metric].rank(ascending=False, method="average")
            corr = float(rank_a.corr(rank_b, method="spearman"))
            rows.append({
                "dataset": dataset, "condition": condition, "metric": metric,
                "native_protocol": native, "restricted_protocol": restricted,
                "spearman_rank_correlation": corr,
                "native_order": " > ".join(a.loc[common, metric].sort_values(ascending=False).index),
                "restricted_order": " > ".join(b.loc[common, metric].sort_values(ascending=False).index),
                "rank_reversal": bool(corr < 0.999999),
            })
    return pd.DataFrame(rows)


def matched_filter_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in summary["dataset"].unique():
        for model in summary["model"].unique():
            for protocol in summary["protocol"].unique():
                for radius in (10, 20):
                    for mode, short in (("low_pass", "lp"), ("high_pass", "hp")):
                        hard_key = f"hard_{short}_r{radius:03d}_linear"
                        gauss_key = f"gauss_{short}_r{radius:03d}_linear"
                        a = summary[(summary.dataset == dataset) & (summary.model == model) &
                                    (summary.protocol == protocol) & (summary.condition == hard_key)]
                        b = summary[(summary.dataset == dataset) & (summary.model == model) &
                                    (summary.protocol == protocol) & (summary.condition == gauss_key)]
                        if a.empty or b.empty:
                            continue
                        ar, br = a.iloc[-1], b.iloc[-1]
                        rows.append({
                            "dataset": dataset, "model": model, "protocol": protocol,
                            "radius": radius, "mode": mode,
                            "hard_accuracy": ar.accuracy, "gaussian_accuracy": br.accuracy,
                            "delta_accuracy_gaussian_minus_hard": br.accuracy - ar.accuracy,
                            "hard_acms": ar.acms, "gaussian_acms": br.acms,
                            "delta_acms_gaussian_minus_hard": br.acms - ar.acms,
                        })
    return pd.DataFrame(rows)


def clipping_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clipped = summary[summary["postprocess"] == "clip"]
    for _, clip in clipped.iterrows():
        prefix = "hard" if clip.filter_family == "hard" else "gauss"
        short = "lp" if clip.mode == "low_pass" else "hp"
        linear_key = f"{prefix}_{short}_r{int(clip.radius):03d}_linear"
        linear = summary[(summary.dataset == clip.dataset) & (summary.model == clip.model) &
                         (summary.protocol == clip.protocol) & (summary.condition == linear_key)]
        if linear.empty:
            continue
        lin = linear.iloc[-1]
        rows.append({
            "dataset": clip.dataset, "model": clip.model, "protocol": clip.protocol,
            "filter_family": clip.filter_family, "mode": clip.mode, "radius": clip.radius,
            "delta_accuracy_clip_minus_linear": clip.accuracy - lin.accuracy,
            "delta_acms_clip_minus_linear": clip.acms - lin.acms,
            "delta_ece_clip_minus_linear": clip.ece - lin.ece,
        })
    return pd.DataFrame(rows)


def write_report(run_dir: Path, summary: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    completed_jobs = summary[["dataset", "model", "condition"]].drop_duplicates().shape[0]
    lines = [
        "# Stage-1 analysis report",
        "",
        f"- Summary rows: {len(summary)}",
        f"- Completed unique dataset-model-condition jobs: {completed_jobs}",
        f"- Datasets: {', '.join(sorted(summary['dataset'].unique())) if not summary.empty else 'none'}",
        f"- Models: {', '.join(sorted(summary['model'].unique())) if not summary.empty else 'none'}",
        "",
        "## Prespecified decision questions",
        "",
        "1. Does the accuracy advantage of the low-energy high-pass component at r=10 replicate across checkpoint pairs?",
        "2. Is the low-pass advantage preserved in the second checkpoint of the same architecture?",
        "3. Do the native and restricted output spaces change the ACMS ranking?",
        "4. Do hard and complementary Gaussian masks yield the same qualitative conclusion?",
        "5. Is the clipping effect large and model-specific, especially under high-pass conditions?",
        "",
    ]
    hp = tables.get("hp_lp", pd.DataFrame())
    if not hp.empty:
        positive = hp[hp["accuracy_ci_low"] > 0]
        lines += [
            "## r=10 HP-LP accuracy difference",
            "",
            f"- Comparisons analyzed: {len(hp)}",
            f"- Comparisons with a 95% CI entirely above zero: {len(positive)}",
            "",
        ]
    ranks = tables.get("protocol_ranks", pd.DataFrame())
    if not ranks.empty:
        acms = ranks[ranks.metric == "acms"]
        lines += [
            "## Output-protocol sensitivity",
            "",
            f"- Conditions evaluated for ACMS: {len(acms)}",
            f"- Conditions with a rank reversal: {int(acms['rank_reversal'].sum())}",
            "",
        ]
    lines += [
        "This report is generated only from completed jobs. Do not make a final scientific decision while phases are missing.",
    ]
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis" / "STAGE1_ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    run_dir = args.run_dir
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "checkpoint_pairs": checkpoint_pair_tests(run_dir, summary, args.bootstrap_reps, args.seed),
        "hp_lp": hp_lp_tests(run_dir, summary, args.bootstrap_reps, args.seed),
        "protocol_ranks": protocol_rank_table(summary),
        "matched_filters": matched_filter_table(summary),
        "clipping": clipping_table(summary),
    }
    for name, frame in tables.items():
        frame.to_csv(analysis_dir / f"{name}.csv", index=False)
    write_report(run_dir, summary, tables)
    print("Stage-1 analysis outputs:", analysis_dir)
    for name, frame in tables.items():
        print(f"  {name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
