"""Metrics for the focused next-run diagnostic suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import has_model_results, load_results, nullable_float, to_jsonable


def write_next_run_summary(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not has_model_results(df):
        summary = {"has_model_results": False, "message": "No logits found; generated next-run samples only."}
        (output_dir / "next_run_summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    summary = {
        "has_model_results": True,
        "exp4_v2": summarize_exp4_v2(df, output_dir),
        "unrelated_conflict": summarize_unrelated_conflict(df, output_dir),
        "exp2b": summarize_exp2b(df, output_dir),
        "duplicate_controls": summarize_duplicate_controls(df, output_dir),
    }
    (output_dir / "next_run_summary_metrics.json").write_text(
        json.dumps(to_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


def summarize_exp4_v2(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "next_exp4_v2_order_permutation"].copy()
    if exp.empty:
        return {}

    pattern_means = group_rates(exp, ["multiset", "pattern", "condition"])
    pattern_means.to_csv(output_dir / "next_exp4_v2_pattern_means.csv", index=False)

    balanced = exp[exp["pattern"].isin(["+-", "-+"])].copy()
    balanced_means = group_rates(balanced, ["pattern", "condition"])
    balanced_means.to_csv(output_dir / "next_exp4_v2_balanced_diagnostic.csv", index=False)

    perm = exp[exp["multiset"].isin(["balanced", "positive_imbalance", "negative_imbalance"])]
    order_variance = (
        perm.groupby(["base_event_id", "multiset"])
        .agg(order_std=("R", lambda s: float(np.std(s, ddof=0))), order_range=("R", lambda s: float(s.max() - s.min())), n=("R", "count"))
        .reset_index()
    )
    order_variance.to_csv(output_dir / "next_exp4_v2_order_variance.csv", index=False)

    regression = linear_regression(exp, "R", ["q", "last_sign", "mixed"])
    regression_with_neg = linear_regression(exp, "R", ["q", "last_sign", "mixed", "has_neg"])

    mixed = exp[exp["mixed"] == 1]
    pivot = balanced.pivot_table(index="base_event_id", columns="pattern", values="R", aggfunc="mean")
    order_effect = nullable_float((pivot["+-"] - pivot["-+"]).mean()) if {"+-", "-+"}.issubset(pivot.columns) else None

    return {
        "balanced_order_effect_mean_R_plus_minus_minus_plus": order_effect,
        "mean_order_std_by_multiset": order_variance.groupby("multiset")["order_std"].mean().apply(nullable_float).to_dict(),
        "mean_order_range_by_multiset": order_variance.groupby("multiset")["order_range"].mean().apply(nullable_float).to_dict(),
        "regression_R_q_last_mixed": regression,
        "regression_R_q_last_mixed_has_neg": regression_with_neg,
        "mixed_rows": summarize_rates(mixed),
        "n": int(len(exp)),
    }


def summarize_unrelated_conflict(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "next_unrelated_conflict"].copy()
    if exp.empty:
        return {}

    by_pattern = group_rates(exp, ["pattern", "condition"])
    by_pattern.to_csv(output_dir / "next_unrelated_conflict_by_pattern.csv", index=False)
    pivot = exp.pivot_table(index="base_event_id", columns="pattern", values="R", aggfunc="mean")
    order_effect = nullable_float((pivot["+-"] - pivot["-+"]).mean()) if {"+-", "-+"}.issubset(pivot.columns) else None
    return summarize_rates(exp) | {"order_effect_mean_R_plus_minus_minus_plus": order_effect}


def summarize_exp2b(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "next_exp2b_counterbalanced_overlap"].copy()
    if exp.empty:
        return {}

    condition_means = group_rates(exp, ["overlap_type", "phase_combo", "phase_relation"])
    condition_means.to_csv(output_dir / "next_exp2b_condition_means.csv", index=False)

    rows = []
    for overlap_type, group in exp.groupby("overlap_type"):
        reg = linear_regression(group, "R", ["phase_cos"])
        rows.append(
            {
                "overlap_type": overlap_type,
                "overlap_count": int(group["overlap_count"].max()),
                "beta0": reg.get("beta0"),
                "beta_phase_cos": reg.get("beta_phase_cos"),
                "r2": reg.get("r2"),
                "mean_R": group["R"].mean(),
                "mean_U_gap": group["U_gap"].mean(),
                "T_rate": (group["pred_label"] == "T").mean(),
                "F_rate": (group["pred_label"] == "F").mean(),
                "U_rate": (group["pred_label"] == "U").mean(),
                "n": len(group),
            }
        )
    slopes = pd.DataFrame(rows)
    slopes.to_csv(output_dir / "next_exp2b_phase_slopes.csv", index=False)

    slope_map = dict(zip(slopes["overlap_type"], slopes["beta_phase_cos"]))
    svo = slope_map.get("SVO", np.nan)
    partial = [slope_map.get(name, np.nan) for name in ["SV", "VO", "S-only", "none"]]
    max_partial = np.nanmax(partial) if partial else np.nan
    atomicity = 1 - max_partial / svo if np.isfinite(svo) and abs(svo) > 1e-12 else np.nan
    none = slopes[slopes["overlap_type"] == "none"]

    return {
        "beta_phase_cos_by_overlap": {str(k): nullable_float(v) for k, v in slope_map.items()},
        "atomicity_from_beta_phase_cos": nullable_float(atomicity),
        "none_overlap_beta_phase_cos": nullable_float(none["beta_phase_cos"].iloc[0] if not none.empty else np.nan),
        "none_overlap_F_rate": nullable_float(none["F_rate"].iloc[0] if not none.empty else np.nan),
        "n": int(len(exp)),
    }


def summarize_duplicate_controls(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    dup = df[df["experiment"] == "next_duplicate_controls"].copy()
    exp4 = df[df["experiment"] == "next_exp4_v2_order_permutation"].copy()
    if dup.empty or exp4.empty:
        return {}

    controls = pd.concat([dup, exp4[exp4["source_only"] == 1]], ignore_index=True)
    by_pattern = group_rates(controls, ["pattern", "condition"])
    by_pattern.to_csv(output_dir / "next_duplicate_controls_pattern_means.csv", index=False)

    pivot = controls.pivot_table(index="base_event_id", columns="pattern", values="R", aggfunc="mean")
    needed = {"+", "-", "++", "--"}
    if needed.issubset(pivot.columns):
        out = pivot.reset_index()[["base_event_id", "+", "++", "-", "--"]].copy()
        out["delta_pp"] = out["++"] - out["+"]
        out["delta_mm"] = out["--"] - out["-"]
        out.to_csv(output_dir / "next_duplicate_controls_deltas.csv", index=False)
        return {
            "mean_R_plus": nullable_float(out["+"].mean()),
            "mean_R_pp": nullable_float(out["++"].mean()),
            "mean_delta_pp": nullable_float(out["delta_pp"].mean()),
            "mean_R_minus": nullable_float(out["-"].mean()),
            "mean_R_mm": nullable_float(out["--"].mean()),
            "mean_delta_mm": nullable_float(out["delta_mm"].mean()),
            "n": int(len(out)),
        }
    return {"n": int(len(dup)), "message": "Missing source-only controls needed for duplicate deltas."}


def group_rates(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(keys, dropna=False)
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            mean_R=("R", "mean"),
            median_R=("R", "median"),
            mean_U_gap=("U_gap", "mean"),
            T_rate=("pred_label", lambda s: (s == "T").mean()),
            F_rate=("pred_label", lambda s: (s == "F").mean()),
            U_rate=("pred_label", lambda s: (s == "U").mean()),
            U_gap_positive_rate=("U_gap", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )


def summarize_rates(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return {
        "mean_R": nullable_float(df["R"].mean()),
        "median_R": nullable_float(df["R"].median()),
        "mean_U_gap": nullable_float(df["U_gap"].mean()),
        "T_rate": nullable_float((df["pred_label"] == "T").mean()),
        "F_rate": nullable_float((df["pred_label"] == "F").mean()),
        "U_rate": nullable_float((df["pred_label"] == "U").mean()),
        "U_gap_positive_rate": nullable_float((df["U_gap"] > 0).mean()),
        "accuracy": nullable_float(df["is_correct"].mean()),
        "n": int(len(df)),
    }


def linear_regression(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict[str, Any]:
    cols = [y_col] + x_cols
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < len(x_cols) + 1:
        return {}
    x = np.column_stack([np.ones(len(data)), *[data[col].to_numpy(dtype=float) for col in x_cols]])
    y = data[y_col].to_numpy(dtype=float)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_total = float(((y - y.mean()) ** 2).sum())
    ss_resid = float(((y - pred) ** 2).sum())
    out = {"beta0": nullable_float(beta[0]), "rank": int(rank), "n": int(len(data))}
    for col, value in zip(x_cols, beta[1:]):
        out[f"beta_{col}"] = nullable_float(value)
    out["r2"] = nullable_float(1 - ss_resid / ss_total if ss_total > 1e-12 else np.nan)
    return out


__all__ = ["load_results", "write_next_run_summary"]
