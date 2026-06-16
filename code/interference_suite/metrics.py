"""Aggregate metrics for the interference experiments and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def has_model_results(df: pd.DataFrame) -> bool:
    return {"R", "U_gap", "pred_label"}.issubset(df.columns) and df["R"].notna().any()


def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ["logit_T", "logit_F", "logit_U", "R", "U_gap", "expected_R_sign"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def write_summary_outputs(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not has_model_results(df):
        summary = {"has_model_results": False, "message": "No logits found; generated samples only."}
        (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    summary: dict[str, Any] = {"has_model_results": True}
    summary["exp1_phase_flip"] = summarize_exp1(df, output_dir)
    summary["exp2_carrier_overlap"] = summarize_exp2(df, output_dir)
    summary["exp3_clean_selection"] = summarize_exp3(df, output_dir)
    summary["exp4_cancellation"] = summarize_exp4(df, output_dir)
    summary["exp5_object_bound_phase"] = summarize_exp5(df, output_dir)
    if has_next_diagnostic_rows(df):
        summary["next_diagnostics"] = summarize_next_diagnostics(df, output_dir)
    (output_dir / "summary_metrics.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    return summary


def summarize_exp1(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp1_phase_flip"].copy()
    if exp.empty:
        return {}
    exp["R_sign"] = np.sign(exp["R"])
    exp["sign_correct"] = exp["R_sign"] == exp["expected_R_sign"]
    means = exp.groupby(["condition", "phase_relation"], dropna=False).agg(
        mean_R=("R", "mean"),
        median_R=("R", "median"),
        mean_U_gap=("U_gap", "mean"),
        accuracy=("is_correct", "mean"),
        n=("sample_id", "count"),
    )
    means.to_csv(output_dir / "exp1_condition_means.csv")

    same = exp[exp["phase_relation"] == "same"]["R"].mean()
    opposite = exp[exp["phase_relation"] == "opposite"]["R"].mean()
    corr = safe_corr(exp["R"], exp["phase_cos"])
    return {
        "phase_sign_acc": float(exp["sign_correct"].mean()),
        "mean_R_same_polarity": nullable_float(same),
        "mean_R_opposite_polarity": nullable_float(opposite),
        "phase_effect": nullable_float(same - opposite),
        "corr_R_phase_cos": nullable_float(corr),
        "n": int(len(exp)),
    }


def summarize_exp2(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp2_carrier_overlap"].copy()
    if exp.empty:
        return {}

    aggregates = []
    for overlap_type, group in exp.groupby("overlap_type", dropna=False):
        positive = group[group["source_polarity"] == "positive"]
        negative = group[group["source_polarity"] == "negative"]
        mean_r_positive = positive["R"].mean()
        mean_r_negative = negative["R"].mean()
        aggregates.append(
            {
                "overlap_type": overlap_type,
                "overlap_count": group["overlap_count"].max(),
                "mean_R_positive": mean_r_positive,
                "mean_R_negative": mean_r_negative,
                "inner_product_proxy": (mean_r_positive - mean_r_negative) / 2,
                "mean_U_gap_positive": positive["U_gap"].mean(),
                "mean_U_gap_negative": negative["U_gap"].mean(),
                "U_rate_positive": (positive["pred_label"] == "U").mean(),
                "U_rate_negative": (negative["pred_label"] == "U").mean(),
                "F_rate_negative": (negative["pred_label"] == "F").mean(),
                "n_positive": len(positive),
                "n_negative": len(negative),
            }
        )
    proxy_df = pd.DataFrame(aggregates)
    proxy_df.to_csv(output_dir / "exp2_carrier_proxy.csv", index=False)

    proxy = dict(zip(proxy_df["overlap_type"], proxy_df["inner_product_proxy"]))
    svo = proxy.get("SVO", np.nan)
    partial_values = [proxy.get(name, np.nan) for name in ("SV", "VO", "S-only", "none")]
    max_partial = np.nanmax(partial_values) if partial_values else np.nan
    atomicity = 1 - (max_partial / svo) if np.isfinite(svo) and abs(svo) > 1e-12 else np.nan

    no_overlap_negative = exp[(exp["overlap_type"] == "none") & (exp["source_polarity"] == "negative")]
    regression = role_regression(proxy_df)
    return {
        "atomicity_index": nullable_float(atomicity),
        "role_regression": regression,
        "no_overlap_negation_leakage": {
            "mean_R_none_negative": nullable_float(no_overlap_negative["R"].mean()),
            "F_rate_none_negative": nullable_float((no_overlap_negative["pred_label"] == "F").mean()),
            "mean_U_gap_none_negative": nullable_float(no_overlap_negative["U_gap"].mean()),
            "U_gap_positive_rate_none_negative": nullable_float((no_overlap_negative["U_gap"] > 0).mean()),
            "n": int(len(no_overlap_negative)),
        },
    }


def summarize_exp3(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp3_clean_selection"].copy()
    if exp.empty:
        return {}
    exp["R_sign"] = np.sign(exp["R"])
    exp["sign_correct"] = exp["R_sign"] == exp["expected_R_sign"]
    means = exp.groupby(["match_idx", "match_polarity", "sanity_type"], dropna=False).agg(
        mean_R=("R", "mean"),
        mean_U_gap=("U_gap", "mean"),
        sign_acc=("sign_correct", "mean"),
        label_acc=("is_correct", "mean"),
        n=("sample_id", "count"),
    )
    means.to_csv(output_dir / "exp3_position_means.csv")

    position_bias: dict[str, float | None] = {}
    for polarity, group in exp.groupby("match_polarity"):
        overall = group["R"].mean()
        by_idx = group.groupby("match_idx")["R"].mean()
        position_bias[str(polarity)] = nullable_float((by_idx - overall).abs().max())

    return {
        "match_following_acc": float(exp["sign_correct"].mean()),
        "position_bias": position_bias,
        "label_acc": nullable_float(exp["is_correct"].mean()),
        "n": int(len(exp)),
    }


def summarize_exp4(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp4_cancellation"].copy()
    if exp.empty:
        return {}

    core = exp[exp["source_only"] != 1].copy()
    pivot = core.pivot_table(index="base_event_id", columns="pattern", values="R", aggfunc="mean")
    required = ["+--", "+-", "++-"]
    if set(required).issubset(pivot.columns):
        pivot["ordering_correct"] = (pivot["+--"] < pivot["+-"]) & (pivot["+-"] < pivot["++-"])
        ordering_acc = pivot["ordering_correct"].mean()
    else:
        ordering_acc = np.nan
    pivot.to_csv(output_dir / "exp4_ordering_by_base.csv")

    regression = simple_regression(core["q"], core["R"])
    balanced = core[core["pattern"] == "+-"]
    cancellation = cancellation_ratios(exp)
    if not cancellation.empty:
        cancellation.to_csv(output_dir / "exp4_cancellation_ratios.csv", index=False)

    return {
        "ordering_accuracy": nullable_float(ordering_acc),
        "net_phase_regression": regression,
        "mean_cancellation_ratio": nullable_float(cancellation["cancel_ratio"].mean() if not cancellation.empty else np.nan),
        "balanced_case": {
            "mean_R": nullable_float(balanced["R"].mean()),
            "median_R": nullable_float(balanced["R"].median()),
            "mean_U_gap": nullable_float(balanced["U_gap"].mean()),
            "U_gap_positive_rate": nullable_float((balanced["U_gap"] > 0).mean()),
            "U_gap_negative_rate": nullable_float((balanced["U_gap"] < 0).mean()),
            "n": int(len(balanced)),
        },
    }


def summarize_exp5(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp5_object_bound_phase"].copy()
    if exp.empty:
        return {}

    pair = exp.pivot_table(index=["base_event_id", "order_pattern"], columns="claim_object_role", values="R", aggfunc="mean")
    if {"positive_object", "negative_object"}.issubset(pair.columns):
        pair["object_binding_correct"] = (pair["positive_object"] > 0) & (pair["negative_object"] < 0)
        pair["object_phase_gap"] = pair["positive_object"] - pair["negative_object"]
        object_binding_acc = pair["object_binding_correct"].mean()
        object_phase_gap = pair["object_phase_gap"].mean()
    else:
        object_binding_acc = np.nan
        object_phase_gap = np.nan
    pair.to_csv(output_dir / "exp5_object_binding_pairs.csv")

    exp1_simple = df[(df["experiment"] == "exp1_phase_flip") & (df["condition"] == "A+ C+")][["base_event_id", "R"]]
    compound_positive = exp[exp["claim_object_role"] == "positive_object"][["base_event_id", "order_pattern", "R"]]
    leak = compound_positive.merge(exp1_simple, on="base_event_id", suffixes=("_compound", "_simple"))
    if not leak.empty:
        leak["delta_leak"] = leak["R_compound"] - leak["R_simple"]
        leak.to_csv(output_dir / "exp5_global_leakage.csv", index=False)

    order_means = exp.groupby(["order_pattern", "claim_object_role"]).agg(
        mean_R=("R", "mean"),
        mean_U_gap=("U_gap", "mean"),
        label_acc=("is_correct", "mean"),
        n=("sample_id", "count"),
    )
    order_means.to_csv(output_dir / "exp5_order_means.csv")

    return {
        "object_binding_accuracy": nullable_float(object_binding_acc),
        "mean_object_phase_gap": nullable_float(object_phase_gap),
        "mean_delta_leak": nullable_float(leak["delta_leak"].mean() if not leak.empty else np.nan),
        "positive_object_flip_rate": nullable_float((compound_positive["R"] < 0).mean()),
        "label_acc": nullable_float(exp["is_correct"].mean()),
        "n": int(len(exp)),
    }


def role_regression(proxy_df: pd.DataFrame) -> dict[str, Any]:
    design = {
        "SVO": (1, 1, 1),
        "SV": (1, 1, 0),
        "VO": (0, 1, 1),
        "S-only": (1, 0, 0),
        "none": (0, 0, 0),
    }
    rows = []
    y_values = []
    labels = []
    for _, row in proxy_df.iterrows():
        overlap_type = row["overlap_type"]
        if overlap_type not in design or pd.isna(row["inner_product_proxy"]):
            continue
        rows.append(design[overlap_type])
        y_values.append(float(row["inner_product_proxy"]))
        labels.append(overlap_type)
    if len(rows) < 3:
        return {}
    x = np.asarray(rows, dtype=float)
    y = np.asarray(y_values, dtype=float)
    weights, residuals, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ weights
    ss_total = float(((y - y.mean()) ** 2).sum())
    ss_resid = float(((y - prediction) ** 2).sum())
    r2 = 1 - ss_resid / ss_total if ss_total > 1e-12 else np.nan
    return {
        "w_subject": nullable_float(weights[0]),
        "w_verb": nullable_float(weights[1]),
        "w_object": nullable_float(weights[2]),
        "r2": nullable_float(r2),
        "rank": int(rank),
        "rows": labels,
    }


def cancellation_ratios(exp: pd.DataFrame) -> pd.DataFrame:
    values = exp.pivot_table(index="base_event_id", columns="pattern", values="R", aggfunc="mean")
    required = {"+-", "+", "-"}
    if not required.issubset(values.columns):
        return pd.DataFrame()
    out = values.reset_index()[["base_event_id", "+-", "+", "-"]].copy()
    out["cancel_ratio"] = out["+-"].abs() / (1 + 0.5 * (out["+"].abs() + out["-"].abs()))
    return out


def simple_regression(x_series: pd.Series, y_series: pd.Series) -> dict[str, float | None]:
    data = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
    if len(data) < 2:
        return {}
    x = np.column_stack([np.ones(len(data)), data["x"].to_numpy(dtype=float)])
    y = data["y"].to_numpy(dtype=float)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_total = float(((y - y.mean()) ** 2).sum())
    ss_resid = float(((y - pred) ** 2).sum())
    return {
        "beta0": nullable_float(beta[0]),
        "beta1": nullable_float(beta[1]),
        "r2": nullable_float(1 - ss_resid / ss_total if ss_total > 1e-12 else np.nan),
    }


def safe_corr(x_series: pd.Series, y_series: pd.Series) -> float:
    data = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
    if len(data) < 2 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan
    return float(data["x"].corr(data["y"]))


def nullable_float(value: Any) -> float | None:
    try:
        if value is None or not np.isfinite(value):
            return None
        return float(value)
    except TypeError:
        return None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return nullable_float(value)
    return value


def has_next_diagnostic_rows(df: pd.DataFrame) -> bool:
    return "experiment" in df.columns and df["experiment"].astype(str).str.startswith("next_").any()


def summarize_next_diagnostics(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "exp4_v2": summarize_exp4_v2(df, output_dir),
        "unrelated_conflict": summarize_unrelated_conflict(df, output_dir),
        "exp2b": summarize_exp2b(df, output_dir),
        "duplicate_controls": summarize_duplicate_controls(df, output_dir),
    }


def write_next_run_summary(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not has_model_results(df):
        summary = {"has_model_results": False, "message": "No logits found; generated next-run samples only."}
        (output_dir / "next_run_summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    summary = {"has_model_results": True} | summarize_next_diagnostics(df, output_dir)
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
