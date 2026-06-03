"""Aggregate metrics for the interference experiment suite."""

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
