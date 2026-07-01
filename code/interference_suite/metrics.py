"""Aggregate metrics for the interference experiments and supplements."""

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
    numeric_columns = [
        "logit_T",
        "logit_F",
        "logit_U",
        "R",
        "R_claim",
        "R_axis",
        "U_gap",
        "expected_R_sign",
        "claim_axis_sign",
        "is_correct",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    ensure_axis_columns(df)
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
    summary["exp6_negation_phase"] = summarize_exp6(df, output_dir)
    if has_supplemental_rows(df):
        summary["supplements"] = summarize_supplements(df, output_dir)
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


def summarize_exp6(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp6_negation_phase"].copy()
    if exp.empty:
        return {}
    ensure_axis_columns(exp)
    return {
        "exp6a_absolute_negation": summarize_exp6a(exp, output_dir),
        "exp6b_frequency_negation": summarize_exp6b(exp, output_dir),
    }


def summarize_exp6a(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = exp6_subframe(df, "exp6a_absolute_negation")
    if exp.empty:
        return {}

    by_form = exp6_by_form(exp, output_dir / "exp6a_by_form.csv")
    wide_R = exp6_wide(exp, "R")
    wide_axis = exp6_wide(exp, "R_axis")
    deltas = exp6a_deltas(wide_R)
    if not deltas.empty:
        deltas.to_csv(output_dir / "exp6a_deltas.csv", index=False)

    coeff_detail, coeff_summary = exp6_coefficients(
        wide=wide_axis,
        assumption_forms=["AFF", "DID_NOT", "DID_NOT_EVER", "NEVER"],
        positive_claim="C_POS",
        negative_claim="C_DID_NOT",
        positive_anchor="AFF",
        negative_anchor="DID_NOT",
    )
    if not coeff_detail.empty:
        coeff_detail.to_csv(output_dir / "exp6a_coefficients_by_base.csv", index=False)
    if not coeff_summary.empty:
        coeff_summary.to_csv(output_dir / "exp6a_coefficients.csv", index=False)

    anchor = exp6a_anchor_control(wide_R)
    if not anchor.empty:
        anchor.to_csv(output_dir / "exp6a_anchor_control.csv", index=False)

    hard_rows = exp[exp.get("label_confidence", "hard") != "diagnostic"]
    return {
        "n": int(len(exp)),
        "main_accuracy_excluding_diagnostic": nullable_float(hard_rows["is_correct"].mean() if not hard_rows.empty else np.nan),
        "by_form_rows": int(len(by_form)),
        "sign_consistency": {
            "never_sign_acc": paired_sign_acc(wide_value(wide_R, "NEVER", "C_POS"), wide_value(wide_R, "NEVER", "C_DID_NOT")),
            "did_not_ever_sign_acc": paired_sign_acc(wide_value(wide_R, "DID_NOT_EVER", "C_POS"), wide_value(wide_R, "DID_NOT_EVER", "C_DID_NOT")),
        },
        "delta_means": dataframe_means(deltas, exclude={"base_event_id"}),
        "coefficient_means": coefficient_summary_dict(coeff_summary),
        "anchor_contamination_means": dataframe_means(anchor, exclude={"base_event_id"}),
        "interpretation_rules": exp6a_interpretation_rules(),
    }


def summarize_exp6b(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = exp6_subframe(df, "exp6b_frequency_negation")
    if exp.empty:
        return {}

    by_form = exp6_by_form(exp, output_dir / "exp6b_by_form.csv")
    wide = exp6_wide(exp, "R_axis")
    coeff_detail, coeff_summary = exp6_coefficients(
        wide=wide,
        assumption_forms=["OFTEN", "DID_NOT_OFTEN", "RARELY", "SELDOM", "HARDLY_EVER"],
        positive_claim="C_OFTEN",
        negative_claim="C_DID_NOT_OFTEN",
        positive_anchor="OFTEN",
        negative_anchor="DID_NOT_OFTEN",
    )
    if not coeff_detail.empty:
        coeff_detail.to_csv(output_dir / "exp6b_coefficients_by_base.csv", index=False)
    if not coeff_summary.empty:
        coeff_summary.to_csv(output_dir / "exp6b_coefficients.csv", index=False)

    u_by_assumption = (
        exp.groupby("assumption_form", dropna=False)
        .agg(
            mean_U_gap=("U_gap", "mean"),
            U_rate=("pred_label", lambda s: (s == "U").mean()),
            n=("sample_id", "count"),
        )
        .reset_index()
    )
    u_by_assumption.to_csv(output_dir / "exp6b_u_by_assumption.csv", index=False)

    directional_rows = exp[exp.get("label_confidence", "hard") != "diagnostic"]
    return {
        "n": int(len(exp)),
        "accuracy_including_directional_labels": nullable_float(directional_rows["is_correct"].mean() if not directional_rows.empty else np.nan),
        "by_form_rows": int(len(by_form)),
        "coefficient_means": coefficient_summary_dict(coeff_summary),
        "mean_U_gap_by_assumption": {
            str(row["assumption_form"]): nullable_float(row["mean_U_gap"])
            for _, row in u_by_assumption.iterrows()
        },
        "outcome_rules": {
            "stable_graded_coefficient": "Approximate forms have kappa_hat < 0 with low E_reuse.",
            "u_dominant": "High U_gap or U_rate means the model treats the claim as underdetermined.",
            "reuse_failure": "High E_reuse means no reusable frequency-polarity coefficient.",
        },
    }


def ensure_axis_columns(df: pd.DataFrame) -> None:
    if "R_claim" not in df.columns and "R" in df.columns:
        df["R_claim"] = df["R"]
    if "R" not in df.columns and "R_claim" in df.columns:
        df["R"] = df["R_claim"]
    if "claim_axis_sign" not in df.columns:
        df["claim_axis_sign"] = 1
    df["claim_axis_sign"] = pd.to_numeric(df["claim_axis_sign"], errors="coerce").fillna(1)
    if "R_claim" in df.columns:
        df["R_claim"] = pd.to_numeric(df["R_claim"], errors="coerce")
    if "R" in df.columns:
        df["R"] = pd.to_numeric(df["R"], errors="coerce")
    if "R_axis" not in df.columns:
        df["R_axis"] = df["claim_axis_sign"] * df.get("R", np.nan)
    else:
        df["R_axis"] = pd.to_numeric(df["R_axis"], errors="coerce")
        missing = df["R_axis"].isna() & df.get("R", pd.Series(np.nan, index=df.index)).notna()
        df.loc[missing, "R_axis"] = df.loc[missing, "claim_axis_sign"] * df.loc[missing, "R"]


def exp6_subframe(df: pd.DataFrame, subexperiment: str) -> pd.DataFrame:
    if "subexperiment" not in df.columns:
        return pd.DataFrame()
    exp = df[(df["experiment"] == "exp6_negation_phase") & (df["subexperiment"] == subexperiment)].copy()
    if exp.empty:
        return exp
    ensure_axis_columns(exp)
    ensure_assumption_form(exp)
    return exp


def ensure_assumption_form(df: pd.DataFrame) -> None:
    if "assumption_form" not in df.columns and "source_form" in df.columns:
        df["assumption_form"] = df["source_form"]


def exp6_by_form(exp: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    for (assumption_form, claim_form), group in exp.groupby(["assumption_form", "claim_form"], dropna=False):
        non_diagnostic = group[group.get("label_confidence", "hard") != "diagnostic"]
        rows.append(
            {
                "assumption_form": assumption_form,
                "claim_form": claim_form,
                "label_confidence": ",".join(sorted(str(v) for v in group.get("label_confidence", pd.Series(dtype=str)).dropna().unique())),
                "mean_R": group["R"].mean(),
                "median_R": group["R"].median(),
                "mean_R_axis": group["R_axis"].mean(),
                "mean_U_gap": group["U_gap"].mean(),
                "T_rate": (group["pred_label"] == "T").mean(),
                "F_rate": (group["pred_label"] == "F").mean(),
                "U_rate": (group["pred_label"] == "U").mean(),
                "accuracy": group["is_correct"].mean(),
                "main_accuracy": non_diagnostic["is_correct"].mean() if not non_diagnostic.empty else np.nan,
                "n": len(group),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def exp6_wide(exp: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if exp.empty or value_col not in exp.columns:
        return pd.DataFrame()
    return exp.pivot_table(index="base_event_id", columns=["assumption_form", "claim_form"], values=value_col, aggfunc="mean")


def wide_value(wide: pd.DataFrame, assumption_form: str, claim_form: str) -> pd.Series:
    if wide.empty:
        return pd.Series(dtype=float)
    key = (assumption_form, claim_form)
    if key in wide.columns:
        return wide[key]
    return pd.Series(np.nan, index=wide.index, dtype=float)


def exp6a_deltas(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "base_event_id": wide.index,
            "delta_R_never_notever_C_POS": wide_value(wide, "NEVER", "C_POS") - wide_value(wide, "DID_NOT_EVER", "C_POS"),
            "delta_R_never_notever_C_DID_NOT": wide_value(wide, "NEVER", "C_DID_NOT") - wide_value(wide, "DID_NOT_EVER", "C_DID_NOT"),
            "delta_R_notever_not_C_POS": wide_value(wide, "DID_NOT_EVER", "C_POS") - wide_value(wide, "DID_NOT", "C_POS"),
            "delta_R_notever_not_C_DID_NOT": wide_value(wide, "DID_NOT_EVER", "C_DID_NOT") - wide_value(wide, "DID_NOT", "C_DID_NOT"),
        }
    ).reset_index(drop=True)


def exp6a_anchor_control(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "base_event_id": wide.index,
            "G_anchor_DIDNOT": wide_value(wide, "DID_NOT", "C_DID_NOT") - wide_value(wide, "DID_NOT", "C_NEVER"),
            "G_anchor_NEVER": wide_value(wide, "NEVER", "C_NEVER") - wide_value(wide, "NEVER", "C_DID_NOT"),
            "G_anchor_DID_NOT_EVER": wide_value(wide, "DID_NOT_EVER", "C_NEVER") - wide_value(wide, "DID_NOT_EVER", "C_DID_NOT"),
        }
    ).reset_index(drop=True)


def exp6_coefficients(
    wide: pd.DataFrame,
    assumption_forms: list[str],
    positive_claim: str,
    negative_claim: str,
    positive_anchor: str,
    negative_anchor: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()

    pos_anchor_pos_claim = wide_value(wide, positive_anchor, positive_claim)
    neg_anchor_pos_claim = wide_value(wide, negative_anchor, positive_claim)
    pos_anchor_neg_claim = wide_value(wide, positive_anchor, negative_claim)
    neg_anchor_neg_claim = wide_value(wide, negative_anchor, negative_claim)

    beta_pos = (pos_anchor_pos_claim - neg_anchor_pos_claim) / 2
    gamma_pos = (pos_anchor_pos_claim + neg_anchor_pos_claim) / 2
    beta_neg = (pos_anchor_neg_claim - neg_anchor_neg_claim) / 2
    gamma_neg = (pos_anchor_neg_claim + neg_anchor_neg_claim) / 2

    rows = []
    for assumption_form in assumption_forms:
        kappa_pos = safe_divide(wide_value(wide, assumption_form, positive_claim) - gamma_pos, beta_pos)
        kappa_neg = safe_divide(wide_value(wide, assumption_form, negative_claim) - gamma_neg, beta_neg)
        detail = pd.DataFrame(
            {
                "base_event_id": wide.index,
                "assumption_form": assumption_form,
                "beta_pos": beta_pos.to_numpy(),
                "gamma_pos": gamma_pos.to_numpy(),
                "beta_neg": beta_neg.to_numpy(),
                "gamma_neg": gamma_neg.to_numpy(),
                "kappa_pos": kappa_pos.to_numpy(),
                "kappa_neg": kappa_neg.to_numpy(),
            }
        )
        detail["kappa_hat"] = (detail["kappa_pos"] + detail["kappa_neg"]) / 2
        detail["E_reuse"] = (detail["kappa_pos"] - detail["kappa_neg"]).abs()
        detail["exceeds_negative_anchor"] = (detail["kappa_hat"] < -1).astype(int)
        rows.append(detail)

    detail_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if detail_df.empty:
        return detail_df, pd.DataFrame()
    summary = (
        detail_df.groupby("assumption_form", dropna=False)
        .agg(
            mean_beta_pos=("beta_pos", "mean"),
            mean_gamma_pos=("gamma_pos", "mean"),
            mean_beta_neg=("beta_neg", "mean"),
            mean_gamma_neg=("gamma_neg", "mean"),
            mean_kappa_pos=("kappa_pos", "mean"),
            mean_kappa_neg=("kappa_neg", "mean"),
            mean_kappa_hat=("kappa_hat", "mean"),
            median_kappa_hat=("kappa_hat", "median"),
            mean_E_reuse=("E_reuse", "mean"),
            exceeds_negative_anchor_rate=("exceeds_negative_anchor", "mean"),
            n=("base_event_id", "count"),
        )
        .reset_index()
    )
    return detail_df, summary


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.copy()
    denom = denom.where(denom.abs() > 1e-12)
    return numerator / denom


def paired_sign_acc(positive_claim_scores: pd.Series, negative_claim_scores: pd.Series) -> float | None:
    if positive_claim_scores.empty or negative_claim_scores.empty:
        return None
    data = pd.DataFrame({"positive": positive_claim_scores, "negative": negative_claim_scores}).dropna()
    if data.empty:
        return None
    return nullable_float(((data["positive"] < 0) & (data["negative"] > 0)).mean())


def dataframe_means(df: pd.DataFrame, exclude: set[str] | None = None) -> dict[str, float | None]:
    if df.empty:
        return {}
    exclude = exclude or set()
    out = {}
    for column in df.columns:
        if column in exclude:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            out[column] = nullable_float(numeric.mean())
    return out


def coefficient_summary_dict(summary: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if summary.empty:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    for _, row in summary.iterrows():
        assumption_form = str(row["assumption_form"])
        out[assumption_form] = {
            "mean_kappa_hat": nullable_float(row.get("mean_kappa_hat")),
            "mean_E_reuse": nullable_float(row.get("mean_E_reuse")),
            "mean_beta_pos": nullable_float(row.get("mean_beta_pos")),
            "mean_gamma_pos": nullable_float(row.get("mean_gamma_pos")),
            "mean_beta_neg": nullable_float(row.get("mean_beta_neg")),
            "mean_gamma_neg": nullable_float(row.get("mean_gamma_neg")),
            "exceeds_negative_anchor_rate": nullable_float(row.get("exceeds_negative_anchor_rate")),
        }
    return out


def exp6a_interpretation_rules() -> dict[str, str]:
    return {
        "strong_support": "DID_NOT, DID_NOT_EVER, and NEVER have similar kappa_hat, low E_reuse, and small G_anchor.",
        "temporal_scope_contribution": "DID_NOT_EVER differs from DID_NOT while NEVER tracks DID_NOT_EVER with low E_reuse.",
        "surface_lexical_leakage": "NEVER differs from DID_NOT_EVER, suggesting lexical form affects readout.",
        "no_stable_coefficient": "High E_reuse means the form does not reuse across claim polarity.",
        "anchor_contamination": "Large C_DID_NOT versus C_NEVER gaps indicate claim wording or lexical overlap effects.",
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



def has_supplemental_rows(df: pd.DataFrame) -> bool:
    if "experiment" not in df.columns:
        return False
    supplemental = {
        "exp2_counterbalanced_overlap",
        "exp4_order_permutation",
        "exp4_unrelated_conflict",
        "exp4_duplicate_controls",
    }
    return df["experiment"].isin(supplemental).any()


def summarize_supplements(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "exp2_counterbalanced_overlap": summarize_exp2_counterbalanced(df, output_dir),
        "exp4_order_permutation": summarize_exp4_order_permutation(df, output_dir),
        "exp4_unrelated_conflict": summarize_exp4_unrelated_conflict(df, output_dir),
        "exp4_duplicate_controls": summarize_exp4_duplicate_controls(df, output_dir),
    }


def summarize_exp4_order_permutation(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp4_order_permutation"].copy()
    if exp.empty:
        return {}

    pattern_means = group_rates(exp, ["multiset", "pattern", "condition"])
    pattern_means.to_csv(output_dir / "exp4_order_permutation_pattern_means.csv", index=False)

    balanced = exp[exp["pattern"].isin(["+-", "-+"])].copy()
    balanced_means = group_rates(balanced, ["pattern", "condition"])
    balanced_means.to_csv(output_dir / "exp4_order_permutation_balanced_cases.csv", index=False)

    perm = exp[exp["multiset"].isin(["balanced", "positive_imbalance", "negative_imbalance"])]
    order_variance = (
        perm.groupby(["base_event_id", "multiset"])
        .agg(order_std=("R", lambda s: float(np.std(s, ddof=0))), order_range=("R", lambda s: float(s.max() - s.min())), n=("R", "count"))
        .reset_index()
    )
    order_variance.to_csv(output_dir / "exp4_order_permutation_order_variance.csv", index=False)

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


def summarize_exp4_unrelated_conflict(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp4_unrelated_conflict"].copy()
    if exp.empty:
        return {}

    by_pattern = group_rates(exp, ["pattern", "condition"])
    by_pattern.to_csv(output_dir / "exp4_unrelated_conflict_by_pattern.csv", index=False)
    pivot = exp.pivot_table(index="base_event_id", columns="pattern", values="R", aggfunc="mean")
    order_effect = nullable_float((pivot["+-"] - pivot["-+"]).mean()) if {"+-", "-+"}.issubset(pivot.columns) else None
    return summarize_rates(exp) | {"order_effect_mean_R_plus_minus_minus_plus": order_effect}


def summarize_exp2_counterbalanced(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    exp = df[df["experiment"] == "exp2_counterbalanced_overlap"].copy()
    if exp.empty:
        return {}

    condition_means = group_rates(exp, ["overlap_type", "phase_combo", "phase_relation"])
    condition_means.to_csv(output_dir / "exp2_counterbalanced_condition_means.csv", index=False)

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
    slopes.to_csv(output_dir / "exp2_counterbalanced_phase_slopes.csv", index=False)

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


def summarize_exp4_duplicate_controls(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    dup = df[df["experiment"] == "exp4_duplicate_controls"].copy()
    exp4 = df[df["experiment"] == "exp4_order_permutation"].copy()
    if dup.empty or exp4.empty:
        return {}

    controls = pd.concat([dup, exp4[exp4["source_only"] == 1]], ignore_index=True)
    by_pattern = group_rates(controls, ["pattern", "condition"])
    by_pattern.to_csv(output_dir / "exp4_duplicate_controls_pattern_means.csv", index=False)

    pivot = controls.pivot_table(index="base_event_id", columns="pattern", values="R", aggfunc="mean")
    needed = {"+", "-", "++", "--"}
    if needed.issubset(pivot.columns):
        out = pivot.reset_index()[["base_event_id", "+", "++", "-", "--"]].copy()
        out["delta_pp"] = out["++"] - out["+"]
        out["delta_mm"] = out["--"] - out["-"]
        out.to_csv(output_dir / "exp4_duplicate_controls_deltas.csv", index=False)
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
