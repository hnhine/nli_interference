"""Plot helpers for scored interference results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import has_model_results, summarize_exp2, summarize_exp6a, summarize_exp6b


def plot_all(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not has_model_results(df):
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    paths: list[Path] = []
    paths.extend(plot_exp1(df, output_dir, plt, sns))
    paths.extend(plot_exp2(df, output_dir, plt, sns))
    paths.extend(plot_exp3(df, output_dir, plt, sns))
    paths.extend(plot_exp4(df, output_dir, plt, sns))
    paths.extend(plot_exp5(df, output_dir, plt, sns))
    paths.extend(plot_exp6a(df, output_dir, plt, sns))
    paths.extend(plot_exp6b(df, output_dir, plt, sns))
    paths.extend(plot_exp2_counterbalanced(df, output_dir, plt, sns))
    paths.extend(plot_exp4_order_permutation(df, output_dir, plt, sns))
    paths.extend(plot_exp4_unrelated_conflict(df, output_dir, plt, sns))
    paths.extend(plot_exp4_duplicate_controls(df, output_dir, plt, sns))
    plt.close("all")
    return paths


def save_current(path: Path, plt) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_exp1(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    exp = df[df["experiment"] == "exp1_phase_flip"]
    if exp.empty:
        return []
    order = ["A+ C+", "A- C+", "A+ C-", "A- C-"]
    plt.figure(figsize=(7, 4))
    sns.barplot(data=exp, x="condition", y="R", order=order, errorbar="se")
    sns.stripplot(data=exp, x="condition", y="R", order=order, color="black", alpha=0.35, size=2)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 1: R by phase condition")
    return [save_current(output_dir / "exp1_R_by_condition.png", plt)]


def plot_exp2(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    exp = df[df["experiment"] == "exp2_carrier_overlap"]
    if exp.empty:
        return []
    paths = []
    order = ["SVO", "SV", "VO", "S-only", "none"]
    plt.figure(figsize=(8, 4))
    sns.barplot(data=exp, x="overlap_type", y="R", hue="source_polarity", order=order, errorbar="se")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 2: R by overlap and source polarity")
    paths.append(save_current(output_dir / "exp2_R_by_overlap_polarity.png", plt))

    summarize_exp2(df, output_dir)
    proxy = pd.read_csv(output_dir / "exp2_carrier_proxy.csv")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=proxy, x="overlap_type", y="inner_product_proxy", order=order)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 2: empirical carrier inner-product proxy")
    paths.append(save_current(output_dir / "exp2_inner_product_proxy.png", plt))

    none_negative = exp[(exp["overlap_type"] == "none") & (exp["source_polarity"] == "negative")]
    if not none_negative.empty:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=none_negative, x="U_gap", bins=20)
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Exp 2: U_gap for no-overlap negative")
        paths.append(save_current(output_dir / "exp2_no_overlap_negative_U_gap.png", plt))
    return paths


def plot_exp3(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    exp = df[df["experiment"] == "exp3_clean_selection"]
    if exp.empty:
        return []
    paths = []
    plt.figure(figsize=(6, 4))
    sns.barplot(data=exp, x="match_polarity", y="R", errorbar="se")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 3: R by matched polarity")
    paths.append(save_current(output_dir / "exp3_R_by_matched_polarity.png", plt))

    heat = exp.pivot_table(index="match_polarity", columns="match_idx", values="R", aggfunc="mean")
    plt.figure(figsize=(6, 3.5))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Exp 3: mean R by match position")
    paths.append(save_current(output_dir / "exp3_position_heatmap.png", plt))

    pairs = build_exp3_pair_effects(exp)
    if not pairs.empty:
        effect_palette = {"Target flip": "#4C78A8", "Distractor flip": "#F58518"}
        plt.figure(figsize=(7.2, 4.6))
        ax = sns.barplot(
            data=pairs,
            x="match_idx",
            y="abs_delta_R",
            hue="pair_label",
            hue_order=["Target flip", "Distractor flip"],
            errorbar="se",
            capsize=0.08,
            palette=effect_palette,
        )
        ax.set_title("Exp 3: target vs distractor flip effect")
        ax.set_xlabel("Target position")
        ax.set_ylabel(r"Mean $|\Delta R|$")
        ax.legend(title="")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
        max_effect = pairs.groupby(["match_idx", "pair_label"])["abs_delta_R"].mean().max()
        ax.set_ylim(0, float(max_effect) * 1.2)
        paths.append(save_current(output_dir / "exp3_flip_effects_by_position.png", plt))

        success_order = ["Target flip", "Distractor flip"]
        success = (
            pairs.groupby("pair_label", as_index=False)["pair_success"]
            .mean()
            .set_index("pair_label")
            .reindex(success_order)
            .reset_index()
        )
        plt.figure(figsize=(6.2, 4.2))
        ax = plt.gca()
        bars = ax.bar(
            success["pair_label"],
            success["pair_success"],
            color=[effect_palette[label] for label in success["pair_label"]],
        )
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("")
        ax.set_ylabel("Rate")
        ax.set_title("Exp 3: directional accuracy / prediction invariance")
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in success["pair_success"]], padding=3)
        paths.append(save_current(output_dir / "exp3_flip_success_rates.png", plt))
    return paths


def build_exp3_pair_effects(exp: pd.DataFrame) -> pd.DataFrame:
    required = {
        "base_event_id",
        "match_idx",
        "match_polarity",
        "exp3_distractor_config",
        "R",
        "pred_label",
    }
    if not required.issubset(exp.columns):
        return pd.DataFrame()
    core = exp[exp["exp3_distractor_config"].notna()].copy()
    if core.empty:
        return pd.DataFrame()

    target_keys = ["base_event_id", "match_idx", "exp3_distractor_config"]
    positive = core[core["match_polarity"] == "positive"][
        target_keys + ["R", "pred_label"]
    ].rename(columns={"R": "R_positive", "pred_label": "pred_positive"})
    negative = core[core["match_polarity"] == "negative"][
        target_keys + ["R", "pred_label"]
    ].rename(columns={"R": "R_negative", "pred_label": "pred_negative"})
    target = positive.merge(negative, on=target_keys, how="inner", validate="one_to_one")
    target["abs_delta_R"] = (target["R_positive"] - target["R_negative"]).abs()
    target["pair_success"] = target["R_positive"] > target["R_negative"]
    target["pair_label"] = "Target flip"

    distractor_keys = ["base_event_id", "match_idx", "match_polarity"]
    anchor = core[core["exp3_distractor_config"] == "anchor"][
        distractor_keys + ["R", "pred_label"]
    ].rename(columns={"R": "R_anchor", "pred_label": "pred_anchor"})
    flips = core[core["exp3_distractor_config"] != "anchor"][
        distractor_keys + ["R", "pred_label"]
    ].rename(columns={"R": "R_flipped", "pred_label": "pred_flipped"})
    distractor = flips.merge(anchor, on=distractor_keys, how="inner", validate="many_to_one")
    distractor["abs_delta_R"] = (distractor["R_flipped"] - distractor["R_anchor"]).abs()
    distractor["pair_success"] = distractor["pred_flipped"] == distractor["pred_anchor"]
    distractor["pair_label"] = "Distractor flip"

    columns = ["base_event_id", "match_idx", "abs_delta_R", "pair_success", "pair_label"]
    return pd.concat([target[columns], distractor[columns]], ignore_index=True)


def plot_exp4(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    if "source_only" not in df.columns:
        return []
    exp = df[(df["experiment"] == "exp4_cancellation") & (df["source_only"] != 1)].copy()
    if exp.empty:
        return []
    order = ["+--", "+-", "++-"]
    exp["pattern"] = pd.Categorical(exp["pattern"], categories=order, ordered=True)
    exp = exp.sort_values("pattern")
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=exp, x="pattern", y="R", hue="base_event_id", legend=False, alpha=0.35)
    sns.pointplot(data=exp, x="pattern", y="R", order=order, color="black", errorbar="se")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 4: destructive cancellation ordering")
    return [save_current(output_dir / "exp4_cancellation_ordering.png", plt)]


def plot_exp5(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    exp = df[df["experiment"] == "exp5_object_bound_phase"]
    if exp.empty:
        return []
    order = ["positive_object", "negative_object"]
    hue_order = ["pos_then_neg", "neg_then_pos"]
    role_labels = {
        "positive_object": "Positive object\n(expected T)",
        "negative_object": "Negative object\n(expected F)",
    }
    order_labels = {
        "pos_then_neg": "Positive first",
        "neg_then_pos": "Negative first",
    }

    display = exp.copy()
    display["claim_object_role_label"] = display["claim_object_role"].map(role_labels)
    display["order_pattern_label"] = display["order_pattern"].map(order_labels)

    plt.figure(figsize=(10, 5.2))
    ax = sns.barplot(
        data=display,
        x="claim_object_role_label",
        y="R",
        hue="order_pattern_label",
        order=[role_labels[value] for value in order],
        hue_order=[order_labels[value] for value in hue_order],
        errorbar="se",
        capsize=0.12,
        palette=["#4C78A8", "#F58518"],
    )
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title("Exp 5: object-bound phase")
    ax.set_xlabel("Query object")
    ax.set_ylabel("R = logit(T) - logit(F)")
    ax.set_ylim(min(-3.6, float(exp["R"].min()) - 0.45), max(1.9, float(exp["R"].max()) + 0.45))
    ax.legend(title="Assumption order", loc="upper right", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    sns.despine()
    return [save_current(output_dir / "exp5_object_bound_phase.png", plt)]



def plot_exp6a(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    if "subexperiment" not in df.columns:
        return []
    exp = df[(df["experiment"] == "exp6_negation_phase") & (df["subexperiment"] == "exp6a_absolute_negation")].copy()
    if exp.empty:
        return []

    if "R" not in exp.columns and "R_claim" in exp.columns:
        exp["R"] = exp["R_claim"]
    if "R_axis" not in exp.columns:
        exp["claim_axis_sign"] = pd.to_numeric(exp.get("claim_axis_sign", 1), errors="coerce").fillna(1)
        exp["R_axis"] = exp["claim_axis_sign"] * exp["R"]
    if "assumption_form" not in exp.columns and "source_form" in exp.columns:
        exp["assumption_form"] = exp["source_form"]

    paths = []
    assumption_order = ["AFF", "DID_NOT", "DID_NOT_EVER", "NEVER"]
    claim_order = ["C_POS", "C_DID_NOT", "C_NEVER"]

    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=exp, x="assumption_form", y="R", hue="claim_form", order=assumption_order, hue_order=claim_order, errorbar="se")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 6A: R by assumption and claim form")
    plt.ylabel("R = L_T - L_F")
    paths.append(save_current(output_dir / "exp6a_R_by_form.png", plt))

    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=exp, x="assumption_form", y="R_axis", hue="claim_form", order=assumption_order, hue_order=claim_order, errorbar="se")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 6A: R_axis by assumption and claim form")
    plt.ylabel("R_axis = s_c R")
    paths.append(save_current(output_dir / "exp6a_R_axis_by_form.png", plt))

    summarize_exp6a(df, output_dir)

    deltas_path = output_dir / "exp6a_deltas.csv"
    if deltas_path.exists():
        deltas = pd.read_csv(deltas_path)
        delta_cols = ["delta_R_never_notever_C_POS", "delta_R_never_notever_C_DID_NOT"]
        delta_means = pd.DataFrame(
            {"delta": delta_cols, "mean_R_delta": [deltas[col].mean() for col in delta_cols if col in deltas.columns]}
        )
        if not delta_means.empty:
            plt.figure(figsize=(6.5, 4))
            sns.barplot(data=delta_means, x="delta", y="mean_R_delta")
            plt.axhline(0, color="black", linewidth=1)
            plt.xticks(rotation=20, ha="right")
            plt.title("Exp 6A: NEVER minus DID_NOT_EVER")
            plt.ylabel("Mean R delta")
            paths.append(save_current(output_dir / "exp6a_never_notever_delta.png", plt))

    coefficients_path = output_dir / "exp6a_coefficients.csv"
    if coefficients_path.exists():
        coefficients = pd.read_csv(coefficients_path)
        if not coefficients.empty:
            coefficients["assumption_form"] = pd.Categorical(coefficients["assumption_form"], categories=assumption_order, ordered=True)
            coefficients = coefficients.sort_values("assumption_form")
            plt.figure(figsize=(7.5, 4))
            sns.barplot(data=coefficients, x="assumption_form", y="mean_kappa_hat", order=assumption_order)
            plt.axhline(0, color="black", linewidth=1)
            plt.axhline(-1, color="gray", linewidth=1, linestyle="--")
            plt.axhline(1, color="gray", linewidth=1, linestyle="--")
            plt.title("Exp 6A: normalized polarity coefficient κ_hat")
            plt.ylabel("Mean κ_hat")
            paths.append(save_current(output_dir / "exp6a_kappa_hat_coefficients.png", plt))

            kappa_long = coefficients.melt(
                id_vars=["assumption_form"],
                value_vars=["mean_kappa_pos", "mean_kappa_neg"],
                var_name="estimate",
                value_name="mean_kappa",
            )
            kappa_long["estimate"] = kappa_long["estimate"].map(
                {
                    "mean_kappa_pos": "κ_pos from C_POS",
                    "mean_kappa_neg": "κ_neg from C_DID_NOT",
                }
            )
            plt.figure(figsize=(8, 4.2))
            sns.barplot(data=kappa_long, x="assumption_form", y="mean_kappa", hue="estimate", order=assumption_order)
            plt.axhline(0, color="black", linewidth=1)
            plt.axhline(-1, color="gray", linewidth=1, linestyle="--")
            plt.axhline(1, color="gray", linewidth=1, linestyle="--")
            plt.title("Exp 6A: κ_pos and κ_neg by assumption form")
            plt.ylabel("Mean κ")
            plt.legend(title="")
            paths.append(save_current(output_dir / "exp6a_kappa_pos_neg.png", plt))

            plt.figure(figsize=(7.5, 4))
            sns.barplot(data=coefficients, x="assumption_form", y="mean_E_reuse", order=assumption_order)
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Exp 6A: coefficient reuse error E_reuse")
            plt.ylabel("Mean E_reuse = |κ_pos - κ_neg|")
            paths.append(save_current(output_dir / "exp6a_E_reuse.png", plt))

    detail_path = output_dir / "exp6a_coefficients_by_base.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        if not detail.empty:
            detail["assumption_form"] = pd.Categorical(detail["assumption_form"], categories=assumption_order, ordered=True)
            plt.figure(figsize=(6.2, 5.4))
            sns.scatterplot(data=detail, x="kappa_pos", y="kappa_neg", hue="assumption_form", hue_order=assumption_order, alpha=0.75, s=35)
            finite = detail[["kappa_pos", "kappa_neg"]].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
            finite = finite[pd.notna(finite)]
            if len(finite):
                lo = min(-1.2, float(finite.min()) - 0.1)
                hi = max(1.2, float(finite.max()) + 0.1)
                plt.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
                plt.xlim(lo, hi)
                plt.ylim(lo, hi)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.axvline(0, color="black", linewidth=0.8)
            plt.title("Exp 6A: coefficient reuse across claim polarity")
            plt.xlabel("κ_pos from C_POS")
            plt.ylabel("κ_neg from C_DID_NOT")
            paths.append(save_current(output_dir / "exp6a_kappa_reuse_scatter.png", plt))

    anchor_path = output_dir / "exp6a_anchor_control.csv"
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path)
        value_cols = [col for col in anchor.columns if col != "base_event_id"]
        if value_cols:
            anchor_means = pd.DataFrame({"anchor_gap": value_cols, "mean_gap": [anchor[col].mean() for col in value_cols]})
            plt.figure(figsize=(7, 4))
            sns.barplot(data=anchor_means, x="anchor_gap", y="mean_gap")
            plt.axhline(0, color="black", linewidth=1)
            plt.xticks(rotation=20, ha="right")
            plt.title("Exp 6A: anchor contamination G_anchor")
            plt.ylabel("Mean G_anchor")
            paths.append(save_current(output_dir / "exp6a_anchor_contamination.png", plt))

    return paths


def plot_exp6b(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    if "subexperiment" not in df.columns:
        return []
    exp = df[(df["experiment"] == "exp6_negation_phase") & (df["subexperiment"] == "exp6b_frequency_negation")].copy()
    if exp.empty:
        return []

    if "R" not in exp.columns and "R_claim" in exp.columns:
        exp["R"] = exp["R_claim"]
    if "R_axis" not in exp.columns:
        exp["claim_axis_sign"] = pd.to_numeric(exp.get("claim_axis_sign", 1), errors="coerce").fillna(1)
        exp["R_axis"] = exp["claim_axis_sign"] * exp["R"]
    if "assumption_form" not in exp.columns and "source_form" in exp.columns:
        exp["assumption_form"] = exp["source_form"]

    paths = []
    assumption_order = ["OFTEN", "DID_NOT_OFTEN", "RARELY", "SELDOM", "HARDLY_EVER"]
    claim_order = ["C_OFTEN", "C_DID_NOT_OFTEN"]

    plt.figure(figsize=(10, 4.8))
    sns.barplot(data=exp, x="assumption_form", y="R_axis", hue="claim_form", order=assumption_order, hue_order=claim_order, errorbar="se")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=15, ha="right")
    plt.title("Exp 6B: R_axis by assumption and claim form")
    plt.ylabel("R_axis = s_c R")
    paths.append(save_current(output_dir / "exp6b_R_axis_by_form.png", plt))

    plt.figure(figsize=(8.5, 4.4))
    sns.barplot(data=exp, x="assumption_form", y="U_gap", order=assumption_order, errorbar="se")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=15, ha="right")
    plt.title("Exp 6B: U_gap by assumption form")
    plt.ylabel("U_gap = logit(U) - max(logit(T), logit(F))")
    paths.append(save_current(output_dir / "exp6b_U_gap_by_assumption.png", plt))

    summarize_exp6b(df, output_dir)

    coefficients_path = output_dir / "exp6b_coefficients.csv"
    if coefficients_path.exists():
        coefficients = pd.read_csv(coefficients_path)
        if not coefficients.empty:
            coefficients["assumption_form"] = pd.Categorical(coefficients["assumption_form"], categories=assumption_order, ordered=True)
            coefficients = coefficients.sort_values("assumption_form")
            plt.figure(figsize=(8.5, 4.4))
            sns.barplot(data=coefficients, x="assumption_form", y="mean_kappa_hat", order=assumption_order)
            plt.axhline(0, color="black", linewidth=1)
            plt.axhline(-1, color="gray", linewidth=1, linestyle="--")
            plt.axhline(1, color="gray", linewidth=1, linestyle="--")
            plt.xticks(rotation=15, ha="right")
            plt.title("Exp 6B: frequency polarity coefficient spectrum κ_hat")
            plt.ylabel("Mean κ_hat")
            paths.append(save_current(output_dir / "exp6b_kappa_hat_spectrum.png", plt))

            kappa_long = coefficients.melt(
                id_vars=["assumption_form"],
                value_vars=["mean_kappa_pos", "mean_kappa_neg"],
                var_name="estimate",
                value_name="mean_kappa",
            )
            kappa_long["estimate"] = kappa_long["estimate"].map(
                {
                    "mean_kappa_pos": "κ_pos from C_OFTEN",
                    "mean_kappa_neg": "κ_neg from C_DID_NOT_OFTEN",
                }
            )
            plt.figure(figsize=(9, 4.4))
            sns.barplot(data=kappa_long, x="assumption_form", y="mean_kappa", hue="estimate", order=assumption_order)
            plt.axhline(0, color="black", linewidth=1)
            plt.axhline(-1, color="gray", linewidth=1, linestyle="--")
            plt.axhline(1, color="gray", linewidth=1, linestyle="--")
            plt.xticks(rotation=15, ha="right")
            plt.title("Exp 6B: κ_pos and κ_neg by assumption form")
            plt.ylabel("Mean κ")
            plt.legend(title="")
            paths.append(save_current(output_dir / "exp6b_kappa_pos_neg.png", plt))

    detail_path = output_dir / "exp6b_coefficients_by_base.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        if not detail.empty:
            detail["assumption_form"] = pd.Categorical(detail["assumption_form"], categories=assumption_order, ordered=True)
            plt.figure(figsize=(6.2, 5.4))
            sns.scatterplot(data=detail, x="kappa_pos", y="kappa_neg", hue="assumption_form", hue_order=assumption_order, alpha=0.75, s=35)
            finite = detail[["kappa_pos", "kappa_neg"]].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
            finite = finite[pd.notna(finite)]
            if len(finite):
                lo = min(-1.2, float(finite.min()) - 0.1)
                hi = max(1.2, float(finite.max()) + 0.1)
                plt.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
                plt.xlim(lo, hi)
                plt.ylim(lo, hi)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.axvline(0, color="black", linewidth=0.8)
            plt.title("Exp 6B: coefficient reuse across claim polarity")
            plt.xlabel("κ_pos from C_OFTEN")
            plt.ylabel("κ_neg from C_DID_NOT_OFTEN")
            paths.append(save_current(output_dir / "exp6b_kappa_reuse_scatter.png", plt))

    return paths


def plot_exp2_counterbalanced(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    exp = df[df["experiment"] == "exp2_counterbalanced_overlap"].copy()
    if exp.empty:
        return []
    paths = []
    order = ["SVO", "SV", "VO", "S-only", "none"]
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=exp, x="overlap_type", y="R", hue="phase_relation", order=order, errorbar="se")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 2 supplement: counterbalanced claim polarity")
    plt.ylabel("R = logit(T) - logit(F)")
    paths.append(save_current(output_dir / "exp2_counterbalanced_R_by_overlap_phase.png", plt))

    slope_rows = []
    for overlap_type, group in exp.groupby("overlap_type"):
        same = group[group["phase_cos"] == 1]["R"].mean()
        opposite = group[group["phase_cos"] == -1]["R"].mean()
        slope_rows.append({"overlap_type": overlap_type, "beta_phase_cos": (same - opposite) / 2})
    slopes = pd.DataFrame(slope_rows)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=slopes, x="overlap_type", y="beta_phase_cos", order=order)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 2 supplement: phase slope by overlap")
    plt.ylabel("Beta phase cos")
    paths.append(save_current(output_dir / "exp2_counterbalanced_phase_slopes.png", plt))
    return paths


def plot_exp4_order_permutation(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    exp = df[df["experiment"] == "exp4_order_permutation"].copy()
    if exp.empty:
        return []
    order = ["+--", "-+-", "--+", "+-", "-+", "++-", "+-+", "-++", "+", "-"]
    exp["pattern"] = pd.Categorical(exp["pattern"], categories=order, ordered=True)
    exp = exp.sort_values("pattern")
    plt.figure(figsize=(10, 4.8))
    sns.pointplot(data=exp, x="pattern", y="R", hue="multiset", order=order, errorbar="se", dodge=0.25)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 4 supplement: source-order permutations")
    plt.ylabel("R = logit(T) - logit(F)")
    return [save_current(output_dir / "exp4_order_permutation_R_by_pattern.png", plt)]


def plot_exp4_unrelated_conflict(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    exp = df[df["experiment"] == "exp4_unrelated_conflict"].copy()
    if exp.empty:
        return []
    order = ["+-", "-+"]
    plt.figure(figsize=(5.5, 4))
    sns.barplot(data=exp, x="pattern", y="R", order=order, errorbar="se")
    sns.stripplot(data=exp, x="pattern", y="R", order=order, color="black", alpha=0.35, size=2)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 4 supplement: unrelated conflict")
    plt.ylabel("R = logit(T) - logit(F)")
    return [save_current(output_dir / "exp4_unrelated_conflict_R_by_pattern.png", plt)]


def plot_exp4_duplicate_controls(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
    dup = df[df["experiment"] == "exp4_duplicate_controls"].copy()
    order_perm = df[df["experiment"] == "exp4_order_permutation"].copy()
    order_source_only = order_perm[order_perm["source_only"] == 1] if "source_only" in order_perm.columns else order_perm.iloc[0:0]
    controls = pd.concat([dup, order_source_only], ignore_index=True)
    if controls.empty:
        return []
    order = ["+", "++", "-", "--"]
    controls["pattern"] = pd.Categorical(controls["pattern"], categories=order, ordered=True)
    controls = controls.sort_values("pattern")
    plt.figure(figsize=(6.5, 4))
    sns.barplot(data=controls, x="pattern", y="R", order=order, errorbar="se")
    sns.stripplot(data=controls, x="pattern", y="R", order=order, color="black", alpha=0.35, size=2)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Exp 4 supplement: duplicate controls")
    plt.ylabel("R = logit(T) - logit(F)")
    return [save_current(output_dir / "exp4_duplicate_controls_R_by_pattern.png", plt)]
