"""Plot helpers for scored interference results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import has_model_results, summarize_exp2


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
    return paths


def plot_exp4(df: pd.DataFrame, output_dir: Path, plt, sns) -> list[Path]:
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
    controls = pd.concat([dup, order_perm[order_perm["source_only"] == 1]], ignore_index=True)
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
