"""Create publication-ready figures for the Exp2b structural-overlap gate.

The confidence intervals in the main panel are obtained by bootstrapping base
events, so the four polarity combinations of an event are kept together.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("/workspace/nhi/nli_interference")
OUTPUT_DIR = ROOT / "data" / "comparison" / "plots"
MODELS = {
    "Qwen3-8B": (
        ROOT / "data" / "qwen3_8_1000" / "samples.csv",
        "next_exp2b_counterbalanced_overlap",
    ),
    "Phi-4-mini": (
        ROOT / "data" / "phi4_mini_1000" / "samples.csv",
        "exp2_counterbalanced_overlap",
    ),
}

OVERLAP_ORDER = ["SVO", "VO", "SV", "S-only", "none"]
MISMATCH_ORDER = ["VO", "SV", "S-only", "none"]
PHASE_ORDER = ["A+ C+", "A+ C-", "A- C+", "A- C-"]
DISPLAY_LABELS = {
    "SVO": "Full match\n(SVO)",
    "VO": "Subject\nmismatch (VO)",
    "SV": "Object\nmismatch (SV)",
    "S-only": "Verb + object\nmismatch (S-only)",
    "none": "No content\nmatch",
}
COLORS = {"Qwen3-8B": "#31688E", "Phi-4-mini": "#E07A3F"}


def load_exp2b() -> pd.DataFrame:
    """Load the counterbalanced Exp2b rows for both models."""
    frames = []
    columns = [
        "base_event_id",
        "experiment",
        "overlap_type",
        "phase_combo",
        "U_gap",
        "pred_label",
    ]
    for model, (path, experiment) in MODELS.items():
        frame = pd.read_csv(path, usecols=columns, low_memory=False)
        frame = frame.loc[frame["experiment"] == experiment].copy()
        if frame.empty:
            raise RuntimeError(f"No rows found for {model}: {experiment}")
        frame["model"] = model
        frame["is_U"] = frame["pred_label"].eq("U").astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cluster_bootstrap_summary(
    data: pd.DataFrame, n_boot: int = 5000, seed: int = 20260718
) -> pd.DataFrame:
    """Summarize mean U-gap with a base-event bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    rows = []
    for (model, overlap), group in data.groupby(["model", "overlap_type"], sort=False):
        event_means = group.groupby("base_event_id", sort=False)["U_gap"].mean().to_numpy()
        n_events = len(event_means)
        boot_means = np.empty(n_boot)
        for start in range(0, n_boot, 250):
            stop = min(start + 250, n_boot)
            sampled = rng.integers(0, n_events, size=(stop - start, n_events))
            boot_means[start:stop] = event_means[sampled].mean(axis=1)
        low, high = np.quantile(boot_means, [0.025, 0.975])
        rows.append(
            {
                "model": model,
                "overlap_type": overlap,
                "mean_G_U": event_means.mean(),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def plot_gate_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    show_panel_title: bool = True,
    compact_paper: bool = False,
) -> None:
    x = np.arange(len(OVERLAP_ORDER), dtype=float)
    offsets = {"Qwen3-8B": -0.12, "Phi-4-mini": 0.12}
    value_fontsize = 11.0 if compact_paper else 8.2
    annotation_fontsize = 11.0 if compact_paper else 9.5
    axis_fontsize = 11.5 if compact_paper else None
    tick_fontsize = 10.5 if compact_paper else None
    legend_fontsize = 10.5 if compact_paper else None

    ax.axhspan(0, 4.2, color="#E8F3F7", alpha=0.72, zorder=0)
    ax.axhspan(-4.8, 0, color="#F8ECE7", alpha=0.72, zorder=0)
    ax.axhline(0, color="#333333", linewidth=1.15, zorder=1)
    ax.axvline(0.5, color="#777777", linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)

    for model in MODELS:
        model_rows = summary.loc[summary["model"] == model].set_index("overlap_type")
        model_rows = model_rows.loc[OVERLAP_ORDER]
        means = model_rows["mean_G_U"].to_numpy()
        low = model_rows["ci_low"].to_numpy()
        high = model_rows["ci_high"].to_numpy()
        positions = x + offsets[model]
        ax.errorbar(
            positions,
            means,
            yerr=np.vstack([means - low, high - means]),
            fmt="o",
            markersize=9.0 if compact_paper else 7.5,
            capsize=3.5,
            elinewidth=1.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=COLORS[model],
            label=model,
            zorder=3,
        )
        for position, value in zip(positions, means):
            vertical_offset = (
                0.42 if compact_paper else (0.22 if value >= 0 else -0.32)
            )
            ax.text(
                position,
                value + vertical_offset,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=value_fontsize,
                color=COLORS[model],
                fontweight="semibold",
            )

    ax.text(4.48, 3.78, "U preferred", ha="right", va="top", fontsize=annotation_fontsize, color="#275D70")
    ax.text(4.48, -4.38, "T/F preferred", ha="right", va="bottom", fontsize=annotation_fontsize, color="#8A4B32")
    ax.text(0.25, 4.08, "structural gate", ha="center", va="top", fontsize=10.5 if compact_paper else 9, color="#555555")
    ax.set_xticks(x, [DISPLAY_LABELS[item] for item in OVERLAP_ORDER])
    ylabel = (
        "Unknown-label margin"
        if compact_paper
        else r"Unknown-label margin  $G_U = L_U - \max(L_T,L_F)$"
    )
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    ax.set_ylim(-4.8, 4.2)
    ax.set_xlim(-0.48, 4.52)
    ax.legend(
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.50, 0.015),
        fontsize=legend_fontsize,
    )
    if compact_paper:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(axis="y", color="white", linewidth=1.1)
    ax.grid(axis="x", visible=False)
    if show_panel_title:
        ax.set_title(
            "A   Structural agreement gates polarity-based inference",
            loc="left",
            fontweight="bold",
        )


def polarity_heatmap(data: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = data.loc[
        (data["model"] == model) & data["overlap_type"].isin(MISMATCH_ORDER)
    ]
    heat = subset.pivot_table(
        index="overlap_type", columns="phase_combo", values="is_U", aggfunc="mean"
    )
    return heat.reindex(index=MISMATCH_ORDER, columns=PHASE_ORDER) * 100


def draw_heatmap(ax: plt.Axes, heat: pd.DataFrame, title: str, show_y: bool, cbar_ax=None) -> None:
    annotations = heat.map(lambda value: f"{value:.1f}%")
    sns.heatmap(
        heat,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap=sns.light_palette("#237A8B", as_cmap=True),
        vmin=85,
        vmax=100,
        linewidths=1.0,
        linecolor="white",
        cbar=cbar_ax is not None,
        cbar_ax=cbar_ax,
        cbar_kws={"label": "U accuracy (%)"} if cbar_ax is not None else None,
        annot_kws={"fontsize": 8.5},
    )
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.set_xlabel("Premise–claim polarity")
    if show_y:
        ax.set_ylabel("Mismatch type")
        ax.set_yticklabels(
            ["Subject (VO)", "Object (SV)", "Verb + object", "No content match"],
            rotation=0,
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_xticklabels(PHASE_ORDER, rotation=0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlepad": 10,
            "axes.labelpad": 7,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    data = load_exp2b()
    summary = cluster_bootstrap_summary(data)

    # Main standalone panel for layouts that use a single-column result figure.
    main_fig, main_ax = plt.subplots(figsize=(10.4, 3.5))
    plot_gate_panel(
        main_ax,
        summary,
        show_panel_title=False,
        compact_paper=True,
    )
    main_fig.tight_layout()
    save_figure(main_fig, "exp2b_gate_GU_model_comparison")
    plt.close(main_fig)

    # Complete multi-panel figure: gate contrast plus polarity stability.
    fig = plt.figure(figsize=(11.8, 9.0))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.18, 0.82],
        width_ratios=[1, 1, 0.045],
        hspace=0.42,
        wspace=0.18,
    )
    gate_ax = fig.add_subplot(grid[0, :2])
    plot_gate_panel(gate_ax, summary)

    qwen_ax = fig.add_subplot(grid[1, 0])
    phi_ax = fig.add_subplot(grid[1, 1])
    cbar_ax = fig.add_subplot(grid[1, 2])
    draw_heatmap(qwen_ax, polarity_heatmap(data, "Qwen3-8B"), "Qwen3-8B", True)
    draw_heatmap(
        phi_ax,
        polarity_heatmap(data, "Phi-4-mini"),
        "Phi-4-mini",
        False,
        cbar_ax=cbar_ax,
    )
    qwen_ax.text(
        -0.12,
        1.13,
        "B   U accuracy remains high across polarity changes",
        transform=qwen_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    save_figure(fig, "exp2b_gate_combined")
    plt.close(fig)

    summary.to_csv(OUTPUT_DIR / "exp2b_gate_GU_bootstrap_summary.csv", index=False)
    print(f"Saved figures and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
