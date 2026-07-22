"""Plot a cross-model comparison for scored Exp3 minimal-pair runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PAIR_ORDER = ["target_flip", "distractor_flip"]
PAIR_LABELS = {"target_flip": "Target flip", "distractor_flip": "Distractor flip"}
PAIR_COLORS = {"target_flip": "#4C78A8", "distractor_flip": "#F58518"}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must have the form LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.exists():
        raise argparse.ArgumentTypeError(f"Invalid run specification: {value}")
    return label, path


def load_runs(runs: list[tuple[str, Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_frames = []
    summary_rows = []
    for model, root in runs:
        pair_path = root / "summary" / "exp3_intervention_pairs.csv"
        summary_path = root / "summary" / "summary_metrics.json"
        pairs = pd.read_csv(pair_path)
        pairs["model"] = model
        pair_frames.append(pairs)

        exp3 = json.loads(summary_path.read_text())["exp3_clean_selection"]
        minimal = exp3["minimal_pairs"]
        summary_rows.extend([
            {
                "model": model,
                "pair_type": "target_flip",
                "mean_abs_delta_R": minimal["target_flip"]["mean_abs_delta_R"],
                "success_rate": minimal["target_flip"]["directional_rate"],
                "ratio": minimal["target_to_distractor_mean_abs_effect_ratio"],
            },
            {
                "model": model,
                "pair_type": "distractor_flip",
                "mean_abs_delta_R": minimal["distractor_flip"]["mean_abs_delta_R"],
                "success_rate": minimal["distractor_flip"]["prediction_invariance_rate"],
                "ratio": minimal["target_to_distractor_mean_abs_effect_ratio"],
            },
        ])
    return pd.concat(pair_frames, ignore_index=True), pd.DataFrame(summary_rows)


def plot_comparison(pairs: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    for ax, model in zip(axes[0], models):
        subset = pairs[pairs["model"] == model].copy()
        subset["pair_label"] = subset["pair_type"].map(PAIR_LABELS)
        sns.barplot(
            data=subset,
            x="match_idx",
            y="abs_delta_R",
            hue="pair_label",
            hue_order=[PAIR_LABELS[pair] for pair in PAIR_ORDER],
            errorbar="se",
            capsize=0.08,
            palette=[PAIR_COLORS[pair] for pair in PAIR_ORDER],
            ax=ax,
        )
        ax.set_title(model)
        ax.set_xlabel("Target position")
        ax.set_ylabel(r"Mean $|\Delta R|$")
        ax.legend(title="", loc="lower right", fontsize=10)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
        ax.set_ylim(0, float(subset.groupby(["match_idx", "pair_type"])["abs_delta_R"].mean().max()) * 1.2)

    overall = axes[1, 0]
    display = summary.copy()
    display["pair_label"] = display["pair_type"].map(PAIR_LABELS)
    sns.barplot(
        data=display,
        x="model",
        y="mean_abs_delta_R",
        hue="pair_label",
        hue_order=[PAIR_LABELS[pair] for pair in PAIR_ORDER],
        palette=[PAIR_COLORS[pair] for pair in PAIR_ORDER],
        ax=overall,
    )
    overall.set_title("Overall polarity-flip effect")
    overall.set_xlabel("")
    overall.set_ylabel(r"Mean $|\Delta R|$")
    overall.legend(title="", fontsize=10)
    for container in overall.containers:
        overall.bar_label(container, fmt="%.2f", padding=3, fontsize=10)
    overall.set_ylim(0, float(display["mean_abs_delta_R"].max()) * 1.25)
    for model_index, model in enumerate(models):
        row = display[(display["model"] == model) & (display["pair_type"] == "target_flip")].iloc[0]
        overall.text(model_index, row["mean_abs_delta_R"] + 1.25, f"{row['ratio']:.1f}×", ha="center", fontsize=11)

    success_ax = axes[1, 1]
    sns.barplot(
        data=display,
        x="model",
        y="success_rate",
        hue="pair_label",
        hue_order=[PAIR_LABELS[pair] for pair in PAIR_ORDER],
        palette=[PAIR_COLORS[pair] for pair in PAIR_ORDER],
        ax=success_ax,
    )
    success_ax.set_ylim(0.85, 1.015)
    success_ax.set_title("Directional accuracy / prediction invariance")
    success_ax.set_xlabel("")
    success_ax.set_ylabel("Rate")
    success_ax.legend(title="", fontsize=10)
    for container in success_ax.containers:
        success_ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)

    fig.suptitle("Exp3: selecting matched-event polarity while ignoring distractors", fontsize=18, y=1.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        parser.error("exactly two --run arguments are required for the 2x2 comparison")
    pairs, summary = load_runs(args.run)
    plot_comparison(pairs, summary, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
