"""Plot per-cell joint-gate composition results for Qwen and Phi-4."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


CELL_ORDER = (
    "open_to_F",
    "open_to_T",
    "close_from_T",
    "close_from_F",
    "rho_flip_T_to_F",
    "rho_flip_F_to_T",
)
CELL_LABELS = {
    "open_to_F": "Open gate\nU → F",
    "open_to_T": "Open gate\nU → T",
    "close_from_T": "Close gate\nT → U",
    "close_from_F": "Close gate\nF → U",
    "rho_flip_T_to_F": r"Keep $m=1$\nT → F",
    "rho_flip_F_to_T": r"Keep $m=1$\nF → T",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    summary_path: Path
    dark: str
    medium: str
    light: str


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("scope") == "all"]


def values_for(rows: list[dict[str, str]], cell: str) -> dict[str, float]:
    cell_rows = [row for row in rows if row["cell_type"] == cell]

    def selected(family: str, regime: str | None = None) -> list[dict[str, str]]:
        out = [row for row in cell_rows if row["condition_family"] == family]
        if regime is not None:
            out = [row for row in out if str(row["rho_source_m"]) == regime]
        return out

    joint_m0 = selected("joint_constrained", "0")
    joint_m1 = selected("joint_constrained", "1")
    if len(joint_m0) != 1 or len(joint_m1) != 1:
        raise ValueError(f"Expected one joint row per rho-source regime for {cell}")

    m_single = selected("m_only")
    rho_single = selected("rho_only")
    best_single = max(
        mean(float(row["joint_target_accuracy"]) for row in m_single),
        mean(float(row["joint_target_accuracy"]) for row in rho_single),
    )
    random_rows = selected("joint_random_m") + selected("joint_random_rho")
    same_rows = selected("joint_same_value")
    none_rows = selected("none")
    return {
        "joint_m0": float(joint_m0[0]["IIA"]),
        "joint_m1": float(joint_m1[0]["IIA"]),
        "joint_m0_lo": float(joint_m0[0]["IIA_ci_low"]),
        "joint_m0_hi": float(joint_m0[0]["IIA_ci_high"]),
        "joint_m1_lo": float(joint_m1[0]["IIA_ci_low"]),
        "joint_m1_hi": float(joint_m1[0]["IIA_ci_high"]),
        "best_single": best_single,
        "random": mean(float(row["joint_target_accuracy"]) for row in random_rows),
        "same": mean(float(row["IIA"]) for row in same_rows),
        "none": mean(float(row["IIA"]) for row in none_rows),
    }


def plot(specs: list[ModelSpec], output: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(specs), figsize=(14.6, 6.8), sharey=True)
    if len(specs) == 1:
        axes = [axes]
    x = np.arange(len(CELL_ORDER))
    width = 0.18

    for ax, spec in zip(axes, specs):
        rows = read_summary(spec.summary_path)
        data = [values_for(rows, cell) for cell in CELL_ORDER]
        m0 = np.array([row["joint_m0"] for row in data])
        m1 = np.array([row["joint_m1"] for row in data])
        best = np.array([row["best_single"] for row in data])
        random = np.array([row["random"] for row in data])
        m0_err = np.vstack([
            m0 - np.array([row["joint_m0_lo"] for row in data]),
            np.array([row["joint_m0_hi"] for row in data]) - m0,
        ])
        m1_err = np.vstack([
            m1 - np.array([row["joint_m1_lo"] for row in data]),
            np.array([row["joint_m1_hi"] for row in data]) - m1,
        ])

        ax.bar(x - 1.5 * width, m0, width, color=spec.dark, yerr=m0_err, capsize=2.5)
        ax.bar(x - 0.5 * width, m1, width, color=spec.medium, yerr=m1_err, capsize=2.5)
        ax.bar(x + 0.5 * width, best, width, color=spec.light)
        ax.bar(x + 1.5 * width, random, width, color="#B8B8B8")
        ax.scatter(x, [row["same"] for row in data], marker="D", s=32, facecolors="white", edgecolors="#222222", zorder=5)
        ax.scatter(x, [row["none"] for row in data], marker="_", s=180, color="#222222", linewidths=1.5, zorder=5)

        ax.set_title(spec.name)
        ax.set_xticks(x, [CELL_LABELS[cell] for cell in CELL_ORDER])
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy for the condition-specific target")
    fig.suptitle(r"Joint causal composition of match ($m$) and polarity relation ($\rho$)", fontsize=16)
    legend = [
        Patch(facecolor=specs[0].dark, label=r"Joint, $m_B=0$"),
        Patch(facecolor=specs[0].medium, label=r"Joint, $m_B=1$"),
        Patch(facecolor=specs[0].light, label="Best single patch on joint target"),
        Patch(facecolor="#B8B8B8", label="Random-subspace replacement"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#222222", label="Same-value purity control"),
        Line2D([0], [0], marker="_", color="#222222", linestyle="none", markersize=14, label="Clean base"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-summary", type=Path, required=True)
    parser.add_argument("--phi4-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/das/joint_gate_qwen_phi4.png"))
    parser.add_argument("--dpi", type=int, default=320)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = [
        ModelSpec("Qwen3-8B", args.qwen_summary, "#E85D04", "#F3A261", "#FAD7B5"),
        ModelSpec("Phi-4 Mini Instruct", args.phi4_summary, "#1D4ED8", "#7AA7E8", "#C7DCF7"),
    ]
    plot(specs, args.output, args.dpi)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
