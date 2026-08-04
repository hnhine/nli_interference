"""Plot Phi-4 rank robustness at four claim-final layers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd


ROOT = Path("/workspace/nhi/nli_interference")
DATA_ROOT = ROOT / "data/das/rank_robustness"
OUT = ROOT / "data/das/rank_robustness_phi4_claim_layers_0_10_20_30"

RANKS = [16, 128, 256, 512]
LAYERS = [0, 10, 20, 30]
RANK_COLORS = {
    16: "#BFD7EA",
    128: "#78A9DC",
    256: "#377EB8",
    512: "#174A7E",
}

VARIABLES = {
    "pc": {
        "combo": "phi4_pc",
        "title": r"Claim polarity $p_c$",
        "controls": ["main", "probe_flip_both", "probe_flip_pi"],
    },
    "pi": {
        "combo": "phi4_pi",
        "title": r"Premise polarity $p_i$",
        "controls": [
            "main",
            "active_source_m0",
            "probe_flip_both",
            "probe_flip_pc",
        ],
    },
    "rho": {
        "combo": "phi4_rho",
        "title": r"Polarity relation $\rho$",
        "controls": [
            "flip_pi",
            "flip_pc",
            "hold_both",
            "source_m0",
            "gate_m0",
            "label_copy_trap",
        ],
    },
    "m": {
        "combo": "phi4_m",
        "title": r"Match gate $m$",
        "controls": [
            "match_to_nomatch",
            "nomatch_to_match",
            "label_copy_trap",
            "label_copy_trap_same_m1",
        ],
    },
}


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable, spec in VARIABLES.items():
        for rank in RANKS:
            for layer in LAYERS:
                cell = (
                    DATA_ROOT
                    / spec["combo"]
                    / f"r{rank}"
                    / "seed0"
                    / f"L{layer:02d}_claim_final"
                    / "summary_metrics.json"
                )
                if not cell.is_file():
                    raise FileNotFoundError(cell)
                summary = json.loads(cell.read_text())
                by_control = summary["test"]["by_control"]
                values = {
                    control: float(by_control[control]["IIA"])
                    for control in spec["controls"]
                }
                bottleneck = min(values, key=values.get)
                rows.append(
                    {
                        "model": "Phi-4 Mini Instruct",
                        "variable": variable,
                        "layer": layer,
                        "rank": rank,
                        "strict_IIA": values[bottleneck],
                        "bottleneck_control": bottleneck,
                        **{f"{name}_IIA": value for name, value in values.items()},
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    frame = load_rows()
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.65), sharex=True, sharey=True)

    for panel, (ax, (variable, spec)) in zip("ABCD", zip(axes.flat, VARIABLES.items())):
        subset = frame[frame["variable"] == variable]
        for rank in RANKS:
            profile = subset[subset["rank"] == rank].sort_values("layer")
            ax.plot(
                profile["layer"],
                profile["strict_IIA"],
                color=RANK_COLORS[rank],
                linewidth=2.15,
                marker="o",
                markersize=3.8,
                solid_capstyle="round",
            )
        ax.set_title(f"{panel}   {spec['title']}", loc="left", fontweight="semibold")
        ax.set_xlim(-1, 31)
        ax.set_xticks(LAYERS)
        ax.set_ylim(-0.02, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, color="#DDDDDD", linewidth=0.65, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Confirmatory IIA")

    handles = [
        Line2D(
            [0],
            [0],
            color=RANK_COLORS[rank],
            marker="o",
            markersize=3.8,
            lw=2.15,
            label=f"Rank {rank}",
        )
        for rank in RANKS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
    )
    frame.to_csv(OUT.with_suffix(".csv"), index=False)
    print(f"Wrote {OUT.with_suffix('.png')}")
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
