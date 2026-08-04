"""Plot rank robustness for Phi-4 and Qwen at four claim-final layers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd


ROOT = Path("/workspace/nhi/nli_interference")
DATA_ROOT = ROOT / "data/das/rank_robustness"
OUT = ROOT / "data/das/rank_robustness_qwen_phi4_claim_layers_0_10_20_30"

RANKS = [16, 64, 128, 256, 512]
LAYERS = [0, 10, 20, 30]
RANK_COLORS = {
    16: "#C6DBEF",
    64: "#9ECAE1",
    128: "#6BAED6",
    256: "#3182BD",
    512: "#08519C",
}

MODELS = {
    "Phi-4 Mini Instruct": "phi4",
    "Qwen3-8B": "qwen",
}

R64_SOURCES = {
    "phi4_pc": ROOT / "data/das/seed_sweep_phi4_pc/seed0/relay_map.csv",
    "phi4_pi": ROOT / "data/das/seed_sweep_phi4_pi/seed0/relay_map.csv",
    "phi4_rho": ROOT / "data/das/seed_sweep_phi4_rho/seed0/relay_map.csv",
    "phi4_m": ROOT / "data/das/seed_sweep_phi4_m/seed0/relay_map.csv",
    "qwen_pc": ROOT / "data/das/seed_sweep_qwen_pc_r64/seed0/relay_map.csv",
    "qwen_pi": ROOT / "data/das/seed_sweep_qwen_pi_r64/seed0/relay_map.csv",
    "qwen_rho": ROOT / "data/das/seed_sweep_qwen_rho_r64/seed0/relay_map.csv",
    "qwen_m": ROOT / "data/das/seed_sweep_qwen_m_r64/seed0/relay_map.csv",
}

VARIABLES = {
    "pc": {
        "title": r"Claim polarity $p_c$",
        "controls": ["main_IIA", "probe_flip_both_IIA", "probe_flip_pi_IIA"],
    },
    "pi": {
        "title": r"Premise polarity $p_i$",
        "controls": [
            "main_IIA",
            "active_source_m0_IIA",
            "probe_flip_both_IIA",
            "probe_flip_pc_IIA",
        ],
    },
    "rho": {
        "title": r"Polarity relation $\rho$",
        "controls": [
            "flip_pi_IIA",
            "flip_pc_IIA",
            "hold_both_IIA",
            "source_m0_IIA",
            "gate_m0_IIA",
            "label_copy_trap_IIA",
        ],
    },
    "m": {
        "title": r"Match gate $m$",
        "controls": [
            "match_to_nomatch_IIA",
            "nomatch_to_match_IIA",
            "label_copy_trap_IIA",
            "label_copy_trap_same_m1_IIA",
        ],
    },
}


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, prefix in MODELS.items():
        for variable, spec in VARIABLES.items():
            for rank in RANKS:
                combo = f"{prefix}_{variable}"
                path = (
                    R64_SOURCES[combo]
                    if rank == 64
                    else DATA_ROOT / combo / f"r{rank}" / "seed0" / "relay_map.csv"
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                frame["site"] = frame["site"].replace({"row": "claim_final"})
                for layer in LAYERS:
                    cell = frame.loc[
                        (frame["layer"] == layer) & (frame["site"] == "claim_final")
                    ]
                    if len(cell) != 1:
                        raise ValueError(
                            f"Expected one completed L{layer} claim-final cell in {path}; "
                            f"found {len(cell)}"
                        )
                    values = {
                        control: float(cell.iloc[0][control])
                        for control in spec["controls"]
                    }
                    if any(pd.isna(value) for value in values.values()):
                        raise ValueError(f"Missing strict control at {path}, L{layer}: {values}")
                    bottleneck = min(values, key=values.get)
                    rows.append(
                        {
                            "model": model,
                            "variable": variable,
                            "rank": rank,
                            "layer": layer,
                            "strict_IIA": values[bottleneck],
                            "bottleneck_control": bottleneck.removesuffix("_IIA"),
                            **values,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> int:
    frame = load_rows()
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.4,
            "legend.fontsize": 8.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(10.8, 5.05), sharex=True, sharey=True)
    panel_labels = iter("ABCDEFGH")

    for row_index, model in enumerate(MODELS):
        for column_index, (variable, spec) in enumerate(VARIABLES.items()):
            ax = axes[row_index, column_index]
            subset = frame.loc[
                (frame["model"] == model) & (frame["variable"] == variable)
            ]
            for rank in RANKS:
                profile = subset.loc[subset["rank"] == rank].sort_values("layer")
                ax.plot(
                    profile["layer"],
                    profile["strict_IIA"],
                    color=RANK_COLORS[rank],
                    linewidth=2.05,
                    marker="o",
                    markersize=3.6,
                    solid_capstyle="round",
                )

            label = next(panel_labels)
            ax.set_title(f"{label}   {spec['title']}", loc="left", fontweight="semibold")
            ax.set_xlim(-1, 31)
            ax.set_xticks(LAYERS)
            ax.set_ylim(-0.02, 1.03)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.grid(True, color="#DDDDDD", linewidth=0.65, alpha=0.75)
            ax.spines[["top", "right"]].set_visible(False)

        axes[row_index, 0].set_ylabel(f"{model}\nConfirmatory IIA")

    for ax in axes[-1]:
        ax.set_xlabel("Layer")

    handles = [
        Line2D(
            [0],
            [0],
            color=RANK_COLORS[rank],
            marker="o",
            markersize=3.8,
            linewidth=2.05,
            label=f"Rank {rank}",
        )
        for rank in RANKS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=1.15, h_pad=1.25)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    frame.to_csv(OUT.with_suffix(".csv"), index=False)
    print(f"Wrote {OUT.with_suffix('.png')}")
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
