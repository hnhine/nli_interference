"""Plot panel D using the minimum across every M-v4 evaluation control."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd


ROOT = Path("/workspace/nhi/nli_interference")
OUT = ROOT / "data/das/causal_flow_panel_d_match_gate_m_full_min_qwen_phi4"

MODELS = {
    "Qwen3-8B": {
        "layers": 36,
        "claim": "#D95F02",
        "answer": "#F4A261",
        "relay_csv": ROOT / "data/das/qwen_m_v4_r16_stride2_1ep_b32/relay_map.csv",
    },
    "Phi-4 Mini Instruct": {
        "layers": 32,
        "claim": "#225EA8",
        "answer": "#78A9DC",
        "relay_csv": ROOT / "data/das/phi4_m_v4_r64_stride2/relay_map.csv",
    },
}

M_CONTROL_COLUMNS = {
    "match_to_nomatch": "match_to_nomatch_IIA",
    "nomatch_to_match": "nomatch_to_match_IIA",
    "label_copy_trap": "label_copy_trap_IIA",
    "label_copy_trap_same_m1": "label_copy_trap_same_m1_IIA",
}


def main() -> int:
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.15, 3.35))
    source_rows: list[pd.DataFrame] = []

    for model, spec in MODELS.items():
        frame = pd.read_csv(spec["relay_csv"]).copy()
        frame = frame[frame["site"].isin(["claim_final", "answer_token"])].copy()
        columns = list(M_CONTROL_COLUMNS.values())
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"{spec['relay_csv']} is missing columns: {missing}")

        frame["identified_IIA"] = frame[columns].min(axis=1)
        frame["bottleneck_control"] = (
            frame[columns].idxmin(axis=1).map({value: key for key, value in M_CONTROL_COLUMNS.items()})
        )
        frame["normalized_depth"] = frame["layer"] / spec["layers"]
        frame["model"] = model

        source_rows.append(
            frame[
                [
                    "model",
                    "layer",
                    "normalized_depth",
                    "site",
                    *columns,
                    "identified_IIA",
                    "bottleneck_control",
                ]
            ]
        )

        for site, color, width in (
            ("claim_final", spec["claim"], 2.35),
            ("answer_token", spec["answer"], 2.05),
        ):
            subset = frame[frame["site"] == site].sort_values("normalized_depth")
            ax.plot(
                subset["normalized_depth"],
                subset["identified_IIA"],
                color=color,
                linewidth=width,
                solid_capstyle="round",
            )

    ax.set_title(r"D   Match gate $m$ — all-control minimum", loc="left", fontweight="semibold")
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Confirmatory IIA")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.02, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, color="#DDDDDD", linewidth=0.65, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([0], [0], color="#D95F02", lw=2.35, label="Qwen3-8B — claim final"),
        Line2D([0], [0], color="#F4A261", lw=2.05, label="Qwen3-8B — answer token"),
        Line2D([0], [0], color="#225EA8", lw=2.35, label="Phi-4 Mini — claim final"),
        Line2D([0], [0], color="#78A9DC", lw=2.05, label="Phi-4 Mini — answer token"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))

    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    pd.concat(source_rows, ignore_index=True).to_csv(OUT.with_suffix(".csv"), index=False)
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")
    print(f"Wrote {OUT.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
