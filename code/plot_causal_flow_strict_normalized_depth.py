"""Plot confirmatory DAS profiles for p_c, p_i, rho, and m.

The figure deliberately uses variable-identifying scores instead of an active
average: minima over the raw-polarity controls, the full rho audit minimum,
and the mean of the two causal m-transfer directions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd


ROOT = Path("/workspace/nhi/nli_interference")
OUT = ROOT / "data/das/causal_flow_strict_normalized_depth_qwen_phi4"

MODELS = {
    "Qwen3-8B": {
        "layers": 36,
        "claim": "#D95F02",
        "answer": "#F4A261",
        "files": {
            "pc": "data/das/qwen_pc_v4_r16_stride2_1ep/relay_map.csv",
            "pi": "data/das/qwen_pi_v5_rawid_r16_stride2_1ep_b20/relay_map.csv",
            "rho": "data/das/qwen_rho_v1_r16_stride2_1ep_b32/relay_map.csv",
            "m": "data/das/qwen_m_v4_r16_stride2_1ep_b32/relay_map.csv",
        },
    },
    "Phi-4 Mini Instruct": {
        "layers": 32,
        "claim": "#225EA8",
        "answer": "#78A9DC",
        "files": {
            "pc": "data/das/phi4_pc_v4_r64_stride2_1ep/relay_map.csv",
            "pi": "data/das/phi4_pi_v5_rawid_r64_stride2_1ep_b20/relay_map.csv",
            "rho": "data/das/phi4_rho_r64_stride2_1ep/relay_map.csv",
            "m": "data/das/phi4_m_v4_r64_stride2/relay_map.csv",
        },
    },
}

TITLES = {
    "pc": r"Claim polarity $p_c$",
    "pi": r"Premise polarity $p_i$",
    "rho": r"Polarity relation $\rho$",
    "m": r"Match gate $m$",
}


def identified_score(frame: pd.DataFrame, variable: str) -> pd.Series:
    if variable == "pc":
        return frame[["main_IIA", "probe_flip_both_IIA", "probe_flip_pi_IIA"]].min(axis=1)
    if variable == "pi":
        return frame[
            ["main_IIA", "active_source_m0_IIA", "probe_flip_both_IIA", "probe_flip_pc_IIA"]
        ].min(axis=1)
    if variable == "rho":
        return frame["rho_full_audit_min_IIA"]
    if variable == "m":
        return frame[["match_to_nomatch_IIA", "nomatch_to_match_IIA"]].mean(axis=1)
    raise ValueError(variable)


def main() -> int:
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), sharex=True, sharey=True)
    panels = ["pc", "pi", "rho", "m"]
    source_rows: list[pd.DataFrame] = []

    for panel_label, (ax, variable) in zip("ABCD", zip(axes.flat, panels)):
        for model, spec in MODELS.items():
            frame = pd.read_csv(ROOT / spec["files"][variable]).copy()
            if variable == "rho" and model == "Phi-4 Mini Instruct":
                frame["site"] = frame["site"].replace({"row": "claim_final"})
            frame = frame[frame["site"].isin(["claim_final", "answer_token"])].copy()
            frame["identified_IIA"] = identified_score(frame, variable)
            frame["normalized_depth"] = frame["layer"] / spec["layers"]
            frame["model"] = model
            frame["variable"] = variable
            source_rows.append(
                frame[["model", "variable", "layer", "normalized_depth", "site", "identified_IIA"]]
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

        ax.set_title(f"{panel_label}   {TITLES[variable]}", loc="left", fontweight="semibold")
        ax.set_xlim(-0.01, 0.96)
        ax.set_ylim(-0.02, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, color="#DDDDDD", linewidth=0.65, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized depth")
    for ax in axes[:, 0]:
        ax.set_ylabel("Confirmatory IIA")

    handles = [
        Line2D([0], [0], color="#D95F02", lw=2.35, label="Qwen3-8B - claim final"),
        Line2D([0], [0], color="#F4A261", lw=2.05, label="Qwen3-8B - answer token"),
        Line2D([0], [0], color="#225EA8", lw=2.35, label="Phi-4 Mini Instruct - claim final"),
        Line2D([0], [0], color="#78A9DC", lw=2.05, label="Phi-4 Mini Instruct - answer token"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.105, 1, 1))
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    pd.concat(source_rows, ignore_index=True).to_csv(OUT.with_suffix(".csv"), index=False)
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")
    print(f"Wrote {OUT.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
