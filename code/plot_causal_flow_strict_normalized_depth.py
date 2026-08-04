"""Plot confirmatory DAS profiles for p_c, p_i, rho, and m.

The figure deliberately uses variable-identifying scores instead of an active
average: minima over the raw-polarity controls, the full rho audit minimum,
and the full M-v4 audit minimum over both causal-transfer directions and both
label-copy controls. For multi-seed inputs, each control is averaged across seeds at a layer/site before the minimum is taken across controls.
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
        "claim": "#1F5A94",
        "answer": "#C55400",
        "files": {
            "pc": [
                f"data/das/seed_sweep_qwen_pc_r64/seed{seed}/relay_map.csv"
                for seed in range(3)
            ],
            "pi": [
                f"data/das/seed_sweep_qwen_pi_r64/seed{seed}/relay_map.csv"
                for seed in range(3)
            ],
            "rho": [
                f"data/das/seed_sweep_qwen_rho_r64/seed{seed}/relay_map.csv"
                for seed in range(3)
            ],
            # Seed 0 is not yet a complete full sweep. Use the two complete
            # r64 replications for the Qwen m profile.
            "m": [
                f"data/das/seed_sweep_qwen_m_r64/seed{seed}/relay_map.csv"
                for seed in (1, 2)
            ],
        },
    },
    "Phi-4 Mini Instruct": {
        "layers": 32,
        "claim": "#8DBDDD",
        "answer": "#F2A65A",
        "files": {
            "pc": [f"data/das/seed_sweep_phi4_pc/seed{seed}/relay_map.csv" for seed in range(3)],
            "pi": [f"data/das/seed_sweep_phi4_pi/seed{seed}/relay_map.csv" for seed in range(3)],
            "rho": [f"data/das/seed_sweep_phi4_rho/seed{seed}/relay_map.csv" for seed in range(3)],
            "m": [f"data/das/seed_sweep_phi4_m/seed{seed}/relay_map.csv" for seed in range(3)],
        },
    },
}

TITLES = {
    "pc": r"Claim polarity $p_c$",
    "pi": r"Assumption polarity $p_i$",
    "rho": r"Polarity relation $\rho$",
    "m": r"Match gate $m$",
}


CONTROL_COLUMNS = {
    "pc": ["main_IIA", "probe_flip_both_IIA", "probe_flip_pi_IIA"],
    "pi": ["main_IIA", "active_source_m0_IIA", "probe_flip_both_IIA", "probe_flip_pc_IIA"],
    "rho": [
        "flip_pi_IIA",
        "flip_pc_IIA",
        "hold_both_IIA",
        "source_m0_IIA",
        "gate_m0_IIA",
        "label_copy_trap_IIA",
    ],
    "m": [
        "match_to_nomatch_IIA",
        "nomatch_to_match_IIA",
        "label_copy_trap_IIA",
        "label_copy_trap_same_m1_IIA",
    ],
}


def load_identified_profile(
    paths: str | list[str], *, variable: str, model: str, n_layers: int
) -> pd.DataFrame:
    if isinstance(paths, str):
        paths = [paths]

    controls = CONTROL_COLUMNS[variable]
    seed_frames: list[pd.DataFrame] = []
    for seed_index, relative_path in enumerate(paths):
        frame = pd.read_csv(ROOT / relative_path).copy()
        frame["site"] = frame["site"].replace({"row": "claim_final"})
        frame = frame[frame["site"].isin(["claim_final", "answer_token"])].copy()
        missing = [column for column in controls if column not in frame.columns]
        if missing:
            raise ValueError(f"{relative_path} is missing controls for {variable}: {missing}")
        frame["seed"] = seed_index
        seed_frames.append(frame[["layer", "site", "seed", *controls]])

    per_seed = pd.concat(seed_frames, ignore_index=True)
    control_means = (
        per_seed.groupby(["layer", "site"], as_index=False)[controls]
        .mean()
        .sort_values(["layer", "site"])
    )
    seed_counts = (
        per_seed.groupby(["layer", "site"], as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
    )
    profile = control_means.merge(seed_counts, on=["layer", "site"], how="left")
    profile["identified_IIA"] = profile[controls].min(axis=1)
    profile["normalized_depth"] = profile["layer"] / n_layers
    profile["model"] = model
    profile["variable"] = variable
    return profile[
        ["layer", "site", "identified_IIA", "n_seeds", "normalized_depth", "model", "variable"]
    ]


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
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.43), sharex=True, sharey=True)
    panels = ["pc", "pi", "rho", "m"]
    source_rows: list[pd.DataFrame] = []

    for panel_label, (ax, variable) in zip("ABCD", zip(axes.flat, panels)):
        for model, spec in MODELS.items():
            frame = load_identified_profile(
                spec["files"][variable],
                variable=variable,
                model=model,
                n_layers=spec["layers"],
            )
            source_rows.append(frame)

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
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(-0.02, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, color="#DDDDDD", linewidth=0.65, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized depth")
    for ax in axes[:, 0]:
        ax.set_ylabel("Confirmatory IIA")

    handles = [
        Line2D([0], [0], color="#1F5A94", lw=2.35, label="claim final - Qwen3-8B"),
        Line2D([0], [0], color="#8DBDDD", lw=2.35, label="claim final - Phi-4 Mini"),
        Line2D([0], [0], color="#C55400", lw=2.05, label="answer token - Qwen3-8B"),
        Line2D([0], [0], color="#F2A65A", lw=2.05, label="answer token - Phi-4 Mini"),
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
