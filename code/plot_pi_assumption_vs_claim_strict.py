"""Compare strict p_i DAS profiles at assumption-final and claim-final sites.

For every layer, site, and identifying control, IIA is first averaged across
the three DAS training seeds. Confirmatory IIA is then the minimum over the
raw-p_i controls, matching the causal-flow figure's aggregation rule.
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
OUT = ROOT / "data/das/pi_assumption_vs_claim_strict_normalized_depth_qwen_phi4"

CONTROLS = [
    "main_IIA",
    "active_source_m0_IIA",
    "probe_flip_both_IIA",
    "probe_flip_pc_IIA",
]

MODELS = {
    "Phi-4 Mini Instruct": {
        "layers": 32,
        "claim_color": "#225EA8",
        "assumption_color": "#78A9DC",
        "claim_files": [
            f"data/das/seed_sweep_phi4_pi/seed{seed}/relay_map.csv"
            for seed in range(3)
        ],
        "assumption_files": [
            "data/das/seed_sweep_phi4_pi_r64_matched_assumption_final/"
            f"seed{seed}/relay_map.csv"
            for seed in range(3)
        ],
    },
    "Qwen3-8B": {
        "layers": 36,
        "claim_color": "#D95F02",
        "assumption_color": "#F4A261",
        "claim_files": [
            f"data/das/seed_sweep_qwen_pi_r64/seed{seed}/relay_map.csv"
            for seed in range(3)
        ],
        "assumption_files": [
            "data/das/seed_sweep_qwen_pi_r64_matched_assumption_final/"
            f"seed{seed}/relay_map.csv"
            for seed in range(3)
        ],
    },
}


def load_profile(
    paths: list[str],
    *,
    source_site: str,
    display_site: str,
    model: str,
    n_layers: int,
) -> pd.DataFrame:
    seed_frames: list[pd.DataFrame] = []
    for seed, relative_path in enumerate(paths):
        path = ROOT / relative_path
        frame = pd.read_csv(path).copy()
        frame["site"] = frame["site"].replace({"row": "claim_final"})
        frame = frame[frame["site"] == source_site].copy()
        if frame.empty:
            raise ValueError(f"{path} contains no rows for site={source_site!r}")
        missing = [column for column in CONTROLS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing p_i controls: {missing}")
        frame["seed"] = seed
        seed_frames.append(frame[["layer", "seed", *CONTROLS]])

    per_seed = pd.concat(seed_frames, ignore_index=True)
    control_means = (
        per_seed.groupby("layer", as_index=False)[CONTROLS]
        .mean()
        .sort_values("layer")
    )
    seed_counts = (
        per_seed.groupby("layer", as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
    )
    profile = control_means.merge(seed_counts, on="layer", how="left")
    if not (profile["n_seeds"] == 3).all():
        raise ValueError(
            f"{model}/{display_site} does not have three seeds at every layer"
        )
    profile["identified_IIA"] = profile[CONTROLS].min(axis=1)
    profile["normalized_depth"] = profile["layer"] / n_layers
    profile["model"] = model
    profile["site"] = display_site
    return profile[
        [
            "model",
            "site",
            "layer",
            "normalized_depth",
            "n_seeds",
            "identified_IIA",
            *CONTROLS,
        ]
    ]


def annotate_peak(ax: plt.Axes, profile: pd.DataFrame, color: str, *, above: bool) -> None:
    peak = profile.loc[profile["identified_IIA"].idxmax()]
    x = float(peak["normalized_depth"])
    y = float(peak["identified_IIA"])
    layer = int(peak["layer"])
    ax.scatter([x], [y], s=28, color=color, edgecolor="white", linewidth=0.8, zorder=5)
    offset = (6, 7) if above else (6, -15)
    ax.annotate(
        f"L{layer}  {y:.1%}",
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=8.4,
        fontweight="semibold",
    )


def main() -> int:
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.15, 4.8))
    source_rows: list[pd.DataFrame] = []

    for model, spec in MODELS.items():
        claim = load_profile(
            spec["claim_files"],
            source_site="claim_final",
            display_site="claim_final",
            model=model,
            n_layers=spec["layers"],
        )
        assumption = load_profile(
            spec["assumption_files"],
            source_site="matched_assumption_final",
            display_site="target_assumption_final",
            model=model,
            n_layers=spec["layers"],
        )
        source_rows.extend([claim, assumption])

        ax.plot(
            assumption["normalized_depth"],
            assumption["identified_IIA"],
            color=spec["assumption_color"],
            linewidth=2.35,
            solid_capstyle="round",
        )
        ax.plot(
            claim["normalized_depth"],
            claim["identified_IIA"],
            color=spec["claim_color"],
            linewidth=2.35,
            solid_capstyle="round",
        )
    ax.set_title(r"Premise polarity $p_i$", loc="left", fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.02, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, color="#DDDDDD", linewidth=0.65, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(r"Confirmatory IIA for $p_i$")
    ax.set_xlabel("Normalized depth")
    handles = [
        Line2D(
            [0],
            [0],
            color="#F4A261",
            lw=2.35,
            label="Qwen3-8B - target assumption final",
        ),
        Line2D(
            [0], [0], color="#D95F02", lw=2.35,
            label="Qwen3-8B - claim final",
        ),
        Line2D(
            [0], [0], color="#78A9DC", lw=2.35,
            label="Phi-4 Mini Instruct - target assumption final",
        ),
        Line2D(
            [0], [0], color="#225EA8", lw=2.35,
            label="Phi-4 Mini Instruct - claim final",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.105, 1, 1))
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
    )
    pd.concat(source_rows, ignore_index=True).to_csv(
        OUT.with_suffix(".csv"),
        index=False,
    )
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")
    print(f"Wrote {OUT.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
