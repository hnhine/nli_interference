"""Plot the updated p_c DAS relay profiles for Qwen3-8B and Phi-4 Mini.

Unlike the earlier figure, the primary curve is pc_active_IIA: the mean over
main, flip_both, and flip_pi.  The main-only curve is retained as a dashed
diagnostic.  A third panel uses the identity audit to show whether answer-token
behavior is better explained by raw p_c or by REL = p_i p_c.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path("/workspace/nhi/nli_interference")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    relay_csv: Path
    identity_csv: Path
    num_layers: int
    color: str
    light_color: str


SPECS = (
    ModelSpec(
        name="Qwen3-8B",
        relay_csv=ROOT / "data/das/qwen_pc_v4_r16_stride2_1ep/relay_map.csv",
        identity_csv=ROOT / "data/das/identity_qwen_pc_v4_answer/identity_sweep.csv",
        num_layers=36,
        color="#D95F02",
        light_color="#F4A261",
    ),
    ModelSpec(
        name="Phi-4 Mini Instruct",
        relay_csv=ROOT / "data/das/phi4_pc_v4_r64_stride2_1ep/relay_map.csv",
        identity_csv=ROOT / "data/das/identity_phi4_pc_v4_answer/identity_sweep.csv",
        num_layers=32,
        color="#225EA8",
        light_color="#78A9DC",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot updated p_c v4 normalized-depth results.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/das/pc_v4_normalized_depth_qwen_phi4.png",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def load_relay(spec: ModelSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.relay_csv)
    required = {
        "layer",
        "site",
        "main_IIA",
        "probe_flip_both_IIA",
        "probe_flip_pi_IIA",
        "pc_active_IIA",
        "pc_inactive_IIA",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{spec.name} relay map is missing: {sorted(missing)}")
    frame = frame.loc[frame["site"].isin(["claim_final", "answer_token"])].copy()
    frame["model"] = spec.name
    frame["normalized_depth"] = frame["layer"] / spec.num_layers
    return frame


def load_identity(spec: ModelSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.identity_csv)
    required = {"layer", "H_pc_macro", "H_rel_macro", "H_pc_strict_min", "H_rel_strict_min"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{spec.name} identity audit is missing: {sorted(missing)}")
    frame = frame.copy()
    frame["model"] = spec.name
    frame["normalized_depth"] = frame["layer"] / spec.num_layers
    frame["pc_minus_rel"] = frame["H_pc_macro"] - frame["H_rel_macro"]
    frame["pc_minus_rel_strict"] = frame["H_pc_strict_min"] - frame["H_rel_strict_min"]
    return frame


def annotate_peak(ax: plt.Axes, frame: pd.DataFrame, spec: ModelSpec, site: str) -> None:
    subset = frame.loc[frame["site"] == site]
    row = subset.loc[subset["pc_active_IIA"].idxmax()]
    x = float(row["normalized_depth"])
    y = float(row["pc_active_IIA"])
    layer = int(row["layer"])
    if site == "claim_final":
        offsets = {"Qwen3-8B": (8, -32), "Phi-4 Mini Instruct": (-78, -32)}
    else:
        offsets = {"Qwen3-8B": (10, 14), "Phi-4 Mini Instruct": (-86, 14)}
    ax.scatter([x], [y], s=55, color=spec.color, edgecolor="white", linewidth=0.9, zorder=5)
    ax.annotate(
        f"{spec.name}: {y:.1%} (L{layer})",
        xy=(x, y),
        xytext=offsets[spec.name],
        textcoords="offset points",
        color=spec.color,
        fontsize=8.7,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-", "color": spec.color, "linewidth": 0.8},
    )


def plot_relay_panel(
    ax: plt.Axes,
    relay_frames: dict[str, pd.DataFrame],
    site: str,
    panel_label: str,
) -> None:
    for spec in SPECS:
        frame = relay_frames[spec.name]
        subset = frame.loc[frame["site"] == site].sort_values("normalized_depth")
        x = subset["normalized_depth"]
        ax.plot(
            x,
            subset["pc_inactive_IIA"],
            color=spec.light_color,
            linewidth=1.25,
            linestyle=(0, (1.2, 2.0)),
            alpha=0.72,
            zorder=1,
        )
        ax.plot(
            x,
            subset["main_IIA"],
            color=spec.color,
            linewidth=1.65,
            linestyle=(0, (4.5, 2.3)),
            alpha=0.75,
            zorder=2,
        )
        ax.plot(
            x,
            subset["pc_active_IIA"],
            color=spec.color,
            linewidth=2.9,
            solid_capstyle="round",
            zorder=3,
        )
        annotate_peak(ax, frame, spec, site)

    site_name = "Claim-final site" if site == "claim_final" else "Answer-token site"
    ax.set_title(f"{panel_label}   {site_name}", loc="left", fontweight="bold")
    ax.set_xlabel("Normalized depth (layer / number of layers)")
    ax.set_xlim(-0.015, 0.965)
    ax.set_ylim(-0.03, 1.055)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.75)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot_identity_panel(ax: plt.Axes, identity_frames: dict[str, pd.DataFrame]) -> None:
    ax.axhspan(0, 0.65, color="#EAF4EA", alpha=0.72, zorder=0)
    ax.axhspan(-0.75, 0, color="#F7ECE8", alpha=0.72, zorder=0)
    ax.axhline(0, color="#333333", linewidth=1.0, zorder=1)

    for spec in SPECS:
        frame = identity_frames[spec.name].sort_values("normalized_depth")
        ax.plot(
            frame["normalized_depth"],
            frame["pc_minus_rel"],
            color=spec.color,
            linewidth=2.7,
            label=spec.name,
            zorder=3,
        )
        positive = frame["pc_minus_rel"] > 0
        ax.scatter(
            frame.loc[positive, "normalized_depth"],
            frame.loc[positive, "pc_minus_rel"],
            color=spec.color,
            edgecolor="white",
            linewidth=0.7,
            s=35,
            zorder=4,
        )

    ax.text(0.02, 0.60, r"raw $p_c$ favored", color="#356B35", fontsize=9.2, va="top")
    ax.text(0.02, -0.70, r"$REL=p_i p_c$ favored", color="#8B4A38", fontsize=9.2, va="bottom")
    ax.set_title("C   Answer-token identity audit", loc="left", fontweight="bold")
    ax.set_xlabel("Normalized depth (layer / number of layers)")
    ax.set_ylabel(r"Identity margin  $H_{p_c}-H_{REL}$")
    ax.set_xlim(-0.015, 0.965)
    ax.set_ylim(-0.75, 0.65)
    ax.grid(axis="x", color="white", linewidth=0.8)
    ax.grid(axis="y", color="white", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def write_source_csv(
    relay_frames: dict[str, pd.DataFrame], identity_frames: dict[str, pd.DataFrame], path: Path
) -> None:
    relay_columns = [
        "model",
        "layer",
        "normalized_depth",
        "site",
        "main_IIA",
        "probe_flip_both_IIA",
        "probe_flip_pi_IIA",
        "pc_active_IIA",
        "pc_inactive_IIA",
        "pc_macro_IIA",
    ]
    identity_columns = [
        "model",
        "layer",
        "normalized_depth",
        "H_pc_macro",
        "H_rel_macro",
        "pc_minus_rel",
        "H_pc_strict_min",
        "H_rel_strict_min",
        "pc_minus_rel_strict",
    ]
    relay = pd.concat(relay_frames.values(), ignore_index=True)[relay_columns]
    relay.insert(0, "record_type", "relay")
    identity = pd.concat(identity_frames.values(), ignore_index=True)[identity_columns]
    identity.insert(0, "record_type", "identity")
    combined = pd.concat([relay, identity], ignore_index=True, sort=False)
    combined.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    relay_frames = {spec.name: load_relay(spec) for spec in SPECS}
    identity_frames = {spec.name: load_identity(spec) for spec in SPECS}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.3,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.65))
    plot_relay_panel(axes[0], relay_frames, "claim_final", "A")
    plot_relay_panel(axes[1], relay_frames, "answer_token", "B")
    plot_identity_panel(axes[2], identity_frames)
    axes[0].set_ylabel("Interchange Intervention Accuracy (IIA)")

    model_handles = [
        Line2D([0], [0], color=spec.color, linewidth=3.0, label=spec.name) for spec in SPECS
    ]
    metric_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.9, label="Active mean (3 controls)"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=1.65,
            linestyle=(0, (4.5, 2.3)),
            label="Main only",
        ),
        Line2D(
            [0],
            [0],
            color="#777777",
            linewidth=1.25,
            linestyle=(0, (1.2, 2.0)),
            label="Inactive controls",
        ),
    ]
    fig.legend(
        handles=model_handles + metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        handlelength=3.2,
        columnspacing=2.0,
    )
    fig.suptitle(
        r"Claim polarity ($p_c$): causal relay and answer-token identity",
        fontsize=15.5,
        fontweight="semibold",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.98), w_pad=2.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    source_csv = args.output.with_suffix(".csv")
    write_source_csv(relay_frames, identity_frames, source_csv)
    print(f"Wrote {args.output}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {source_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
