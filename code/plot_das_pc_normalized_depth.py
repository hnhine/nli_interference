"""Compare p_c DAS profiles for Qwen and Phi-4 on normalized depth.

The two source sweeps use different aggregate schemas.  To keep the plotted
quantities comparable, this script reconstructs three metrics from controls
shared by both runs:

    core     = main
    inactive = mean(gate_m0, label_copy_trap)
    macro    = mean(main, gate_m0, label_copy_trap)

Claim-final and answer-token intervention sites are shown in separate panels.
Normalized depth is layer / num_layers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class ModelSpec:
    name: str
    csv_path: Path
    num_layers: int
    column_prefix: str
    colors: dict[str, str]


METRIC_STYLES = {
    "core": {"label": "Active*", "linewidth": 2.8},
    "macro": {"label": "Macro", "linewidth": 2.0},
    "inactive": {"label": "Inactive controls", "linewidth": 1.7},
}
PLOT_ORDER = ("inactive", "macro", "core")
SITES = ("claim_final", "answer_token")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Qwen and Phi-4 p_c IIA against normalized model depth."
    )
    parser.add_argument(
        "--qwen-csv",
        type=Path,
        default=Path("data/das/qwen_pc_forced_stride2_r16/pc_forced_sweep.csv"),
    )
    parser.add_argument(
        "--phi4-csv",
        type=Path,
        default=Path("data/das/phi4_pc_relay_r16_stride2/relay_map.csv"),
    )
    parser.add_argument("--qwen-layers", type=int, default=36)
    parser.add_argument("--phi4-layers", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/das/pc_normalized_depth_qwen_phi4.png"),
    )
    parser.add_argument(
        "--normalized-csv",
        type=Path,
        default=Path("data/das/pc_normalized_depth_qwen_phi4.csv"),
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def required_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"Missing {key!r} at layer={row.get('layer')} site={row.get('site')}")
    return float(value)


def load_model(spec: ModelSpec) -> list[dict[str, float | int | str]]:
    with spec.csv_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, float | int | str]] = []
    for raw in raw_rows:
        site = raw.get("site", "")
        if site not in SITES:
            continue
        prefix = spec.column_prefix
        core = required_float(raw, f"{prefix}main_IIA")
        gate = required_float(raw, f"{prefix}gate_m0_IIA")
        label_copy = required_float(raw, f"{prefix}label_copy_trap_IIA")
        inactive = mean((gate, label_copy))
        macro = mean((core, gate, label_copy))
        layer = int(raw["layer"])
        rows.append(
            {
                "model": spec.name,
                "num_layers": spec.num_layers,
                "layer": layer,
                "normalized_depth": layer / spec.num_layers,
                "site": site,
                "core": core,
                "inactive": inactive,
                "macro": macro,
            }
        )

    if not rows:
        raise ValueError(f"No claim_final/answer_token records found in {spec.csv_path}")
    return sorted(rows, key=lambda row: (str(row["site"]), int(row["layer"])))


def write_normalized_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "num_layers",
        "layer",
        "normalized_depth",
        "site",
        "core",
        "inactive",
        "macro",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(models: list[tuple[ModelSpec, list[dict]]], output: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), sharex=True, sharey=True)

    site_titles = {
        "claim_final": "Claim-final intervention site",
        "answer_token": "Answer-token intervention site",
    }
    for ax, site in zip(axes, SITES):
        for spec, rows in models:
            site_rows = [row for row in rows if row["site"] == site]
            x = [float(row["normalized_depth"]) for row in site_rows]
            for metric in PLOT_ORDER:
                style = METRIC_STYLES[metric]
                y = [float(row[metric]) for row in site_rows]
                ax.plot(
                    x,
                    y,
                    color=spec.colors[metric],
                    linestyle="-",
                    linewidth=style["linewidth"],
                    alpha=0.97,
                )

        ax.set_title(site_titles[site])
        ax.set_xlabel("Normalized depth (layer / number of layers)")
        ax.set_xlim(-0.015, 1.0)
        ax.set_ylim(-0.02, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Interchange Intervention Accuracy (IIA)")
    fig.suptitle(r"Claim polarity ($p_c$) across normalized model depth", fontsize=16)

    legend_handles = []
    for spec, _ in models:
        for metric, style in METRIC_STYLES.items():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=spec.colors[metric],
                    linestyle="-",
                    linewidth=style["linewidth"],
                    label=f"{spec.name} — {style['label']}",
                )
            )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        columnspacing=2.8,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    specs = [
        ModelSpec(
            "Qwen3-8B",
            args.qwen_csv,
            args.qwen_layers,
            "test_",
            {"macro": "#F3A261", "core": "#E85D04", "inactive": "#FAD7B5"},
        ),
        ModelSpec(
            "Phi-4 Mini",
            args.phi4_csv,
            args.phi4_layers,
            "",
            {"macro": "#7AA7E8", "core": "#1D4ED8", "inactive": "#C7DCF7"},
        ),
    ]
    model_rows = [(spec, load_model(spec)) for spec in specs]
    all_rows = [row for _, rows in model_rows for row in rows]
    write_normalized_csv(all_rows, args.normalized_csv)
    plot(model_rows, args.output, args.dpi)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.normalized_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
