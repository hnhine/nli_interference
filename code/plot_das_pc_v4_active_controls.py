"""Plot the three active p_c controls across normalized model depth.

Claim-final and answer-token intervention sites are shown in separate panels.
Within each model color family, main is darkest, flip_both is medium, and
flip_pi is lightest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    csv_path: Path
    num_layers: int
    colors: dict[str, str]


METRIC_STYLES = {
    "main": {"label": "Main", "linewidth": 2.8},
    "flip_both": {"label": "Flip both", "linewidth": 2.0},
    "flip_pi": {"label": r"Flip $p_i$", "linewidth": 1.7},
}
PLOT_ORDER = ("flip_pi", "flip_both", "main")
SITES = ("claim_final", "answer_token")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the three active p_c controls for Qwen and Phi-4."
    )
    parser.add_argument(
        "--qwen-csv",
        type=Path,
        default=Path("data/das/qwen_pc_v4_r16_stride2_1ep/relay_map.csv"),
    )
    parser.add_argument(
        "--phi4-csv",
        type=Path,
        default=Path("data/das/phi4_pc_v4_r64_stride2_1ep/relay_map.csv"),
    )
    parser.add_argument("--qwen-layers", type=int, default=36)
    parser.add_argument("--phi4-layers", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/das/pc_v4_active_controls_qwen_phi4.png"),
    )
    parser.add_argument(
        "--normalized-csv",
        type=Path,
        default=Path("data/das/pc_v4_active_controls_qwen_phi4.csv"),
    )
    parser.add_argument("--dpi", type=int, default=320)
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
        layer = int(raw["layer"])
        rows.append(
            {
                "model": spec.name,
                "num_layers": spec.num_layers,
                "layer": layer,
                "normalized_depth": layer / spec.num_layers,
                "site": site,
                "main": required_float(raw, "main_IIA"),
                "flip_both": required_float(raw, "probe_flip_both_IIA"),
                "flip_pi": required_float(raw, "probe_flip_pi_IIA"),
            }
        )

    if not rows:
        raise ValueError(f"No claim_final/answer_token rows found in {spec.csv_path}")
    return sorted(rows, key=lambda row: (str(row["site"]), int(row["layer"])))


def write_normalized_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "num_layers",
        "layer",
        "normalized_depth",
        "site",
        "main",
        "flip_both",
        "flip_pi",
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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
    fig.suptitle(r"Claim polarity ($p_c$): active controls across normalized depth", fontsize=16)

    legend_handles = []
    for spec, _ in models:
        for metric in ("main", "flip_both", "flip_pi"):
            style = METRIC_STYLES[metric]
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
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    specs = [
        ModelSpec(
            "Qwen3-8B",
            args.qwen_csv,
            args.qwen_layers,
            {"main": "#E85D04", "flip_both": "#F3A261", "flip_pi": "#FAD7B5"},
        ),
        ModelSpec(
            "Phi-4 Mini",
            args.phi4_csv,
            args.phi4_layers,
            {"main": "#1D4ED8", "flip_both": "#7AA7E8", "flip_pi": "#C7DCF7"},
        ),
    ]
    model_rows = [(spec, load_model(spec)) for spec in specs]
    all_rows = [row for _, rows in model_rows for row in rows]
    write_normalized_csv(all_rows, args.normalized_csv)
    plot(model_rows, args.output, args.dpi)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.normalized_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
