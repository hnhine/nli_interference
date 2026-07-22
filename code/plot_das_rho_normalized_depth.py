"""Plot identified polarity-relation DAS profiles over normalized depth.

The Qwen sweep names its primary site claim_final.  The Phi-4 sweep names it
row, which tells the runner to use each pair's stored site; rho pairs store
claim_final for both base and source.  Thus both sweeps intervene at the
claim-final token even though their relay-map site labels differ.
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
    relation_site: str
    colors: dict[str, str]


METRIC_STYLES = {
    "identified": {"label": r"Identified $\rho$ (strict)", "linewidth": 2.8},
    "active": {"label": "Active mean", "linewidth": 2.2},
    "inactive": {"label": "Inactive controls", "linewidth": 1.7},
}
PLOT_ORDER = ("inactive", "active", "identified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Qwen and Phi-4 rho IIA against normalized model depth."
    )
    parser.add_argument(
        "--qwen-csv",
        type=Path,
        default=Path("data/das/qwen_rho_v1_r16_stride2_1ep_b32/relay_map.csv"),
    )
    parser.add_argument(
        "--phi4-csv",
        type=Path,
        default=Path("data/das/phi4_rho_r64_stride2_1ep/relay_map.csv"),
    )
    parser.add_argument("--qwen-layers", type=int, default=36)
    parser.add_argument("--phi4-layers", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/das/rho_normalized_depth_qwen_phi4.png"),
    )
    parser.add_argument(
        "--normalized-csv",
        type=Path,
        default=Path("data/das/rho_normalized_depth_qwen_phi4.csv"),
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
    valid_sites = {spec.relation_site, "answer_token"}
    for raw in raw_rows:
        source_site = raw.get("site", "")
        if source_site not in valid_sites:
            continue
        panel = "relation_site" if source_site == spec.relation_site else "answer_token"
        layer = int(raw["layer"])
        rows.append(
            {
                "model": spec.name,
                "num_layers": spec.num_layers,
                "layer": layer,
                "normalized_depth": layer / spec.num_layers,
                "panel": panel,
                "source_site": source_site,
                "identified": required_float(raw, "rho_identification_min_IIA"),
                "active": required_float(raw, "rho_active_IIA"),
                "inactive": required_float(raw, "rho_inactive_IIA"),
            }
        )

    for panel in ("relation_site", "answer_token"):
        if not any(row["panel"] == panel for row in rows):
            raise ValueError(f"No {panel} records found in {spec.csv_path}")
    return sorted(rows, key=lambda row: (str(row["panel"]), int(row["layer"])))


def write_normalized_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "num_layers",
        "layer",
        "normalized_depth",
        "panel",
        "source_site",
        "identified",
        "active",
        "inactive",
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
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0), sharex=True, sharey=True)
    panels = ("relation_site", "answer_token")
    panel_titles = {
        "relation_site": "Claim-final intervention site",
        "answer_token": "Answer-token intervention site",
    }

    for ax, panel in zip(axes, panels):
        for spec, rows in models:
            panel_rows = [row for row in rows if row["panel"] == panel]
            x = [float(row["normalized_depth"]) for row in panel_rows]
            for metric in PLOT_ORDER:
                style = METRIC_STYLES[metric]
                ax.plot(
                    x,
                    [float(row[metric]) for row in panel_rows],
                    color=spec.colors[metric],
                    linewidth=style["linewidth"],
                    alpha=0.97,
                )

        ax.set_title(panel_titles[panel])
        ax.set_xlabel("Normalized depth (layer / number of layers)")
        ax.set_xlim(-0.015, 1.0)
        ax.set_ylim(-0.02, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Interchange Intervention Accuracy (IIA)")
    fig.suptitle(r"Polarity relation ($\rho$) across normalized model depth", fontsize=16)

    legend_handles = []
    for spec, _ in models:
        for metric in ("identified", "active", "inactive"):
            style = METRIC_STYLES[metric]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=spec.colors[metric],
                    linewidth=style["linewidth"],
                    label=f"{spec.name} - {style['label']}",
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
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
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
            "claim_final",
            {"identified": "#E85D04", "active": "#F3A261", "inactive": "#FAD7B5"},
        ),
        ModelSpec(
            "Phi-4 Mini Instruct",
            args.phi4_csv,
            args.phi4_layers,
            "row",
            {"identified": "#1D4ED8", "active": "#7AA7E8", "inactive": "#C7DCF7"},
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
