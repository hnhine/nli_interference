"""Compare matched M-v2 anti-copy DAS sweeps on normalized model depth.

Both source runs use rank 16 and the same three control types.  The plotted
metrics are reconstructed as:

    active*   = mean(match_to_nomatch, nomatch_to_match)
    anti-copy = label_copy_trap
    macro     = mean(match_to_nomatch, nomatch_to_match, label_copy_trap)

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
    colors: dict[str, str]


METRIC_STYLES = {
    "active": {"label": "Active*", "linewidth": 2.8},
    "macro": {"label": "Macro", "linewidth": 2.0},
    "anti_copy": {"label": "Anti-copy control", "linewidth": 1.7},
}
PLOT_ORDER = ("anti_copy", "macro", "active")
SITES = ("claim_final", "answer_token")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Qwen and Phi-4 event-claim match IIA against normalized depth."
    )
    parser.add_argument(
        "--qwen-csv",
        type=Path,
        default=Path("data/das/qwen_m_v2_anticopy_relay_r16_stride2/relay_map.csv"),
    )
    parser.add_argument(
        "--phi4-csv",
        type=Path,
        default=Path("data/das/phi4_m_v2_anticopy_relay_r16_stride2/relay_map.csv"),
    )
    parser.add_argument("--qwen-layers", type=int, default=36)
    parser.add_argument("--phi4-layers", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/das/m_normalized_depth_qwen_phi4.png"),
    )
    parser.add_argument(
        "--normalized-csv",
        type=Path,
        default=Path("data/das/m_normalized_depth_qwen_phi4.csv"),
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
        match_to_nomatch = required_float(raw, "match_to_nomatch_IIA")
        nomatch_to_match = required_float(raw, "nomatch_to_match_IIA")
        anti_copy = required_float(raw, "label_copy_trap_IIA")
        active = mean((match_to_nomatch, nomatch_to_match))
        macro = mean((match_to_nomatch, nomatch_to_match, anti_copy))
        recorded_core = required_float(raw, "m_core_IIA")
        if abs(recorded_core - active) > 1e-9:
            raise ValueError(
                f"m_core_IIA={recorded_core} disagrees with reconstructed active={active} "
                f"at layer={raw.get('layer')} site={site}"
            )
        layer = int(raw["layer"])
        rows.append(
            {
                "model": spec.name,
                "num_layers": spec.num_layers,
                "layer": layer,
                "normalized_depth": layer / spec.num_layers,
                "site": site,
                "active": active,
                "anti_copy": anti_copy,
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
        "active",
        "anti_copy",
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
    fig.suptitle(r"Event–claim match ($m$) across normalized model depth", fontsize=16)

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
            "Qwen3-8B (M-v2, r=16)",
            args.qwen_csv,
            args.qwen_layers,
            {"macro": "#F3A261", "active": "#E85D04", "anti_copy": "#FAD7B5"},
        ),
        ModelSpec(
            "Phi-4 Mini (M-v2, r=16)",
            args.phi4_csv,
            args.phi4_layers,
            {"macro": "#7AA7E8", "active": "#1D4ED8", "anti_copy": "#C7DCF7"},
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
