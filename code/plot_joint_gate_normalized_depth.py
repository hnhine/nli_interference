"""Plot joint gate (m, rho) compositional control across normalized depth.

For each model the sweep directory holds <site>/L<layer>/joint_gate_summary.csv.
Joint IIA is the mean over cells of the joint_constrained condition; purity is
the mean of joint_same_value.  Phi-4 uses the held-out test150 sweep; Qwen uses
the joint_gate_sweep_qwen directory (claim_final coverage starts at L18).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    sweep_dir: Path
    num_layers: int
    colors: dict[str, str]


SITES = ("claim_final", "answer_token")
SITE_TITLES = {
    "claim_final": "Claim-final site",
    "answer_token": "Answer-token site",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot joint gate IIA against normalized model depth."
    )
    parser.add_argument(
        "--qwen-dir",
        type=Path,
        default=Path("data/das/joint_gate_sweep_qwen"),
    )
    parser.add_argument(
        "--phi4-dir",
        type=Path,
        default=Path("data/das/joint_gate_test150_sweep_phi4"),
    )
    parser.add_argument("--qwen-layers", type=int, default=36)
    parser.add_argument("--phi4-layers", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/das/joint_gate_normalized_depth_qwen_phi4.png"),
    )
    parser.add_argument(
        "--normalized-csv",
        type=Path,
        default=Path("data/das/joint_gate_normalized_depth_qwen_phi4.csv"),
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def summary_mean(path: Path, condition: str) -> float | None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("condition") == condition and row.get("scope") == "all"
        ]
    if not rows:
        return None
    return sum(float(row["IIA"]) for row in rows) / len(rows)


def load_model(spec: ModelSpec) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for site in SITES:
        for summary in sorted(spec.sweep_dir.glob(f"{site}/L*/joint_gate_summary.csv")):
            layer = int(summary.parent.name.lstrip("L"))
            joint = summary_mean(summary, "joint_constrained")
            purity = summary_mean(summary, "joint_same_value")
            if joint is None:
                continue
            rows.append(
                {
                    "model": spec.name,
                    "num_layers": spec.num_layers,
                    "layer": layer,
                    "normalized_depth": layer / spec.num_layers,
                    "site": site,
                    "joint_iia": joint,
                    "purity": purity if purity is not None else "",
                }
            )
    if not rows:
        raise ValueError(f"No joint_gate_summary.csv found under {spec.sweep_dir}")
    return sorted(rows, key=lambda row: (str(row["site"]), int(row["layer"])))


def write_normalized_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "num_layers",
        "layer",
        "normalized_depth",
        "site",
        "joint_iia",
        "purity",
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
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharex=True, sharey=True)

    for ax, (spec, rows) in zip(axes, models):
        for site in SITES:
            site_rows = [row for row in rows if row["site"] == site]
            x = [float(row["normalized_depth"]) for row in site_rows]
            ax.plot(
                x,
                [float(row["joint_iia"]) for row in site_rows],
                color=spec.colors[site],
                linewidth=2.6,
                alpha=0.97,
            )
            purity_pairs = [
                (float(row["normalized_depth"]), float(row["purity"]))
                for row in site_rows
                if row["purity"] != ""
            ]
            if purity_pairs:
                ax.plot(
                    [pair[0] for pair in purity_pairs],
                    [pair[1] for pair in purity_pairs],
                    color=spec.colors[site],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.85,
                )

        ax.axhline(1.0 / 3.0, color="#8C8C8C", linewidth=1.1, linestyle=":")
        ax.set_title(spec.name)
        ax.set_xlabel("Normalized depth (layer / number of layers)")
        ax.set_xlim(-0.015, 1.0)
        ax.set_ylim(-0.02, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Joint interchange accuracy")
    fig.suptitle(
        r"Joint $(m,\rho)$ gate control across normalized model depth", fontsize=16
    )

    legend_handles = []
    for spec, _ in models:
        for site in SITES:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=spec.colors[site],
                    linewidth=2.6,
                    label=f"{spec.name} — {SITE_TITLES[site]} (joint)",
                )
            )
    legend_handles.append(
        Line2D([0], [0], color="#555555", linewidth=1.5, linestyle="--", label="Same-value purity")
    )
    legend_handles.append(
        Line2D([0], [0], color="#8C8C8C", linewidth=1.1, linestyle=":", label="Chance (3 labels)")
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        columnspacing=2.4,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    specs = [
        ModelSpec(
            "Qwen3-8B",
            args.qwen_dir,
            args.qwen_layers,
            {"claim_final": "#E85D04", "answer_token": "#F3A261"},
        ),
        ModelSpec(
            "Phi-4 Mini Instruct",
            args.phi4_dir,
            args.phi4_layers,
            {"claim_final": "#1D4ED8", "answer_token": "#7AA7E8"},
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
