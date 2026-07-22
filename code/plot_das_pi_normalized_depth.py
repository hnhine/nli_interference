"""Compare p_i DAS regimes across models on a normalized depth axis.

The relay-map CSV produced by older Qwen runs does not contain the explicit
``pi_*_IIA`` columns.  Those regimes can be reconstructed from the underlying
control groups using the same definitions as the newer summaries:

    active   = mean(main, active_source_m0)
    inactive = mean(gate_m0, label_copy_trap)
    locality = distractor
    macro    = mean(active, inactive, locality)

By default, normalized depth is layer / num_layers, matching the convention
used in the accompanying cross-model report.
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
    linestyle: str
    marker: str | None
    colors: dict[str, str]


METRIC_STYLES = {
    "macro": {"label": "Macro", "linewidth": 2.0},
    "active": {"label": "Active (core)", "linewidth": 2.8},
    "inactive": {"label": "Inactive", "linewidth": 1.7},
}
PLOT_ORDER = ("inactive", "active", "macro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Qwen and Phi-4 p_i IIA against normalized model depth."
    )
    parser.add_argument(
        "--qwen-csv",
        type=Path,
        default=Path("data/das/qwen_pi_v4_r16_stride2_1ep/relay_map.csv"),
    )
    parser.add_argument(
        "--phi4-csv",
        type=Path,
        default=Path("data/das/phi4_pi_v4_r64_stride2_1ep_b20/relay_map.csv"),
    )
    parser.add_argument("--qwen-layers", type=int, default=36)
    parser.add_argument("--phi4-layers", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/das/pi_normalized_depth_qwen_phi4.png"),
    )
    parser.add_argument(
        "--normalized-csv",
        type=Path,
        default=Path("data/das/pi_normalized_depth_qwen_phi4.csv"),
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return float(value) if value else None


def available_mean(row: dict[str, str], keys: tuple[str, ...]) -> float:
    values = [value for key in keys if (value := optional_float(row, key)) is not None]
    if len(values) != len(keys):
        missing = [key for key in keys if optional_float(row, key) is None]
        raise ValueError(f"Missing columns/values needed for p_i regime: {missing}")
    return mean(values)


def regime_value(
    row: dict[str, str], explicit_key: str, fallback_keys: tuple[str, ...]
) -> float:
    explicit = optional_float(row, explicit_key)
    reconstructed = available_mean(row, fallback_keys)
    if explicit is not None and abs(explicit - reconstructed) > 1e-9:
        raise ValueError(
            f"{explicit_key}={explicit} disagrees with reconstructed value "
            f"{reconstructed} at layer {row.get('layer')}"
        )
    return reconstructed


def load_model(spec: ModelSpec) -> list[dict[str, float | int | str]]:
    with spec.csv_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, float | int | str]] = []
    for raw in raw_rows:
        if raw.get("site") != "row":
            continue
        layer = int(raw["layer"])
        active = regime_value(
            raw,
            "pi_active_IIA",
            ("main_IIA", "active_source_m0_IIA"),
        )
        inactive = regime_value(
            raw,
            "pi_inactive_IIA",
            ("gate_m0_IIA", "label_copy_trap_IIA"),
        )
        locality = regime_value(raw, "pi_locality_IIA", ("distractor_IIA",))
        macro = mean((active, inactive, locality))
        explicit_macro = optional_float(raw, "pi_regime_macro_IIA")
        if explicit_macro is not None and abs(explicit_macro - macro) > 1e-9:
            raise ValueError(
                f"pi_regime_macro_IIA={explicit_macro} disagrees with reconstructed "
                f"value {macro} at layer {layer}"
            )
        rows.append(
            {
                "model": spec.name,
                "num_layers": spec.num_layers,
                "layer": layer,
                "normalized_depth": layer / spec.num_layers,
                "active": active,
                "inactive": inactive,
                "locality": locality,
                "macro": macro,
            }
        )

    if not rows:
        raise ValueError(f"No site='row' records found in {spec.csv_path}")
    return sorted(rows, key=lambda row: int(row["layer"]))


def write_normalized_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "num_layers",
        "layer",
        "normalized_depth",
        "active",
        "inactive",
        "locality",
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
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 7.0))

    for spec, rows in models:
        x = [float(row["normalized_depth"]) for row in rows]
        for metric in PLOT_ORDER:
            style = METRIC_STYLES[metric]
            y = [float(row[metric]) for row in rows]
            ax.plot(
                x,
                y,
                color=spec.colors[metric],
                linestyle=spec.linestyle,
                marker=spec.marker,
                markersize=4.8,
                linewidth=style["linewidth"],
                markerfacecolor="white",
                markeredgewidth=1.2,
                alpha=0.95,
            )

    ax.set_title(r"Assumption polarity ($p_i$) across normalized model depth")
    ax.set_xlabel(r"Normalized depth (layer / number of layers)")
    ax.set_ylabel("Interchange Intervention Accuracy (IIA)")
    ax.set_xlim(-0.015, 1.0)
    ax.set_ylim(-0.02, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = []
    for spec, _ in models:
        for metric, style in METRIC_STYLES.items():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=spec.colors[metric],
                    linestyle=spec.linestyle,
                    marker=spec.marker,
                    markerfacecolor="white",
                    linewidth=style["linewidth"],
                    label=f"{spec.name} — {style['label']}",
                )
            )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        columnspacing=2.4,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
            "-",
            None,
            {"macro": "#F3A261", "active": "#E85D04", "inactive": "#FAD7B5"},
        ),
        ModelSpec(
            "Phi-4 Mini",
            args.phi4_csv,
            args.phi4_layers,
            "-",
            None,
            {"macro": "#7AA7E8", "active": "#1D4ED8", "inactive": "#C7DCF7"},
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
