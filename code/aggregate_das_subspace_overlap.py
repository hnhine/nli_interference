"""Aggregate same-layer m/rho overlap audits across DAS training seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


SUMMARY_FIELDS = (
    "frobenius_squared",
    "normalized_frobenius_squared",
    "overlap_to_random_expectation_ratio",
)
ANGLE_FIELDS = (
    "principal_cosine",
    "principal_cosine_squared",
    "principal_angle_degrees",
)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.input_root).resolve()
    summary_paths = sorted(root.glob("**/overlap_summary.json"))
    summary_paths = [path for path in summary_paths if "aggregate" not in path.parts]
    if not summary_paths:
        raise FileNotFoundError(f"No overlap summaries under {root}")

    grouped: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        key = (
            summary["model_name"],
            int(summary["layer"]),
            summary["site"],
            int(summary["rank"]),
            int(summary["hidden_size"]),
        )
        grouped[key].append((path, summary))

    aggregate_rows = []
    spectrum_rows = []
    nested: dict[str, Any] = {}
    for key, runs in sorted(grouped.items()):
        model, layer, site, rank, hidden = key
        seeds = [int(summary["training_seed"]) for _, summary in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate training seeds for {key}: {seeds}")
        label = f"{model}|L{layer}|{site}"
        row: dict[str, Any] = {
            "model_name": model,
            "layer": layer,
            "site": site,
            "rank": rank,
            "hidden_size": hidden,
            "n_training_seeds": len(runs),
            "training_seeds": ",".join(str(seed) for seed in seeds),
            "random_subspace_expectation_k_over_d": rank / hidden,
        }
        nested[label] = {
            "model_name": model,
            "layer": layer,
            "site": site,
            "rank": rank,
            "hidden_size": hidden,
            "training_seeds": seeds,
        }
        for field in SUMMARY_FIELDS:
            values = [float(summary[field]) for _, summary in runs]
            result = stats(values)
            nested[label][field] = {"by_seed": values, **result}
            for statistic, value in result.items():
                row[f"{field}_{statistic}"] = value
        aggregate_rows.append(row)

        spectra = [read_csv(path.parent / "principal_angles.csv") for path, _ in runs]
        if any(len(spectrum) != rank for spectrum in spectra):
            raise ValueError(f"Principal-angle spectrum length mismatch for {key}")
        for index in range(rank):
            angle_row: dict[str, Any] = {
                "model_name": model,
                "layer": layer,
                "site": site,
                "rank": rank,
                "hidden_size": hidden,
                "principal_index": index + 1,
                "n_training_seeds": len(runs),
            }
            for field in ANGLE_FIELDS:
                values = [float(spectrum[index][field]) for spectrum in spectra]
                result = stats(values)
                for statistic, value in result.items():
                    angle_row[f"{field}_{statistic}"] = value
            spectrum_rows.append(angle_row)

    output = Path(args.output_dir).resolve() if args.output_dir else root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "overlap_by_layer_site.csv", aggregate_rows)
    write_csv(output / "principal_angle_spectrum.csv", spectrum_rows)
    (output / "overlap_aggregate.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_type": "das_same_layer_m_rho_overlap_aggregate",
                "by_layer_site": nested,
                "interpretation_scope": (
                    "Low overlap argues against direct linear m/rho leakage only; "
                    "it does not address a shared downstream T/F decision axis."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote overlap aggregate to {output}")
    for row in aggregate_rows:
        print(
            f"{row['model_name']} L{row['layer']}/{row['site']}: "
            f"normalized overlap={row['normalized_frobenius_squared_mean']:.6f} "
            f"+/- {row['normalized_frobenius_squared_sd']:.6f}, "
            f"random={row['random_subspace_expectation_k_over_d']:.6f}, "
            f"ratio={row['overlap_to_random_expectation_ratio_mean']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
