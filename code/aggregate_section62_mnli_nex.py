"""Aggregate Section 6.2 MNLI resample NEx across DAS training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from run_section62_ablation import write_csv_atomic, write_json_atomic


FIELDS = (
    "base_accuracy",
    "learned_accuracy",
    "random_accuracy_mean",
    "NEx_accuracy",
    "NEx_accuracy_pp",
    "base_M_TF",
    "learned_M_TF",
    "random_M_TF_mean",
    "NEx_M_TF",
    "base_gold_margin_3",
    "learned_gold_margin_3",
    "random_gold_margin_3_mean",
    "NEx_gold_margin_3",
)


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.input_root).resolve()
    paths = sorted(root.glob("seed*/L*_claim_final/nex_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No seed summaries under {root}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    contracts = {
        (
            run["model_name"],
            run["site"],
            int(run["layer"]),
            int(run["rank"]),
            tuple(run["random_seeds"]),
            int(run["donor_audit"]["seed"]),
        )
        for run in runs
    }
    if len(contracts) != 1:
        raise ValueError(f"Incompatible NEx runs: {sorted(contracts)}")
    rotation_seeds = [int(run["rotation_training_seed"]) for run in runs]
    if len(rotation_seeds) != len(set(rotation_seeds)):
        raise ValueError(f"Duplicate rotation seeds: {rotation_seeds}")

    scopes = ("overall", "same_rho_donor", "opposite_rho_donor")
    rows = []
    nested: dict[str, Any] = {}
    for scope in scopes:
        row: dict[str, Any] = {
            "scope": scope,
            "n_rotation_seeds": len(runs),
            "rotation_seeds": ",".join(str(seed) for seed in rotation_seeds),
            "n_rows_per_seed": int(runs[0]["by_scope"][scope]["n"]),
        }
        nested[scope] = {}
        for field in FIELDS:
            values = [float(run["by_scope"][scope][field]) for run in runs]
            stats = aggregate(values)
            nested[scope][field] = {"by_rotation_seed": values, **stats}
            row[f"{field}_mean"] = stats["mean"]
            row[f"{field}_sd"] = stats["sd"]
        rows.append(row)

    model_name, site, layer, rank, random_seeds, donor_seed = next(iter(contracts))
    summary = {
        "schema_version": 1,
        "run_type": "section62_mnli_resample_nex_aggregate",
        "model_name": model_name,
        "site": site,
        "layer": layer,
        "rank": rank,
        "rotation_seeds": rotation_seeds,
        "random_seeds": list(random_seeds),
        "donor_seed": donor_seed,
        "source_summaries": [str(path) for path in paths],
        "by_scope": nested,
    }
    output = Path(args.output_dir).resolve() if args.output_dir else root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(rows, output / "aggregate_by_scope.csv")
    write_json_atomic(summary, output / "aggregate_summary.json")
    overall = nested["overall"]
    print(f"Wrote MNLI NEx aggregate to {output}")
    print(
        "NEx accuracy = "
        f"{overall['NEx_accuracy_pp']['mean']:+.2f} +/- "
        f"{overall['NEx_accuracy_pp']['sd']:.2f} pp "
        f"over {len(runs)} rotation seeds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
