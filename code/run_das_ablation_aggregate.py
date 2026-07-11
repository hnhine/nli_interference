"""Aggregate full metrics from existing ablation outputs (no GPU, no re-runs).

The sweep CSVs only carried resample accuracy; the per-cell JSONs already hold
all five conditions with accuracy, mean_R and prediction distributions. This
reads every {cell}.json in one or more ablation output dirs and writes a wide
CSV: per condition x control -> accuracy, mean_R, U-rate, plus derived deltas.

Example:
    python code/run_das_ablation_aggregate.py \
        --dirs data/das/ablation_forced_qwen data/das/ablation_granite_full \
               data/das/ablation_pi_r16_qwen data/das/ablation_stride2_qwen \
        --output data/das/ablation_metrics_full.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONDITIONS = (
    "none",
    "das_zero",
    "das_resample",
    "das_resample_same",
    "das_resample_opposite",
    "rand_zero",
    "rand_resample",
)


def main() -> int:
    args = build_parser().parse_args()
    records = []
    for d in args.dirs:
        for jf in sorted(Path(d).glob("L*.json")):
            summary = json.loads(jf.read_text())
            if "none" not in summary:
                continue
            rec = {
                "source_dir": d,
                "cell": jf.stem,
                "layer": summary.get("layer"),
                "site": summary.get("site"),
                "rank": summary.get("rank"),
                "condition_on": summary.get("condition_on"),
            }
            controls = sorted(summary["none"].get("by_control", {}))
            for cond in CONDITIONS:
                cond_summary = summary.get(cond)
                if not cond_summary:
                    continue
                for ctrl in controls:
                    stats = cond_summary["by_control"].get(ctrl)
                    if not stats:
                        continue
                    key = f"{cond}.{ctrl}"
                    rec[f"{key}.acc"] = round(stats["accuracy"], 4)
                    rec[f"{key}.mean_R"] = round(stats["mean_R"], 3)
                    rec[f"{key}.U_rate"] = round(stats["pred_dist"].get("U", 0.0), 4)
            # derived deltas on main
            for cond in (
                "das_zero",
                "das_resample",
                "das_resample_same",
                "das_resample_opposite",
                "rand_resample",
            ):
                a = rec.get(f"{cond}.main.acc")
                b = rec.get("none.main.acc")
                r_c, r_n = rec.get(f"{cond}.main.mean_R"), rec.get("none.main.mean_R")
                if a is not None and b is not None:
                    rec[f"delta_acc.main.{cond}"] = round(b - a, 4)
                if r_c is not None and r_n is not None:
                    rec[f"delta_meanR.main.{cond}"] = round(r_n - r_c, 3)
            same = rec.get("das_resample_same.main.acc")
            opposite = rec.get("das_resample_opposite.main.acc")
            baseline = rec.get("none.main.acc")
            if same is not None and baseline is not None:
                rec["purity_drop"] = round(baseline - same, 4)
            if opposite is not None:
                rec["backup_rescue"] = round(opposite, 4)
            records.append(rec)

    if not records:
        raise SystemExit("No cell JSONs found in the given dirs")
    fieldnames: list[str] = []
    for rec in records:
        for k in rec:
            if k not in fieldnames:
                fieldnames.append(k)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"Wrote {len(records)} cells x {len(fieldnames)} columns to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate ablation cell JSONs into a wide metrics CSV.")
    p.add_argument("--dirs", nargs="+", required=True, help="Ablation output dirs containing L*.json")
    p.add_argument("--output", default="data/das/ablation_metrics_full.csv")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
