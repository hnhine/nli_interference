"""Offline breakdown of m relay/interchange scored CSVs.

The DAS trainer's ``test_scored.csv`` already contains pair metadata.  The
baseline scorer is intentionally compact, so this script joins it to the
original pairs by ``sample_id`` (and ``side`` when present) before grouping.
No model is loaded.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from interference_suite.io_utils import read_rows_csv
from interference_suite.das_pyvene import mean


DEFAULT_GROUPS = ("control_type", "mismatch_type", "matched_idx", "m_base")


def infer_mode(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if "baseline" in path.name:
        return "baseline"
    if "ablation" in path.name:
        return "ablation"
    return "interchange"


def enrich(scored: list[dict[str, Any]], sample_rows: dict[str, dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in scored:
        sample = sample_rows.get(str(row.get("sample_id")), {})
        merged = dict(sample)
        merged.update(row)
        for key, value in sample.items():
            if merged.get(key) in (None, ""):
                merged[key] = value
        condition = str(row.get("condition", "none"))
        base_label = str(merged.get("base_label", row.get("true_label", "")))
        target_label = str(merged.get("target_label", ""))
        if mode == "baseline":
            side = str(row.get("side", "base"))
            expected = str(sample.get(f"{side}_label", row.get("true_label", "")))
            if side == "source":
                for stem in ("m", "p_i", "p_c"):
                    source_key = f"{stem}_src"
                    if sample.get(source_key) not in (None, ""):
                        merged[f"{stem}_base"] = sample[source_key]
        elif mode == "ablation" and condition == "das_resample_opposite":
            expected = target_label or base_label
        elif mode == "interchange":
            expected = target_label or base_label
        else:
            expected = base_label
        merged["_mode"] = mode
        merged["_condition"] = condition
        merged["_expected"] = expected
        merged["_correct"] = int(str(row.get("pred_label", "")) == expected)
        merged["_counterfactual_correct"] = int(bool(target_label) and str(row.get("pred_label", "")) == target_label)
        out.append(merged)
    return out


def summarize(rows: list[dict[str, Any]], groups: tuple[str, ...], source: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(group, "")) for group in groups)
        buckets[key].append(row)
    records: list[dict[str, Any]] = []
    for key, values in sorted(buckets.items()):
        preds = [str(row.get("pred_label", "")) for row in values]
        record: dict[str, Any] = {
            "source": source,
            "mode": values[0]["_mode"],
            "condition": values[0]["_condition"],
            **{group: value for group, value in zip(groups, key)},
            "n": len(values),
            "accuracy": mean(bool(row["_correct"]) for row in values),
            "counterfactual_acc": mean(bool(row["_counterfactual_correct"]) for row in values),
            "U_rate": mean(pred == "U" for pred in preds),
            "T_rate": mean(pred == "T" for pred in preds),
            "F_rate": mean(pred == "F" for pred in preds),
        }
        for metric in ("R", "U_gap"):
            nums = [float(row[metric]) for row in values if row.get(metric) not in (None, "")]
            if nums:
                record[f"mean_{metric}"] = sum(nums) / len(nums)
        records.append(record)
    return records


def main() -> int:
    args = build_parser().parse_args()
    sample_rows = {str(row["sample_id"]): row for row in read_rows_csv(args.samples)}
    groups = tuple(x.strip() for x in args.group_by.split(",") if x.strip())
    records: list[dict[str, Any]] = []
    for filename in args.scored:
        path = Path(filename)
        scored = read_rows_csv(path)
        mode = infer_mode(path, args.mode)
        enriched = enrich(scored, sample_rows, mode)
        records.extend(summarize(enriched, groups, str(path)))
    if not records:
        raise ValueError("No scored rows found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    print(f"Wrote {len(records)} grouped records to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline m breakdown by direction/type/slot.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--scored", nargs="+", required=True)
    parser.add_argument("--mode", choices=["auto", "baseline", "interchange", "ablation"], default="auto")
    parser.add_argument("--group-by", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--output", default="data/das/m_breakdown.csv")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
