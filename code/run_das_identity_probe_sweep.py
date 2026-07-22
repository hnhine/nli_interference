"""Eval-only identification of trained DAS subspaces as raw pi, raw pc, or REL."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

from interference_suite.das_data import high_level_label
from interference_suite.das_pyvene import (
    build_intervenable,
    evaluate_pyvene_das,
    get_input_device,
    import_runtime,
    load_hf_model,
    mean,
    set_intervenable_device,
    to_jsonable,
)
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, resolve_label_tokens
from run_das_handle_probe import load_rotation


HYPOTHESES = ("pi", "pc", "rel", "noop", "label")
DIAGNOSTIC_CONTROLS = {
    "pc": {
        "probe_flip_both": "flip_both",
        "probe_flip_pi": "flip_pi",
    },
    "pi": {
        "main": "flip_pi",
        "probe_flip_both": "flip_both",
        "probe_flip_pc": "flip_pc",
    },
}
GATE_CONTROLS = ("gate_m0", "label_copy_trap")


def as_int(row: dict[str, Any], column: str) -> int:
    value = row.get(column)
    if value in (None, ""):
        raise ValueError(f"Missing {column!r} on sample {row.get('sample_id')!r}")
    return int(float(str(value)))


def annotate_hypotheses(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    m_base = as_int(row, "m_base")
    p_i_base = as_int(row, "p_i_base")
    p_i_src = as_int(row, "p_i_src")
    p_c_base = as_int(row, "p_c_base")
    p_c_src = as_int(row, "p_c_src")
    source_rel = p_i_src * p_c_src
    result["pred_H_pi"] = high_level_label(m_base, p_i_src, p_c_base)
    result["pred_H_pc"] = high_level_label(m_base, p_i_base, p_c_src)
    result["pred_H_rel"] = "U" if m_base == 0 else ("T" if source_rel == 1 else "F")
    result["pred_H_noop"] = str(row["base_label"])
    result["pred_H_label"] = str(row["source_label"])
    return result


def load_identity_rows(path: str, target_var: str, split: str) -> list[dict[str, Any]]:
    controls = DIAGNOSTIC_CONTROLS[target_var]
    selected = set(controls) | set(GATE_CONTROLS)
    rows: list[dict[str, Any]] = []
    for row in read_rows_csv(path):
        if row.get("target_var") != target_var or row.get("split") != split:
            continue
        control = str(row.get("control_type", ""))
        if control not in selected:
            continue
        annotated = annotate_hypotheses(row)
        annotated["identity_condition"] = controls.get(control, control)
        annotated["target_label"] = annotated["pred_H_rel"]
        rows.append(annotated)
    missing = sorted(set(controls) - {row["control_type"] for row in rows})
    if missing:
        raise ValueError(f"Probe samples lack required {target_var} controls: {missing}")
    if not rows:
        raise ValueError(f"No identity rows found for target_var={target_var!r}, split={split!r}")
    return rows


def summarize_identity(scored: list[dict[str, Any]], target_var: str) -> dict[str, Any]:
    conditions = tuple(DIAGNOSTIC_CONTROLS[target_var].values())
    by_condition: dict[str, Any] = {}
    for condition in (*conditions, *GATE_CONTROLS):
        values = [row for row in scored if row["identity_condition"] == condition]
        if not values:
            continue
        stats: dict[str, Any] = {
            "n": len(values),
            "change_rate": mean(row["pred_label"] != row["base_label"] for row in values),
            "U_rate": mean(row["pred_label"] == "U" for row in values),
            "global_top_in_TFU_rate": mean(int(row.get("global_top_in_TFU", 1)) for row in values),
        }
        for hypothesis in HYPOTHESES:
            stats[f"match_H_{hypothesis}"] = mean(
                row["pred_label"] == row[f"pred_H_{hypothesis}"] for row in values
            )
        by_condition[condition] = stats

    macro: dict[str, float] = {}
    strict_min: dict[str, float] = {}
    for hypothesis in HYPOTHESES:
        scores = [by_condition[condition][f"match_H_{hypothesis}"] for condition in conditions]
        macro[hypothesis] = mean(scores)
        strict_min[hypothesis] = min(scores)
    ranked = sorted((macro[name], name) for name in ("pi", "pc", "rel"))
    return {
        "conditions": by_condition,
        "macro": macro,
        "strict_min": strict_min,
        "identity_winner": ranked[-1][1],
        "identity_margin": ranked[-1][0] - ranked[-2][0],
    }


def flatten_record(rotation_dir: Path, metadata: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "cell": rotation_dir.name,
        "layer": int(metadata["layer"]),
        "site": str(metadata.get("site", "claim_final")),
        "rank": int(metadata["rank"]),
        "identity_winner": summary["identity_winner"],
        "identity_margin": summary["identity_margin"],
    }
    for hypothesis in HYPOTHESES:
        record[f"H_{hypothesis}_macro"] = summary["macro"][hypothesis]
        record[f"H_{hypothesis}_strict_min"] = summary["strict_min"][hypothesis]
    for condition, stats in summary["conditions"].items():
        for key, value in stats.items():
            if key != "n":
                record[f"{condition}_{key}"] = value
    return record


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = build_parser().parse_args()
    rows = load_identity_rows(args.probe_samples, args.target_var, args.split)
    print(f"Loaded {len(rows)} identity rows for {args.target_var}/{args.split}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    pending: list[Path] = []
    for raw_dir in args.rotation_dirs:
        rotation_dir = Path(raw_dir)
        metadata_path = rotation_dir / "rotation_weight_metadata.json"
        if not metadata_path.exists():
            if args.skip_missing:
                print(f"Skip missing rotation: {rotation_dir}")
                continue
            raise FileNotFoundError(metadata_path)
        cached_path = output_dir / f"{rotation_dir.name}.json"
        if args.resume and cached_path.exists():
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            records.append(cached["record"])
            print(f"Skip completed identity cell: {rotation_dir.name}")
        else:
            pending.append(rotation_dir)

    if pending:
        torch, pv, amc, atc = import_runtime()
        tokenizer, model = load_hf_model(
            torch=torch,
            auto_model_cls=amc,
            auto_tokenizer_cls=atc,
            model_name=args.model_name,
            device=args.device,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
        input_device = get_input_device(model, torch, args.device)

        for rotation_dir in pending:
            metadata = json.loads((rotation_dir / "rotation_weight_metadata.json").read_text(encoding="utf-8"))
            layer = int(metadata["layer"])
            rank = int(metadata["rank"])
            site = str(metadata.get("site", "claim_final"))
            component = str(metadata.get("component", "block_output"))
            print(f"\n=== identity {rotation_dir.name}: L{layer}/{site}, r={rank} ===")
            intervenable = build_intervenable(pv, model, layer=layer, rank=rank, component=component)
            set_intervenable_device(intervenable, input_device)
            load_rotation(intervenable, torch, rotation_dir)
            _, scored = evaluate_pyvene_das(
                intervenable,
                rows,
                tokenizer,
                torch,
                input_device,
                label_tokens.token_ids,
                args.eval_batch_size,
                site,
                "identity probes",
            )
            identity = summarize_identity(scored, args.target_var)
            record = flatten_record(rotation_dir, metadata, identity)
            records.append(record)
            payload = {"record": record, "identity": identity}
            (output_dir / f"{rotation_dir.name}.json").write_text(
                json.dumps(to_jsonable(payload), indent=2), encoding="utf-8"
            )
            if args.save_scored:
                write_rows_csv(scored, output_dir / f"{rotation_dir.name}_scored.csv")
            write_csv(records, output_dir / "identity_sweep.csv")
            print(
                f"winner={record['identity_winner']} margin={record['identity_margin']:.3f} "
                f"H_pi={record['H_pi_macro']:.3f} H_pc={record['H_pc_macro']:.3f} "
                f"H_REL={record['H_rel_macro']:.3f}"
            )
            del intervenable
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if records:
        records.sort(key=lambda record: (int(record["layer"]), str(record["site"])))
        write_csv(records, output_dir / "identity_sweep.csv")
    print(f"\nWrote {len(records)} cells to {output_dir / 'identity_sweep.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eval-only raw-pi/raw-pc/REL identity sweep.")
    parser.add_argument("--probe-samples", required=True)
    parser.add_argument("--target-var", required=True, choices=["pi", "pc"])
    parser.add_argument("--rotation-dirs", nargs="+", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--label-token-style", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--save-scored", action="store_true")
    parser.add_argument("--output-dir", default="data/das/identity_probe_sweep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
