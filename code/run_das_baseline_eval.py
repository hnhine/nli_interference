"""Score DAS pair prompts WITHOUT any intervention.

Two uses:
  1. Task accuracy of a model on the DAS prompt distribution (pred vs
     base_label / source_label) — e.g. to vet a new model before DAS.
  2. No-intervention baseline logits for Delta metrics: join baseline_scored.csv
     with an intervened *_scored.csv on sample_id and compute
     delta_R = R_intervened - R_baseline, delta_U_gap likewise.

Example:
    python code/run_das_baseline_eval.py --samples data/das/pc_1000_v2/pairs.csv \
        --model-name ibm-granite/granite-4.1-8b --split test --local-files-only \
        --output-dir data/das/baseline_granite41_pc_test
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Any

from interference_suite.das_pyvene import (
    encode_to_device,
    import_runtime,
    load_hf_model,
    mean,
    to_jsonable,
)
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, resolve_label_tokens


def score_side(rows: list[dict[str, Any]], side: str, model: Any, tokenizer: Any, torch: Any, device: Any,
               label_token_ids: dict[str, int], batch_size: int) -> list[dict[str, Any]]:
    from interference_suite.model import progress_iter

    prompt_key = f"{side}_prompt"
    label_key = f"{side}_label"
    label_id_set = set(label_token_ids.values())
    scored: list[dict[str, Any]] = []
    batches = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]
    for batch_rows in progress_iter(batches, total=ceil(len(rows) / batch_size), desc=f"Baseline {side}"):
        texts = [str(row[prompt_key]) for row in batch_rows]
        inputs = encode_to_device(tokenizer, texts, device)
        with torch.no_grad():
            logits = model(**inputs).logits
        final_idx = inputs["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        next_logits = logits[batch_idx, final_idx]
        top_ids = next_logits.argmax(dim=-1)
        for row, row_logits, top_id in zip(batch_rows, next_logits, top_ids):
            values = {label: float(row_logits[tid].detach().cpu()) for label, tid in label_token_ids.items()}
            pred = max(values, key=values.get)
            top_token_id = int(top_id.detach().cpu())
            true_label = str(row.get(label_key, ""))
            scored.append(
                {
                    "sample_id": row.get("sample_id"),
                    "side": side,
                    "split": row.get("split"),
                    "control_type": row.get("control_type"),
                    "target_var": row.get("target_var"),
                    "true_label": true_label,
                    "pred_label": pred,
                    "is_correct": int(pred == true_label),
                    "logit_T": values["T"],
                    "logit_F": values["F"],
                    "logit_U": values["U"],
                    "R": values["T"] - values["F"],
                    "U_gap": values["U"] - max(values["T"], values["F"]),
                    "global_top_token_id": top_token_id,
                    "global_top_token": tokenizer.decode([top_token_id]),
                    "global_top_in_TFU": int(top_token_id in label_id_set),
                }
            )
    return scored


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in sorted({row["side"] for row in scored}):
        side_rows = [row for row in scored if row["side"] == side]
        by_control: dict[str, Any] = {}
        for control in sorted({row["control_type"] for row in side_rows}):
            control_rows = [row for row in side_rows if row["control_type"] == control]
            by_control[control] = {
                "n": len(control_rows),
                "accuracy": mean(float(row["is_correct"]) for row in control_rows),
                "mean_R": mean(float(row["R"]) for row in control_rows),
                "mean_U_gap": mean(float(row["U_gap"]) for row in control_rows),
                "global_top_in_TFU_rate": mean(float(row["global_top_in_TFU"]) for row in control_rows),
            }
        out[side] = {
            "n": len(side_rows),
            "accuracy": mean(float(row["is_correct"]) for row in side_rows),
            "global_top_in_TFU_rate": mean(float(row["global_top_in_TFU"]) for row in side_rows),
            "by_control": by_control,
        }
    return out


def main() -> int:
    args = build_parser().parse_args()
    rows = [
        row
        for row in read_rows_csv(args.samples)
        if row.get("target_var") == args.target_var and (args.split == "all" or row.get("split") == args.split)
    ]
    if not rows:
        raise ValueError("No rows matched the filters")
    print(f"Scoring {len(rows)} rows (target_var={args.target_var}, split={args.split}, sides={args.sides})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, _, auto_model_cls, auto_tokenizer_cls = import_runtime()
    tokenizer, model = load_hf_model(
        torch=torch,
        auto_model_cls=auto_model_cls,
        auto_tokenizer_cls=auto_tokenizer_cls,
        model_name=args.model_name,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    device = next(model.parameters()).device

    scored: list[dict[str, Any]] = []
    for side in args.sides:
        scored.extend(score_side(rows, side, model, tokenizer, torch, device, label_tokens.token_ids, args.eval_batch_size))

    summary = {
        "model_name": args.model_name,
        "samples": args.samples,
        "target_var": args.target_var,
        "split": args.split,
        "sides": args.sides,
        "label_token_style": label_tokens.style,
        **summarize(scored),
    }
    write_rows_csv(scored, output_dir / "baseline_scored.csv")
    (output_dir / "baseline_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")

    print("\n=== No-intervention baseline ===")
    for side in args.sides:
        stats = summary[side]
        print(f"[{side}] n={stats['n']} accuracy={stats['accuracy']:.4f} topTFU={stats['global_top_in_TFU_rate']:.4f}")
        for control, cstats in stats["by_control"].items():
            print(
                f"  {control:18s} acc={cstats['accuracy']:.4f}  mean_R={cstats['mean_R']:+.3f}  "
                f"mean_U_gap={cstats['mean_U_gap']:+.3f}  n={cstats['n']}"
            )
    print(f"\nWrote {output_dir / 'baseline_scored.csv'} and baseline_summary.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score DAS pair prompts without intervention.")
    parser.add_argument("--samples", required=True, help="DAS pairs CSV from das-generate.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-var", default="pc")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--sides", nargs="+", default=["base"], choices=["base", "source"])
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--label-token-style", default="auto")
    parser.add_argument("--output-dir", default="data/das/baseline_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
