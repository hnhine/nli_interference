"""Necessity ablation across many trained rotations (relay-map cells), one model load.

Each rotation dir supplies its own layer + site (read from metadata), so this
maps where p_c is a NECESSARY bottleneck along the relay path (claim_final early,
answer_token late). Aggregates one row per cell into ablation_sweep.csv.

Example:
    python code/run_das_ablation_sweep.py \
        --samples data/das/pc_1000_v2/pairs.csv --model-name Qwen/Qwen3-8B \
        --split test --local-files-only --output-dir data/das/ablation_sweep_qwen \
        --rotation-dirs data/das/relay_map_v3/L11_claim_final \
                        data/das/relay_map_v3/L14_claim_final \
                        data/das/relay_map_v3/L17_claim_final \
                        data/das/relay_map_v3/L20_claim_final \
                        data/das/relay_map_v3/L23_answer_token \
                        data/das/relay_map_v3/L26_answer_token \
                        data/das/relay_map_v3/L29_answer_token \
                        data/das/relay_map_v3/L35_answer_token
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from interference_suite.das_pyvene import import_runtime, load_hf_model, to_jsonable
from interference_suite.io_utils import read_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, resolve_label_tokens
from run_das_ablation import (
    CONDITION_NAMES,
    ablate_one,
    build_donor_indices,
    get_decoder_layers,
    print_ablation_table,
    resolve_condition_column,
)


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    rows = [r for r in read_rows_csv(args.samples)
            if r.get("target_var") == args.target_var and (args.split == "all" or r.get("split") == args.split)]
    rng.shuffle(rows)
    if not rows:
        raise ValueError("No rows matched filters")
    condition_on = resolve_condition_column(args.target_var, args.condition_on)
    donor_indices = None if args.skip_conditioned_donors else build_donor_indices(rows, condition_on, args.seed)

    torch, _, amc, atc = import_runtime()
    tokenizer, model = load_hf_model(torch=torch, auto_model_cls=amc, auto_tokenizer_cls=atc,
        model_name=args.model_name, device=args.device, device_map=args.device_map,
        torch_dtype=args.torch_dtype, trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    device = next(model.parameters()).device
    layers = get_decoder_layers(model)
    hidden = model.config.hidden_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for rdir in args.rotation_dirs:
        name = Path(rdir).name
        cell_json = output_dir / f"{name}.json"
        cached = json.loads(cell_json.read_text()) if cell_json.exists() else None
        cache_complete = cached is not None and (
            donor_indices is None or "das_resample_opposite" in cached
        )
        if cache_complete:
            print(f"\n=========== skip {name}: da co ket qua, doc lai tu {cell_json.name} ===========")
            summary = cached
        elif not (Path(rdir) / "rotation_weight_metadata.json").exists():
            print(f"\n=========== SKIP {name}: chua co rotation (train chua xong?) ===========")
            continue
        else:
            print(f"\n=========== ablate {rdir} ===========")
            summary, all_scored = ablate_one(model, layers, tokenizer, torch, device, hidden, rows,
                                             rdir, label_tokens.token_ids, args.eval_batch_size, args.seed,
                                             donor_indices=donor_indices, condition_on=condition_on)
            print_ablation_table(summary, all_scored)
            cell_json.write_text(json.dumps(to_jsonable(summary), indent=2))

        records.append(build_record(name, summary))
        with (output_dir / "ablation_sweep.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)

    print("\n============  NECESSITY ALONG THE RELAY (main accuracy)  ============")
    hdr = f"{'cell':22s} {'none':>7s} {'das_res':>8s} {'rand_res':>9s} {'necessity':>10s} {'gate_das':>9s}"
    print(hdr); print("-" * len(hdr))
    for r in records:
        print(f"{r['cell']:22s} {fmt(r['main_none'])} {fmt(r['main_das_resample'])} "
              f"{fmt(r['main_rand_resample']):>9s} {fmt(r.get('necessity_excess')):>10s} {fmt(r['gate_m0_das_resample']):>9s}")
    print(f"\nWrote {output_dir / 'ablation_sweep.csv'}")
    return 0


def build_record(name: str, summary: dict) -> dict:
    def acc(cond, ctrl):
        return (summary[cond]["by_control"].get(ctrl) or {}).get("accuracy")

    rec = {
        "cell": name,
        "layer": summary["layer"],
        "site": summary["site"],
        "rank": summary.get("rank"),
        "condition_on": summary.get("condition_on"),
    }
    for ctrl in ("main", "gate_m0"):
        rec[f"{ctrl}_none"] = acc("none", ctrl)
        rec[f"{ctrl}_das_resample"] = acc("das_resample", ctrl)
        if "das_resample_same" in summary:
            rec[f"{ctrl}_das_resample_same"] = acc("das_resample_same", ctrl)
            rec[f"{ctrl}_das_resample_opposite"] = acc("das_resample_opposite", ctrl)
        rec[f"{ctrl}_rand_resample"] = acc("rand_resample", ctrl)
    if rec["main_none"] is not None:
        rec["necessity_excess"] = rec["main_rand_resample"] - rec["main_das_resample"]
        if "main_das_resample_same" in rec:
            rec["purity_drop"] = rec["main_none"] - rec["main_das_resample_same"]
            rec["backup_rescue"] = rec["main_das_resample_opposite"]
    return rec


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "   -  "


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep necessity ablation over relay-map cells.")
    p.add_argument("--rotation-dirs", nargs="+", required=True, help="Cell dirs, each with rotation_weight.npy + metadata.")
    p.add_argument("--samples", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--target-var", default="pc")
    p.add_argument("--condition-on", default="auto",
                   help="Column defining same/opposite donor classes; auto maps pc/pi/m to p_c_base/p_i_base/m_base.")
    p.add_argument("--skip-conditioned-donors", action="store_true",
                   help="Run only the legacy zero/random-resample conditions.")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--label-token-style", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/das/ablation_sweep")
    p.add_argument("--device", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
