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
import hashlib
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
    make_donor_spec,
    print_ablation_table,
    resolve_condition_column,
    resolve_condition_hold_columns,
    resolve_donor_event_policy,
    resolve_donor_pool,
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
    condition_hold = resolve_condition_hold_columns(args.target_var, args.condition_hold)
    if args.split == "all" and "split" not in condition_hold:
        condition_hold.append("split")
    donor_pool = resolve_donor_pool(args.target_var, args.donor_pool)
    donor_event_policy = resolve_donor_event_policy(args.target_var, args.donor_event_policy)
    donor_spec = make_donor_spec(condition_on, condition_hold, donor_pool, donor_event_policy)
    donor_spec["seed"] = args.seed
    donor_spec["row_signature"] = hashlib.sha256(
        "\n".join(str(row.get("sample_id", "")) for row in rows).encode("utf-8")
    ).hexdigest()[:16]
    donor_indices = None if args.skip_conditioned_donors else build_donor_indices(
        rows,
        condition_on,
        args.seed,
        condition_hold=condition_hold,
        donor_pool=donor_pool,
        donor_event_policy=donor_event_policy,
    )

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
            donor_indices is None
            or ("das_resample_opposite" in cached and cached.get("donor_spec") == donor_spec)
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
                                             donor_indices=donor_indices, condition_on=condition_on,
                                             donor_spec=None if donor_indices is None else donor_spec)
            print_ablation_table(summary, all_scored)
            cell_json.write_text(json.dumps(to_jsonable(summary), indent=2))

        records.append(build_record(name, summary))
        with (output_dir / "ablation_sweep.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)

    if args.target_var == "m":
        print("\n============  NECESSITY FOR m BY DIRECTION  ============")
        hdr = (f"{'cell':22s} {'control':19s} {'none':>7s} {'das_res':>8s} {'rand_res':>9s} "
               f"{'necessity':>10s} {'same':>7s} {'opp_cf':>7s}")
        print(hdr); print("-" * len(hdr))
        for r in records:
            for ctrl in ("match_to_nomatch", "nomatch_to_match", "label_copy_trap"):
                print(
                    f"{r['cell']:22s} {ctrl:19s} {fmt(r.get(f'{ctrl}_none'))} "
                    f"{fmt(r.get(f'{ctrl}_das_resample'))} {fmt(r.get(f'{ctrl}_rand_resample')):>9s} "
                    f"{fmt(r.get(f'necessity_excess_{ctrl}')):>10s} "
                    f"{fmt(r.get(f'{ctrl}_das_resample_same'))} "
                    f"{fmt(r.get(f'{ctrl}_das_resample_opposite_cf'))}"
                )
    else:
        print("\n============  NECESSITY ALONG THE RELAY (main accuracy)  ============")
        hdr = f"{'cell':22s} {'none':>7s} {'das_res':>8s} {'rand_res':>9s} {'necessity':>10s} {'gate_das':>9s}"
        print(hdr); print("-" * len(hdr))
        for r in records:
            print(f"{r['cell']:22s} {fmt(r.get('main_none'))} {fmt(r.get('main_das_resample'))} "
                  f"{fmt(r.get('main_rand_resample')):>9s} {fmt(r.get('necessity_excess')):>10s} "
                  f"{fmt(r.get('gate_m0_das_resample')):>9s}")
    print(f"\nWrote {output_dir / 'ablation_sweep.csv'}")
    return 0


def build_record(name: str, summary: dict) -> dict:
    rec = {
        "cell": name,
        "layer": summary["layer"],
        "site": summary["site"],
        "rank": summary.get("rank"),
        "condition_on": summary.get("condition_on"),
        "condition_hold": ",".join(summary.get("condition_hold", [])),
        "donor_pool": summary.get("donor_pool"),
        "donor_event_policy": summary.get("donor_event_policy"),
    }
    controls = sorted(summary["none"].get("by_control", {}))
    for ctrl in controls:
        for cond in CONDITION_NAMES:
            stats = (summary.get(cond) or {}).get("by_control", {}).get(ctrl)
            if not stats:
                continue
            rec[f"{ctrl}_{cond}"] = stats.get("accuracy")
            rec[f"{ctrl}_{cond}_cf"] = stats.get("counterfactual_accuracy")
            rec[f"{ctrl}_{cond}_U_rate"] = (stats.get("pred_dist") or {}).get("U")
        none = rec.get(f"{ctrl}_none")
        das = rec.get(f"{ctrl}_das_resample")
        rand = rec.get(f"{ctrl}_rand_resample")
        same = rec.get(f"{ctrl}_das_resample_same")
        if das is not None and rand is not None:
            rec[f"necessity_excess_{ctrl}"] = rand - das
        if none is not None and same is not None:
            rec[f"purity_drop_{ctrl}"] = none - same

    # Stable legacy aliases used by existing pc/pi analysis notebooks.
    if rec.get("main_none") is not None:
        rec["necessity_excess"] = rec.get("necessity_excess_main")
        rec["purity_drop"] = rec.get("purity_drop_main")
        rec["backup_rescue"] = rec.get("main_das_resample_opposite")
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
    p.add_argument(
        "--condition-hold",
        default="auto",
        help=("Comma-separated nuisance columns fixed for donors. Auto uses none for pc/pi and "
              "p_i_base,p_c_base,mismatch_type,matched_idx for m."),
    )
    p.add_argument("--donor-pool", default="auto",
                   choices=["auto", "within_control", "m_main", "all"])
    p.add_argument("--donor-event-policy", default="auto",
                   choices=["auto", "prefer", "require", "ignore"])
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
