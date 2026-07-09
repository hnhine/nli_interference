"""Handle-probe sweep: is the DAS subspace p_c or REL, across many cells, one model load.

Runs the two probe conditions (flip_both / flip_pi) for every rotation dir,
reusing the probe generator and evaluator from run_das_handle_probe. Loads the
base model once. Aggregates per-cell hypothesis leverage into a CSV so you can
read the p_c->REL fusion curve along the relay.

The load-bearing columns are the ACTIVE ones (a "stay" prediction is inflated by
an impotent intervention):
    flip_both -> H_pc   (p_c leverage: injecting p_c flips the label)
    flip_pi   -> H_rel  (REL leverage: injecting REL flips the label)

Example:
    python code/run_das_handle_probe_sweep.py \
        --model-name ibm-granite/granite-4.1-8b \
        --sanity-samples data/das/pc_1000_v2/pairs.csv --split test --local-files-only \
        --output-dir data/das/granite_probe_sweep \
        --rotation-dirs data/das/granite41_relay_claim/L14_claim_final \
                        data/das/granite41_relay_claim/L17_claim_final \
                        data/das/granite41_relay_claim/L19_claim_final \
                        data/das/granite41_relay_claim/L21_claim_final \
                        data/das/granite41_relay_claim/L25_claim_final \
                        data/das/granite41_relay_claim/L31_claim_final
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from interference_suite.das_pyvene import (
    build_intervenable,
    evaluate_pyvene_das,
    get_input_device,
    import_runtime,
    load_hf_model,
    set_intervenable_device,
    to_jsonable,
)
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, resolve_label_tokens
from run_das_handle_probe import generate_probe_rows, hypothesis_match_rates, load_rotation


def main() -> int:
    args = build_parser().parse_args()
    probe_rows = generate_probe_rows(args.n_base_events, args.seed, args.split, args.train_fraction, args.val_fraction)
    print(f"Generated {len(probe_rows)} probe rows (split={args.split})")
    sanity_all = None
    if args.sanity_samples:
        sanity_all = [r for r in read_rows_csv(args.sanity_samples)
                      if r.get("target_var") == "pc" and r.get("split") == args.split]

    torch, pv, amc, atc = import_runtime()
    tokenizer, model = load_hf_model(torch=torch, auto_model_cls=amc, auto_tokenizer_cls=atc,
        model_name=args.model_name, device=args.device, device_map=args.device_map,
        torch_dtype=args.torch_dtype, trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    input_device = get_input_device(model, torch, args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for rdir in args.rotation_dirs:
        meta = json.loads((Path(rdir) / "rotation_weight_metadata.json").read_text())
        layer, rank = int(meta["layer"]), int(meta["rank"])
        site = str(meta.get("site", "claim_final"))
        component = str(meta.get("component", "block_output"))
        print(f"\n=== probe {Path(rdir).name} (L{layer}/{site}) ===")
        intervenable = build_intervenable(pv, model, layer=layer, rank=rank, component=component)
        set_intervenable_device(intervenable, input_device)
        load_rotation(intervenable, torch, Path(rdir))

        rec = {"cell": Path(rdir).name, "layer": layer, "site": site}
        if sanity_all is not None:
            sm, _ = evaluate_pyvene_das(intervenable, sanity_all, tokenizer, torch, input_device,
                                        label_tokens.token_ids, args.eval_batch_size, site, "sanity")
            rec["sanity_main"] = (sm.get("by_control", {}).get("main", {}) or {}).get("IIA")

        _, probe_scored = evaluate_pyvene_das(intervenable, probe_rows, tokenizer, torch, input_device,
                                              label_tokens.token_ids, args.eval_batch_size, site, "probe")
        hm = hypothesis_match_rates(probe_scored)
        rec["flip_both_Hpc"] = hm["probe_flip_both"]["match_H_pc"]
        rec["flip_both_Hrel"] = hm["probe_flip_both"]["match_H_rel"]
        rec["flip_pi_Hpc"] = hm["probe_flip_pi"]["match_H_pc"]
        rec["flip_pi_Hrel"] = hm["probe_flip_pi"]["match_H_rel"]
        rec["flip_both_U"] = hm["probe_flip_both"]["pred_counts"].get("U", 0) / hm["probe_flip_both"]["n"]
        records.append(rec)
        write_rows_csv(probe_scored, output_dir / f"{Path(rdir).name}_probe_scored.csv")
        with (output_dir / "probe_sweep.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
        print(f"  p_c leverage (flip_both->Hpc)={rec['flip_both_Hpc']:.3f}  "
              f"REL leverage (flip_pi->Hrel)={rec['flip_pi_Hrel']:.3f}"
              + (f"  sanity_main={rec['sanity_main']:.3f}" if rec.get('sanity_main') is not None else ""))

    print("\n========  HANDLE ALONG THE RELAY  ========")
    hdr = f"{'cell':22s} {'p_c lev':>8s} {'REL lev':>8s} {'flip_both_U':>12s}"
    print(hdr); print("-" * len(hdr))
    for r in records:
        print(f"{r['cell']:22s} {r['flip_both_Hpc']:>8.3f} {r['flip_pi_Hrel']:>8.3f} {r['flip_both_U']:>12.3f}")
    print(f"\nWrote {output_dir / 'probe_sweep.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep handle-probe (p_c vs REL) over rotation cells.")
    p.add_argument("--rotation-dirs", nargs="+", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--sanity-samples", default=None, help="Original pairs CSV; re-evaluated per cell to verify rotation load.")
    p.add_argument("--n-base-events", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--label-token-style", default="auto")
    p.add_argument("--output-dir", default="data/das/probe_sweep")
    p.add_argument("--device", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
