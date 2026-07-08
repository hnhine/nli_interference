"""Train a DAS rotation that is FORCED to find a raw p_c subspace, if one exists.

The standard training pairs cannot distinguish p_c from REL = p_i*p_c (they flip
together), so DAS is free to latch onto REL. Here the two probe conditions are
added to the TRAINING set with H_pc counterfactual targets:

  probe_flip_both (source flips p_i and p_c; REL unchanged) -> target FLIPS
  probe_flip_pi   (source flips p_i only; REL flips)        -> target STAYS

A REL-carrying subspace fails both conditions, an inert one fails flip_both,
a p_i-carrying one fails flip_pi. Only a subspace carrying claim-local polarity
can satisfy the full set. If training cannot reach high probe IIA on val/test,
no usable p_c variable exists at this site/layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from interference_suite.das_data import generate_das_pairs
from interference_suite.das_pyvene import run_pyvene_das
from interference_suite.model import DEFAULT_CACHE_DIR
from run_das_handle_probe import generate_probe_rows


def main() -> int:
    args = build_parser().parse_args()

    rows = generate_probe_rows(args.n_base_events, args.seed, "all", args.train_fraction, args.val_fraction)
    n_probe = len(rows)
    if not args.probes_only:
        original = generate_das_pairs(
            n_base_events=args.n_base_events,
            seed=args.seed,
            targets=("pc",),
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        rows = original + rows
    for idx, row in enumerate(rows):
        row["row_id"] = idx
    print(f"Training pool: {len(rows)} rows ({n_probe} probe rows{'' if args.probes_only else ' + original pc pairs'})")

    summary = run_pyvene_das(
        rows=rows,
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        target_var="pc",
        layer=args.layer,
        rank=args.rank,
        site="claim_final",
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        eval_interval=args.eval_interval,
        train_control_types=["all"],
        export_rotation_weight=True,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        eval_train=False,
    )

    print("\n=== p_c-forced training verdict (test split) ===")
    by_control = (summary.get("test") or {}).get("by_control") or {}
    for control in ("probe_flip_both", "probe_flip_pi", "main", "gate_m0", "label_copy_trap"):
        stats = by_control.get(control)
        if stats:
            print(f"  {control:18s} IIA={stats['IIA']:.4f}  n={stats['n']}")
    both = (by_control.get("probe_flip_both") or {}).get("IIA")
    pi = (by_control.get("probe_flip_pi") or {}).get("IIA")
    if both is not None and pi is not None:
        if both > 0.9 and pi > 0.9:
            print("-> A raw p_c subspace EXISTS at this layer (both probe conditions satisfied).")
        elif both < 0.5:
            print("-> Training could NOT make interventions flip with p_c alone: no usable raw p_c variable here.")
        else:
            print("-> Partial: inspect by-control numbers; the subspace mixes p_c with other content.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DAS with probe pairs in the training set (p_c-forced).")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--n-base-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--probes-only", action="store_true", help="Train on probe pairs only, without the original main/gate/trap pairs.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", default="data/das/pc_forced")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
