"""Sweep p_c-forced DAS training over multiple layers.

Same experiment as run_das_pc_forced.py (probe pairs with H_pc targets inside
the training set), but loads the base model once, loops over layers, and
aggregates per-layer test metrics into pc_forced_sweep.csv/json (rewritten
after every layer, so partial sweeps survive crashes).

Example:
    python code/run_das_pc_forced_sweep.py --layers 8 11 14 17 20 23 --eval-interval 0 --local-files-only
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from interference_suite.das_data import generate_das_pairs
from interference_suite.das_pyvene import import_runtime, load_hf_model, run_pyvene_das, to_jsonable
from interference_suite.model import DEFAULT_CACHE_DIR
from run_das_handle_probe import generate_probe_rows

CONTROLS = ("probe_flip_both", "probe_flip_pi", "main", "gate_m0", "label_copy_trap")


def build_rows(args: argparse.Namespace) -> list[dict]:
    rows = generate_probe_rows(args.n_base_events, args.seed, "all", args.train_fraction, args.val_fraction)
    n_probe = len(rows)
    if not args.probes_only:
        rows = generate_das_pairs(
            n_base_events=args.n_base_events,
            seed=args.seed,
            targets=("pc",),
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ) + rows
    for idx, row in enumerate(rows):
        row["row_id"] = idx
    print(f"Training pool: {len(rows)} rows ({n_probe} probe rows{'' if args.probes_only else ' + original pc pairs'})")
    return rows


def cell_record(layer: int, summary: dict) -> dict:
    record: dict = {"layer": layer}
    for split in ("val", "test"):
        by_control = (summary.get(split) or {}).get("by_control") or {}
        for control in CONTROLS:
            stats = by_control.get(control) or {}
            record[f"{split}_{control}_IIA"] = stats.get("IIA")
        record[f"{split}_top_in_TFU"] = (summary.get(split) or {}).get("global_top_in_TFU_rate")
    return record


def write_outputs(records: list[dict], output_dir: Path) -> None:
    (output_dir / "pc_forced_sweep.json").write_text(json.dumps(to_jsonable(records), indent=2), encoding="utf-8")
    import csv

    with (output_dir / "pc_forced_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def print_table(records: list[dict]) -> None:
    header = "layer | " + " | ".join(f"{control:>18s}" for control in CONTROLS) + " (test IIA)"
    print("\n" + header)
    print("-" * len(header))
    for record in sorted(records, key=lambda r: r["layer"]):
        cells = []
        for control in CONTROLS:
            value = record.get(f"test_{control}_IIA")
            cells.append(f"{value:>18.4f}" if value is not None else f"{'-':>18s}")
        print(f"{record['layer']:>5} | " + " | ".join(cells))


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args)

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

    records: list[dict] = []
    for layer in args.layers:
        print(f"\n=== p_c-forced training @ layer {layer} ===")
        summary = run_pyvene_das(
            rows=rows,
            output_dir=output_dir / f"L{layer:02d}",
            model_name=args.model_name,
            target_var="pc",
            layer=layer,
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
            model=model,
            tokenizer=tokenizer,
            eval_train=False,
        )
        records.append(cell_record(layer, summary))
        write_outputs(records, output_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_table(records)
    print(f"\nWrote {len(records)} layers to {output_dir / 'pc_forced_sweep.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep p_c-forced DAS training over layers.")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 11, 14, 17, 20, 23])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--eval-interval", type=int, default=0, help="0 disables mid-training val evals.")
    parser.add_argument("--n-base-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--probes-only", action="store_true", help="Train on probe pairs only, without the original main/gate/trap pairs.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", default="data/das/pc_forced_sweep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
