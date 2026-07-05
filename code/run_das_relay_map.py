"""Train one DAS intervention per (layer, site) cell and aggregate a relay map.

Loads the base model once and reuses it across cells. Each cell gets its own
subdirectory with the usual summary_metrics.json / *_scored.csv, and the
aggregate table is rewritten after every cell so partial sweeps survive crashes.

Example:
    python code/run_das_relay_map.py \
        --samples data/das/pc_1000_v2/pairs.csv \
        --layers 0 5 11 17 23 29 35 \
        --sites claim_final answer_token \
        --steps 500 --local-files-only
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from interference_suite.das_pyvene import import_runtime, load_hf_model, run_pyvene_das, to_jsonable
from interference_suite.io_utils import read_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows_csv(args.samples)
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

    cells: list[dict] = []
    aggregate_path = output_dir / "relay_map.json"
    for layer in args.layers:
        for site in args.sites:
            cell_name = f"L{layer:02d}_{site}"
            print(f"\n=== relay map cell {cell_name} ===")
            summary = run_pyvene_das(
                rows=rows,
                output_dir=output_dir / cell_name,
                model_name=args.model_name,
                target_var=args.target_var,
                layer=layer,
                rank=args.rank,
                site=site,
                steps=args.steps,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                eval_interval=args.eval_interval,
                train_control_types=args.train_control_types,
                model=model,
                tokenizer=tokenizer,
                eval_train=False,
            )
            cells.append(cell_record(layer, site, summary))
            aggregate_path.write_text(json.dumps(to_jsonable(cells), indent=2), encoding="utf-8")
            write_csv(cells, output_dir / "relay_map.csv")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_table(cells, args.sites)
    print(f"\nWrote {len(cells)} cells to {aggregate_path} and relay_map.csv")
    return 0


def cell_record(layer: int, site: str, summary: dict) -> dict:
    test = summary.get("test") or {}
    by_control = test.get("by_control") or {}

    def control_iia(name: str):
        return (by_control.get(name) or {}).get("IIA")

    return {
        "layer": layer,
        "site": site,
        "test_IIA": test.get("IIA"),
        "main_IIA": control_iia("main"),
        "gate_m0_IIA": control_iia("gate_m0"),
        "label_copy_trap_IIA": control_iia("label_copy_trap"),
        "global_top_in_TFU_rate": test.get("global_top_in_TFU_rate"),
        "val_IIA": (summary.get("val") or {}).get("IIA"),
        "n_test": test.get("n"),
    }


def write_csv(cells: list[dict], path: Path) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0].keys()))
        writer.writeheader()
        writer.writerows(cells)


def print_table(cells: list[dict], sites: list[str]) -> None:
    by_key = {(cell["layer"], cell["site"]): cell for cell in cells}
    layers = sorted({cell["layer"] for cell in cells})
    header = "layer | " + " | ".join(f"main IIA @{site}" for site in sites)
    print("\n" + header)
    print("-" * len(header))
    for layer in layers:
        values = []
        for site in sites:
            cell = by_key.get((layer, site))
            value = cell.get("main_IIA") if cell else None
            values.append(f"{value:.4f}" if value is not None else "-")
        print(f"{layer:>5} | " + " | ".join(f"{value:>{len(f'main IIA @{site}')}}" for value, site in zip(values, sites)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAS layer/site relay map sweep.")
    parser.add_argument("--samples", required=True, help="DAS pairs CSV from das-generate.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--target-var", default="pc")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11, 17, 23, 29, 35])
    parser.add_argument("--sites", nargs="+", default=["claim_final", "answer_token"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--eval-interval", type=int, default=250, help="0 disables mid-training val evals.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-control-types", nargs="+", default=["all"])
    parser.add_argument("--output-dir", default="data/das/relay_map")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
