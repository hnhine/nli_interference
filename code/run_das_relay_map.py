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
import math
from pathlib import Path

from interference_suite.das_pyvene import (
    drop_relaxed_rows,
    filter_train_rows,
    import_runtime,
    load_hf_model,
    rows_for_split,
    run_pyvene_das,
    to_jsonable,
)
from interference_suite.io_utils import read_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR


PI_V4_DEFAULT_CONTROL_PROPORTIONS = {
    "main": 40.0,
    "active_source_m0": 40.0,
    "gate_m0": 5.0,
    "label_copy_trap": 5.0,
    "distractor": 10.0,
}

PI_V5_DEFAULT_CONTROL_PROPORTIONS = {
    "main": 30.0,
    "active_source_m0": 30.0,
    "probe_flip_both": 10.0,
    "probe_flip_pc": 10.0,
    "gate_m0": 5.0,
    "label_copy_trap": 5.0,
    "distractor": 10.0,
}

RHO_DEFAULT_CONTROL_PROPORTIONS = {
    "flip_pi": 20.0,
    "flip_pc": 20.0,
    "hold_both": 20.0,
    "source_m0": 20.0,
    "gate_m0": 10.0,
    "label_copy_trap": 10.0,
}


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows_csv(args.samples)
    train_control_proportions = resolve_control_proportions(
        rows=rows,
        target_var=args.target_var,
        train_control_types=args.train_control_types,
        values=args.train_control_proportions,
    )
    train_control_types = args.train_control_types
    if (
        args.train_control_proportions is None
        and train_control_proportions is not None
        and train_control_types == ["auto"]
    ):
        train_control_types = list(train_control_proportions)
    steps = resolve_training_steps(
        rows=rows,
        target_var=args.target_var,
        train_control_types=train_control_types,
        include_relaxed=args.include_relaxed,
        batch_size=args.batch_size,
        steps=args.steps,
        epochs=args.epochs,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_dir / "relay_map.json"
    cells = load_existing_cells(aggregate_path) if args.resume else []
    pending_cells: list[tuple[int, str]] = []
    for layer in args.layers:
        for site in args.sites:
            cell_name = f"L{layer:02d}_{site}"
            summary_path = output_dir / cell_name / "summary_metrics.json"
            if args.resume and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                cells = upsert_cell(cells, cell_record(layer, site, summary))
                print(f"=== skip completed relay map cell {cell_name} ===")
            else:
                pending_cells.append((layer, site))

    if args.resume and cells:
        write_aggregate(cells, aggregate_path)
    if not pending_cells:
        print_table(cells, args.sites)
        print(f"\nAll requested cells already exist; {len(cells)} total cells in {aggregate_path}")
        return 0

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

    for layer, site in pending_cells:
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
            steps=steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            eval_interval=args.eval_interval,
            train_control_types=train_control_types,
            train_control_proportions=train_control_proportions,
            include_relaxed=args.include_relaxed,
            model=model,
            tokenizer=tokenizer,
            eval_train=False,
            export_rotation_weight=True,
        )
        cells = upsert_cell(cells, cell_record(layer, site, summary))
        write_aggregate(cells, aggregate_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_table(cells, args.sites)
    print(f"\nWrote {len(cells)} cells to {aggregate_path} and relay_map.csv")
    return 0


def load_existing_cells(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cells = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
        raise ValueError(f"Expected a list of relay-map cells in {path}")
    return cells


def upsert_cell(cells: list[dict], record: dict) -> list[dict]:
    key = (record.get("layer"), record.get("site"))
    updated = [cell for cell in cells if (cell.get("layer"), cell.get("site")) != key]
    updated.append(record)
    site_order = {"row": 0, "row_lexical_final": 1, "claim_final": 2, "answer_token": 3}
    return sorted(
        updated,
        key=lambda cell: (
            int(cell.get("layer", -1)),
            site_order.get(str(cell.get("site", "")), 99),
            str(cell.get("site", "")),
        ),
    )


def write_aggregate(cells: list[dict], path: Path) -> None:
    path.write_text(json.dumps(to_jsonable(cells), indent=2), encoding="utf-8")
    write_csv(cells, path.with_suffix(".csv"))


def cell_record(layer: int, site: str, summary: dict) -> dict:
    test = summary.get("test") or {}
    by_control = test.get("by_control") or {}
    by_pi_regime = test.get("by_pi_regime") or {}
    by_rho_regime = test.get("by_rho_regime") or {}

    def control_iia(name: str):
        return (by_control.get(name) or {}).get("IIA")

    def weighted_control_iia(names: tuple[str, ...]):
        values = [
            (by_control.get(name) or {})
            for name in names
            if (by_control.get(name) or {}).get("IIA") is not None
        ]
        total = sum(int(value.get("n", 0)) for value in values)
        return (
            sum(float(value["IIA"]) * int(value.get("n", 0)) for value in values) / total
            if total else None
        )

    record = {
        "layer": layer,
        "site": site,
        "test_IIA": test.get("IIA"),
        "pi_regime_macro_IIA": test.get("pi_regime_macro_IIA"),
        "pi_active_IIA": (by_pi_regime.get("active") or {}).get("IIA"),
        "pi_inactive_IIA": (by_pi_regime.get("inactive") or {}).get("IIA"),
        "pi_locality_IIA": (by_pi_regime.get("locality") or {}).get("IIA"),
        "rho_regime_macro_IIA": test.get("rho_regime_macro_IIA"),
        "rho_active_regime_IIA": (by_rho_regime.get("active") or {}).get("IIA"),
        "rho_inactive_regime_IIA": (by_rho_regime.get("inactive") or {}).get("IIA"),
        "main_IIA": control_iia("main"),
        "probe_flip_both_IIA": control_iia("probe_flip_both"),
        "probe_flip_pi_IIA": control_iia("probe_flip_pi"),
        "probe_flip_pc_IIA": control_iia("probe_flip_pc"),
        "active_source_m0_IIA": control_iia("active_source_m0"),
        "flip_pi_IIA": control_iia("flip_pi"),
        "flip_pc_IIA": control_iia("flip_pc"),
        "hold_both_IIA": control_iia("hold_both"),
        "source_m0_IIA": control_iia("source_m0"),
        "match_to_nomatch_IIA": control_iia("match_to_nomatch"),
        "nomatch_to_match_IIA": control_iia("nomatch_to_match"),
        "distractor_IIA": control_iia("distractor"),
        "gate_m0_IIA": control_iia("gate_m0"),
        "label_copy_trap_IIA": control_iia("label_copy_trap"),
        "label_copy_trap_same_m1_IIA": control_iia("label_copy_trap_same_m1"),
        "global_top_in_TFU_rate": test.get("global_top_in_TFU_rate"),
        "val_IIA": (summary.get("val") or {}).get("IIA"),
        "val_pi_regime_macro_IIA": (summary.get("val") or {}).get("pi_regime_macro_IIA"),
        "val_rho_regime_macro_IIA": (summary.get("val") or {}).get("rho_regime_macro_IIA"),
        "n_test": test.get("n"),
    }

    record["pc_active_IIA"] = weighted_control_iia(
        ("main", "probe_flip_both", "probe_flip_pi")
    )
    record["pc_inactive_IIA"] = weighted_control_iia(
        ("gate_m0", "label_copy_trap")
    )
    record["pc_macro_IIA"] = weighted_control_iia(
        ("main", "probe_flip_both", "probe_flip_pi", "gate_m0", "label_copy_trap")
    )

    record["rho_active_IIA"] = weighted_control_iia(
        ("flip_pi", "flip_pc", "hold_both", "source_m0")
    )
    record["rho_inactive_IIA"] = weighted_control_iia(
        ("gate_m0", "label_copy_trap")
    )
    identifying = [
        control_iia(name)
        for name in ("flip_pi", "flip_pc", "hold_both", "source_m0", "label_copy_trap")
    ]
    record["rho_identification_min_IIA"] = (
        min(float(value) for value in identifying)
        if all(value is not None for value in identifying)
        else None
    )
    full_audit = [*identifying, control_iia("gate_m0")]
    record["rho_full_audit_min_IIA"] = (
        min(float(value) for value in full_audit)
        if all(value is not None for value in full_audit)
        else None
    )

    m_values = [
        (by_control.get(name) or {})
        for name in ("match_to_nomatch", "nomatch_to_match")
        if (by_control.get(name) or {}).get("IIA") is not None
    ]
    if m_values:
        total = sum(int(value.get("n", 0)) for value in m_values)
        record["m_core_IIA"] = (
            sum(float(value["IIA"]) * int(value.get("n", 0)) for value in m_values) / total
            if total else None
        )
    else:
        record["m_core_IIA"] = None
    return record


def write_csv(cells: list[dict], path: Path) -> None:
    import csv

    fieldnames = list(dict.fromkeys(key for cell in cells for key in cell))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cells)


def print_table(cells: list[dict], sites: list[str]) -> None:
    by_key = {(cell["layer"], cell["site"]): cell for cell in cells}
    layers = sorted({cell["layer"] for cell in cells})
    if any(cell.get("rho_identification_min_IIA") is not None for cell in cells):
        metric = "rho_identification_min_IIA"
        label = "rho-min IIA"
    elif any(cell.get("pi_regime_macro_IIA") is not None for cell in cells):
        metric = "pi_regime_macro_IIA"
        label = "pi-macro IIA"
    elif any(cell.get("m_core_IIA") is not None for cell in cells):
        metric = "m_core_IIA"
        label = "m-core IIA"
    else:
        metric = "main_IIA"
        label = "main IIA"
    header = "layer | " + " | ".join(f"{label} @{site}" for site in sites)
    print("\n" + header)
    print("-" * len(header))
    for layer in layers:
        values = []
        for site in sites:
            cell = by_key.get((layer, site))
            value = cell.get(metric) if cell else None
            values.append(f"{value:.4f}" if value is not None else "-")
        print(f"{layer:>5} | " + " | ".join(f"{value:>{len(f'{label} @{site}')}}" for value, site in zip(values, sites)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAS layer/site relay map sweep.")
    parser.add_argument("--samples", required=True, help="DAS pairs CSV from das-generate.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--target-var", default="pc")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11, 17, 23, 29, 35])
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["claim_final", "answer_token"],
        help=("Token sites to intervene on. Use row for pair-specific boundary sites, "
              "or row_lexical_final to preserve each row choice while skipping "
              "trailing punctuation/newlines."),
    )
    parser.add_argument("--rank", type=int, default=16)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, default=None)
    duration.add_argument(
        "--epochs",
        type=float,
        default=None,
        help=("Train for this many epoch-equivalents, converted to "
              "ceil(epochs * n_train / batch_size) steps. Sampling remains with replacement."),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--eval-interval", type=int, default=250, help="0 disables mid-training val evals.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-control-types",
        nargs="+",
        default=["auto"],
        help=("Training controls. For m, auto uses both transfer directions. For rho, "
              "auto trains all six controls with the default 20/20/20/20/10/10 mix."),
    )
    parser.add_argument(
        "--train-control-proportions",
        nargs="+",
        default=None,
        metavar="CONTROL=WEIGHT",
        help=("Stratify every training batch by control type. pi V4 defaults to "
              "main=40 active_source_m0=40 gate_m0=5 label_copy_trap=5 distractor=10; "
              "pi V5 defaults to main=30 active_source_m0=30 flip_both=10 "
              "flip_pc=10 gate_m0=5 label_copy_trap=5 distractor=10; "
              "rho defaults to flip_pi=20 flip_pc=20 hold_both=20 source_m0=20 "
              "gate_m0=10 label_copy_trap=10; "
              "other datasets retain their existing unstratified default. Example override: "
              "main=0.5 distractor=0.1666666667 gate_m0=0.1666666667 "
              "label_copy_trap=0.1666666667. Weights are normalized."),
    )
    parser.add_argument(
        "--include-relaxed",
        action="store_true",
        help="Keep rows flagged mismatch_exclusion_relaxed=1 (excluded by default to avoid cross-split event leakage).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=("Keep existing relay-map cells in the output directory, recover completed "
              "cells from summary_metrics.json, and train only missing layer/site cells."),
    )
    parser.add_argument("--output-dir", default="data/das/relay_map")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def parse_control_proportions(values: list[str] | None) -> dict[str, float] | None:
    if values is None:
        return None
    result: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONTROL=WEIGHT, got {value!r}")
        name, raw_weight = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate control name in {value!r}")
        try:
            result[name] = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid weight in {value!r}") from exc
    return result


def resolve_control_proportions(
    *,
    rows: list[dict],
    target_var: str,
    train_control_types: list[str] | None,
    values: list[str] | None,
) -> dict[str, float] | None:
    """Resolve default stratified mixes for identified pi and rho designs."""
    if values is not None:
        return parse_control_proportions(values)
    control_mode = train_control_types or ["auto"]
    if target_var == "rho":
        if control_mode != ["auto"]:
            return None
        controls = {
            str(row.get("control_type", ""))
            for row in rows
            if row.get("target_var") == "rho"
        }
        if set(RHO_DEFAULT_CONTROL_PROPORTIONS) <= controls:
            return dict(RHO_DEFAULT_CONTROL_PROPORTIONS)
        return None
    if target_var != "pi" or control_mode not in (["auto"], ["all"]):
        return None

    target_rows = [row for row in rows if row.get("target_var") == "pi"]
    variants = {str(row.get("pi_variant", "")) for row in target_rows}
    controls = {str(row.get("control_type", "")) for row in target_rows}
    if variants == {"v5"} and set(PI_V5_DEFAULT_CONTROL_PROPORTIONS) <= controls:
        return dict(PI_V5_DEFAULT_CONTROL_PROPORTIONS)
    if variants == {"v4"} and set(PI_V4_DEFAULT_CONTROL_PROPORTIONS) <= controls:
        return dict(PI_V4_DEFAULT_CONTROL_PROPORTIONS)
    return None


def resolve_training_steps(
    *,
    rows: list[dict],
    target_var: str,
    train_control_types: list[str] | None,
    include_relaxed: bool,
    batch_size: int,
    steps: int | None,
    epochs: float | None,
) -> int:
    """Resolve CLI duration while matching run_pyvene_das's train-row filtering."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if steps is not None:
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        return steps
    if epochs is None:
        return 500
    if not math.isfinite(epochs) or epochs <= 0:
        raise ValueError(f"epochs must be a positive finite number, got {epochs}")

    target_rows = [row for row in rows if row.get("target_var") == target_var]
    if not include_relaxed:
        target_rows = drop_relaxed_rows(target_rows)
    train_rows = filter_train_rows(rows_for_split(target_rows, "train"), target_var, train_control_types)
    if not train_rows:
        raise ValueError(f"No train rows found for target_var={target_var!r}")
    resolved_steps = max(1, math.ceil(epochs * len(train_rows) / batch_size))
    print(
        f"Resolved --epochs {epochs:g} to {resolved_steps} steps "
        f"({len(train_rows)} train rows, batch_size={batch_size})."
    )
    return resolved_steps


if __name__ == "__main__":
    raise SystemExit(main())
