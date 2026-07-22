"""Eval-only composition of trained DAS rotations at multiple layers.

Loads exported rotation weights from relay-map cell dirs (rotation_weight.npy +
rotation_weight_metadata.json) and applies them simultaneously in one forward
pass, so e.g. an L14 cell and an L18 cell can patch the same row token at both
depths. Also re-evaluates each cell alone as a loading sanity check.

Example:
    python code/run_das_multilayer_eval.py \
        --samples data/das/pi_1000_v2/pairs.csv \
        --model-name ibm-granite/granite-4.1-8b \
        --target-var pi \
        --rotation-dirs data/das/granite_pi_relay_r16_early/L14_row \
                        data/das/granite_pi_relay_r16_peaks/L18_row \
        --local-files-only \
        --output-dir data/das/granite_pi_multilayer_L14_L18
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from interference_suite.das_pyvene import (
    INDEX_TO_LABEL,
    call_intervenable,
    chunks,
    collate_rows,
    drop_relaxed_rows,
    import_runtime,
    load_hf_model,
    make_stable_low_rank_intervention_type,
    next_token_logits,
    register_pyvene_model_mappings,
    rows_for_split,
    set_intervenable_device,
    get_input_device,
    summarize_scored,
    tfu_logits_from_next,
    to_jsonable,
)
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, progress_iter, resolve_label_tokens


def load_specs(rotation_dirs: list[str]) -> list[dict[str, Any]]:
    specs = []
    for rotation_dir in rotation_dirs:
        rotation_dir = Path(rotation_dir)
        meta = json.loads((rotation_dir / "rotation_weight_metadata.json").read_text())
        specs.append(
            {
                "dir": rotation_dir,
                "layer": int(meta["layer"]),
                "rank": int(meta["rank"]),
                "site": str(meta.get("site", "row")),
                "component": str(meta.get("component", "block_output")),
                "npy": rotation_dir / "rotation_weight.npy",
            }
        )
    sites = {spec["site"] for spec in specs}
    if len(sites) > 1:
        raise ValueError(f"All rotation dirs must share one site, got {sites}")
    layers = [spec["layer"] for spec in specs]
    if len(set(layers)) != len(layers):
        raise ValueError(f"Duplicate layers in rotation dirs: {layers}")
    return sorted(specs, key=lambda spec: spec["layer"])


def build_multi_intervenable(pv: Any, model: Any, specs: list[dict[str, Any]]) -> Any:
    register_pyvene_model_mappings(model)
    intervention_type = make_stable_low_rank_intervention_type(pv)
    representations = [
        pv.RepresentationConfig(
            spec["layer"],
            spec["component"],
            "pos",
            1,
            low_rank_dimension=spec["rank"],
        )
        for spec in specs
    ]
    config = pv.IntervenableConfig(
        model_type=type(model),
        representations=representations,
        intervention_types=intervention_type,
    )
    try:
        intervenable = pv.IntervenableModel(config, model=model, use_fast=False)
    except TypeError:
        intervenable = pv.IntervenableModel(config, model=model)
    if hasattr(intervenable, "disable_model_gradients"):
        intervenable.disable_model_gradients()
    return intervenable


def load_rotation_weights(intervenable: Any, specs: list[dict[str, Any]], torch: Any) -> None:
    import numpy as np

    interventions = list(intervenable.interventions.items())
    if len(interventions) != len(specs):
        raise RuntimeError(f"Expected {len(specs)} interventions, pyvene built {len(interventions)}")
    for spec, (key, intervention) in zip(specs, interventions):
        if f"layer.{spec['layer']}." not in key and f"layer_{spec['layer']}_" not in key:
            raise RuntimeError(f"Intervention key {key!r} does not match layer {spec['layer']}")
        weight = intervention.rotate_layer.weight
        tensor = torch.tensor(np.load(spec["npy"]), dtype=torch.float32)
        if tuple(tensor.shape) != tuple(weight.shape):
            raise ValueError(f"{spec['npy']}: shape {tuple(tensor.shape)} != {tuple(weight.shape)}")
        with torch.no_grad():
            weight.copy_(tensor.to(device=weight.device, dtype=weight.dtype))


def evaluate_multi(
    intervenable: Any,
    n_interventions: int,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    torch: Any,
    device: Any,
    label_token_ids: dict[str, int],
    batch_size: int,
    site_override: str,
    progress_desc: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from math import ceil

    scored: list[dict[str, Any]] = []
    label_id_set = set(label_token_ids.values())
    row_chunks = progress_iter(chunks(rows, batch_size), total=ceil(len(rows) / batch_size), desc=progress_desc)
    for batch_rows in row_chunks:
        batch = collate_rows(batch_rows, tokenizer, torch, device, site_override)
        source_locs, base_locs = batch["unit_locations"]["sources->base"]
        batch["unit_locations"] = {"sources->base": (source_locs * n_interventions, base_locs * n_interventions)}
        with torch.no_grad():
            outputs = call_intervenable(intervenable, batch)
            next_logits = next_token_logits(torch, outputs, batch["input_lengths"])
            tfu_logits = tfu_logits_from_next(torch, next_logits, label_token_ids)
            global_top_ids = next_logits.argmax(dim=-1)
        preds = tfu_logits.argmax(dim=-1)
        for row, row_logits, pred_idx, top_id in zip(batch_rows, tfu_logits, preds, global_top_ids):
            pred_label = INDEX_TO_LABEL[int(pred_idx.detach().cpu())]
            top_token_id = int(top_id.detach().cpu())
            out = dict(row)
            out.update(
                {
                    "logit_T": float(row_logits[0].detach().cpu()),
                    "logit_F": float(row_logits[1].detach().cpu()),
                    "logit_U": float(row_logits[2].detach().cpu()),
                    "R": float(row_logits[0].detach().cpu()) - float(row_logits[1].detach().cpu()),
                    "U_gap": float(row_logits[2].detach().cpu()) - max(float(row_logits[0].detach().cpu()), float(row_logits[1].detach().cpu())),
                    "pred_label": pred_label,
                    "is_correct": int(pred_label == row["target_label"]),
                    "global_top_token_id": top_token_id,
                    "global_top_token": tokenizer.decode([top_token_id]),
                    "global_top_in_TFU": int(top_token_id in label_id_set),
                }
            )
            scored.append(out)
    return summarize_scored(scored), scored


def polarity_split(scored: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cells: dict[str, list[float]] = {}
    for row in scored:
        if str(row.get("control_type")) != "main":
            continue
        key = f"p_i_base={row.get('p_i_base')},claim={row.get('base_claim_polarity')}"
        cells.setdefault(key, []).append(float(row["is_correct"]))
    return {
        key: {"IIA": sum(values) / len(values), "n": len(values)}
        for key, values in sorted(cells.items())
    }


def run_config(
    name: str,
    specs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    torch: Any,
    pv: Any,
    device: Any,
    label_tokens: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    site = specs[0]["site"]
    print(f"\n=== multilayer eval {name} (layers {[spec['layer'] for spec in specs]}, site {site}) ===")
    intervenable = build_multi_intervenable(pv, model, specs)
    set_intervenable_device(intervenable, device)
    load_rotation_weights(intervenable, specs, torch)
    metrics, scored = evaluate_multi(
        intervenable=intervenable,
        n_interventions=len(specs),
        rows=rows,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        label_token_ids=label_tokens.token_ids,
        batch_size=args.eval_batch_size,
        site_override=site,
        progress_desc=f"Eval {name}",
    )
    config_dir = output_dir / name
    config_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(scored, config_dir / f"{args.split}_scored.csv")
    split = polarity_split(scored)
    summary = {
        "name": name,
        "layers": [spec["layer"] for spec in specs],
        "rotation_dirs": [str(spec["dir"]) for spec in specs],
        "site": site,
        "split": args.split,
        "metrics": metrics,
        "main_polarity_split": split,
    }
    (config_dir / "summary_metrics.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    main_iia = (metrics.get("by_control", {}).get("main") or {}).get("IIA")
    print(f"{name}: main IIA = {main_iia}")
    for key, cell in split.items():
        print(f"  {key}: {cell['IIA']:.3f} (n={cell['n']})")
    del intervenable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = build_parser().parse_args()
    specs = load_specs(args.rotation_dirs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_rows_csv(args.samples) if row.get("target_var") == args.target_var]
    if not args.include_relaxed:
        rows = drop_relaxed_rows(rows)
    rows = rows_for_split(rows, args.split)
    if not rows:
        raise ValueError(f"No rows for target_var={args.target_var!r} split={args.split!r}")
    print(f"Evaluating {len(rows)} rows ({args.split} split)")

    torch, pv, auto_model_cls, auto_tokenizer_cls = import_runtime()
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
    device = get_input_device(model, torch, args.device)

    summaries = []
    if not args.skip_singles:
        for spec in specs:
            summaries.append(
                run_config(f"single_L{spec['layer']:02d}", [spec], rows, model, tokenizer, torch, pv, device, label_tokens, args, output_dir)
            )
    combined_name = "combined_" + "_".join(f"L{spec['layer']:02d}" for spec in specs)
    summaries.append(
        run_config(combined_name, specs, rows, model, tokenizer, torch, pv, device, label_tokens, args, output_dir)
    )

    (output_dir / "multilayer_summary.json").write_text(json.dumps(to_jsonable(summaries), indent=2), encoding="utf-8")
    print(f"\nWrote {len(summaries)} configs to {output_dir / 'multilayer_summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-var", default="pi")
    parser.add_argument("--rotation-dirs", nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--skip-singles", action="store_true")
    parser.add_argument("--include-relaxed", action="store_true")
    parser.add_argument("--label-token-style", default="auto")
    parser.add_argument("--output-dir", default="data/das/multilayer_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
