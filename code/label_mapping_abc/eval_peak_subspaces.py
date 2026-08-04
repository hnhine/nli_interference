#!/usr/bin/env python3
"""Evaluate frozen peak DAS subspaces after remapping T/F/U to A/B/C.

This is an evaluation-only diagnostic. It never trains or modifies a rotation.
The semantic mapping is fixed as T->A, F->B, U->C, while the natural-language
definitions remain "must be true", "must be false", and "cannot be determined".
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from interference_suite.das_pyvene import (
    build_intervenable,
    call_intervenable,
    chunks,
    collect_rotation_weights,
    collate_rows,
    drop_relaxed_rows,
    encode_to_device,
    get_input_device,
    import_runtime,
    load_hf_model,
    next_token_logits,
    rows_for_split,
    set_intervenable_device,
    to_jsonable,
)
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, progress_iter


TFU_TO_ABC = {"T": "A", "F": "B", "U": "C"}
ABC_TO_TFU = {value: key for key, value in TFU_TO_ABC.items()}
ABC_ORDER = ("A", "B", "C")

TFU_LEGEND = (
    "Choose exactly one:\n"
    "T = must be true\n"
    "F = must be false\n"
    "U = cannot be determined"
)
ABC_LEGEND = (
    "Choose exactly one:\n"
    "A = must be true\n"
    "B = must be false\n"
    "C = cannot be determined"
)

IDENTIFYING_CONTROLS = {
    "pc": ("main", "probe_flip_both", "probe_flip_pi"),
    "pi": ("main", "active_source_m0", "probe_flip_both", "probe_flip_pc"),
    "rho": (
        "flip_pi",
        "flip_pc",
        "hold_both",
        "source_m0",
        "gate_m0",
        "label_copy_trap",
    ),
    "m": (
        "match_to_nomatch",
        "nomatch_to_match",
        "label_copy_trap",
        "label_copy_trap_same_m1",
    ),
}

MODEL_SPECS = {
    "phi4": {
        "model_name": "microsoft/Phi-4-mini-instruct",
        "datasets": {
            "pc": "data/das/pc_v4/pairs.csv",
            "pi": "data/das/pi_v5/pairs.csv",
            "rho": "data/das/rho_v1/pairs.csv",
            "m": "data/das/m_v4/pairs.csv",
        },
        "rotation_roots": {
            "pc": "data/das/seed_sweep_phi4_pc",
            "pi": "data/das/seed_sweep_phi4_pi",
            "rho": "data/das/seed_sweep_phi4_rho",
            "m": "data/das/seed_sweep_phi4_m",
        },
        "peaks": {
            "pc": {"claim_final": 10, "answer_token": 16},
            "pi": {"claim_final": 10, "answer_token": 16},
            "rho": {"claim_final": 12, "answer_token": 16},
            "m": {"claim_final": 12, "answer_token": 16},
        },
    },
    "qwen": {
        "model_name": "Qwen/Qwen3-8B",
        "datasets": {
            "pc": "data/das/pc_v4/pairs.csv",
            "pi": "data/das/pi_v5/pairs.csv",
            "rho": "data/das/rho_v1/pairs.csv",
            "m": "data/das/m_v4/pairs.csv",
        },
        "rotation_roots": {
            "pc": "data/das/seed_sweep_qwen_pc_r64",
            "pi": "data/das/seed_sweep_qwen_pi_r64",
            "rho": "data/das/seed_sweep_qwen_rho_r64",
            "m": "data/das/seed_sweep_qwen_m_r64",
        },
        "peaks": {
            "pc": {"claim_final": 14, "answer_token": 22},
            "pi": {"claim_final": 16, "answer_token": 22},
            "rho": {"claim_final": 18, "answer_token": 24},
            "m": {"claim_final": 16, "answer_token": 22},
        },
    },
}


def remap_prompt(prompt: str) -> str:
    """Replace only the label symbols, preserving definitions and char length."""

    if prompt.count(TFU_LEGEND) != 1:
        raise ValueError("Prompt does not contain exactly one canonical T/F/U legend")
    remapped = prompt.replace(TFU_LEGEND, ABC_LEGEND)
    if len(remapped) != len(prompt):
        raise AssertionError("A/B/C remapping unexpectedly changed character offsets")
    return remapped


def remap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["base_prompt"] = remap_prompt(str(row["base_prompt"]))
        item["source_prompt"] = remap_prompt(str(row["source_prompt"]))
        item["base_symbol"] = TFU_TO_ABC[str(row["base_label"])]
        item["source_symbol"] = TFU_TO_ABC[str(row["source_label"])]
        item["target_symbol"] = TFU_TO_ABC[str(row["target_label"])]
        output.append(item)
    return output


def balanced_limit(rows: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    if max_rows is None or max_rows >= len(rows):
        return rows
    if max_rows < 1:
        raise ValueError("--max-rows must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("control_type", ""))].append(row)
    quota = math.ceil(max_rows / len(groups))
    selected = [row for name in sorted(groups) for row in groups[name][:quota]]
    selected.sort(key=lambda row: int(row.get("row_id", 0)))
    return selected[:max_rows]


def resolve_abc_tokens(tokenizer: Any, requested_style: str) -> dict[str, Any]:
    styles = ["bare", "space", "newline"] if requested_style == "auto" else [requested_style]
    builders = {
        "bare": lambda value: value,
        "space": lambda value: f" {value}",
        "newline": lambda value: f"\n{value}",
    }
    for style in styles:
        token_ids: dict[str, int] = {}
        token_texts: dict[str, str] = {}
        for symbol in ABC_ORDER:
            text = builders[style](symbol)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                break
            token_ids[symbol] = int(ids[0])
            token_texts[symbol] = text
        if len(token_ids) == 3 and len(set(token_ids.values())) == 3:
            return {"style": style, "token_ids": token_ids, "token_texts": token_texts}
    raise ValueError("Could not resolve A/B/C as three distinct single tokens")


def abc_logits(torch: Any, next_logits: Any, token_ids: dict[str, int]) -> Any:
    return torch.stack([next_logits[:, token_ids[symbol]] for symbol in ABC_ORDER], dim=-1)


def prediction_record(
    *,
    row_logits: Any,
    pred_index: Any,
    top_id: Any,
    tokenizer: Any,
    token_ids: dict[str, int],
    expected_symbol: str,
) -> dict[str, Any]:
    values = [float(value.detach().cpu()) for value in row_logits]
    pred_symbol = ABC_ORDER[int(pred_index.detach().cpu())]
    top_token_id = int(top_id.detach().cpu())
    return {
        "logit_A": values[0],
        "logit_B": values[1],
        "logit_C": values[2],
        "R_AB": values[0] - values[1],
        "C_gap": values[2] - max(values[0], values[1]),
        "pred_symbol": pred_symbol,
        "pred_semantic_label": ABC_TO_TFU[pred_symbol],
        "is_correct": int(pred_symbol == expected_symbol),
        "global_top_token_id": top_token_id,
        "global_top_token": tokenizer.decode([top_token_id]),
        "global_top_in_ABC": int(top_token_id in set(token_ids.values())),
    }


def score_baseline(
    *,
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    token_ids: dict[str, int],
    batch_size: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    total = math.ceil(len(rows) / batch_size)
    iterator = progress_iter(chunks(rows, batch_size), total=total, desc="ABC baseline")
    for batch_rows in iterator:
        inputs = encode_to_device(tokenizer, [str(row["base_prompt"]) for row in batch_rows], device)
        lengths = inputs["attention_mask"].sum(dim=1)
        with torch.no_grad():
            outputs = model(**inputs)
            next_logits = next_token_logits(torch, outputs, lengths)
            selected = abc_logits(torch, next_logits, token_ids)
            predictions = selected.argmax(dim=-1)
            top_ids = next_logits.argmax(dim=-1)
        for row, logits, pred, top_id in zip(batch_rows, selected, predictions, top_ids):
            out = dict(row)
            out.update(
                prediction_record(
                    row_logits=logits,
                    pred_index=pred,
                    top_id=top_id,
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    expected_symbol=str(row["base_symbol"]),
                )
            )
            scored.append(out)
    return scored


def load_rotation(
    *,
    rotation_dir: Path,
    model_name: str,
    target: str,
    layer: int,
    site: str,
) -> tuple[dict[str, Any], Any]:
    import numpy as np

    metadata_path = rotation_dir / "rotation_weight_metadata.json"
    weight_path = rotation_dir / "rotation_weight.npy"
    summary_path = rotation_dir / "summary_metrics.json"
    for path in (metadata_path, weight_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "model_name": model_name,
        "target_var": target,
        "layer": layer,
        "site": site,
        "rank": 64,
    }
    for field, value in expected.items():
        actual = metadata.get(field)
        if str(actual) != str(value):
            raise ValueError(f"{metadata_path}: {field}={actual!r}, expected={value!r}")
    return metadata, np.load(weight_path)


def install_rotation(intervenable: Any, array: Any, torch: Any) -> None:
    weights = collect_rotation_weights(intervenable)
    if len(weights) != 1:
        raise RuntimeError(f"Expected one intervention weight, found {list(weights)}")
    weight = next(iter(weights.values()))
    tensor = torch.tensor(array, dtype=torch.float32)
    if tuple(tensor.shape) != tuple(weight.shape):
        raise ValueError(f"Rotation shape {tuple(tensor.shape)} != runtime shape {tuple(weight.shape)}")
    with torch.no_grad():
        weight.copy_(tensor.to(device=weight.device, dtype=weight.dtype))


def score_intervention(
    *,
    rows: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    intervenable: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    token_ids: dict[str, int],
    batch_size: int,
    site: str,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    total = math.ceil(len(rows) / batch_size)
    iterator = progress_iter(chunks(rows, batch_size), total=total, desc=f"ABC patch {site}")
    for batch_rows in iterator:
        batch = collate_rows(batch_rows, tokenizer, torch, device, site)
        with torch.no_grad():
            outputs = call_intervenable(intervenable, batch)
            next_logits = next_token_logits(torch, outputs, batch["input_lengths"])
            selected = abc_logits(torch, next_logits, token_ids)
            predictions = selected.argmax(dim=-1)
            top_ids = next_logits.argmax(dim=-1)
        for row, logits, pred, top_id in zip(batch_rows, selected, predictions, top_ids):
            sample_id = str(row["sample_id"])
            baseline = baseline_by_id[sample_id]
            out = dict(row)
            out.update(
                {
                    "base_pred_symbol": baseline["pred_symbol"],
                    "base_pred_semantic_label": baseline["pred_semantic_label"],
                    "base_is_correct_abc": baseline["is_correct"],
                }
            )
            out.update(
                prediction_record(
                    row_logits=logits,
                    pred_index=pred,
                    top_id=top_id,
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    expected_symbol=str(row["target_symbol"]),
                )
            )
            scored.append(out)
    return scored


def safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return mean(values) if values else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_control: dict[str, dict[str, Any]] = {}
    for control in sorted({str(row["control_type"]) for row in rows}):
        group = [row for row in rows if str(row["control_type"]) == control]
        eligible = [row for row in group if int(row.get("base_is_correct_abc", 0)) == 1]
        by_control[control] = {
            "n": len(group),
            "IIA_ABC": safe_mean(float(row["is_correct"]) for row in group),
            "n_base_correct": len(eligible),
            "IIA_ABC_base_correct": safe_mean(float(row["is_correct"]) for row in eligible),
            "global_top_in_ABC_rate": safe_mean(float(row["global_top_in_ABC"]) for row in group),
            "mean_R_AB": safe_mean(float(row["R_AB"]) for row in group),
            "mean_C_gap": safe_mean(float(row["C_gap"]) for row in group),
        }
    eligible = [row for row in rows if int(row.get("base_is_correct_abc", 0)) == 1]
    return {
        "n": len(rows),
        "IIA_ABC": safe_mean(float(row["is_correct"]) for row in rows),
        "n_base_correct": len(eligible),
        "IIA_ABC_base_correct": safe_mean(float(row["is_correct"]) for row in eligible),
        "global_top_in_ABC_rate": safe_mean(float(row["global_top_in_ABC"]) for row in rows),
        "by_control": by_control,
    }


def summarize_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy_ABC": safe_mean(float(row["is_correct"]) for row in rows),
        "global_top_in_ABC_rate": safe_mean(float(row["global_top_in_ABC"]) for row in rows),
        "prediction_counts": {
            symbol: sum(str(row["pred_symbol"]) == symbol for row in rows)
            for symbol in ABC_ORDER
        },
    }


def reference_validation(rotation_dir: Path) -> dict[str, Any]:
    summary = json.loads((rotation_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    validation = summary.get("val")
    if not isinstance(validation, dict):
        raise ValueError(f"{rotation_dir}: missing validation summary")
    return validation


def preflight(args: argparse.Namespace, spec: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for target in args.targets:
        dataset = Path(spec["datasets"][target])
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        for site in args.sites:
            layer = int(spec["peaks"][target][site])
            for seed in args.seeds:
                rotation_dir = Path(spec["rotation_roots"][target]) / f"seed{seed}" / f"L{layer}_{site}"
                metadata, _ = load_rotation(
                    rotation_dir=rotation_dir,
                    model_name=str(spec["model_name"]),
                    target=target,
                    layer=layer,
                    site=site,
                )
                cells.append(
                    {
                        "target": target,
                        "site": site,
                        "layer": layer,
                        "seed": seed,
                        "rotation_dir": rotation_dir,
                        "metadata": metadata,
                    }
                )
    return cells


def aggregate_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        key = (str(cell["model_alias"]), str(cell["target"]), str(cell["site"]), int(cell["layer"]))
        grouped[key].append(cell)

    output: list[dict[str, Any]] = []
    for (model_alias, target, site, layer), group in sorted(grouped.items()):
        controls = IDENTIFYING_CONTROLS[target]
        abc_control_means = {
            control: mean(float(cell["abc_by_control"][control]["IIA_ABC"]) for cell in group)
            for control in controls
        }
        tfu_control_means = {
            control: mean(float(cell["tfu_by_control"][control]["IIA"]) for cell in group)
            for control in controls
        }
        abc_values = [float(cell["abc_iia"]) for cell in group]
        tfu_values = [float(cell["tfu_val_iia"]) for cell in group]
        tfu_reference_same_rows = all(bool(cell["tfu_reference_same_rows"]) for cell in group)
        confirmatory_delta = min(abc_control_means.values()) - min(tfu_control_means.values()) if tfu_reference_same_rows else None
        output.append(
            {
                "model": model_alias,
                "target": target,
                "site": site,
                "layer": layer,
                "n_seeds": len(group),
                "seeds": " ".join(str(cell["seed"]) for cell in sorted(group, key=lambda value: value["seed"])),
                "ABC_baseline_accuracy_mean": mean(float(cell["abc_baseline_accuracy"]) for cell in group),
                "ABC_IIA_mean": mean(abc_values),
                "ABC_IIA_sd": pstdev(abc_values) if len(abc_values) > 1 else 0.0,
                "ABC_IIA_base_correct_mean": safe_mean(
                    float(cell["abc_iia_base_correct"])
                    for cell in group
                    if cell["abc_iia_base_correct"] is not None
                ),
                "TFU_reference_val_IIA_mean": mean(tfu_values),
                "confirmatory_ABC_IIA": min(abc_control_means.values()),
                "confirmatory_TFU_reference_IIA": min(tfu_control_means.values()),
                "confirmatory_delta_ABC_minus_TFU": confirmatory_delta,
                "ABC_control_means": json.dumps(abc_control_means, sort_keys=True),
                "TFU_reference_control_means": json.dumps(tfu_control_means, sort_keys=True),
            }
        )
    return output


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[args.model]
    requested_cells = preflight(args, spec)
    output_root = Path(args.output_dir) / args.model

    if args.dry_run:
        printable = [
            {
                "target": cell["target"],
                "site": cell["site"],
                "layer": cell["layer"],
                "seed": cell["seed"],
                "rotation_dir": str(cell["rotation_dir"]),
            }
            for cell in requested_cells
        ]
        print(json.dumps(printable, indent=2))
        return 0

    torch, pv, auto_model_cls, auto_tokenizer_cls = import_runtime()
    tokenizer, model = load_hf_model(
        torch=torch,
        auto_model_cls=auto_model_cls,
        auto_tokenizer_cls=auto_tokenizer_cls,
        model_name=str(spec["model_name"]),
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    device = get_input_device(model, torch, args.device)
    label_tokens = resolve_abc_tokens(tokenizer, args.label_token_style)
    print(f"A/B/C label tokens: {label_tokens}")

    cell_summaries: list[dict[str, Any]] = []
    cells_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in requested_cells:
        cells_by_target[str(cell["target"])].append(cell)

    for target in args.targets:
        rows = read_rows_csv(spec["datasets"][target])
        rows = [row for row in rows if str(row.get("target_var")) == target]
        rows = rows_for_split(drop_relaxed_rows(rows), args.split)
        rows = balanced_limit(rows, args.max_rows)
        rows = remap_rows(rows)
        print(f"\n=== {args.model} {target}: {len(rows)} remapped {args.split} rows ===")

        baseline_scored = score_baseline(
            rows=rows,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            token_ids=label_tokens["token_ids"],
            batch_size=args.eval_batch_size,
        )
        baseline_summary = summarize_baseline(baseline_scored)
        target_root = output_root / target
        target_root.mkdir(parents=True, exist_ok=True)
        write_rows_csv(baseline_scored, target_root / f"{args.split}_abc_baseline_scored.csv")
        (target_root / f"{args.split}_abc_baseline_summary.json").write_text(
            json.dumps(to_jsonable(baseline_summary), indent=2), encoding="utf-8"
        )
        baseline_by_id = {str(row["sample_id"]): row for row in baseline_scored}

        for cell in sorted(cells_by_target[target], key=lambda value: (value["site"], value["seed"])):
            rotation_dir = Path(cell["rotation_dir"])
            metadata, rotation = load_rotation(
                rotation_dir=rotation_dir,
                model_name=str(spec["model_name"]),
                target=target,
                layer=int(cell["layer"]),
                site=str(cell["site"]),
            )
            print(
                f"\n--- {args.model} {target} seed{cell['seed']} "
                f"L{cell['layer']}_{cell['site']} ---"
            )
            intervenable = build_intervenable(
                pv,
                model,
                int(metadata["layer"]),
                int(metadata["rank"]),
                str(metadata.get("component", "block_output")),
            )
            set_intervenable_device(intervenable, device)
            install_rotation(intervenable, rotation, torch)
            scored = score_intervention(
                rows=rows,
                baseline_by_id=baseline_by_id,
                intervenable=intervenable,
                tokenizer=tokenizer,
                torch=torch,
                device=device,
                token_ids=label_tokens["token_ids"],
                batch_size=args.eval_batch_size,
                site=str(cell["site"]),
            )
            abc_summary = summarize_rows(scored)
            tfu_reference = reference_validation(rotation_dir)
            cell_dir = target_root / f"seed{cell['seed']}" / f"L{cell['layer']}_{cell['site']}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            write_rows_csv(scored, cell_dir / f"{args.split}_abc_intervention_scored.csv")
            summary = {
                "experiment": "frozen_peak_label_mapping_abc",
                "model_alias": args.model,
                "model_name": spec["model_name"],
                "target": target,
                "split": args.split,
                "seed": int(cell["seed"]),
                "site": str(cell["site"]),
                "layer": int(cell["layer"]),
                "rank": int(metadata["rank"]),
                "rotation_dir": str(rotation_dir),
                "mapping": TFU_TO_ABC,
                "definitions": {
                    "A": "must be true",
                    "B": "must be false",
                    "C": "cannot be determined",
                },
                "label_tokens": label_tokens,
                "abc_baseline": baseline_summary,
                "abc_intervention": abc_summary,
                "tfu_validation_reference": tfu_reference,
                "tfu_reference_is_same_rows": args.max_rows is None,
            }
            (cell_dir / "summary.json").write_text(
                json.dumps(to_jsonable(summary), indent=2), encoding="utf-8"
            )
            cell_summaries.append(
                {
                    "model_alias": args.model,
                    "target": target,
                    "site": str(cell["site"]),
                    "layer": int(cell["layer"]),
                    "seed": int(cell["seed"]),
                    "abc_baseline_accuracy": baseline_summary["accuracy_ABC"],
                    "abc_iia": abc_summary["IIA_ABC"],
                    "abc_iia_base_correct": abc_summary["IIA_ABC_base_correct"],
                    "abc_by_control": abc_summary["by_control"],
                    "tfu_val_iia": tfu_reference["IIA"],
                    "tfu_by_control": tfu_reference["by_control"],
                    "tfu_reference_same_rows": args.max_rows is None,
                }
            )
            del intervenable
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = aggregate_cells(cell_summaries)
    write_csv(aggregate, output_root / "aggregate.csv")
    (output_root / "aggregate.json").write_text(
        json.dumps(to_jsonable(aggregate), indent=2), encoding="utf-8"
    )
    print(f"\nWrote aggregate: {output_root / 'aggregate.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--targets", nargs="+", choices=tuple(IDENTIFYING_CONTROLS), default=list(IDENTIFYING_CONTROLS))
    parser.add_argument("--sites", nargs="+", choices=("claim_final", "answer_token"), default=["claim_final", "answer_token"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--label-token-style", default="auto", choices=("auto", "bare", "space", "newline"))
    parser.add_argument("--output-dir", default="data/das/label_mapping_abc")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
