"""Generate and evaluate the preregistered Section 6.1 experiment suite."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from interference_suite.das_pyvene import import_runtime, load_hf_model
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, evaluate_rows, resolve_label_tokens
from interference_suite.negation_forms import NEW_FORM_KEYS, TIER1_FORM_KEYS
from interference_suite.section61_data import (
    RHO_CONTROLS,
    anchor_mismatches,
    derive_double_negation,
    derive_local_form,
    derive_m_form,
    derive_rho_cross,
    derive_rho_mixed,
    derive_rho_within,
    filter_das_rows,
    generate_behavioral_rows,
    source_as_base,
    unique_test_events,
)
from interference_suite.section61_metrics import (
    behavioral_summary,
    das_summary,
    flatten_das_summary,
    main_table_rows,
)
from run_das_ablation import collect_subspace_coordinates, get_decoder_layers, run_condition


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv_records(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        write_rows_csv(rows, path)


def generate_command(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rho_all = read_rows_csv(args.rho_samples)
    rho_rows = filter_das_rows(rho_all, "rho", RHO_CONTROLS, args.split)
    events = unique_test_events(rho_all, args.n_events)

    e1 = generate_behavioral_rows(events)
    e7_behavior = generate_behavioral_rows(
        events, forms=("double_negation",), double_negation=True
    )
    e2 = [
        row
        for form in TIER1_FORM_KEYS
        for row in derive_rho_within(rho_rows, form)
    ]
    canonical = [row for row in e2 if row["form"] == "did_not"]
    if anchor_mismatches(rho_rows, canonical):
        raise RuntimeError("Canonical did_not renderer changed original rho prompts")

    e3 = [
        row
        for form in NEW_FORM_KEYS
        for row in derive_rho_cross(rho_rows, form)
    ]
    e6 = [
        *derive_rho_mixed(rho_rows, "never", "did_not"),
        *derive_rho_mixed(rho_rows, "did_not", "never"),
    ]
    e7 = derive_double_negation(rho_rows)

    outputs: dict[str, list[dict[str, Any]]] = {
        "e1_behavioral.csv": e1,
        "e2_rho_within.csv": e2,
        "e3_rho_cross.csv": e3,
        "e6_rho_mixed.csv": e6,
        "e7_behavioral.csv": e7_behavior,
        "e7_rho_double_negation.csv": e7,
    }

    if args.pi_samples:
        pi_rows = filter_das_rows(read_rows_csv(args.pi_samples), "pi", split=args.split)
        outputs["e4_pi_within.csv"] = [
            row
            for form in TIER1_FORM_KEYS
            for row in derive_local_form(pi_rows, "pi", form)
        ]
    if args.pc_samples:
        pc_rows = filter_das_rows(read_rows_csv(args.pc_samples), "pc", split=args.split)
        outputs["e4_pc_within.csv"] = [
            row
            for form in TIER1_FORM_KEYS
            for row in derive_local_form(pc_rows, "pc", form)
        ]
    if args.m_samples:
        m_rows = filter_das_rows(read_rows_csv(args.m_samples), "m", split=args.split)
        m_rows = [
            row
            for row in m_rows
            if str(row.get("mismatch_exclusion_relaxed", "0") or "0") != "1"
        ]
        for form in TIER1_FORM_KEYS:
            outputs[f"e5_m_{form}.csv"] = derive_m_form(m_rows, form)

    for filename, rows in outputs.items():
        write_rows_csv(rows, output / filename)

    manifest = {
        "split": args.split,
        "n_events": len(events),
        "event_ids": [base_id for base_id, _ in events],
        "inputs": {
            "rho": args.rho_samples,
            "pi": args.pi_samples,
            "pc": args.pc_samples,
            "m": args.m_samples,
        },
        "forms": list(TIER1_FORM_KEYS),
        "counts": {name: len(rows) for name, rows in outputs.items()},
        "rho_controls": list(RHO_CONTROLS),
        "anchor_mismatches": 0,
    }
    write_json(manifest, output / "manifest.json")
    print(f"Wrote Section 6.1 datasets and manifest to {output}")
    return 0


def behavioral_command(args: argparse.Namespace) -> int:
    rows = read_rows_csv(args.samples)
    scored = evaluate_rows(
        rows,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        label_token_style=args.label_token_style,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(scored, output / "scored.csv")
    summary = behavioral_summary(
        scored,
        model_name=args.model_name,
        gate_threshold=args.gate_threshold,
    )
    write_json(summary, output / "behavioral_summary.json")
    write_csv_records(summary["by_form_cell"], output / "behavioral_by_form_cell.csv")
    write_csv_records(summary["by_form"], output / "behavioral_by_form.csv")
    print(f"Wrote behavioral scores and gate summary to {output}")
    return 0


def load_rotation(path: Path, model_name: str, target_var: str) -> tuple[dict[str, Any], Any]:
    import numpy as np

    metadata_path = path / "rotation_weight_metadata.json"
    weight_path = path / "rotation_weight.npy"
    if not metadata_path.exists() or not weight_path.exists():
        raise FileNotFoundError(f"Missing rotation metadata/weight under {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("target_var")) != target_var:
        raise ValueError(
            f"Rotation target={metadata.get('target_var')!r}, samples target={target_var!r}"
        )
    if str(metadata.get("model_name")).lower() != model_name.lower():
        raise ValueError(
            f"Rotation model={metadata.get('model_name')!r}, requested={model_name!r}"
        )
    return metadata, np.load(weight_path)


def das_command(args: argparse.Namespace) -> int:
    rows = [
        row
        for samples_path in args.samples
        for row in read_rows_csv(samples_path)
    ]
    target_vars = {str(row.get("target_var")) for row in rows}
    if len(target_vars) != 1:
        raise ValueError(f"One DAS run requires one target_var, got {sorted(target_vars)}")
    target_var = next(iter(target_vars))
    if args.require_gate_summary:
        gate_payload = json.loads(
            Path(args.require_gate_summary).read_text(encoding="utf-8")
        )
        eligible = {
            str(record["form"]): bool(record["eligible"])
            for record in gate_payload.get("by_form", [])
        }
        requested_forms = {str(row.get("form")) for row in rows}
        failed = sorted(form for form in requested_forms if not eligible.get(form, False))
        if failed:
            raise RuntimeError(f"Behavioral gate failed or is missing for forms: {failed}")
    rotation_dir = Path(args.rotation_dir)
    metadata, raw_rotation = load_rotation(rotation_dir, args.model_name, target_var)

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
    labels = resolve_label_tokens(tokenizer, args.label_token_style)
    device = next(model.parameters()).device
    layers = get_decoder_layers(model)
    hidden = int(model.config.hidden_size)
    layer = int(metadata["layer"])
    rank = int(metadata["rank"])
    site = str(metadata.get("site", "claim_final"))
    if tuple(raw_rotation.shape) != (hidden, rank):
        raise ValueError(
            f"Rotation shape={tuple(raw_rotation.shape)}, expected={(hidden, rank)}"
        )
    rotation = torch.tensor(raw_rotation, dtype=torch.float32, device=device)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("section61_experiment")),
            str(row.get("form")),
            str(row.get("direction", "")),
        )
        grouped[key].append(row)

    rng = random.Random(args.seed)
    all_scored: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        donors = [source_as_base(row) for row in group]
        donor_z = collect_subspace_coordinates(
            model, layer, site, rotation, donors, tokenizer, torch, device,
            args.eval_batch_size,
        )
        scored = run_condition(
            model, layers, layer, site, rotation, "resample_same", group,
            tokenizer, torch, device, labels.token_ids, args.eval_batch_size,
            rng, donor_z=donor_z,
        )
        for source_row, scored_row in zip(group, scored):
            for column in (
                "section61_experiment", "form", "direction", "base_form",
                "source_form", "base_site", "source_site", "rho_base", "rho_src",
            ):
                scored_row[column] = source_row.get(column, "")
        all_scored.extend(scored)
        summaries.append(
            das_summary(scored, model_name=args.model_name, rotation=metadata)
        )
        print(f"Completed {key}: n={len(scored)}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(all_scored, output / "scored.csv")
    write_json(
        {
            "model_name": args.model_name,
            "rotation_dir": str(rotation_dir),
            "rotation": metadata,
            "cells": summaries,
        },
        output / "das_summary.json",
    )
    write_csv_records(
        [flatten_das_summary(summary) for summary in summaries],
        output / "das_summary.csv",
    )
    print(f"Wrote DAS scores and summaries to {output}")
    return 0


def read_summary_files(paths: list[str], key: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        value = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if key is None:
            records.append(value)
        else:
            records.extend(value.get(key, []))
    return records


def latex_value(value: Any) -> str:
    if value is None or value == "":
        return "--"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (float, int)):
        return f"{float(value):.3f}"
    return str(value).replace("_", r"\_")


def write_latex_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Negation form & \multicolumn{2}{c}{Behavior min} & \multicolumn{2}{c}{Within IIA} & \multicolumn{2}{c}{Cross IIA} \\",
        r" & Phi & Qwen & Phi & Qwen & Phi & Qwen \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_value(row["form"]),
                    latex_value(row.get("phi_behavior_min")),
                    latex_value(row.get("qwen_behavior_min")),
                    latex_value(row.get("phi_within")),
                    latex_value(row.get("qwen_within")),
                    latex_value(row.get("phi_cross")),
                    latex_value(row.get("qwen_cross")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_command(args: argparse.Namespace) -> int:
    behavioral = read_summary_files(args.behavioral_summaries)
    das = read_summary_files(args.das_summaries, "cells")
    table = main_table_rows(behavioral, das)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(table, output / "section61_main_table.csv")
    write_latex_table(table, output / "section61_main_table.tex")
    write_csv_records(
        [flatten_das_summary(record) for record in das],
        output / "section61_das_appendix.csv",
    )
    control_rows = [
        {
            **flatten_das_summary(record),
            **control,
        }
        for record in das
        for control in record.get("by_control", [])
    ]
    write_csv_records(control_rows, output / "section61_das_by_control.csv")
    e6_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in das:
        if str(record.get("section61_experiment")) == "E6":
            e6_groups[
                (
                    str(record.get("model_name")),
                    str(record.get("reported_site")),
                    int(record.get("layer", -1)),
                )
            ].append(record)
    e6_min = [
        {"model_name": key[0], "site": key[1], "layer": key[2],
         "worst_direction_IIA": min(float(item["rho_full_audit_min_IIA"]) for item in values)}
        for key, values in sorted(e6_groups.items())
    ]
    write_csv_records(e6_min, output / "section61_e6_worst_direction.csv")
    write_csv_records(
        [cell for record in behavioral for cell in record.get("by_form_cell", [])],
        output / "section61_behavioral_by_cell.csv",
    )
    write_json(
        {"behavioral": behavioral, "das": das, "main_table": table},
        output / "section61_summary.json",
    )
    print(f"Wrote Section 6.1 aggregate table and summary to {output}")
    return 0


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--label-token-style", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate")
    generate.add_argument("--rho-samples", required=True)
    generate.add_argument("--pi-samples")
    generate.add_argument("--pc-samples")
    generate.add_argument("--m-samples")
    generate.add_argument("--split", default="test")
    generate.add_argument("--n-events", type=int, default=150)
    generate.add_argument("--output-dir", required=True)

    behavioral = commands.add_parser("behavioral")
    behavioral.add_argument("--samples", required=True)
    behavioral.add_argument("--gate-threshold", type=float, default=0.90)
    behavioral.add_argument("--output-dir", required=True)
    add_model_args(behavioral)

    das = commands.add_parser("das")
    das.add_argument("--samples", nargs="+", required=True)
    das.add_argument("--rotation-dir", required=True)
    das.add_argument("--seed", type=int, default=0)
    das.add_argument("--output-dir", required=True)
    das.add_argument(
        "--require-gate-summary", help="Abort unless every requested form passed its behavioral gate."
    )
    add_model_args(das)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--behavioral-summaries", nargs="+", required=True)
    summarize.add_argument("--das-summaries", nargs="+", required=True)
    summarize.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return {
        "generate": generate_command,
        "behavioral": behavioral_command,
        "das": das_command,
        "summarize": summarize_command,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
