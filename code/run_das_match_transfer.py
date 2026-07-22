"""Match-transfer diagnostics for the atomic DAS variable ``m``.

The m generator provides exact reverse minimal pairs for two core controls:
``match_to_nomatch`` and ``nomatch_to_match``.  This script reuses the trained
rotation and reports, per layer/site:

* A: natural behavioral gate by direction, mismatch type, slot and m;
* B: geometry of match vs the four no-match types;
* C: paired counterfactual interchange in both directions (plus label trap);
* D: 4x4 same-m=0 mismatch-type donor purity;
* E: 3x3 same-m cross-slot donor purity for m=0 and m=1.

No brittle automatic hypothesis label is emitted: raw metrics are the
diagnostic.  The donor matrices are intentionally strict: same polarity,
same slot/type where requested, and a different base event.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from interference_suite.das_pyvene import import_runtime, load_hf_model, mean, to_jsonable
from interference_suite.das_match import (
    CORE_M_CONTROLS as CORE_CONTROLS,
    MISMATCH_TYPES,
    source_as_base,
    validate_and_pair_core_rows,
)
from interference_suite.das_spans import resolve_token_site
from interference_suite.io_utils import read_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, resolve_label_tokens
from run_das_ablation import collect_subspace_coordinates, get_decoder_layers, run_condition


def select_rows(rows: list[dict[str, Any]], split: str, n_events: int,
                include_relaxed: bool) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row.get("target_var") == "m"
        and (split == "all" or row.get("split") == split)
        and (include_relaxed or str(row.get("mismatch_exclusion_relaxed", "0")) != "1")
        and row.get("control_type") in (*CORE_CONTROLS, "label_copy_trap")
    ]
    if n_events:
        event_ids = sorted({str(row["base_event_id"]) for row in selected})[:n_events]
        selected = [row for row in selected if str(row["base_event_id"]) in event_ids]
    if not selected:
        raise ValueError("No m rows matched split/filters")
    return selected


def validate_reverse_pairs(rows: list[dict[str, Any]]) -> None:
    validate_and_pair_core_rows(rows)


def group_rows(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return out


def metric_records(scored: list[dict[str, Any]], test: str, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group, values in group_rows(scored, *keys).items():
        preds = [str(row.get("pred_label")) for row in values]
        target = [str(row.get("counterfactual_label")) for row in values]
        base = [str(row.get("true_label")) for row in values]
        source = [str(row.get("source_label")) for row in values]
        record = {
            "test": test,
            **{key: value for key, value in zip(keys, group)},
            "n": len(values),
            "accuracy": mean(str(row.get("pred_label")) == str(row.get("true_label")) for row in values),
            "counterfactual_acc": mean(pred == wanted for pred, wanted in zip(preds, target) if wanted),
            "original_retention": mean(pred == wanted for pred, wanted in zip(preds, base)),
            "U_rate": mean(pred == "U" for pred in preds),
            "T_rate": mean(pred == "T" for pred in preds),
            "F_rate": mean(pred == "F" for pred in preds),
        }
        if any(source):
            record["source_copy_rate"] = mean(pred == wanted for pred, wanted in zip(preds, source) if wanted)
        records.append(record)
    return records


def geometry_records(z_match, z_nomatch, nomatch_rows: list[dict[str, Any]], cell: str) -> list[dict[str, Any]]:
    import torch

    mu_match = z_match.mean(0)
    mu_nomatch = z_nomatch.mean(0)
    axis = mu_match - mu_nomatch
    axis_len = float(axis.norm())
    if axis_len < 1e-8:
        raise ValueError(f"Degenerate match axis at {cell}")
    unit = axis / axis_len
    residuals = torch.cat((z_match - mu_match, z_nomatch - mu_nomatch), dim=0)
    within_orth = float((residuals - torch.outer(residuals @ unit, unit)).norm(dim=1).mean())

    records = [{
        "cell": cell, "test": "B_geometry", "group": "m=1", "n": len(z_match),
        "axis_coord_rel": 1.0, "orth_drift_over_within": 0.0,
        "centroid_classified_match": 1.0, "overlap_count": 3,
    }]
    for mismatch in MISMATCH_TYPES:
        mask = [str(row.get("mismatch_type")) == mismatch for row in nomatch_rows]
        if not any(mask):
            continue
        z = z_nomatch[mask]
        coord = float(((z - mu_nomatch) @ unit).mean()) / axis_len
        drift = z.mean(0) - mu_nomatch
        drift_orth = drift - (drift @ unit) * unit
        d_match = ((z - mu_match) ** 2).sum(1)
        d_nomatch = ((z - mu_nomatch) ** 2).sum(1)
        records.append({
            "cell": cell, "test": "B_geometry", "group": f"m=0:{mismatch}", "n": int(sum(mask)),
            "axis_coord_rel": round(coord, 4),
            "orth_drift_over_within": round(float(drift_orth.norm()) / max(within_orth, 1e-8), 4),
            "centroid_classified_match": round(float((d_match < d_nomatch).float().mean()), 4),
            "overlap_count": 0 if mismatch == "no_overlap" else 2,
        })
    return records


def strict_candidates(rows: list[dict[str, Any]], *, m_value: str, mismatch_type: str | None = None,
                      slot: str | None = None, pi: str | None = None, pc: str | None = None,
                      exclude_event: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if str(row.get("m_base")) == m_value
        and (mismatch_type is None or str(row.get("mismatch_type")) == mismatch_type)
        and (slot is None or str(row.get("matched_idx")) == slot)
        and (pi is None or str(row.get("p_i_base")) == pi)
        and (pc is None or str(row.get("p_c_base")) == pc)
        and str(row.get("base_event_id")) != exclude_event
    ]


def add_matrix_rows(base_rows: list[dict[str, Any]], donor_rows: list[dict[str, Any]], z_lookup: dict[str, Any],
                    *, test: str, matrix_m: str, base_field: str, donor_field: str,
                    candidate_kwargs, rng: random.Random) -> tuple[list[dict[str, Any]], list[Any]]:
    rows_out: list[dict[str, Any]] = []
    donor_z: list[Any] = []
    for base in base_rows:
        kwargs = dict(candidate_kwargs(base))
        candidates = strict_candidates(donor_rows, exclude_event=str(base.get("base_event_id")), **kwargs)
        if not candidates:
            raise ValueError(f"No strict donor for {test} base={base.get('sample_id')} kwargs={kwargs}")
        donor = rng.choice(candidates)
        row = dict(base)
        row["base_mismatch_type"] = base.get("mismatch_type")
        row["donor_mismatch_type"] = donor.get("mismatch_type")
        row["base_slot"] = base.get("matched_idx")
        row["donor_slot"] = donor.get("matched_idx")
        row["matrix_m"] = matrix_m
        rows_out.append(row)
        donor_z.append(z_lookup[str(donor["sample_id"])])
    return rows_out, donor_z


def cache_payload(config: dict[str, Any], **data: Any) -> dict[str, Any]:
    return {"config": config, **data}


def load_cache(path: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the cached payload only if it was produced under the same config.

    Legacy files (pre-config formats) and config mismatches are treated as
    stale so changed --split/--n-events/--include-relaxed never reuse them.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("config") != config:
        print(f"stale cache {path.name}: recomputing")
        return None
    return payload


def write_outputs(records: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    if records:
        fields: list[str] = []
        for record in records:
            for key in record:
                if key not in fields:
                    fields.append(key)
        with (output_dir / "match_transfer.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    (output_dir / "match_transfer_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    rows = select_rows(read_rows_csv(args.samples), args.split, args.n_events, args.include_relaxed)
    validate_reverse_pairs(rows)
    core_rows = [row for row in rows if row.get("control_type") in CORE_CONTROLS]
    state_rows = core_rows
    match_rows = [row for row in state_rows if str(row.get("m_base")) == "1"]
    nomatch_rows = [row for row in state_rows if str(row.get("m_base")) == "0"]
    trap_rows = [row for row in rows if row.get("control_type") == "label_copy_trap"]
    print(f"m rows={len(rows)} core={len(core_rows)} match={len(match_rows)} nomatch={len(nomatch_rows)} trap={len(trap_rows)}")

    torch, _, amc, atc = import_runtime()
    tokenizer, model = load_hf_model(torch=torch, auto_model_cls=amc, auto_tokenizer_cls=atc,
        model_name=args.model_name, device=args.device, device_map=args.device_map,
        torch_dtype=args.torch_dtype, trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    device = next(model.parameters()).device
    layers = get_decoder_layers(model)
    hidden = model.config.hidden_size
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"model": args.model_name, "target_var": "m", "split": args.split,
                               "n_events": args.n_events, "include_relaxed": args.include_relaxed,
                               "cells": {}}

    # Behavioral gate is independent of rotation and cached globally.
    gate_config = {"samples": str(args.samples), "model_name": args.model_name, "split": args.split,
                   "n_events": args.n_events, "include_relaxed": args.include_relaxed}
    gate_path = output_dir / "behavioral_gate.json"
    gate_cache = load_cache(gate_path, gate_config)
    if gate_cache is not None:
        summary["behavioral_gate"] = gate_cache["records"]
    else:
        gate_records: list[dict[str, Any]] = []
        for side, source in (("base", state_rows), ("source", [source_as_base(row) for row in state_rows])):
            if side == "base":
                source = [dict(row, eval_side="base") for row in source]
            scored = run_condition(model, layers, 0, "claim_final", None, "none", source,
                                   tokenizer, torch, device, label_tokens.token_ids,
                                   args.eval_batch_size, rng)
            for record in metric_records(scored, "A_gate", ("eval_side", "m_base", "mismatch_type", "matched_idx")):
                gate_records.append(record)
        summary["behavioral_gate"] = gate_records
        gate_path.write_text(json.dumps(to_jsonable(cache_payload(gate_config, records=gate_records)), indent=2))
    
    import numpy as np
    for rotation_dir in args.rotation_dirs:
        cell = Path(rotation_dir).name
        meta_path = Path(rotation_dir) / "rotation_weight_metadata.json"
        if not meta_path.exists():
            print(f"SKIP {cell}: no rotation metadata")
            continue
        meta = json.loads(meta_path.read_text())
        layer, rank = int(meta["layer"]), int(meta["rank"])
        site = str(meta.get("site", "claim_final"))
        cell_config = {**gate_config, "seed": args.seed,
                       "skip_mismatch_matrix": args.skip_mismatch_matrix,
                       "skip_slot_matrix": args.skip_slot_matrix,
                       "layer": layer, "site": site, "rank": rank}
        cell_path = output_dir / f"{cell}.json"
        cached = load_cache(cell_path, cell_config)
        if cached is not None:
            summary["cells"][cell] = cached["cell_out"]
            records.extend(cached["records"])
            continue
        R = torch.tensor(np.load(Path(rotation_dir) / "rotation_weight.npy"), dtype=torch.float32, device=device)
        if tuple(R.shape) != (hidden, rank):
            raise ValueError(f"{cell}: rotation shape {tuple(R.shape)} != ({hidden},{rank})")
        print(f"\n=== match-transfer {cell} ===")
        cell_records: list[dict[str, Any]] = []
        z_state = collect_subspace_coordinates(model, layer, site, R, state_rows, tokenizer, torch, device, args.eval_batch_size)
        z_lookup = {str(row["sample_id"]): z for row, z in zip(state_rows, z_state)}
        z_match = z_state[[str(row.get("m_base")) == "1" for row in state_rows]]
        z_nomatch = z_state[[str(row.get("m_base")) == "0" for row in state_rows]]
        cell_records.extend(geometry_records(z_match, z_nomatch, nomatch_rows, cell))

        # C: exact paired source donor; source rows are in the same order as core_rows.
        source_core = [source_as_base(row) for row in core_rows]
        z_source = collect_subspace_coordinates(model, layer, site, R, source_core, tokenizer, torch, device, args.eval_batch_size)
        c_scored = run_condition(model, layers, layer, site, R, "resample_same", core_rows,
                                 tokenizer, torch, device, label_tokens.token_ids, args.eval_batch_size,
                                 rng, donor_z=z_source)
        for row in metric_records(c_scored, "C_interchange", ("control_type", "m_base", "mismatch_type", "matched_idx")):
            row["cell"] = cell; cell_records.append(row)

        if trap_rows:
            trap_source = [source_as_base(row) for row in trap_rows]
            z_trap = collect_subspace_coordinates(model, layer, site, R, trap_source, tokenizer, torch, device, args.eval_batch_size)
            trap_scored = run_condition(model, layers, layer, site, R, "resample_same", trap_rows,
                                        tokenizer, torch, device, label_tokens.token_ids, args.eval_batch_size,
                                        rng, donor_z=z_trap)
            for row in metric_records(trap_scored, "C_label_copy_trap", ("mismatch_type", "matched_idx")):
                row["cell"] = cell; cell_records.append(row)

        if not args.skip_mismatch_matrix:
            d_rows_all: list[dict[str, Any]] = []; d_z_all: list[Any] = []
            for base_type in MISMATCH_TYPES:
                base_type_rows = [row for row in nomatch_rows if str(row.get("mismatch_type")) == base_type]
                for donor_type in MISMATCH_TYPES:
                    pair_rows, pair_z = add_matrix_rows(
                        base_type_rows, nomatch_rows, z_lookup, test="D_donor_same",
                        matrix_m="m=0", base_field=base_type, donor_field=donor_type,
                        candidate_kwargs=lambda row, donor_type=donor_type: {
                            "m_value": "0", "mismatch_type": donor_type,
                            "slot": str(row.get("matched_idx")),
                            "pi": str(row.get("p_i_base")), "pc": str(row.get("p_c_base")),
                        }, rng=rng,
                    )
                    d_rows_all.extend(pair_rows); d_z_all.extend(pair_z)
            d_scored = run_condition(model, layers, layer, site, R, "resample_same", d_rows_all,
                                     tokenizer, torch, device, label_tokens.token_ids, args.eval_batch_size,
                                     rng, donor_z=torch.stack(d_z_all))
            for row in metric_records(d_scored, "D_donor_same", ("base_mismatch_type", "donor_mismatch_type")):
                row["cell"] = cell; cell_records.append(row)

        if not args.skip_slot_matrix:
            e_rows_all: list[dict[str, Any]] = []; e_z_all: list[Any] = []
            for m_value, base_pool in (("0", nomatch_rows), ("1", match_rows)):
                for base_slot in ("0", "1", "2"):
                    base_slot_rows = [row for row in base_pool if str(row.get("matched_idx")) == base_slot]
                    for donor_slot in ("0", "1", "2"):
                        pair_rows, pair_z = add_matrix_rows(
                            base_slot_rows, base_pool, z_lookup, test="E_slot_donor",
                            matrix_m=f"m={m_value}", base_field=base_slot, donor_field=donor_slot,
                            candidate_kwargs=lambda row, donor_slot=donor_slot: {
                                "m_value": m_value, "mismatch_type": str(row.get("mismatch_type")),
                                "slot": donor_slot, "pi": str(row.get("p_i_base")),
                                "pc": str(row.get("p_c_base")),
                            }, rng=rng,
                        )
                        e_rows_all.extend(pair_rows); e_z_all.extend(pair_z)
            e_scored = run_condition(model, layers, layer, site, R, "resample_same", e_rows_all,
                                     tokenizer, torch, device, label_tokens.token_ids, args.eval_batch_size,
                                     rng, donor_z=torch.stack(e_z_all))
            for row in metric_records(e_scored, "E_slot_donor", ("matrix_m", "base_slot", "donor_slot")):
                row["cell"] = cell; cell_records.append(row)

        cell_out = {"layer": layer, "site": site, "rank": rank,
                    "geometry": [row for row in cell_records if row.get("test") == "B_geometry"]}
        summary["cells"][cell] = cell_out
        records.extend(cell_records)
        cell_path.write_text(json.dumps(to_jsonable(cache_payload(cell_config, cell_out=cell_out, records=cell_records)), indent=2))
        write_outputs(records, summary, output_dir)

    write_outputs(records, summary, output_dir)
    print(f"\nWrote {output_dir / 'match_transfer.csv'} and match_transfer_summary.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match-transfer diagnostics for DAS target m.")
    parser.add_argument("--rotation-dirs", nargs="+", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--n-events", type=int, default=60, help="0 = all events")
    parser.add_argument("--include-relaxed", action="store_true")
    parser.add_argument("--skip-mismatch-matrix", action="store_true")
    parser.add_argument("--skip-slot-matrix", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--label-token-style", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="data/das/match_transfer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
