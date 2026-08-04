"""Run Section 6.2 MNLI resample necessity with frozen DAS rotations.

For every MNLI polarity-square row, replace the coordinates of the base hidden
state in either the learned rho subspace or a rank-matched random subspace with
coordinates from the same natural donor hidden state. Donors come from a
different MNLI square and are balanced between same-rho and opposite-rho
sources. The primary statistic is

    NEx = mean_random_subspace_accuracy - learned_subspace_accuracy.

This is an in-distribution resample intervention, not a zero ablation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from interference_suite.das_pyvene import encode_to_device
from interference_suite.joint_gate_intervention import random_orthonormal_basis
from interference_suite.model import DEFAULT_CACHE_DIR, progress_iter
from interference_suite.section62_ablation import load_r1_handoff
from interference_suite.section62_data import sha256_file
from interference_suite.section62_intervention import (
    load_rotation_basis,
    resolve_positions,
    row_batches,
    run_single_subspace_condition,
    score_base_rows,
)
from run_section62_ablation import (
    load_and_validate_model,
    tensor_sha256,
    write_csv_atomic,
    write_json_atomic,
)


SCHEMA_VERSION = 1
RUN_TYPE = "section62_mnli_resample_nex"
METRIC_FIELDS = (
    "logit_T",
    "logit_F",
    "logit_U",
    "R",
    "M_TF",
    "G_U",
    "gold_margin_3",
    "U_gap",
    "pred_label",
    "pred_tf_label",
    "is_correct",
    "is_tf_correct",
    "global_top_token_id",
    "global_top_token",
    "global_top_in_TFU",
    "hidden_l2",
    "projected_l2",
    "intervention_l2",
    "coordinate_change_l2",
)


def collect_site_hidden_states(
    *,
    model: Any,
    layers: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    layer: int,
    site: str,
) -> Any:
    """Collect the natural hidden vector at one layer/site for every row."""

    state: dict[str, Any] = {"positions": None, "hidden": None}

    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        positions = state["positions"].to(hidden_states.device)
        indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        state["hidden"] = (
            hidden_states[indices, positions].to(torch.float32).detach().cpu()
        )
        return output

    handle = layers[layer].register_forward_hook(hook)
    collected = []
    core_model = getattr(model, "model", model)
    try:
        batches = row_batches(rows, batch_size)
        batches = progress_iter(
            batches,
            total=ceil(len(rows) / batch_size),
            desc=f"collect natural hidden L{layer}/{site}",
        )
        for batch_rows in batches:
            texts = [str(row["base_prompt"]) for row in batch_rows]
            encoded = encode_to_device(tokenizer, texts, device)
            state["positions"] = torch.tensor(
                resolve_positions(tokenizer, texts, batch_rows, site),
                dtype=torch.long,
                device=device,
            )
            state["hidden"] = None
            with torch.no_grad():
                core_model(**encoded, use_cache=False)
            if state["hidden"] is None:
                raise RuntimeError(f"Layer {layer} hidden-state hook did not run")
            collected.append(state["hidden"])
    finally:
        handle.remove()
        state["positions"] = None
        state["hidden"] = None
    return torch.cat(collected, dim=0)


def square_key(row: Mapping[str, Any]) -> str:
    value = row.get("square_id") or row.get("base_event_id") or row.get("triple_id")
    if value in (None, ""):
        raise ValueError(f"Row {row.get('sample_id')!r} has no square identifier")
    return str(value)


def rho_value(row: Mapping[str, Any]) -> int:
    value = int(float(row["rho_base"]))
    if value not in {-1, 1}:
        raise ValueError(f"rho_base must be +/-1, found {value}")
    return value


def build_balanced_donors(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> tuple[list[int], list[str], dict[str, Any]]:
    """Choose different-square donors, balanced by base rho and same/opposite."""

    rng = random.Random(seed)
    by_rho = {
        rho: [index for index, row in enumerate(rows) if rho_value(row) == rho]
        for rho in (-1, 1)
    }
    if len(by_rho[-1]) != len(by_rho[1]):
        raise ValueError(f"MNLI rho classes are not balanced: {by_rho}")
    if len(by_rho[-1]) % 2:
        raise ValueError("Each base-rho class must have even size for exact balance")

    desired_relation: dict[int, str] = {}
    for rho in (-1, 1):
        base_indices = list(by_rho[rho])
        rng.shuffle(base_indices)
        half = len(base_indices) // 2
        for position, index in enumerate(base_indices):
            desired_relation[index] = "same" if position < half else "opposite"

    donor_indices: list[int] = []
    relations: list[str] = []
    for base_index, base in enumerate(rows):
        relation = desired_relation[base_index]
        base_rho = rho_value(base)
        donor_rho = base_rho if relation == "same" else -base_rho
        candidates = [
            index
            for index in by_rho[donor_rho]
            if index != base_index and square_key(rows[index]) != square_key(base)
        ]
        if not candidates:
            raise ValueError(
                f"No {relation}-rho different-square donor for {base.get('sample_id')}"
            )
        donor_indices.append(rng.choice(candidates))
        relations.append(relation)

    audit = Counter(
        (rho_value(row), relation)
        for row, relation in zip(rows, relations)
    )
    expected_quarter = len(rows) // 4
    expected = {
        (-1, "same"): expected_quarter,
        (-1, "opposite"): expected_quarter,
        (1, "same"): expected_quarter,
        (1, "opposite"): expected_quarter,
    }
    if dict(audit) != expected:
        raise RuntimeError(f"Donor balance failed: {dict(audit)} != {expected}")
    if any(
        square_key(base) == square_key(rows[donor_index])
        for base, donor_index in zip(rows, donor_indices)
    ):
        raise RuntimeError("A donor came from the same MNLI square as its base")
    return donor_indices, relations, {
        "seed": seed,
        "n_rows": len(rows),
        "same_rho": relations.count("same"),
        "opposite_rho": relations.count("opposite"),
        "by_base_rho_and_relation": {
            f"rho_{rho:+d}_{relation}": audit[(rho, relation)]
            for rho in (-1, 1)
            for relation in ("same", "opposite")
        },
        "different_square_required": True,
    }


def donor_coordinates(
    *,
    torch: Any,
    natural_hidden: Any,
    donor_indices: Sequence[int],
    basis: Any,
    device: Any,
) -> Any:
    indices = torch.tensor(donor_indices, dtype=torch.long)
    donor_hidden = natural_hidden[indices].to(device=device, dtype=torch.float32)
    return (donor_hidden @ basis.to(device=device, dtype=torch.float32)).detach().cpu()


def condition_records(
    *,
    rows: Sequence[Mapping[str, Any]],
    donor_indices: Sequence[int],
    relations: Sequence[str],
    metrics: Sequence[Mapping[str, Any]],
    base_metrics: Sequence[Mapping[str, Any]],
    condition: str,
    model_name: str,
    site: str,
    layer: int,
    rank: int,
    basis_sha256: str,
    random_seed: int | None,
) -> list[dict[str, Any]]:
    if not (
        len(rows)
        == len(donor_indices)
        == len(relations)
        == len(metrics)
        == len(base_metrics)
    ):
        raise ValueError("Rows, donors, relations, base, and metrics are misaligned")
    records = []
    for row, donor_index, relation, current, base in zip(
        rows, donor_indices, relations, metrics, base_metrics
    ):
        donor = rows[donor_index]
        record = {
            "sample_id": row["sample_id"],
            "square_id": square_key(row),
            "cell": row.get("cell", row.get("control_type", "")),
            "expected_label": row["expected_label"],
            "rho_base": rho_value(row),
            "donor_sample_id": donor["sample_id"],
            "donor_square_id": square_key(donor),
            "donor_rho": rho_value(donor),
            "donor_relation": relation,
            "condition": condition,
            "model_name": model_name,
            "site": site,
            "layer": layer,
            "rank": rank,
            "random_seed": "" if random_seed is None else random_seed,
            "basis_sha256": basis_sha256,
        }
        for field in METRIC_FIELDS:
            if field in current:
                record[field] = current[field]
        for field in METRIC_FIELDS:
            if field in base:
                record[f"base_{field}"] = base[field]
        record["R_change"] = float(current["R"]) - float(base["R"])
        record["M_TF_change"] = float(current["M_TF"]) - float(base["M_TF"])
        record["gold_margin_3_change"] = (
            float(current["gold_margin_3"]) - float(base["gold_margin_3"])
        )
        records.append(record)
    return records


def mean_metric(records: Sequence[Mapping[str, Any]], field: str) -> float:
    return fmean(float(row[field]) for row in records)


def summarize_scope(
    *,
    scope: str,
    base_metrics: Sequence[Mapping[str, Any]],
    learned: Sequence[Mapping[str, Any]],
    random_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    indices: Sequence[int],
) -> dict[str, Any]:
    base_here = [base_metrics[index] for index in indices]
    learned_here = [learned[index] for index in indices]
    random_here = {
        seed: [records[index] for index in indices]
        for seed, records in random_by_seed.items()
    }
    random_accuracy = {
        str(seed): mean_metric(records, "is_correct")
        for seed, records in random_here.items()
    }
    random_mtf = {
        str(seed): mean_metric(records, "M_TF")
        for seed, records in random_here.items()
    }
    random_gold_margin = {
        str(seed): mean_metric(records, "gold_margin_3")
        for seed, records in random_here.items()
    }
    learned_accuracy = mean_metric(learned_here, "is_correct")
    learned_mtf = mean_metric(learned_here, "M_TF")
    learned_gold_margin = mean_metric(learned_here, "gold_margin_3")
    random_accuracy_values = list(random_accuracy.values())
    random_mtf_values = list(random_mtf.values())
    random_gold_margin_values = list(random_gold_margin.values())
    return {
        "scope": scope,
        "n": len(indices),
        "base_accuracy": mean_metric(base_here, "is_correct"),
        "learned_accuracy": learned_accuracy,
        "random_accuracy_by_seed": random_accuracy,
        "random_accuracy_mean": fmean(random_accuracy_values),
        "random_accuracy_sd": pstdev(random_accuracy_values),
        "NEx_accuracy": fmean(random_accuracy_values) - learned_accuracy,
        "NEx_accuracy_pp": 100.0 * (fmean(random_accuracy_values) - learned_accuracy),
        "base_M_TF": mean_metric(base_here, "M_TF"),
        "learned_M_TF": learned_mtf,
        "random_M_TF_by_seed": random_mtf,
        "random_M_TF_mean": fmean(random_mtf_values),
        "NEx_M_TF": fmean(random_mtf_values) - learned_mtf,
        "base_gold_margin_3": mean_metric(base_here, "gold_margin_3"),
        "learned_gold_margin_3": learned_gold_margin,
        "random_gold_margin_3_by_seed": random_gold_margin,
        "random_gold_margin_3_mean": fmean(random_gold_margin_values),
        "NEx_gold_margin_3": fmean(random_gold_margin_values) - learned_gold_margin,
    }


def paired_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    donor_indices: Sequence[int],
    relations: Sequence[str],
    base_metrics: Sequence[Mapping[str, Any]],
    learned: Sequence[Mapping[str, Any]],
    random_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for index, (row, donor_index, relation) in enumerate(
        zip(rows, donor_indices, relations)
    ):
        random_accuracy = fmean(
            float(records[index]["is_correct"])
            for records in random_by_seed.values()
        )
        random_mtf = fmean(
            float(records[index]["M_TF"])
            for records in random_by_seed.values()
        )
        random_gold_margin = fmean(
            float(records[index]["gold_margin_3"])
            for records in random_by_seed.values()
        )
        donor = rows[donor_index]
        output.append(
            {
                "sample_id": row["sample_id"],
                "square_id": square_key(row),
                "cell": row.get("cell", row.get("control_type", "")),
                "expected_label": row["expected_label"],
                "rho_base": rho_value(row),
                "donor_sample_id": donor["sample_id"],
                "donor_square_id": square_key(donor),
                "donor_rho": rho_value(donor),
                "donor_relation": relation,
                "base_is_correct": base_metrics[index]["is_correct"],
                "learned_is_correct": learned[index]["is_correct"],
                "random_is_correct_mean": random_accuracy,
                "row_NEx_accuracy": random_accuracy
                - float(learned[index]["is_correct"]),
                "base_M_TF": base_metrics[index]["M_TF"],
                "learned_M_TF": learned[index]["M_TF"],
                "random_M_TF_mean": random_mtf,
                "row_NEx_M_TF": random_mtf - float(learned[index]["M_TF"]),
                "base_gold_margin_3": base_metrics[index]["gold_margin_3"],
                "learned_gold_margin_3": learned[index]["gold_margin_3"],
                "random_gold_margin_3_mean": random_gold_margin,
                "row_NEx_gold_margin_3": random_gold_margin
                - float(learned[index]["gold_margin_3"]),
            }
        )
    return output


def run(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.random_seeds:
        raise ValueError("Provide at least one --random-seeds value")
    if len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError("--random-seeds contains duplicates")
    if args.device and args.device_map not in (None, "none"):
        raise ValueError("--device requires --device-map none")

    output = Path(args.output_dir).resolve()
    completed = output / "nex_summary.json"
    if args.resume and completed.is_file():
        print(f"Completed NEx exists; skipping: {completed}")
        return 0
    if completed.exists() and not args.overwrite:
        raise FileExistsError(
            f"{completed} exists; pass --resume to skip or --overwrite to replace"
        )

    repo_root = Path(args.repo_root).resolve()
    handoff = load_r1_handoff(
        args.behavioral_summary,
        expected_corpora=("MNLI",),
        verify_r0_manifest=True,
        repo_root=repo_root,
    )
    rows = [
        dict(row)
        for row in handoff.rows
        if str(row.get("analysis_stratum")) == "square_valid"
    ]
    if len(rows) != 620:
        raise ValueError(f"Expected all 620 MNLI square rows, found {len(rows)}")
    if Counter(str(row["expected_label"]) for row in rows) != {"T": 310, "F": 310}:
        raise ValueError("MNLI square labels are not balanced 310 T / 310 F")

    donor_indices, relations, donor_audit = build_balanced_donors(
        rows, args.donor_seed
    )
    torch, tokenizer, model, layers, label_token_ids, live_run = load_and_validate_model(
        handoff_summary=handoff.summary,
        model_name=handoff.model_name,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    input_device = next(model.parameters()).device
    hidden_size = int(model.config.hidden_size)
    expected_rotation = {
        "target_var": "rho",
        "model_name": handoff.model_name,
        "layer": args.layer,
        "rank": args.rank,
        "component": "block_output",
        "allowed_metadata_sites": [args.site],
    }
    learned_basis, learned_provenance = load_rotation_basis(
        rotation_dir=repo_root / args.rho_rotation_dir,
        expected=expected_rotation,
        hidden_size=hidden_size,
        torch=torch,
        device=input_device,
    )
    learned_hash = tensor_sha256(learned_basis)

    base_metrics = score_base_rows(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=input_device,
        rows=rows,
        label_token_ids=label_token_ids,
        batch_size=args.batch_size,
    )
    natural_hidden = collect_site_hidden_states(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        torch=torch,
        device=input_device,
        rows=rows,
        batch_size=args.batch_size,
        layer=args.layer,
        site=args.site,
    )
    learned_donor = donor_coordinates(
        torch=torch,
        natural_hidden=natural_hidden,
        donor_indices=donor_indices,
        basis=learned_basis,
        device=input_device,
    )
    learned_metrics = run_single_subspace_condition(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        torch=torch,
        device=input_device,
        rows=rows,
        label_token_ids=label_token_ids,
        batch_size=args.batch_size,
        layer=args.layer,
        site=args.site,
        basis=learned_basis,
        condition=f"{args.site}/learned_rho_resample",
        donor_coordinates=learned_donor,
    )

    random_by_seed: dict[int, list[dict[str, Any]]] = {}
    random_basis_hashes: dict[str, str] = {}
    all_records = condition_records(
        rows=rows,
        donor_indices=donor_indices,
        relations=relations,
        metrics=learned_metrics,
        base_metrics=base_metrics,
        condition="learned_rho_resample",
        model_name=handoff.model_name,
        site=args.site,
        layer=args.layer,
        rank=args.rank,
        basis_sha256=learned_hash,
        random_seed=None,
    )
    for random_seed in args.random_seeds:
        basis = random_orthonormal_basis(
            torch,
            hidden_size,
            args.rank,
            device=input_device,
            seed=random_seed,
        )
        basis_hash = tensor_sha256(basis)
        random_basis_hashes[str(random_seed)] = basis_hash
        donor = donor_coordinates(
            torch=torch,
            natural_hidden=natural_hidden,
            donor_indices=donor_indices,
            basis=basis,
            device=input_device,
        )
        metrics = run_single_subspace_condition(
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            torch=torch,
            device=input_device,
            rows=rows,
            label_token_ids=label_token_ids,
            batch_size=args.batch_size,
            layer=args.layer,
            site=args.site,
            basis=basis,
            condition=f"{args.site}/random_resample_seed{random_seed}",
            donor_coordinates=donor,
        )
        random_by_seed[random_seed] = metrics
        all_records.extend(
            condition_records(
                rows=rows,
                donor_indices=donor_indices,
                relations=relations,
                metrics=metrics,
                base_metrics=base_metrics,
                condition="random_resample",
                model_name=handoff.model_name,
                site=args.site,
                layer=args.layer,
                rank=args.rank,
                basis_sha256=basis_hash,
                random_seed=random_seed,
            )
        )

    scopes = {
        "overall": list(range(len(rows))),
        "same_rho_donor": [
            index for index, relation in enumerate(relations) if relation == "same"
        ],
        "opposite_rho_donor": [
            index for index, relation in enumerate(relations) if relation == "opposite"
        ],
    }
    summaries = [
        summarize_scope(
            scope=scope,
            base_metrics=base_metrics,
            learned=learned_metrics,
            random_by_seed=random_by_seed,
            indices=indices,
        )
        for scope, indices in scopes.items()
    ]
    paired = paired_rows(
        rows=rows,
        donor_indices=donor_indices,
        relations=relations,
        base_metrics=base_metrics,
        learned=learned_metrics,
        random_by_seed=random_by_seed,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_type": RUN_TYPE,
        "model_name": handoff.model_name,
        "site": args.site,
        "layer": args.layer,
        "rank": args.rank,
        "rotation_training_seed": args.rotation_seed,
        "definition": "mean random-subspace resample accuracy minus learned-subspace resample accuracy",
        "population": "all 620 MNLI square-valid polarity rows",
        "donor_audit": donor_audit,
        "random_seeds": args.random_seeds,
        "by_scope": {row["scope"]: row for row in summaries},
    }

    output.mkdir(parents=True, exist_ok=True)
    scored_path = write_csv_atomic(all_records, output / "intervention_scored.csv")
    paired_path = write_csv_atomic(paired, output / "paired_nex.csv")
    table_rows = []
    for row in summaries:
        flat = {
            key: value
            for key, value in row.items()
            if not isinstance(value, dict)
        }
        table_rows.append(flat)
    table_path = write_csv_atomic(table_rows, output / "nex_by_scope.csv")
    summary_path = write_json_atomic(summary, output / "nex_summary.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": RUN_TYPE,
        "behavioral_summary": {
            "path": str(Path(args.behavioral_summary).resolve()),
            "sha256": sha256_file(args.behavioral_summary),
        },
        "live_run": live_run,
        "rotation_training_seed": args.rotation_seed,
        "rho_rotation": learned_provenance,
        "learned_basis_sha256": learned_hash,
        "random_basis_sha256": random_basis_hashes,
        "donor_audit": donor_audit,
        "artifacts": {
            "intervention_scored": str(scored_path),
            "paired_nex": str(paired_path),
            "nex_by_scope": str(table_path),
            "nex_summary": str(summary_path),
        },
    }
    write_json_atomic(manifest, output / "run_manifest.json")

    headline = summary["by_scope"]["overall"]
    print(f"Wrote MNLI resample NEx to {output}")
    print(
        f"base={headline['base_accuracy']:.4f} "
        f"learned={headline['learned_accuracy']:.4f} "
        f"random={headline['random_accuracy_mean']:.4f} "
        f"NEx={headline['NEx_accuracy_pp']:+.2f} pp"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavioral-summary", required=True)
    parser.add_argument("--rho-rotation-dir", required=True)
    parser.add_argument("--rotation-seed", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", choices=["claim_final", "answer_token"], default="claim_final")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="none")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
