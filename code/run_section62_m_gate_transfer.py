"""Run Section 6.2 R3 U-gate transfer on MNLI square-valid/all-U rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from interference_suite.joint_gate_intervention import random_orthonormal_basis
from interference_suite.section62_ablation import (
    FrozenSiteProfile,
    build_analysis_membership,
    load_r1_handoff,
    preflight_frozen_profile,
)
from interference_suite.section62_data import FULL_CELL_ORDER, sha256_file
from interference_suite.section62_intervention import (
    collect_subspace_coordinates,
    load_rotation_basis,
    run_single_subspace_condition,
)
from run_section62_ablation import (
    BASE_METRIC_FIELDS,
    SCHEMA_VERSION,
    _assert_loaded_rotation_matches_preflight,
    _read_csv_records,
    _rotation_expected,
    _selected_profiles,
    artifact_record,
    build_runtime_rows,
    donor_coordinates_for_rows,
    enrich_intervention_records,
    load_and_validate_model,
    stable_generator_seed,
    tensor_sha256,
    write_csv_atomic,
    write_json_atomic,
)

CELL_TARGET = {"++": "T", "-+": "F", "+-": "F", "--": "T"}
TRANSFER_DIRECTIONS = ("match_to_no_overlap", "no_overlap_to_match")
RANDOM_SEEDS = (0, 1, 2)


def _base_prediction(
    row: Mapping[str, Any],
    base_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    sample_id = str(row["sample_id"])
    if sample_id not in base_by_id:
        raise ValueError(f"Missing R2 base score for {sample_id!r}")
    prediction = str(base_by_id[sample_id]["pred_label"])
    if prediction not in {"T", "F", "U"}:
        raise ValueError(f"Invalid base prediction for {sample_id}: {prediction!r}")
    return prediction


def _retarget(
    base: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    direction: str,
    target_label: str,
    base_prediction: str,
) -> dict[str, Any]:
    if str(base["cell"]) != str(donor["cell"]):
        raise ValueError("R3 donor must share the base polarity cell")
    if str(base["triple_id"]) == str(donor["triple_id"]):
        raise ValueError("R3 cross-stratum donor must come from another triple")
    result = dict(base)
    result.update(
        {
            "gate_base_stratum": str(base["analysis_stratum"]),
            "gate_source_stratum": str(donor["analysis_stratum"]),
            "gate_direction": direction,
            "gate_base_prediction": base_prediction,
            "gate_original_expected_label": str(base["expected_label"]),
            "gate_target_label": target_label,
            "gate_pair_id": (
                f"{base['cell']}:{base['triple_id']}:{donor['triple_id']}"
            ),
            "expected_label": target_label,
            "portability_group": "cross_stratum_same_cell",
            "portability_donor_cell": str(donor["cell"]),
            "portability_donor_sample_id": str(donor["sample_id"]),
        }
    )
    return result


def build_gate_transfer_design(
    runtime_rows: Sequence[Mapping[str, Any]],
    base_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build all possible one-to-one state-filtered pairs without length matching."""

    by_cell_stratum: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runtime_rows:
        stratum = str(row["analysis_stratum"])
        if stratum not in {"square_valid", "all_u"}:
            continue
        cell = str(row["cell"])
        if cell not in FULL_CELL_ORDER:
            raise ValueError(f"Unexpected R3 cell {cell!r}")
        by_cell_stratum[(cell, stratum)].append(row)

    transfer_rows: list[dict[str, Any]] = []
    same_rows: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    for cell in FULL_CELL_ORDER:
        square_all = sorted(
            by_cell_stratum[(cell, "square_valid")],
            key=lambda row: (str(row["triple_id"]), str(row["sample_id"])),
        )
        no_overlap_all = sorted(
            by_cell_stratum[(cell, "all_u")],
            key=lambda row: (str(row["triple_id"]), str(row["sample_id"])),
        )
        square = [
            row
            for row in square_all
            if _base_prediction(row, base_by_id) != "U"
        ]
        no_overlap = [
            row
            for row in no_overlap_all
            if _base_prediction(row, base_by_id) == "U"
        ]
        n_pairs = min(len(square), len(no_overlap))
        if n_pairs < 2:
            raise ValueError(
                f"R3 cell {cell} has only {n_pairs} eligible reciprocal pairs"
            )
        square = square[:n_pairs]
        no_overlap = no_overlap[:n_pairs]
        counts.append(
            {
                "cell": cell,
                "n_square_all": len(square_all),
                "n_square_non_u": sum(
                    _base_prediction(row, base_by_id) != "U"
                    for row in square_all
                ),
                "n_all_u_all": len(no_overlap_all),
                "n_all_u_pred_u": sum(
                    _base_prediction(row, base_by_id) == "U"
                    for row in no_overlap_all
                ),
                "n_pairs": n_pairs,
                "n_transfer_rows": 2 * n_pairs,
            }
        )

        cell_transfer: list[dict[str, Any]] = []
        for match_row, no_overlap_row in zip(square, no_overlap):
            cell_transfer.append(
                _retarget(
                    match_row,
                    no_overlap_row,
                    direction="match_to_no_overlap",
                    target_label="U",
                    base_prediction=_base_prediction(match_row, base_by_id),
                )
            )
            cell_transfer.append(
                _retarget(
                    no_overlap_row,
                    match_row,
                    direction="no_overlap_to_match",
                    target_label=CELL_TARGET[cell],
                    base_prediction=_base_prediction(no_overlap_row, base_by_id),
                )
            )
        transfer_rows.extend(cell_transfer)

        ordered_bases = [row for pair in zip(square, no_overlap) for row in pair]
        square_donor = {
            str(row["sample_id"]): square[(index + 1) % n_pairs]
            for index, row in enumerate(square)
        }
        no_overlap_donor = {
            str(row["sample_id"]): no_overlap[(index + 1) % n_pairs]
            for index, row in enumerate(no_overlap)
        }
        for base in ordered_bases:
            stratum = str(base["analysis_stratum"])
            donor = (
                square_donor[str(base["sample_id"])]
                if stratum == "square_valid"
                else no_overlap_donor[str(base["sample_id"])]
            )
            direction = "same_match" if stratum == "square_valid" else "same_no_overlap"
            same_rows.append(
                _retarget(
                    base,
                    donor,
                    direction=direction,
                    target_label=_base_prediction(base, base_by_id),
                    base_prediction=_base_prediction(base, base_by_id),
                )
            )

    transfer_ids = [str(row["sample_id"]) for row in transfer_rows]
    same_ids = [str(row["sample_id"]) for row in same_rows]
    if transfer_ids != same_ids:
        raise ValueError("R3 same-state and transfer rows do not share base order")
    if len(transfer_ids) != len(set(transfer_ids)):
        raise ValueError("R3 reuses a base row within one reciprocal design")
    return transfer_rows, same_rows, {
        "pairing": "same_cell_sorted_one_to_one_no_reuse",
        "length_matching": False,
        "eligibility": {
            "square_valid": "base prediction is not U",
            "all_u": "base prediction is U",
        },
        "by_cell": counts,
        "n_pairs": sum(int(row["n_pairs"]) for row in counts),
        "n_transfer_rows": len(transfer_rows),
        "n_same_rows": len(same_rows),
    }


def _enrich_gate_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for record in records:
        target = str(record["gate_target_label"])
        prediction = str(record["pred_label"])
        record["coordinate_source_sample_id"] = str(
            record["portability_donor_sample_id"]
        )
        record["base_target_correct"] = int(
            str(record["gate_base_prediction"]) == target
        )
        record["gate_success"] = int(
            prediction == "U" if target == "U" else prediction != "U"
        )
        record["target_TF_margin"] = (
            ""
            if target == "U"
            else (1.0 if target == "T" else -1.0) * float(record["R"])
        )
        record["U_gap_delta"] = float(record["U_gap"]) - float(
            record["base_U_gap"]
        )
        record["accuracy_drop"] = ""
        record["gold_margin_3_drop"] = ""
        record["M_TF_drop"] = ""
    return records


def _run_coordinate_patch(
    *,
    condition: str,
    rows: list[dict[str, Any]],
    coordinates: Any,
    basis: Any,
    basis_sha256: str,
    base_by_id: Mapping[str, Mapping[str, Any]],
    profile: FrozenSiteProfile,
    model: Any,
    layers: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    label_token_ids: Mapping[str, int],
    batch_size: int,
    subspace_var: str,
    random_seed: int | None = None,
    generator_seed: int | None = None,
) -> list[dict[str, Any]]:
    donor = donor_coordinates_for_rows(
        torch=torch,
        rows=rows,
        coordinates=coordinates,
    )
    metrics = run_single_subspace_condition(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        rows=rows,
        label_token_ids=dict(label_token_ids),
        batch_size=batch_size,
        layer=profile.layer,
        site=profile.reported_site,
        basis=basis,
        condition=f"{profile.reported_site}/{condition}",
        donor_coordinates=donor,
    )
    records = enrich_intervention_records(
        rows,
        metrics,
        base_metrics_by_id=base_by_id,
        profile=profile,
        condition=condition,
        subspace_var=subspace_var,
        intervention_kind="cross_example_coordinate_replacement",
        basis_sha256=basis_sha256,
        random_seed=random_seed,
        generator_seed=generator_seed,
    )
    return _enrich_gate_records(records)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average empty R3 values")
    return sum(values) / len(values)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = [str(row["pred_label"]) for row in rows]
    tf_values = [
        int(row["is_tf_correct"])
        for row in rows
        if str(row.get("is_tf_correct", "")) != ""
    ]
    margins = [
        float(row["target_TF_margin"])
        for row in rows
        if str(row.get("target_TF_margin", "")) != ""
    ]
    return {
        "n": len(rows),
        "target_IIA": _mean([float(row["is_correct"]) for row in rows]),
        "base_target_accuracy": _mean(
            [float(row["base_target_correct"]) for row in rows]
        ),
        "target_accuracy_gain": _mean(
            [float(row["is_correct"]) for row in rows]
        )
        - _mean([float(row["base_target_correct"]) for row in rows]),
        "gate_success": _mean([float(row["gate_success"]) for row in rows]),
        "TF_IIA": (_mean([float(value) for value in tf_values]) if tf_values else None),
        "mean_target_TF_margin": (_mean(margins) if margins else None),
        "mean_U_gap": _mean([float(row["U_gap"]) for row in rows]),
        "mean_U_gap_delta": _mean([float(row["U_gap_delta"]) for row in rows]),
        "T_rate": predictions.count("T") / len(rows),
        "F_rate": predictions.count("F") / len(rows),
        "U_rate": predictions.count("U") / len(rows),
    }


def summarize_gate_transfer(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("R3 intervention records are empty")
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        condition = str(row["condition"])
        direction = str(row["gate_direction"])
        cell = str(row["cell"])
        random_key = (
            str(row.get("random_seed", ""))
            if condition == "rand_cross_stratum"
            else ""
        )
        grouped[(condition, direction, cell, random_key)].append(row)
    by_group = []
    for (condition, direction, cell, random_key), values in sorted(grouped.items()):
        random_seed: int | str = ""
        if condition == "rand_cross_stratum":
            if random_key == "":
                raise ValueError("R3 random aggregate has no seed")
            random_seed = int(random_key)
        by_group.append(
            {
                "condition": condition,
                "gate_direction": direction,
                "cell": cell,
                "random_seed": random_seed,
                **_aggregate_rows(values),
            }
        )

    by_direction = []
    for condition in ("m_cross_stratum", "rho_cross_stratum"):
        for direction in TRANSFER_DIRECTIONS:
            values = [
                row
                for row in records
                if str(row["condition"]) == condition
                and str(row["gate_direction"]) == direction
            ]
            by_direction.append(
                {
                    "condition": condition,
                    "gate_direction": direction,
                    **_aggregate_rows(values),
                }
            )
    for seed in RANDOM_SEEDS:
        for direction in TRANSFER_DIRECTIONS:
            values = [
                row
                for row in records
                if str(row["condition"]) == "rand_cross_stratum"
                and int(row["random_seed"]) == seed
                and str(row["gate_direction"]) == direction
            ]
            by_direction.append(
                {
                    "condition": "rand_cross_stratum",
                    "gate_direction": direction,
                    "random_seed": seed,
                    **_aggregate_rows(values),
                }
            )

    confirmatory = []
    for direction in TRANSFER_DIRECTIONS:
        m_row = next(
            row
            for row in by_direction
            if row["condition"] == "m_cross_stratum"
            and row["gate_direction"] == direction
        )
        rho_row = next(
            row
            for row in by_direction
            if row["condition"] == "rho_cross_stratum"
            and row["gate_direction"] == direction
        )
        random_rows = [
            row
            for row in by_direction
            if row["condition"] == "rand_cross_stratum"
            and row["gate_direction"] == direction
        ]
        random_iia = _mean([float(row["target_IIA"]) for row in random_rows])
        random_gate = _mean([float(row["gate_success"]) for row in random_rows])
        confirmatory.append(
            {
                "gate_direction": direction,
                "m_target_IIA": m_row["target_IIA"],
                "rho_target_IIA": rho_row["target_IIA"],
                "rand_target_IIA_mean": random_iia,
                "NEx_IIA_vs_rho": float(m_row["target_IIA"])
                - float(rho_row["target_IIA"]),
                "NEx_IIA_vs_rand": float(m_row["target_IIA"])
                - random_iia,
                "m_gate_success": m_row["gate_success"],
                "rho_gate_success": rho_row["gate_success"],
                "rand_gate_success_mean": random_gate,
                "NEx_gate_vs_rho": float(m_row["gate_success"])
                - float(rho_row["gate_success"]),
                "NEx_gate_vs_rand": float(m_row["gate_success"])
                - random_gate,
                "m_mean_U_gap_delta": m_row["mean_U_gap_delta"],
                "rho_mean_U_gap_delta": rho_row["mean_U_gap_delta"],
                "m_TF_IIA": m_row["TF_IIA"],
            }
        )
    same = [
        {
            "gate_direction": direction,
            **_aggregate_rows(
                [
                    row
                    for row in records
                    if str(row["condition"]) == "m_same_stratum"
                    and str(row["gate_direction"]) == direction
                ]
            ),
        }
        for direction in ("same_match", "same_no_overlap")
    ]
    return {
        "condition": "section62_r3_u_gate_transfer",
        "random_seeds": list(RANDOM_SEEDS),
        "by_group": by_group,
        "by_direction": by_direction,
        "confirmatory": confirmatory,
        "same_m_preservation": same,
        "headline": {
            "target_IIA_direction_min": min(
                float(row["m_target_IIA"]) for row in confirmatory
            ),
            "gate_success_direction_min": min(
                float(row["m_gate_success"]) for row in confirmatory
            ),
            "NEx_IIA_vs_rho_direction_min": min(
                float(row["NEx_IIA_vs_rho"]) for row in confirmatory
            ),
            "NEx_IIA_vs_rand_direction_min": min(
                float(row["NEx_IIA_vs_rand"]) for row in confirmatory
            ),
        },
    }


def _load_source_run(
    *,
    existing: Path,
    profile: FrozenSiteProfile,
    rotation_seed: int | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path, Path]:
    manifest_path = existing / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_profile = next(
        (
            value
            for value in manifest.get("frozen_profiles", [])
            if str(value.get("reported_site")) == profile.reported_site
        ),
        None,
    )
    if source_profile is None:
        raise ValueError("Existing R2 run has no matching R3 site profile")
    expected = {
        "model_name": profile.model_name,
        "reported_site": profile.reported_site,
        "layer": profile.layer,
        "rank": profile.rank,
        "rho_rotation_dir": profile.rho_rotation_dir,
        "m_rotation_dir": profile.m_rotation_dir,
    }
    for field, value in expected.items():
        if source_profile.get(field) != value:
            raise ValueError(
                f"Existing R2 {field}={source_profile.get(field)!r}, expected={value!r}"
            )
    if manifest.get("rotation_training_seed") != rotation_seed:
        raise ValueError("Existing R2 rotation seed does not match R3 seed")
    base_path = existing / "base_scored.csv"
    base_record = manifest["artifacts"]["base_scored"]
    if sha256_file(base_path) != str(base_record["sha256"]):
        raise ValueError("Existing R2 base_scored.csv SHA256 mismatch")
    base_rows = _read_csv_records(base_path)
    if len(base_rows) != int(base_record["rows"]):
        raise ValueError("Existing R2 base_scored.csv row-count mismatch")
    base_by_id = {
        str(row["sample_id"]): {field: row[field] for field in BASE_METRIC_FIELDS}
        for row in base_rows
    }
    return manifest, base_by_id, manifest_path, base_path


def run(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if len(args.sites) != 1:
        raise ValueError("R3 requires exactly one --sites value")
    if args.device and args.device_map not in (None, "none"):
        raise ValueError("--device requires --device-map none")
    repo_root = Path(args.repo_root).resolve()
    handoff = load_r1_handoff(
        args.behavioral_summary,
        expected_corpora=("MNLI",),
        verify_r0_manifest=True,
        repo_root=repo_root,
    )
    profile = _selected_profiles(
        handoff.model_name,
        args.sites,
        rho_rotation_dir=args.rho_rotation_dir,
        m_rotation_dir=args.m_rotation_dir,
        rotation_rank=args.rotation_rank,
        rotation_layer=args.rotation_layer,
    )[0]
    if args.rotation_seed is not None and args.rho_rotation_dir is None:
        raise ValueError("--rotation-seed requires explicit rotation overrides")
    preflight = preflight_frozen_profile(profile, repo_root=repo_root)
    existing = Path(args.existing_run_dir).resolve()
    output = Path(args.output_dir).resolve()
    if existing == output:
        raise ValueError("R3 output must not overwrite the existing R2 run")
    source_manifest, base_by_id, source_manifest_path, base_path = _load_source_run(
        existing=existing,
        profile=profile,
        rotation_seed=args.rotation_seed,
    )

    membership = build_analysis_membership(
        handoff.rows,
        analysis_population="full-square",
    )
    runtime_rows = build_runtime_rows(handoff.rows, membership)
    transfer_rows, same_rows, design = build_gate_transfer_design(
        runtime_rows, base_by_id
    )

    torch, tokenizer, model, layers, label_token_ids, live_run = (
        load_and_validate_model(
            handoff_summary=handoff.summary,
            model_name=handoff.model_name,
            device=args.device,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    )
    device = next(model.parameters()).device
    rho_basis, rho_provenance = load_rotation_basis(
        rotation_dir=repo_root / profile.rho_rotation_dir,
        expected=_rotation_expected(
            profile, target_var="rho", metadata_site=profile.rho_metadata_site
        ),
        hidden_size=profile.hidden_size,
        torch=torch,
        device=device,
    )
    m_basis, m_provenance = load_rotation_basis(
        rotation_dir=repo_root / profile.m_rotation_dir,
        expected=_rotation_expected(
            profile, target_var="m", metadata_site=profile.m_metadata_site
        ),
        hidden_size=profile.hidden_size,
        torch=torch,
        device=device,
    )
    _assert_loaded_rotation_matches_preflight(
        rho_provenance, preflight["rho"], label=f"{profile.reported_site} rho"
    )
    _assert_loaded_rotation_matches_preflight(
        m_provenance, preflight["m"], label=f"{profile.reported_site} m"
    )
    rho_hash = tensor_sha256(rho_basis)
    m_hash = tensor_sha256(m_basis)
    all_records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    m_coordinates = collect_subspace_coordinates(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        rows=transfer_rows,
        batch_size=args.batch_size,
        layer=profile.layer,
        site=profile.reported_site,
        basis=m_basis,
    )
    for condition, rows in (
        ("m_cross_stratum", transfer_rows),
        ("m_same_stratum", same_rows),
    ):
        records = _run_coordinate_patch(
            condition=condition,
            rows=rows,
            coordinates=m_coordinates,
            basis=m_basis,
            basis_sha256=m_hash,
            base_by_id=base_by_id,
            profile=profile,
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            label_token_ids=label_token_ids,
            batch_size=args.batch_size,
            subspace_var="m",
        )
        path = write_csv_atomic(records, output / f"{condition}.csv")
        artifacts[condition] = artifact_record(path, rows=len(records))
        all_records.extend(records)

    rho_coordinates = collect_subspace_coordinates(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        rows=transfer_rows,
        batch_size=args.batch_size,
        layer=profile.layer,
        site=profile.reported_site,
        basis=rho_basis,
    )
    rho_records = _run_coordinate_patch(
        condition="rho_cross_stratum",
        rows=transfer_rows,
        coordinates=rho_coordinates,
        basis=rho_basis,
        basis_sha256=rho_hash,
        base_by_id=base_by_id,
        profile=profile,
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        label_token_ids=label_token_ids,
        batch_size=args.batch_size,
        subspace_var="rho",
    )
    path = write_csv_atomic(rho_records, output / "rho_cross_stratum.csv")
    artifacts["rho_cross_stratum"] = artifact_record(path, rows=len(rho_records))
    all_records.extend(rho_records)

    random_provenance = []
    for logical_seed in RANDOM_SEEDS:
        generator_seed = stable_generator_seed(
            model_name=handoff.model_name,
            site=profile.reported_site,
            layer=profile.layer,
            logical_seed=logical_seed,
        )
        basis = random_orthonormal_basis(
            torch,
            profile.hidden_size,
            profile.rank,
            device=torch.device("cpu"),
            seed=generator_seed,
        )
        basis_hash = tensor_sha256(basis)
        coordinates = collect_subspace_coordinates(
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            rows=transfer_rows,
            batch_size=args.batch_size,
            layer=profile.layer,
            site=profile.reported_site,
            basis=basis,
        )
        records = _run_coordinate_patch(
            condition="rand_cross_stratum",
            rows=transfer_rows,
            coordinates=coordinates,
            basis=basis,
            basis_sha256=basis_hash,
            base_by_id=base_by_id,
            profile=profile,
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            label_token_ids=label_token_ids,
            batch_size=args.batch_size,
            subspace_var="random",
            random_seed=logical_seed,
            generator_seed=generator_seed,
        )
        path = write_csv_atomic(
            records, output / f"rand_cross_stratum_seed{logical_seed}.csv"
        )
        artifacts[f"rand_cross_stratum_seed{logical_seed}"] = artifact_record(
            path, rows=len(records)
        )
        random_provenance.append(
            {
                "logical_seed": logical_seed,
                "generator_seed": generator_seed,
                "basis_sha256": basis_hash,
            }
        )
        all_records.extend(records)

    summary = summarize_gate_transfer(all_records)
    combined_path = write_csv_atomic(all_records, output / "intervention_scored.csv")
    by_group_path = write_csv_atomic(summary["by_group"], output / "by_group.csv")
    by_direction_path = write_csv_atomic(
        summary["by_direction"], output / "by_direction.csv"
    )
    confirmatory_path = write_csv_atomic(
        summary["confirmatory"], output / "confirmatory.csv"
    )
    same_path = write_csv_atomic(
        summary["same_m_preservation"], output / "same_m_preservation.csv"
    )
    summary_path = write_json_atomic(summary, output / "summary.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "section62_r3_u_gate_transfer",
        "analysis_scope": "state_filtered_full_sample_no_length_matching",
        "model_name": handoff.model_name,
        "rotation_training_seed": args.rotation_seed,
        "profile": {
            "reported_site": profile.reported_site,
            "layer": profile.layer,
            "rank": profile.rank,
            "rho_rotation_dir": profile.rho_rotation_dir,
            "m_rotation_dir": profile.m_rotation_dir,
        },
        "scope_language": (
            "transfer to MNLI all-U; semantic claim is limited to the "
            "no-overlap end only after a separate semantic audit"
        ),
        "design": design,
        "controls": [
            "rho_cross_stratum_same_cell",
            "rand_cross_stratum_rank_matched",
            "m_same_stratum",
        ],
        "random_provenance": random_provenance,
        "source_r2_manifest": artifact_record(source_manifest_path),
        "source_base_scored": artifact_record(
            base_path, rows=int(source_manifest["artifacts"]["base_scored"]["rows"])
        ),
        "rotation_preflight": preflight,
        "rho_loaded": rho_provenance,
        "m_loaded": m_provenance,
        "rho_applied_qr_basis_sha256": rho_hash,
        "m_applied_qr_basis_sha256": m_hash,
        "live_run": live_run,
        "artifacts": {
            **artifacts,
            "intervention_scored": artifact_record(
                combined_path, rows=len(all_records)
            ),
            "by_group": artifact_record(by_group_path, rows=len(summary["by_group"])),
            "by_direction": artifact_record(
                by_direction_path, rows=len(summary["by_direction"])
            ),
            "confirmatory": artifact_record(
                confirmatory_path, rows=len(summary["confirmatory"])
            ),
            "same_m_preservation": artifact_record(
                same_path, rows=len(summary["same_m_preservation"])
            ),
            "summary": artifact_record(summary_path),
        },
    }
    write_json_atomic(manifest, output / "run_manifest.json")
    print(f"Wrote Section 6.2 R3 U-gate transfer to {output}")
    for row in summary["confirmatory"]:
        print(
            f"{row['gate_direction']}: m_IIA={row['m_target_IIA']:.4f}, "
            f"m_gate={row['m_gate_success']:.4f}, "
            f"NEx_vs_rho={row['NEx_IIA_vs_rho']:.4f}, "
            f"NEx_vs_rand={row['NEx_IIA_vs_rand']:.4f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavioral-summary", required=True)
    parser.add_argument("--existing-run-dir", required=True)
    parser.add_argument(
        "--sites",
        nargs=1,
        choices=["claim_final", "answer_token"],
        required=True,
    )
    parser.add_argument("--rho-rotation-dir")
    parser.add_argument("--m-rotation-dir")
    parser.add_argument("--rotation-rank", type=int)
    parser.add_argument("--rotation-layer", type=int)
    parser.add_argument("--rotation-seed", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", default="/workspace/huggingface/hub")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
