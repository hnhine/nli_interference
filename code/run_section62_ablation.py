"""Run fixed-subspace Section 6.2 ablations on the R1-gated MNLI rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from interference_suite.das_pyvene import load_hf_model
from interference_suite.io_utils import write_rows_csv
from interference_suite.joint_gate_intervention import random_orthonormal_basis
from interference_suite.model import (
    DEFAULT_CACHE_DIR,
    normalize_cache_dir,
    resolve_label_tokens,
)
from interference_suite.section62_ablation import (
    DEFAULT_MIN_BASELINE_CORRECT_PER_CELL,
    AnalysisMembership,
    FrozenSiteProfile,
    build_analysis_membership,
    frozen_profiles_for_model,
    load_r1_handoff,
    preflight_frozen_profile,
    preflight_rotation_metadata,
)
from interference_suite.section62_ablation_metrics import summarize_ablation
from interference_suite.section62_data import sha256_file
from interference_suite.section62_intervention import (
    basis_overlap_metrics,
    collect_subspace_coordinates,
    get_decoder_layers,
    load_rotation_basis,
    run_single_subspace_condition,
    score_base_rows,
)


SCHEMA_VERSION = 1
RUN_TYPE = "section62_fixed_subspace_ablation"
INITIAL_RANDOM_SEEDS = (0, 1, 2)
EXTRA_RANDOM_SEEDS = (3, 4)
DEFAULT_RANDOM_SPREAD_THRESHOLD = 0.02
OPPOSITE_RHO_DONOR = {
    "++": "-+",
    "-+": "++",
    "+-": "--",
    "--": "+-",
}
TF_FLIP = {"T": "F", "F": "T"}

R1_SCORE_FIELDS = (
    "logit_T",
    "logit_F",
    "logit_U",
    "R",
    "M_TF",
    "gold_margin_3",
    "U_gap",
    "pred_label",
    "pred_tf_label",
    "is_correct",
    "is_tf_correct",
    "global_top_token_id",
    "global_top_token",
    "global_top_in_TFU",
)
BASE_METRIC_FIELDS = (
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
)


def write_json_atomic(value: Any, path: str | Path) -> Path:
    """Write JSON via a sibling temporary file, publishing it last."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def write_csv_atomic(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    schema_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Write CSV via a sibling temporary file, publishing it last."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    write_rows_csv(rows, temporary, schema_rows=schema_rows)
    temporary.replace(output)
    return output


def artifact_record(
    path: str | Path,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    artifact = Path(path).resolve()
    return {
        "path": str(artifact),
        **({"rows": rows} if rows is not None else {}),
        "sha256": sha256_file(artifact),
        "bytes": artifact.stat().st_size,
    }


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def summarize_existing_run(args: argparse.Namespace) -> int:
    """Summarize completed condition CSVs without loading a model."""

    output = Path(args.output_dir)
    expected_strata = (
        ("square_valid",)
        if args.analysis_population == "full-square"
        else ("square_valid", "all_u")
    )
    analysis_scope = (
        "full_square_mtf_primary"
        if args.analysis_population == "full-square"
        else "baseline_correct_secondary_only"
    )
    summaries_by_site: dict[str, dict[str, Any]] = {}
    input_artifacts: dict[str, dict[str, Any]] = {}
    model_names: set[str] = set()
    for site in args.sites:
        source = output / site / "intervention_scored.csv"
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing completed intervention artifact: {source}"
            )
        rows = _read_csv_records(source)
        if not rows:
            raise ValueError(f"Completed intervention artifact is empty: {source}")
        model_names.update(str(row.get("model_name", "")) for row in rows)
        seeds = sorted(
            {
                int(row["random_seed"])
                for row in rows
                if str(row.get("condition")) == "rand_zero"
            }
        )
        summaries_by_site[site] = summarize_ablation(
            rows,
            expected_random_seeds=seeds,
            expected_strata=expected_strata,
            accuracy_spread_threshold=args.random_spread_threshold,
        )
        input_artifacts[site] = artifact_record(source, rows=len(rows))
    if len(model_names) != 1 or "" in model_names:
        raise ValueError(
            f"Existing condition files must contain one model, found "
            f"{sorted(model_names)}"
        )
    model_name = next(iter(model_names))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "section62_ablation_summary",
        "analysis_scope": analysis_scope,
        "model_name": model_name,
        "by_site": summaries_by_site,
    }
    summary_path = write_json_atomic(
        summary,
        output / "ablation_summary.json",
    )
    summary_tables = {
        "ablation_by_cell": [
            row
            for site_summary in summaries_by_site.values()
            for row in site_summary["by_cell"]
        ],
        "rand_zero_by_cell": [
            row
            for site_summary in summaries_by_site.values()
            for row in site_summary["rand_zero"]["by_cell"]
        ],
        "confirmatory_by_cell": [
            row
            for site_summary in summaries_by_site.values()
            for row in site_summary["confirmatory"]["by_cell"]
        ],
        "confirmatory_headline": [
            row
            for site_summary in summaries_by_site.values()
            for row in site_summary["confirmatory"]["headline"]
        ],
        "portability_by_group": [
            row
            for site_summary in summaries_by_site.values()
            for row in site_summary["portability"]["by_group"]
        ],
        "portability_pair_outcomes": [
            row
            for site_summary in summaries_by_site.values()
            for row in site_summary["portability"]["pair_outcomes"]
        ],
        "portability_whole_square_outcomes": [
            row
            for site_summary in summaries_by_site.values()
            for row in site_summary["portability"]["whole_square_outcomes"]
        ],
    }
    output_artifacts: dict[str, Any] = {
        "ablation_summary": artifact_record(summary_path),
    }
    for name, rows in summary_tables.items():
        path = write_csv_atomic(rows, output / f"{name}.csv")
        output_artifacts[name] = artifact_record(path, rows=len(rows))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "section62_existing_ablation_summary",
        "analysis_scope": analysis_scope,
        "model_name": model_name,
        "expected_strata": list(expected_strata),
        "source_interventions": input_artifacts,
        "artifacts": output_artifacts,
    }
    manifest_path = write_json_atomic(
        manifest,
        output / "summary_recovery_manifest.json",
    )
    print(f"Summarized existing R2 artifacts to {output}")
    print(f"Recovery manifest: {manifest_path}")
    return 0


def stable_generator_seed(
    *,
    model_name: str,
    site: str,
    layer: int,
    logical_seed: int,
) -> int:
    """Namespace a logical random-control seed to a model/site."""

    payload = (
        f"section62-rand-zero-v1\0{model_name}\0{site}\0"
        f"{int(layer)}\0{int(logical_seed)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().to("cpu").to(dtype=tensor.dtype).contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_runtime_rows(
    r1_rows: Sequence[Mapping[str, Any]],
    membership: AnalysisMembership,
) -> list[dict[str, Any]]:
    """Attach fixed membership while preserving R1 scores under r1_* names."""

    membership_by_id = {
        str(row["sample_id"]): dict(row) for row in membership.rows
    }
    if len(membership_by_id) != len(membership.rows):
        raise ValueError("Analysis membership contains duplicate sample IDs")
    runtime_rows: list[dict[str, Any]] = []
    for original in r1_rows:
        sample_id = str(original["sample_id"])
        member = membership_by_id.get(sample_id)
        if member is None:
            raise ValueError(f"R1 row {sample_id!r} is absent from membership")
        row = dict(original)
        for field in R1_SCORE_FIELDS:
            if field in row:
                row[f"r1_{field}"] = row.pop(field)
        row["r1_condition"] = row.pop("condition", "base")
        for field in ("site_alias", "layer", "subspace_var", "random_seed"):
            row.pop(field, None)
        row.update(member)
        runtime_rows.append(row)
    if len(runtime_rows) != len(membership.rows):
        raise ValueError("R1 and membership row counts differ")
    return runtime_rows


def prefix_metrics(
    metrics: Mapping[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}{field}": metrics.get(field, "")
        for field in BASE_METRIC_FIELDS
    }


def verify_recomputed_base(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require R2 baseline labels to reproduce the frozen R1 membership."""

    if len(rows) != len(metrics):
        raise ValueError("Recomputed baseline metrics do not align with rows")
    mismatches: list[str] = []
    absolute_logit_drift: dict[str, list[float]] = {
        label: [] for label in ("T", "F", "U")
    }
    for row, current in zip(rows, metrics):
        sample_id = str(row["sample_id"])
        r1_prediction = str(row["r1_pred_label"])
        current_prediction = str(current["pred_label"])
        if current_prediction != r1_prediction:
            mismatches.append(
                f"{sample_id}:{r1_prediction}->{current_prediction}"
            )
        for label in ("T", "F", "U"):
            r1_value = float(row[f"r1_logit_{label}"])
            current_value = float(current[f"logit_{label}"])
            absolute_logit_drift[label].append(abs(current_value - r1_value))
    if mismatches:
        preview = ", ".join(mismatches[:10])
        raise RuntimeError(
            f"R2 baseline disagrees with frozen R1 on {len(mismatches)} rows: "
            f"{preview}"
        )
    return {
        "n_rows": len(rows),
        "n_prediction_mismatches": 0,
        "max_abs_logit_drift": {
            label: max(values, default=0.0)
            for label, values in absolute_logit_drift.items()
        },
        "mean_abs_logit_drift": {
            label: (
                sum(values) / len(values)
                if values
                else 0.0
            )
            for label, values in absolute_logit_drift.items()
        },
    }


def attach_base_metrics(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(rows) != len(metrics):
        raise ValueError("Base metrics do not align with rows")
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row, current in zip(rows, metrics):
        sample_id = str(row["sample_id"])
        metric_row = dict(current)
        if sample_id in by_id:
            raise ValueError(f"Duplicate base sample ID {sample_id!r}")
        by_id[sample_id] = metric_row
        record = dict(row)
        record.update(
            {
                "condition": "base",
                "intervention_kind": "none",
                "subspace_var": "",
                "site_alias": "",
                "layer": "",
                "rank": "",
                "random_seed": "",
                "generator_seed": "",
                "basis_sha256": "",
                "coordinate_source_sample_id": "",
            }
        )
        record.update(prefix_metrics(metric_row, "base_"))
        record.update(metric_row)
        records.append(record)
    return records, by_id


def enrich_intervention_records(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    *,
    base_metrics_by_id: Mapping[str, Mapping[str, Any]],
    profile: FrozenSiteProfile,
    condition: str,
    subspace_var: str,
    intervention_kind: str,
    basis_sha256: str,
    random_seed: int | None = None,
    generator_seed: int | None = None,
) -> list[dict[str, Any]]:
    if len(rows) != len(metrics):
        raise ValueError(f"{condition} metrics do not align with rows")
    records: list[dict[str, Any]] = []
    for row, current in zip(rows, metrics):
        sample_id = str(row["sample_id"])
        base = base_metrics_by_id.get(sample_id)
        if base is None:
            raise ValueError(f"No recomputed baseline for {sample_id!r}")
        record = dict(row)
        record.update(
            {
                "condition": condition,
                "intervention_kind": intervention_kind,
                "subspace_var": subspace_var,
                "site_alias": profile.reported_site,
                "layer": profile.layer,
                "rank": profile.rank,
                "random_seed": (
                    "" if random_seed is None else int(random_seed)
                ),
                "generator_seed": (
                    "" if generator_seed is None else int(generator_seed)
                ),
                "basis_sha256": basis_sha256,
                "coordinate_source_sample_id": (
                    str(row["portability_donor_sample_id"])
                    if condition in {
                        "rho_same_cross_cell",
                        "rho_opposite_cross_cell",
                    }
                    else ""
                ),
            }
        )
        record.update(prefix_metrics(base, "base_"))
        record.update(current)
        record["accuracy_drop"] = (
            int(base["is_correct"]) - int(current["is_correct"])
        )
        record["gold_margin_3_drop"] = (
            float(base["gold_margin_3"])
            - float(current["gold_margin_3"])
        )
        if str(row["analysis_stratum"]) == "square_valid":
            record["M_TF_drop"] = (
                float(base["M_TF"]) - float(current["M_TF"])
            )
            record["G_U_drop"] = ""
        else:
            record["M_TF_drop"] = ""
            record["G_U_drop"] = (
                float(base["G_U"]) - float(current["G_U"])
            )
        records.append(record)
    return records


def donor_coordinates_for_rows(
    *,
    torch: Any,
    rows: Sequence[Mapping[str, Any]],
    coordinates: Any,
) -> Any:
    """Reorder captured coordinates from base order into reciprocal donor order."""

    if int(coordinates.shape[0]) != len(rows):
        raise ValueError("Captured coordinates do not align with portability rows")
    index_by_id: dict[str, int] = {}
    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        if sample_id in index_by_id:
            raise ValueError(f"Duplicate portability sample ID {sample_id!r}")
        index_by_id[sample_id] = index
    donor_indices: list[int] = []
    for row in rows:
        donor_id = str(row["portability_donor_sample_id"])
        if donor_id not in index_by_id:
            raise ValueError(
                f"Eligible portability donor {donor_id!r} is absent"
            )
        donor_indices.append(index_by_id[donor_id])
    index_tensor = torch.tensor(donor_indices, dtype=torch.long)
    return coordinates.index_select(0, index_tensor)


def build_opposite_rho_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retarget each square row to an opposite-rho donor in its own triple."""

    by_square: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["corpus"]),
            str(row["analysis_stratum"]),
            str(row["triple_id"]),
        )
        cell = str(row["cell"])
        group = by_square.setdefault(key, {})
        if cell in group:
            raise ValueError(f"Opposite-rho square {key} duplicates cell {cell}")
        group[cell] = row

    output: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row["corpus"]),
            str(row["analysis_stratum"]),
            str(row["triple_id"]),
        )
        group = by_square[key]
        if set(group) != set(OPPOSITE_RHO_DONOR):
            raise ValueError(
                f"Opposite-rho square {key} has cells={sorted(group)}"
            )
        cell = str(row["cell"])
        donor_cell = OPPOSITE_RHO_DONOR[cell]
        donor = group[donor_cell]
        base_label = str(row["expected_label"])
        donor_label = str(donor["expected_label"])
        if base_label not in TF_FLIP or donor_label != TF_FLIP[base_label]:
            raise ValueError(
                f"Opposite-rho labels do not flip for {row['sample_id']}"
            )
        base_rho = float(row["rho_base"])
        donor_rho = float(donor["rho_base"])
        if donor_rho != -base_rho:
            raise ValueError(
                f"Opposite-rho donor does not invert rho for {row['sample_id']}"
            )
        retargeted = dict(row)
        retargeted.update(
            {
                "opposite_base_expected_label": base_label,
                "opposite_target_expected_label": donor_label,
                "opposite_base_rho": base_rho,
                "opposite_target_rho": donor_rho,
                "opposite_direction": (
                    "plus_to_minus" if base_rho > 0 else "minus_to_plus"
                ),
                "expected_label": donor_label,
                "rho_base": donor_rho,
                "portability_group": "opposite_rho",
                "portability_donor_cell": donor_cell,
                "portability_donor_sample_id": str(donor["sample_id"]),
            }
        )
        output.append(retargeted)
    return output


def summarize_opposite_rho(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize bidirectional opposite-rho interchange outcomes."""

    if not rows:
        raise ValueError("Opposite-rho output is empty")
    expected_cells = tuple(OPPOSITE_RHO_DONOR)
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_direction: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_square: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("condition")) != "rho_opposite_cross_cell":
            raise ValueError("Opposite-rho summary received another condition")
        cell = str(row["cell"])
        if cell not in OPPOSITE_RHO_DONOR:
            raise ValueError(f"Unexpected opposite-rho cell {cell!r}")
        if str(row["portability_donor_cell"]) != OPPOSITE_RHO_DONOR[cell]:
            raise ValueError(f"Wrong opposite-rho donor for cell {cell}")
        if str(row["expected_label"]) != str(
            row["opposite_target_expected_label"]
        ):
            raise ValueError("Opposite-rho row was not scored against target label")
        by_cell[cell].append(row)
        by_direction[str(row["opposite_direction"])].append(row)
        by_square[(str(row["analysis_stratum"]), str(row["triple_id"]))].append(row)

    counts = {cell: len(by_cell[cell]) for cell in expected_cells}
    if len(set(counts.values())) != 1 or 0 in counts.values():
        raise ValueError(f"Opposite-rho cell coverage mismatch: {counts}")
    for key, values in by_square.items():
        cells = {str(row["cell"]) for row in values}
        if cells != set(expected_cells) or len(values) != len(expected_cells):
            raise ValueError(f"Opposite-rho square {key} is incomplete")

    def aggregate(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        n = len(values)
        predictions = [str(row["pred_label"]) for row in values]
        return {
            "n": n,
            "IIA": sum(int(row["is_correct"]) for row in values) / n,
            "TF_IIA": sum(int(row["is_tf_correct"]) for row in values) / n,
            "mean_target_M_TF": sum(float(row["M_TF"]) for row in values) / n,
            "mean_R": sum(float(row["R"]) for row in values) / n,
            "T_rate": predictions.count("T") / n,
            "F_rate": predictions.count("F") / n,
            "U_rate": predictions.count("U") / n,
            "base_accuracy": sum(int(row["base_is_correct"]) for row in values) / n,
        }

    cell_rows = []
    for cell in expected_cells:
        values = by_cell[cell]
        cell_rows.append(
            {
                "model_name": str(values[0]["model_name"]),
                "site_alias": str(values[0]["site_alias"]),
                "layer": int(values[0]["layer"]),
                "cell": cell,
                "rho_base": float(values[0]["opposite_base_rho"]),
                "rho_target": float(values[0]["opposite_target_rho"]),
                "base_expected_label": str(
                    values[0]["opposite_base_expected_label"]
                ),
                "target_expected_label": str(
                    values[0]["opposite_target_expected_label"]
                ),
                **aggregate(values),
            }
        )
    direction_rows = [
        {"opposite_direction": direction, **aggregate(values)}
        for direction, values in sorted(by_direction.items())
    ]
    square_success = [
        int(all(int(row["is_correct"]) for row in values))
        for values in by_square.values()
    ]
    return {
        "condition": "rho_opposite_cross_cell",
        "overall": aggregate(list(rows)),
        "by_cell": cell_rows,
        "by_direction": direction_rows,
        "whole_square": {
            "n_squares": len(square_success),
            "n_squares_all_four_correct": sum(square_success),
            "whole_square_IIA": sum(square_success) / len(square_success),
        },
    }


def random_accuracy_spread(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Return the maximum per-cell accuracy range over completed random seeds."""

    grouped: dict[
        tuple[str, str, str, int], list[int]
    ] = defaultdict(list)
    for row in rows:
        if str(row.get("condition")) != "rand_zero":
            continue
        key = (
            str(row["site_alias"]),
            str(row["analysis_stratum"]),
            str(row["cell"]),
            int(row["random_seed"]),
        )
        grouped[key].append(int(row["is_correct"]))
    by_cell: dict[tuple[str, str, str], list[tuple[int, float, int]]] = (
        defaultdict(list)
    )
    for (site, stratum, cell, seed), values in sorted(grouped.items()):
        by_cell[(site, stratum, cell)].append(
            (seed, sum(values) / len(values), len(values))
        )
    details: list[dict[str, Any]] = []
    for (site, stratum, cell), values in sorted(by_cell.items()):
        accuracies = [accuracy for _, accuracy, _ in values]
        details.append(
            {
                "site_alias": site,
                "analysis_stratum": stratum,
                "cell": cell,
                "n_seeds": len(values),
                "seeds": [seed for seed, _, _ in values],
                "n_per_seed": [n for _, _, n in values],
                "accuracy_by_seed": [
                    accuracy for _, accuracy, _ in values
                ],
                "accuracy_min": min(accuracies),
                "accuracy_max": max(accuracies),
                "accuracy_spread": max(accuracies) - min(accuracies),
            }
        )
    return (
        max(
            (float(row["accuracy_spread"]) for row in details),
            default=0.0,
        ),
        details,
    )


def load_and_validate_model(
    *,
    handoff_summary: Mapping[str, Any],
    model_name: str,
    device: str | None,
    device_map: str | None,
    torch_dtype: str | None,
    trust_remote_code: bool,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[Any, Any, Any, Any, dict[str, int], dict[str, Any]]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Install torch and transformers before running R2"
        ) from error

    r1_run = handoff_summary.get("run")
    if not isinstance(r1_run, dict):
        raise ValueError("R1 summary is missing run metadata")
    r1_dtype = str(r1_run.get("torch_dtype", "auto"))
    resolved_dtype_request = torch_dtype or r1_dtype
    if torch_dtype is not None and torch_dtype != r1_dtype:
        raise ValueError(
            f"R2 torch dtype {torch_dtype!r} differs from R1 {r1_dtype!r}"
        )
    tokenizer, model = load_hf_model(
        torch=torch,
        auto_model_cls=AutoModelForCausalLM,
        auto_tokenizer_cls=AutoTokenizer,
        model_name=model_name,
        device=device,
        device_map=device_map,
        torch_dtype=resolved_dtype_request,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    label_style = str(r1_run["label_token_style_resolved"])
    label_tokens = resolve_label_tokens(tokenizer, label_style)
    actual_commit = getattr(model.config, "_commit_hash", None)
    expected_commit = r1_run.get("model_commit_hash")
    if not expected_commit or actual_commit != expected_commit:
        raise RuntimeError(
            f"Live model commit {actual_commit!r} does not match "
            f"R1 commit {expected_commit!r}"
        )
    actual_name = str(getattr(model.config, "_name_or_path", model_name))
    if actual_name != str(r1_run.get("model_name_or_path_resolved")):
        raise RuntimeError(
            f"Live model path/name {actual_name!r} does not match R1"
        )
    if type(model).__name__ != str(r1_run.get("model_class")):
        raise RuntimeError(
            f"Live model class {type(model).__name__!r} does not match R1"
        )
    for field in ("hidden_size", "num_hidden_layers"):
        actual = int(getattr(model.config, field))
        expected = int(r1_run[field])
        if actual != expected:
            raise RuntimeError(
                f"Live model {field}={actual}, R1 recorded {expected}"
            )
    expected_ids = {
        str(label): int(token_id)
        for label, token_id in dict(r1_run["label_token_ids"]).items()
    }
    if dict(label_tokens.token_ids) != expected_ids:
        raise RuntimeError(
            f"Live label-token IDs {label_tokens.token_ids} do not match R1 "
            f"{expected_ids}"
        )
    expected_texts = {
        str(label): str(text)
        for label, text in dict(r1_run["label_token_texts"]).items()
    }
    if dict(label_tokens.token_texts) != expected_texts:
        raise RuntimeError("Live label-token texts do not match R1")
    if tokenizer.padding_side != str(
        r1_run.get("tokenization", {}).get("padding_side")
    ):
        raise RuntimeError("Live tokenizer padding side does not match R1")

    input_device = next(model.parameters()).device
    layers = get_decoder_layers(model)
    if len(layers) != int(r1_run["num_hidden_layers"]):
        raise RuntimeError("Located decoder-layer count does not match R1")
    run_info = {
        "model_name": model_name,
        "model_name_or_path_resolved": actual_name,
        "model_commit_hash": actual_commit,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "batch_device_resolved": str(input_device),
        "parameter_dtype_resolved": str(next(model.parameters()).dtype),
        "torch_dtype_requested": resolved_dtype_request,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "r1_transformers_version": r1_run.get("transformers_version"),
        "trust_remote_code": trust_remote_code,
        "cache_dir": normalize_cache_dir(cache_dir),
        "local_files_only": local_files_only,
        "label_token_style": label_tokens.style,
        "label_token_ids": dict(label_tokens.token_ids),
        "label_token_texts": dict(label_tokens.token_texts),
        "tokenization": {
            "add_special_tokens": False,
            "padding_side": tokenizer.padding_side,
        },
    }
    return (
        torch,
        tokenizer,
        model,
        layers,
        dict(label_tokens.token_ids),
        run_info,
    )


def _rotation_expected(
    profile: FrozenSiteProfile,
    *,
    target_var: str,
    metadata_site: str,
) -> dict[str, Any]:
    return {
        "target_var": target_var,
        "model_name": profile.model_name,
        "layer": profile.layer,
        "rank": profile.rank,
        "component": "block_output",
        "allowed_metadata_sites": [metadata_site],
    }


def _assert_loaded_rotation_matches_preflight(
    loaded: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    label: str,
) -> None:
    checks = (
        ("metadata_sha256", "metadata"),
        ("weight_sha256", "rotation_weight_npy"),
    )
    for loaded_field, preflight_field in checks:
        expected = str(preflight[preflight_field]["sha256"])
        actual = str(loaded[loaded_field])
        if actual != expected:
            raise RuntimeError(
                f"{label} changed after preflight: {actual} != {expected}"
            )


def _selected_profiles(
    model_name: str,
    sites: Iterable[str],
    *,
    rho_rotation_dir: str | None = None,
    m_rotation_dir: str | None = None,
    rotation_rank: int | None = None,
    rotation_layer: int | None = None,
) -> tuple[FrozenSiteProfile, ...]:
    requested = list(sites)
    if len(requested) != len(set(requested)):
        raise ValueError("--sites contains a duplicate")
    by_site = {
        profile.reported_site: profile
        for profile in frozen_profiles_for_model(model_name)
    }
    unknown = sorted(set(requested) - set(by_site))
    if unknown:
        raise ValueError(f"Unknown frozen site(s): {unknown}")

    override_values = (
        rho_rotation_dir,
        m_rotation_dir,
        rotation_rank,
    )
    if not any(value is not None for value in override_values):
        return tuple(by_site[site] for site in requested)
    if not all(value is not None for value in override_values):
        raise ValueError(
            "Rotation override requires --rho-rotation-dir, "
            "--m-rotation-dir, and --rotation-rank together"
        )
    if len(requested) != 1:
        raise ValueError(
            "Rotation override supports exactly one --sites value per run"
        )
    rank = int(rotation_rank)
    if rank < 1:
        raise ValueError("--rotation-rank must be positive")
    site = requested[0]
    frozen = by_site[site]
    layer = frozen.layer if rotation_layer is None else int(rotation_layer)
    if layer < 0:
        raise ValueError("--rotation-layer must be non-negative")
    return (
        FrozenSiteProfile(
            model_name=frozen.model_name,
            model_key=frozen.model_key,
            reported_site=site,
            layer=layer,
            rank=rank,
            hidden_size=frozen.hidden_size,
            rho_rotation_dir=str(rho_rotation_dir),
            m_rotation_dir=str(m_rotation_dir),
            rho_metadata_site=site,
            m_metadata_site=site,
        ),
    )


def _eligible_rows(
    runtime_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    return [dict(row) for row in runtime_rows if int(row[field]) == 1]


def _site_condition_path(
    output: Path,
    profile: FrozenSiteProfile,
    condition: str,
    *,
    random_seed: int | None = None,
) -> Path:
    suffix = (
        f"_seed{int(random_seed)}"
        if random_seed is not None
        else ""
    )
    return output / profile.reported_site / f"{condition}{suffix}.csv"


def run_opposite_rho(args: argparse.Namespace) -> int:
    """Run only opposite-rho interchange using a completed R2 base artifact."""

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.device and args.device_map not in (None, "none"):
        raise ValueError(
            "--device requires --device-map none; do not request both placements"
        )
    if args.analysis_population != "full-square":
        raise ValueError("--opposite-rho-only requires --analysis-population full-square")
    if not args.existing_run_dir:
        raise ValueError("--opposite-rho-only requires --existing-run-dir")
    if len(args.sites) != 1:
        raise ValueError("--opposite-rho-only requires exactly one --sites value")

    repo_root = Path(args.repo_root).resolve()
    handoff = load_r1_handoff(
        args.behavioral_summary,
        expected_corpora=("MNLI",),
        verify_r0_manifest=True,
        repo_root=repo_root,
    )
    if args.rho_rotation_dir is not None and args.m_rotation_dir is None:
        if args.rotation_rank is None or args.rotation_layer is None:
            raise ValueError(
                "rho-only override requires --rho-rotation-dir, "
                "--rotation-rank, and --rotation-layer"
            )
        frozen_by_site = {
            value.reported_site: value
            for value in frozen_profiles_for_model(handoff.model_name)
        }
        site = args.sites[0]
        frozen = frozen_by_site[site]
        profile = FrozenSiteProfile(
            model_name=frozen.model_name,
            model_key=frozen.model_key,
            reported_site=site,
            layer=int(args.rotation_layer),
            rank=int(args.rotation_rank),
            hidden_size=frozen.hidden_size,
            rho_rotation_dir=str(args.rho_rotation_dir),
            m_rotation_dir="",
            rho_metadata_site=site,
            m_metadata_site=site,
        )
        preflight = {
            "rho": preflight_rotation_metadata(
                repo_root / profile.rho_rotation_dir,
                expected_target_var="rho",
                expected_model_name=profile.model_name,
                expected_layer=profile.layer,
                expected_rank=profile.rank,
                expected_hidden_size=profile.hidden_size,
                allowed_metadata_sites=(profile.rho_metadata_site,),
            )
        }
    else:
        profiles = _selected_profiles(
            handoff.model_name,
            args.sites,
            rho_rotation_dir=args.rho_rotation_dir,
            m_rotation_dir=args.m_rotation_dir,
            rotation_rank=args.rotation_rank,
            rotation_layer=args.rotation_layer,
        )
        profile = profiles[0]
        preflight = preflight_frozen_profile(profile, repo_root=repo_root)
    if args.rotation_seed is not None and args.rho_rotation_dir is None:
        raise ValueError("--rotation-seed requires explicit rotation overrides")

    membership = build_analysis_membership(
        handoff.rows,
        min_baseline_correct_per_cell=args.min_baseline_correct_per_cell,
        analysis_population="full-square",
    )
    runtime_rows = build_runtime_rows(handoff.rows, membership)
    square_rows = _eligible_rows(runtime_rows, field="pair_eligible")
    opposite_rows = build_opposite_rho_rows(square_rows)
    if len(opposite_rows) != 620:
        raise ValueError(f"Expected 620 opposite-rho rows, found {len(opposite_rows)}")

    existing = Path(args.existing_run_dir).resolve()
    output = Path(args.output_dir).resolve()
    if existing == output:
        raise ValueError("Opposite-rho output must not overwrite the existing R2 run")
    source_manifest_path = existing / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_profiles = source_manifest.get("frozen_profiles", [])
    source_profile = next(
        (
            value
            for value in source_profiles
            if str(value.get("reported_site")) == profile.reported_site
        ),
        None,
    )
    if source_profile is None:
        raise ValueError("Existing R2 run has no matching site profile")
    source_expected = {
        "model_name": profile.model_name,
        "reported_site": profile.reported_site,
        "rank": profile.rank,
    }
    for field, expected in source_expected.items():
        if source_profile.get(field) != expected:
            raise ValueError(
                f"Existing R2 profile {field}={source_profile.get(field)!r}, "
                f"expected={expected!r}"
            )
    expected_profile = {
        "model_name": profile.model_name,
        "reported_site": profile.reported_site,
        "layer": profile.layer,
        "rank": profile.rank,
        "rho_rotation_dir": profile.rho_rotation_dir,
    }
    if source_manifest.get("rotation_training_seed") != args.rotation_seed:
        raise ValueError("Existing R2 rotation seed does not match requested seed")

    base_path = existing / "base_scored.csv"
    base_record = source_manifest["artifacts"]["base_scored"]
    if sha256_file(base_path) != str(base_record["sha256"]):
        raise ValueError("Existing R2 base_scored.csv SHA256 mismatch")
    base_rows = _read_csv_records(base_path)
    if len(base_rows) != int(base_record["rows"]):
        raise ValueError("Existing R2 base_scored.csv row-count mismatch")
    base_by_id = {
        str(row["sample_id"]): {field: row[field] for field in BASE_METRIC_FIELDS}
        for row in base_rows
    }
    missing_base = sorted(
        str(row["sample_id"])
        for row in opposite_rows
        if str(row["sample_id"]) not in base_by_id
    )
    if missing_base:
        raise ValueError(f"Existing R2 base scores miss IDs: {missing_base[:10]}")

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
    input_device = next(model.parameters()).device
    rho_basis, rho_provenance = load_rotation_basis(
        rotation_dir=repo_root / profile.rho_rotation_dir,
        expected=_rotation_expected(
            profile,
            target_var="rho",
            metadata_site=profile.rho_metadata_site,
        ),
        hidden_size=profile.hidden_size,
        torch=torch,
        device=input_device,
    )
    _assert_loaded_rotation_matches_preflight(
        rho_provenance,
        preflight["rho"],
        label=f"{profile.reported_site} rho",
    )
    rho_hash = tensor_sha256(rho_basis)
    coordinates = collect_subspace_coordinates(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        torch=torch,
        device=input_device,
        rows=opposite_rows,
        batch_size=args.batch_size,
        layer=profile.layer,
        site=profile.reported_site,
        basis=rho_basis,
    )
    donor_coordinates = donor_coordinates_for_rows(
        torch=torch,
        rows=opposite_rows,
        coordinates=coordinates,
    )
    metrics = run_single_subspace_condition(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        torch=torch,
        device=input_device,
        rows=opposite_rows,
        label_token_ids=label_token_ids,
        batch_size=args.batch_size,
        layer=profile.layer,
        site=profile.reported_site,
        basis=rho_basis,
        condition=f"{profile.reported_site}/rho_opposite_cross_cell",
        donor_coordinates=donor_coordinates,
    )
    records = enrich_intervention_records(
        opposite_rows,
        metrics,
        base_metrics_by_id=base_by_id,
        profile=profile,
        condition="rho_opposite_cross_cell",
        subspace_var="rho",
        intervention_kind="opposite_coordinate_replacement",
        basis_sha256=rho_hash,
    )
    for record in records:
        record["accuracy_drop"] = ""
        record["gold_margin_3_drop"] = ""
        record["M_TF_drop"] = ""
    summary = summarize_opposite_rho(records)

    output.mkdir(parents=True, exist_ok=True)
    scored_path = write_csv_atomic(records, output / "rho_opposite_cross_cell.csv")
    by_cell_path = write_csv_atomic(
        summary["by_cell"], output / "opposite_by_cell.csv"
    )
    by_direction_path = write_csv_atomic(
        summary["by_direction"], output / "opposite_by_direction.csv"
    )
    summary_path = write_json_atomic(summary, output / "opposite_summary.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "section62_opposite_rho_interchange",
        "analysis_scope": "full_square_bidirectional_rho_interchange",
        "model_name": handoff.model_name,
        "rotation_training_seed": args.rotation_seed,
        "profile": expected_profile,
        "source_r2_manifest": artifact_record(source_manifest_path),
        "source_base_scored": artifact_record(base_path, rows=len(base_rows)),
        "rho_rotation_preflight": preflight["rho"],
        "rho_loaded": rho_provenance,
        "rho_applied_qr_basis_sha256": rho_hash,
        "live_run": live_run,
        "donor_mapping": OPPOSITE_RHO_DONOR,
        "target_contract": {
            "expected_label": "T<->F",
            "rho_target": "-rho_base",
            "same_triple_required": True,
        },
        "artifacts": {
            "rho_opposite_cross_cell": artifact_record(scored_path, rows=len(records)),
            "opposite_by_cell": artifact_record(by_cell_path, rows=len(summary["by_cell"])),
            "opposite_by_direction": artifact_record(
                by_direction_path, rows=len(summary["by_direction"])
            ),
            "opposite_summary": artifact_record(summary_path),
        },
    }
    write_json_atomic(manifest, output / "run_manifest.json")
    print(
        f"Wrote opposite-rho interchange for {handoff.model_name} "
        f"{profile.reported_site} to {output}"
    )
    print(
        f"IIA={summary['overall']['IIA']:.4f}, "
        f"TF_IIA={summary['overall']['TF_IIA']:.4f}, "
        f"whole-square={summary['whole_square']['whole_square_IIA']:.4f}"
    )
    return 0


def run_ablation(args: argparse.Namespace) -> int:
    if args.opposite_rho_only:
        if args.summarize_existing:
            raise ValueError(
                "--opposite-rho-only and --summarize-existing are exclusive"
            )
        return run_opposite_rho(args)
    if args.summarize_existing:
        return summarize_existing_run(args)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.random_spread_threshold <= 1.0:
        raise ValueError("--random-spread-threshold must lie in [0, 1]")
    if args.device and args.device_map not in (None, "none"):
        raise ValueError(
            "--device requires --device-map none; do not request both placements"
        )

    repo_root = Path(args.repo_root).resolve()
    handoff = load_r1_handoff(
        args.behavioral_summary,
        expected_corpora=("MNLI",),
        verify_r0_manifest=True,
        repo_root=repo_root,
    )
    profiles = _selected_profiles(
        handoff.model_name,
        args.sites,
        rho_rotation_dir=args.rho_rotation_dir,
        m_rotation_dir=args.m_rotation_dir,
        rotation_rank=args.rotation_rank,
        rotation_layer=args.rotation_layer,
    )
    if args.rotation_seed is not None and args.rho_rotation_dir is None:
        raise ValueError(
            "--rotation-seed is provenance for an explicit rotation "
            "override and requires the rotation override arguments"
        )
    all_preflight = {
        profile.reported_site: preflight_frozen_profile(
            profile,
            repo_root=repo_root,
        )
        for profile in profiles
    }
    membership = build_analysis_membership(
        handoff.rows,
        min_baseline_correct_per_cell=(
            args.min_baseline_correct_per_cell
        ),
        analysis_population=args.analysis_population,
    )
    analysis_scope = str(membership.summary["analysis_scope"])
    expected_strata = (
        ("square_valid",)
        if args.analysis_population == "full-square"
        else ("square_valid", "all_u")
    )
    runtime_rows = build_runtime_rows(handoff.rows, membership)
    zero_rows = _eligible_rows(runtime_rows, field="row_eligible")
    portability_rows = _eligible_rows(runtime_rows, field="pair_eligible")
    if not zero_rows or not portability_rows:
        raise ValueError("R2 has no eligible zero or portability rows")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    membership_path = write_csv_atomic(
        [dict(row) for row in membership.rows],
        output / "analysis_membership.csv",
    )

    (
        torch,
        tokenizer,
        model,
        layers,
        label_token_ids,
        live_run,
    ) = load_and_validate_model(
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

    full_base_metrics = score_base_rows(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=input_device,
        rows=runtime_rows,
        label_token_ids=label_token_ids,
        batch_size=args.batch_size,
    )
    baseline_verification = verify_recomputed_base(runtime_rows, full_base_metrics)
    full_base_records, base_metrics_by_id = attach_base_metrics(
        runtime_rows,
        full_base_metrics,
    )
    base_path = write_csv_atomic(full_base_records, output / "base_scored.csv")

    overlap_rows: list[dict[str, Any]] = []
    combined_interventions: list[dict[str, Any]] = []
    site_artifacts: dict[str, Any] = {}
    site_rotation_provenance: dict[str, Any] = {}
    random_seeds_used: dict[str, list[int]] = {}
    random_spread_audit: dict[str, Any] = {}
    summaries_by_site: dict[str, dict[str, Any]] = {}

    for profile in profiles:
        preflight = all_preflight[profile.reported_site]
        rho_basis, rho_provenance = load_rotation_basis(
            rotation_dir=repo_root / profile.rho_rotation_dir,
            expected=_rotation_expected(
                profile,
                target_var="rho",
                metadata_site=profile.rho_metadata_site,
            ),
            hidden_size=profile.hidden_size,
            torch=torch,
            device=input_device,
        )
        m_basis, m_provenance = load_rotation_basis(
            rotation_dir=repo_root / profile.m_rotation_dir,
            expected=_rotation_expected(
                profile,
                target_var="m",
                metadata_site=profile.m_metadata_site,
            ),
            hidden_size=profile.hidden_size,
            torch=torch,
            device=input_device,
        )
        _assert_loaded_rotation_matches_preflight(
            rho_provenance,
            preflight["rho"],
            label=f"{profile.reported_site} rho",
        )
        _assert_loaded_rotation_matches_preflight(
            m_provenance,
            preflight["m"],
            label=f"{profile.reported_site} m",
        )
        rho_applied_basis_sha256 = tensor_sha256(rho_basis)
        m_applied_basis_sha256 = tensor_sha256(m_basis)
        overlap = {
            "model_name": handoff.model_name,
            "site_alias": profile.reported_site,
            "layer": profile.layer,
            **basis_overlap_metrics(torch, rho_basis, m_basis),
        }
        overlap_rows.append(overlap)
        site_rotation_provenance[profile.reported_site] = {
            "profile": {
                "model_name": profile.model_name,
                "model_key": profile.model_key,
                "reported_site": profile.reported_site,
                "layer": profile.layer,
                "rank": profile.rank,
                "hidden_size": profile.hidden_size,
                "rho_rotation_dir": profile.rho_rotation_dir,
                "m_rotation_dir": profile.m_rotation_dir,
                "rho_metadata_site": profile.rho_metadata_site,
                "m_metadata_site": profile.m_metadata_site,
            },
            "preflight": preflight,
            "rho_loaded": rho_provenance,
            "m_loaded": m_provenance,
            "rho_applied_qr_basis_sha256": rho_applied_basis_sha256,
            "m_applied_qr_basis_sha256": m_applied_basis_sha256,
            "model_revision_provenance_gap": (
                "Rotation metadata has no model commit; the live model is "
                "therefore pinned to the hash-verified R1 model commit."
            ),
        }
        artifacts_for_site: dict[str, Any] = {}
        site_rows: list[dict[str, Any]] = []

        for condition, target_var, basis, applied_basis_sha256 in (
            (
                "rho_zero",
                "rho",
                rho_basis,
                rho_applied_basis_sha256,
            ),
            (
                "m_zero",
                "m",
                m_basis,
                m_applied_basis_sha256,
            ),
        ):
            metrics = run_single_subspace_condition(
                model=model,
                layers=layers,
                tokenizer=tokenizer,
                torch=torch,
                device=input_device,
                rows=zero_rows,
                label_token_ids=label_token_ids,
                batch_size=args.batch_size,
                layer=profile.layer,
                site=profile.reported_site,
                basis=basis,
                condition=f"{profile.reported_site}/{condition}",
            )
            records = enrich_intervention_records(
                zero_rows,
                metrics,
                base_metrics_by_id=base_metrics_by_id,
                profile=profile,
                condition=condition,
                subspace_var=target_var,
                intervention_kind="ordinary_zero_projection",
                basis_sha256=applied_basis_sha256,
            )
            condition_path = _site_condition_path(
                output,
                profile,
                condition,
            )
            write_csv_atomic(records, condition_path)
            artifacts_for_site[condition] = artifact_record(
                condition_path,
                rows=len(records),
            )
            site_rows.extend(records)

        site_random_rows: list[dict[str, Any]] = []
        seeds = list(INITIAL_RANDOM_SEEDS)
        for logical_seed in seeds:
            generator_seed = stable_generator_seed(
                model_name=handoff.model_name,
                site=profile.reported_site,
                layer=profile.layer,
                logical_seed=logical_seed,
            )
            random_basis = random_orthonormal_basis(
                torch,
                profile.hidden_size,
                profile.rank,
                device=torch.device("cpu"),
                seed=generator_seed,
            )
            random_hash = tensor_sha256(random_basis)
            metrics = run_single_subspace_condition(
                model=model,
                layers=layers,
                tokenizer=tokenizer,
                torch=torch,
                device=input_device,
                rows=zero_rows,
                label_token_ids=label_token_ids,
                batch_size=args.batch_size,
                layer=profile.layer,
                site=profile.reported_site,
                basis=random_basis,
                condition=(
                    f"{profile.reported_site}/rand_zero/"
                    f"seed{logical_seed}"
                ),
            )
            records = enrich_intervention_records(
                zero_rows,
                metrics,
                base_metrics_by_id=base_metrics_by_id,
                profile=profile,
                condition="rand_zero",
                subspace_var="random",
                intervention_kind="ordinary_zero_projection",
                basis_sha256=random_hash,
                random_seed=logical_seed,
                generator_seed=generator_seed,
            )
            condition_path = _site_condition_path(
                output,
                profile,
                "rand_zero",
                random_seed=logical_seed,
            )
            write_csv_atomic(records, condition_path)
            artifacts_for_site[f"rand_zero_seed{logical_seed}"] = (
                artifact_record(condition_path, rows=len(records))
            )
            site_random_rows.extend(records)

        initial_spread, initial_spread_details = random_accuracy_spread(
            site_random_rows
        )
        expanded = initial_spread >= args.random_spread_threshold
        if expanded:
            for logical_seed in EXTRA_RANDOM_SEEDS:
                generator_seed = stable_generator_seed(
                    model_name=handoff.model_name,
                    site=profile.reported_site,
                    layer=profile.layer,
                    logical_seed=logical_seed,
                )
                random_basis = random_orthonormal_basis(
                    torch,
                    profile.hidden_size,
                    profile.rank,
                    device=torch.device("cpu"),
                    seed=generator_seed,
                )
                random_hash = tensor_sha256(random_basis)
                metrics = run_single_subspace_condition(
                    model=model,
                    layers=layers,
                    tokenizer=tokenizer,
                    torch=torch,
                    device=input_device,
                    rows=zero_rows,
                    label_token_ids=label_token_ids,
                    batch_size=args.batch_size,
                    layer=profile.layer,
                    site=profile.reported_site,
                    basis=random_basis,
                    condition=(
                        f"{profile.reported_site}/rand_zero/"
                        f"seed{logical_seed}"
                    ),
                )
                records = enrich_intervention_records(
                    zero_rows,
                    metrics,
                    base_metrics_by_id=base_metrics_by_id,
                    profile=profile,
                    condition="rand_zero",
                    subspace_var="random",
                    intervention_kind="ordinary_zero_projection",
                    basis_sha256=random_hash,
                    random_seed=logical_seed,
                    generator_seed=generator_seed,
                )
                condition_path = _site_condition_path(
                    output,
                    profile,
                    "rand_zero",
                    random_seed=logical_seed,
                )
                write_csv_atomic(records, condition_path)
                artifacts_for_site[
                    f"rand_zero_seed{logical_seed}"
                ] = artifact_record(condition_path, rows=len(records))
                site_random_rows.extend(records)
                seeds.append(logical_seed)
        final_spread, final_spread_details = random_accuracy_spread(
            site_random_rows
        )
        random_seeds_used[profile.reported_site] = seeds
        random_spread_audit[profile.reported_site] = {
            "threshold": args.random_spread_threshold,
            "expansion_rule": "expand_when_initial_max_spread_gte_threshold",
            "initial_seeds": list(INITIAL_RANDOM_SEEDS),
            "extra_seeds": list(EXTRA_RANDOM_SEEDS),
            "expanded_to_five": expanded,
            "initial_max_accuracy_spread": initial_spread,
            "initial_by_cell": initial_spread_details,
            "final_max_accuracy_spread": final_spread,
            "final_by_cell": final_spread_details,
        }
        site_rows.extend(site_random_rows)

        coordinates = collect_subspace_coordinates(
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            torch=torch,
            device=input_device,
            rows=portability_rows,
            batch_size=args.batch_size,
            layer=profile.layer,
            site=profile.reported_site,
            basis=rho_basis,
        )
        donor_coordinates = donor_coordinates_for_rows(
            torch=torch,
            rows=portability_rows,
            coordinates=coordinates,
        )
        portability_metrics = run_single_subspace_condition(
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            torch=torch,
            device=input_device,
            rows=portability_rows,
            label_token_ids=label_token_ids,
            batch_size=args.batch_size,
            layer=profile.layer,
            site=profile.reported_site,
            basis=rho_basis,
            condition=f"{profile.reported_site}/rho_same_cross_cell",
            donor_coordinates=donor_coordinates,
        )
        portability_records = enrich_intervention_records(
            portability_rows,
            portability_metrics,
            base_metrics_by_id=base_metrics_by_id,
            profile=profile,
            condition="rho_same_cross_cell",
            subspace_var="rho",
            intervention_kind="ordinary_coordinate_replacement",
            basis_sha256=rho_applied_basis_sha256,
        )
        portability_path = _site_condition_path(
            output,
            profile,
            "rho_same_cross_cell",
        )
        write_csv_atomic(portability_records, portability_path)
        artifacts_for_site["rho_same_cross_cell"] = artifact_record(
            portability_path,
            rows=len(portability_records),
        )
        site_rows.extend(portability_records)

        site_combined_path = (
            output / profile.reported_site / "intervention_scored.csv"
        )
        write_csv_atomic(site_rows, site_combined_path)
        artifacts_for_site["intervention_scored"] = artifact_record(
            site_combined_path,
            rows=len(site_rows),
        )
        site_artifacts[profile.reported_site] = artifacts_for_site
        combined_interventions.extend(site_rows)
        site_summary = summarize_ablation(
            site_rows,
            expected_random_seeds=seeds,
            expected_strata=expected_strata,
            accuracy_spread_threshold=args.random_spread_threshold,
        )
        rand_summary = site_summary["rand_zero"]
        final_threshold_met = bool(
            rand_summary["need_extra_seeds"]
        )
        seed_cap = len(INITIAL_RANDOM_SEEDS) + len(EXTRA_RANDOM_SEEDS)
        rand_summary["initial_triggered_extra_seeds"] = expanded
        rand_summary["final_spread_threshold_met"] = final_threshold_met
        rand_summary["seed_cap"] = seed_cap
        rand_summary["seed_cap_reached"] = len(seeds) >= seed_cap
        rand_summary["need_extra_seeds"] = (
            final_threshold_met and len(seeds) < seed_cap
        )
        summaries_by_site[profile.reported_site] = site_summary

    overlap_path = write_csv_atomic(
        overlap_rows,
        output / "basis_overlap.csv",
    )
    combined_path = write_csv_atomic(
        combined_interventions,
        output / "intervention_scored.csv",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "section62_ablation_summary",
        "analysis_scope": analysis_scope,
        "model_name": handoff.model_name,
        "rotation_training_seed": args.rotation_seed,
        "by_site": summaries_by_site,
    }
    summary_path = write_json_atomic(
        summary,
        output / "ablation_summary.json",
    )
    summary_tables = {
        "ablation_by_cell": [
            row
            for site in summaries_by_site.values()
            for row in site["by_cell"]
        ],
        "rand_zero_by_cell": [
            row
            for site in summaries_by_site.values()
            for row in site["rand_zero"]["by_cell"]
        ],
        "confirmatory_by_cell": [
            row
            for site in summaries_by_site.values()
            for row in site["confirmatory"]["by_cell"]
        ],
        "confirmatory_headline": [
            row
            for site in summaries_by_site.values()
            for row in site["confirmatory"]["headline"]
        ],
        "portability_by_group": [
            row
            for site in summaries_by_site.values()
            for row in site["portability"]["by_group"]
        ],
        "portability_pair_outcomes": [
            row
            for site in summaries_by_site.values()
            for row in site["portability"]["pair_outcomes"]
        ],
        "portability_whole_square_outcomes": [
            row
            for site in summaries_by_site.values()
            for row in site["portability"]["whole_square_outcomes"]
        ],
    }
    summary_artifacts: dict[str, Any] = {
        "ablation_summary": artifact_record(summary_path),
    }
    for name, rows in summary_tables.items():
        table_path = write_csv_atomic(rows, output / f"{name}.csv")
        summary_artifacts[name] = artifact_record(
            table_path,
            rows=len(rows),
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": RUN_TYPE,
        "analysis_scope": analysis_scope,
        "rotation_training_seed": args.rotation_seed,
        "r1_handoff": handoff.provenance,
        "membership": membership.summary,
        "frozen_profiles": [
            {
                "model_name": profile.model_name,
                "model_key": profile.model_key,
                "reported_site": profile.reported_site,
                "layer": profile.layer,
                "rank": profile.rank,
                "hidden_size": profile.hidden_size,
                "rho_rotation_dir": profile.rho_rotation_dir,
                "m_rotation_dir": profile.m_rotation_dir,
                "rho_metadata_site": profile.rho_metadata_site,
                "m_metadata_site": profile.m_metadata_site,
            }
            for profile in profiles
        ],
        "rotation_provenance": site_rotation_provenance,
        "live_run": {
            **live_run,
            "batch_size": args.batch_size,
            "device_requested": args.device,
            "device_map": args.device_map,
        },
        "baseline_verification": baseline_verification,
        "interventions": {
            "zero_operator": "h_prime = h - (hU)U^T",
            "portability_operator": (
                "h_prime = h - (hU_rho)U_rho^T + "
                "(h_donor U_rho)U_rho^T"
            ),
            "zero_NEx_definition": (
                "mean(rand_zero accuracy) - rho_zero accuracy"
            ),
            "random_seeds_initial": list(INITIAL_RANDOM_SEEDS),
            "random_seeds_extra_if_unstable": list(EXTRA_RANDOM_SEEDS),
            "random_seeds_used_by_site": random_seeds_used,
            "random_spread_audit": random_spread_audit,
            "random_spread_threshold": args.random_spread_threshold,
            "portability_groups_reported_separately": [
                "negation_count_parity",
                "negation_position",
            ],
        },
        "artifacts": {
            "analysis_membership": artifact_record(
                membership_path,
                rows=len(membership.rows),
            ),
            "base_scored": artifact_record(
                base_path,
                rows=len(full_base_records),
            ),
            "basis_overlap": artifact_record(
                overlap_path,
                rows=len(overlap_rows),
            ),
            "intervention_scored": artifact_record(
                combined_path,
                rows=len(combined_interventions),
            ),
            **summary_artifacts,
            "by_site": site_artifacts,
        },
    }
    manifest_path = output / "run_manifest.json"
    write_json_atomic(manifest, manifest_path)
    print(
        f"Wrote Section 6.2 R2 ablations for {handoff.model_name} "
        f"to {output}"
    )
    for site in (profile.reported_site for profile in profiles):
        spread = random_spread_audit[site]
        print(
            f"{site}: random seeds={random_seeds_used[site]}, "
            f"initial max spread="
            f"{spread['initial_max_accuracy_spread']:.3f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavioral-summary", required=True)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help=(
            "summarize already-written per-site intervention_scored.csv "
            "files without loading a model"
        ),
    )
    parser.add_argument(
        "--opposite-rho-only",
        action="store_true",
        help=(
            "run only same-triple opposite-rho coordinate replacement, "
            "using base scores from --existing-run-dir"
        ),
    )
    parser.add_argument(
        "--existing-run-dir",
        help="completed matching R2 run supplying verified base scores",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=["claim_final", "answer_token"],
        default=["claim_final", "answer_token"],
    )
    parser.add_argument(
        "--rho-rotation-dir",
        help=(
            "override the frozen rho rotation directory; normal R2 requires "
            "--m-rotation-dir and --rotation-rank, while "
            "--opposite-rho-only permits a rho-only override"
        ),
    )
    parser.add_argument(
        "--m-rotation-dir",
        help=(
            "override the frozen m rotation directory; requires exactly "
            "one --sites value plus --rho-rotation-dir and --rotation-rank"
        ),
    )
    parser.add_argument(
        "--rotation-rank",
        type=int,
        help="rank shared by the explicit rho and m rotation overrides",
    )
    parser.add_argument(
        "--rotation-layer",
        type=int,
        help="layer used by explicit rotation overrides",
    )
    parser.add_argument(
        "--rotation-seed",
        type=int,
        help=(
            "training seed provenance for explicit rotation overrides; "
            "does not change the rand-zero seeds"
        ),
    )
    parser.add_argument(
        "--analysis-population",
        choices=["baseline-correct", "full-square"],
        default="baseline-correct",
        help=(
            "baseline-correct preserves the original secondary analysis; "
            "full-square intervenes on all 620 rows of the 155 MNLI "
            "square-valid examples and excludes all-U rows"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--min-baseline-correct-per-cell",
        type=int,
        default=DEFAULT_MIN_BASELINE_CORRECT_PER_CELL,
    )
    parser.add_argument(
        "--random-spread-threshold",
        type=float,
        default=DEFAULT_RANDOM_SPREAD_THRESHOLD,
    )
    parser.add_argument("--device")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    return run_ablation(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
