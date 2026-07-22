"""CSV and JSONL helpers for generated samples and model results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

PREFERRED_COLUMNS = [
    "rho_regime",
    "rho_direction",
    "rho_identity",
    "rho_default_train",
    "rho_base",
    "rho_src",
    "pred_H_rho",
    "pred_H_gated_rel",
    "row_id",
    "run_family",
    "pi_variant",
    "pi_regime",
    "m_variant",
    "m_label_copy_trap_type",
    "m_eval_balanced",
    "sample_id",
    "base_event_id",
    "experiment",
    "subexperiment",
    "axis",
    "condition",
    "assumption",
    "claim",
    "prompt",
    "expected_label",
    "target_label",
    "target_var",
    "control_type",
    "split",
    "base_label",
    "source_label",
    "base_site",
    "source_site",
    "base_prompt",
    "source_prompt",
    "base_assumption",
    "source_assumption",
    "base_claim",
    "source_claim",
    "m_base",
    "m_src",
    "p_i_base",
    "p_i_src",
    "p_c_base",
    "p_c_src",
    "distractor_idx",
    "distractor_p_base",
    "distractor_p_src",
    "mismatch_type",
    "expected_R_sign",
    "label_confidence",
    "claim_subject",
    "claim_verb_base",
    "claim_verb_past",
    "claim_object",
    "claim_arg_type",
    "claim_polarity",
    "claim_form",
    "claim_axis",
    "claim_axis_sign",
    "source_subject",
    "source_verb_base",
    "source_verb_past",
    "source_object",
    "source_arg_type",
    "source_polarity",
    "assumption_form",
    "source_axis_sign",
    "overlap_type",
    "overlap_count",
    "same_subject",
    "same_verb",
    "same_object",
    "match_idx",
    "match_polarity",
    "exp3_design",
    "exp3_distractor_config",
    "exp3_distractor_pattern",
    "exp3_flipped_distractor_idx",
    "exp3_target_pair_id",
    "exp3_distractor_cell_id",
    "pattern",
    "order_pattern",
    "claim_object_role",
    "q",
    "n_pos",
    "n_neg",
    "source_only",
    "sanity_type",
    "phase_relation",
    "phase_cos",
    "verb_allowed_for_frequency",
    "frequency_naturalness",
    "n_assumptions",
    "source1_subject",
    "source1_verb",
    "source1_verb_base",
    "source1_verb_past",
    "source1_object",
    "source1_arg_type",
    "source1_polarity",
    "source1_overlap_type",
    "source2_subject",
    "source2_verb",
    "source2_verb_base",
    "source2_verb_past",
    "source2_object",
    "source2_arg_type",
    "source2_polarity",
    "source2_overlap_type",
    "source3_subject",
    "source3_verb",
    "source3_verb_base",
    "source3_verb_past",
    "source3_object",
    "source3_arg_type",
    "source3_polarity",
    "source3_overlap_type",
    "label_token_T",
    "label_token_F",
    "label_token_U",
    "label_token_style",
    "logit_T",
    "logit_F",
    "logit_U",
    "R",
    "R_claim",
    "R_axis",
    "U_gap",
    "pred_label",
    "is_correct",
]


def write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ordered_columns(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_rows_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    extras = sorted({key for row in rows for key in row} - set(PREFERRED_COLUMNS))
    present_preferred = [column for column in PREFERRED_COLUMNS if any(column in row for row in rows)]
    return present_preferred + extras
