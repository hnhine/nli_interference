"""Derive lexical-form variants of existing DAS rows.

Rebuilds the claim (or the matched assumption) surface form from the row's
event metadata while keeping events, splits, labels and metadata identical to
the canonical rows, so any behavioral/geometry/intervention difference is
attributable to the lexical realization alone.

Form ladder (decreasing surface overlap with the canonical "did not"):
  did_not       anchor; derived rows must be byte-identical to the originals
  did_not_ever  contains the token "not", different template
  never         no "not", truth-conditionally equivalent
  rarely/seldom no "not", graded meaning (NOT truth-equivalent; descriptive only)
"""

from __future__ import annotations

from typing import Any

from .base import did_not_ever, neg, never, pos, rarely, seldom
from .das_spans import add_span_columns, build_prompt_with_spans

NEGATION_FORMS = ("did_not", "did_not_ever", "never", "rarely", "seldom")
UNAMBIGUOUS_FORMS = ("did_not", "did_not_ever", "never")
GRADED_FORMS = ("rarely", "seldom")

_FORM_BUILDERS = {
    "did_not": neg,
    "did_not_ever": did_not_ever,
    "never": never,
    "rarely": rarely,
    "seldom": seldom,
}

KEEP_COLUMNS = (
    "sample_id", "base_event_id", "experiment", "target_var", "control_type", "split",
    "matched_idx", "base_site", "base_label", "m_base", "p_i_base", "p_c_base",
    "base_claim_polarity",
)


def event_dict(row: dict[str, Any], prefix: str) -> tuple[str, dict[str, Any], str]:
    verb = {
        "base": str(row[f"{prefix}_verb_base"]),
        "past": str(row[f"{prefix}_verb_past"]),
        "arg_type": "generic",
        "candidates": [],
    }
    return str(row[f"{prefix}_subject"]), verb, str(row[f"{prefix}_object"])


def sentence_form(row: dict[str, Any], prefix: str, polarity: str, form: str) -> str:
    subject, verb, obj = event_dict(row, prefix)
    if polarity == "positive":
        return pos(subject, verb, obj)
    return _FORM_BUILDERS[form](subject, verb, obj)


def derive_form_row(row: dict[str, Any], form: str, vary: str) -> dict[str, Any]:
    """Rebuild one main row with the negation form applied to the varied slot."""
    matched_idx = int(row["matched_idx"])
    assumptions = []
    for slot in range(3):
        prefix = f"base_source{slot + 1}"
        polarity = str(row[f"{prefix}_polarity"])
        slot_form = form if (vary == "assumption" and slot == matched_idx) else "did_not"
        assumptions.append(sentence_form(row, prefix, polarity, slot_form))

    claim_polarity = str(row["base_claim_polarity"])
    claim_form = form if vary == "claim" else "did_not"
    claim = sentence_form(row, "claim", claim_polarity, claim_form)

    prompt = build_prompt_with_spans(assumptions, claim)
    out = {key: row[key] for key in KEEP_COLUMNS if key in row}
    out["sample_id"] = f"{row['sample_id']}__{form}"
    out["form"] = form
    out["vary"] = vary
    out["base_prompt"] = prompt.prompt
    add_span_columns(out, "base", prompt.spans)
    return out


def derive_form_rows(rows: list[dict[str, Any]], form: str, vary: str) -> list[dict[str, Any]]:
    return [derive_form_row(row, form, vary) for row in rows]


def count_anchor_mismatches(rows: list[dict[str, Any]], vary: str) -> int:
    """Regression guard: form=did_not must reproduce the original prompts exactly."""
    bad = 0
    for row in rows:
        derived = derive_form_row(row, "did_not", vary)
        if derived["base_prompt"] != str(row["base_prompt"]):
            bad += 1
    return bad


def varied_polarity_column(vary: str) -> str:
    return "p_c_base" if vary == "claim" else "p_i_base"


def is_negative_varied(row: dict[str, Any], vary: str) -> bool:
    return str(row[varied_polarity_column(vary)]) == "-1"


SOURCE_KEEP_COLUMNS = (
    "base_event_id", "experiment", "target_var", "control_type", "split", "matched_idx",
    "m_base", "p_i_base", "p_c_base",
)


def derive_source_form_row(row: dict[str, Any], form: str, vary: str) -> dict[str, Any]:
    """Rebuild the SOURCE side of a pair with the form applied to the varied slot.

    The stored source shares all events (and, for pi pairs, distractor
    assumptions) with the base, so C-pairs built from it are true minimal pairs.
    The derived row uses base_* keys so the shared scoring machinery applies.
    """
    matched_idx = int(row["matched_idx"])
    assumptions = []
    for slot in range(3):
        prefix = f"source_source{slot + 1}"
        polarity = str(row[f"{prefix}_polarity"])
        slot_form = form if (vary == "assumption" and slot == matched_idx) else "did_not"
        assumptions.append(sentence_form(row, prefix, polarity, slot_form))

    claim_polarity = str(row["source_claim_polarity"])
    claim_form = form if vary == "claim" else "did_not"
    claim = sentence_form(row, "claim", claim_polarity, claim_form)

    prompt = build_prompt_with_spans(assumptions, claim)
    out = {key: row[key] for key in SOURCE_KEEP_COLUMNS if key in row}
    out["sample_id"] = f"{row['sample_id']}__src_{form}"
    out["form"] = form
    out["vary"] = vary
    out["base_label"] = str(row.get("source_label", ""))
    out["base_site"] = str(row.get("source_site", row.get("base_site", "claim_final")))
    out["base_prompt"] = prompt.prompt
    add_span_columns(out, "base", prompt.spans)
    return out


def derive_source_form_rows(rows: list[dict[str, Any]], form: str, vary: str) -> list[dict[str, Any]]:
    return [derive_source_form_row(row, form, vary) for row in rows]


def count_source_anchor_mismatches(rows: list[dict[str, Any]], vary: str) -> int:
    bad = 0
    for row in rows:
        derived = derive_source_form_row(row, "did_not", vary)
        if derived["base_prompt"] != str(row["source_prompt"]):
            bad += 1
    return bad
