"""Next-run diagnostic dataset generation after the Qwen3-8B pilot."""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import Event, Polarity, VERBS, sentence
from .generation import clean_distractors, choose_other, choose_other_verb, make_sample, sample_base_events

NEXT_SECTIONS = (
    "exp4_v2",
    "unrelated_conflict",
    "exp2b",
    "duplicate_controls",
)

VERB_BY_BASE = {verb.base: verb for verb in VERBS}


def generate_next_run(
    n_base_events: int = 20,
    seed: int = 0,
    base_events_from_csv: str | None = "data/qwen3_8_pilot/samples.csv",
    sections: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate the focused next-run diagnostics without rerunning the whole suite."""

    selected = tuple(sections or NEXT_SECTIONS)
    if selected == ("all",):
        selected = NEXT_SECTIONS
    unknown = sorted(set(selected) - set(NEXT_SECTIONS))
    if unknown:
        raise ValueError(f"Unknown next-run sections: {unknown}")

    rng = random.Random(seed)
    base_items = load_or_sample_base_events(n_base_events, seed, base_events_from_csv)
    rows: list[dict[str, Any]] = []

    for base_id, event in base_items:
        if "exp4_v2" in selected:
            rows.extend(generate_exp4_v2(base_id, event))
        if "unrelated_conflict" in selected:
            rows.extend(generate_unrelated_conflict(base_id, event, rng))
        if "exp2b" in selected:
            rows.extend(generate_exp2b(base_id, event, rng))
        if "duplicate_controls" in selected:
            rows.extend(generate_duplicate_controls(base_id, event))

    for idx, row in enumerate(rows):
        row["row_id"] = idx
    return rows


def load_or_sample_base_events(
    n_base_events: int,
    seed: int,
    base_events_from_csv: str | None,
) -> list[tuple[str, Event]]:
    path = normalize_optional_path(base_events_from_csv)
    if path is not None:
        return load_base_events_from_csv(path, n_base_events)

    rng = random.Random(seed)
    events = sample_base_events(n_base_events, rng)
    return [(f"base_{idx:04d}", event) for idx, event in enumerate(events)]


def load_base_events_from_csv(path: str | Path, n_base_events: int | None = None) -> list[tuple[str, Event]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Base-event CSV not found: {path}. Pass --base-events-from-csv none to sample by seed instead."
        )

    rows: dict[str, Event] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("experiment") != "exp1_phase_flip" or row.get("condition") != "A+ C+":
                continue
            base_id = row["base_event_id"]
            verb = VERB_BY_BASE[row["claim_verb_base"]]
            rows[base_id] = Event(row["claim_subject"], verb, row["claim_object"])

    items = sorted(rows.items())
    if n_base_events is not None:
        items = items[:n_base_events]
    if not items:
        raise ValueError(f"No base events found in {path}; expected Exp1 rows with condition A+ C+.")
    return items


def generate_exp4_v2(base_id: str, event: Event) -> list[dict[str, Any]]:
    patterns = ["+-", "-+", "++-", "+-+", "-++", "+--", "-+-", "--+", "+", "-"]
    rows = []
    for pattern in patterns:
        polarities = polarities_from_pattern(pattern)
        expected_label, expected_sign = expected_from_q(pattern)
        assumptions = [sentence(event, polarity) for polarity in polarities]
        rows.append(
            make_sample(
                sample_id=f"next_exp4v2_{base_id}_{pattern_slug(pattern)}",
                base_id=base_id,
                experiment="next_exp4_v2_order_permutation",
                condition=exp4_condition(pattern),
                assumptions=assumptions,
                sources=[event for _ in polarities],
                source_polarities=polarities,
                claim_event=event,
                claim_polarity="positive",
                expected_label=expected_label,
                expected_R_sign=expected_sign,
                extra=pattern_metadata(pattern) | {
                    "next_run_section": "exp4_v2",
                    "multiset": multiset_name(pattern),
                },
            )
        )
    return rows


def generate_unrelated_conflict(base_id: str, conflict_event: Event, rng: random.Random) -> list[dict[str, Any]]:
    claim_event = clean_distractors(conflict_event, rng, count=1)[0]
    rows = []
    for pattern in ("+-", "-+"):
        polarities = polarities_from_pattern(pattern)
        assumptions = [sentence(conflict_event, polarity) for polarity in polarities]
        rows.append(
            make_sample(
                sample_id=f"next_unrelated_{base_id}_{pattern_slug(pattern)}",
                base_id=base_id,
                experiment="next_unrelated_conflict",
                condition=f"unrelated_conflict_{pattern}",
                assumptions=assumptions,
                sources=[conflict_event for _ in polarities],
                source_polarities=polarities,
                claim_event=claim_event,
                claim_polarity="positive",
                expected_label="U",
                expected_R_sign=0,
                extra=pattern_metadata(pattern) | {
                    "next_run_section": "unrelated_conflict",
                    "conflict_event_subject": conflict_event.subject,
                    "conflict_event_verb_base": conflict_event.verb.base,
                    "conflict_event_object": conflict_event.obj,
                    "unrelated_claim_subject": claim_event.subject,
                    "unrelated_claim_verb_base": claim_event.verb.base,
                    "unrelated_claim_object": claim_event.obj,
                },
            )
        )
    return rows


def generate_exp2b(base_id: str, claim_event: Event, rng: random.Random) -> list[dict[str, Any]]:
    source_by_overlap = exp2b_sources(claim_event, rng)
    phase_combos: list[tuple[Polarity, Polarity, str]] = [
        ("positive", "positive", "A+ C+"),
        ("negative", "positive", "A- C+"),
        ("positive", "negative", "A+ C-"),
        ("negative", "negative", "A- C-"),
    ]

    rows = []
    for overlap_type in ("SVO", "SV", "VO", "S-only", "none"):
        source_event = source_by_overlap[overlap_type]
        for source_polarity, claim_polarity, phase_combo in phase_combos:
            same_phase = source_polarity == claim_polarity
            if overlap_type == "SVO":
                expected_label = "T" if same_phase else "F"
                expected_sign = 1 if same_phase else -1
            else:
                expected_label = "U"
                expected_sign = 0
            rows.append(
                make_sample(
                    sample_id=(
                        f"next_exp2b_{base_id}_{overlap_slug(overlap_type)}_"
                        f"{polarity_short(source_polarity)}_{polarity_short(claim_polarity)}"
                    ),
                    base_id=base_id,
                    experiment="next_exp2b_counterbalanced_overlap",
                    condition=f"{overlap_type}_{phase_combo.replace(' ', '')}",
                    assumptions=[sentence(source_event, source_polarity)],
                    sources=[source_event],
                    source_polarities=[source_polarity],
                    claim_event=claim_event,
                    claim_polarity=claim_polarity,
                    expected_label=expected_label,
                    expected_R_sign=expected_sign,
                    extra={
                        "next_run_section": "exp2b",
                        "phase_combo": phase_combo,
                        "phase_relation": "same" if same_phase else "opposite",
                        "phase_cos": 1 if same_phase else -1,
                    },
                )
            )
    return rows


def generate_duplicate_controls(base_id: str, event: Event) -> list[dict[str, Any]]:
    rows = []
    for pattern in ("++", "--"):
        polarities = polarities_from_pattern(pattern)
        expected_label, expected_sign = expected_from_q(pattern)
        rows.append(
            make_sample(
                sample_id=f"next_duplicate_{base_id}_{pattern_slug(pattern)}",
                base_id=base_id,
                experiment="next_duplicate_controls",
                condition="duplicate_positive" if pattern == "++" else "duplicate_negative",
                assumptions=[sentence(event, polarity) for polarity in polarities],
                sources=[event for _ in polarities],
                source_polarities=polarities,
                claim_event=event,
                claim_polarity="positive",
                expected_label=expected_label,
                expected_R_sign=expected_sign,
                extra=pattern_metadata(pattern) | {
                    "next_run_section": "duplicate_controls",
                    "duplicate_control": 1,
                },
            )
        )
    return rows


def exp2b_sources(claim_event: Event, rng: random.Random) -> dict[str, Event]:
    s_only_verb = choose_other_verb(claim_event.verb, rng, same_arg_type=True)
    none_verb = choose_other_verb(claim_event.verb, rng, same_arg_type=True)
    return {
        "SVO": claim_event,
        "SV": Event(claim_event.subject, claim_event.verb, choose_other(claim_event.verb.candidates, claim_event.obj, rng)),
        "VO": Event(choose_other_person(claim_event.subject, rng), claim_event.verb, claim_event.obj),
        "S-only": Event(claim_event.subject, s_only_verb, choose_other(s_only_verb.candidates, claim_event.obj, rng)),
        "none": Event(
            choose_other_person(claim_event.subject, rng),
            none_verb,
            choose_other(none_verb.candidates, claim_event.obj, rng),
        ),
    }


def choose_other_person(current: str, rng: random.Random) -> str:
    from .base import PERSONS

    return choose_other(PERSONS, current, rng)


def polarities_from_pattern(pattern: str) -> list[Polarity]:
    return ["positive" if char == "+" else "negative" for char in pattern]


def expected_from_q(pattern: str) -> tuple[str, int]:
    q_value = pattern.count("+") - pattern.count("-")
    if q_value > 0:
        return "T", 1
    if q_value < 0:
        return "F", -1
    return "U", 0


def pattern_metadata(pattern: str) -> dict[str, Any]:
    n_pos = pattern.count("+")
    n_neg = pattern.count("-")
    return {
        "pattern": pattern,
        "q": n_pos - n_neg,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "last_sign": 1 if pattern[-1] == "+" else -1,
        "has_neg": int(n_neg > 0),
        "mixed": int(n_pos > 0 and n_neg > 0),
        "source_only": int(len(pattern) == 1),
    }


def exp4_condition(pattern: str) -> str:
    names = {
        "+-": "balanced_+-",
        "-+": "balanced_-+",
        "++-": "positive_imbalance_++-",
        "+-+": "positive_imbalance_+-+",
        "-++": "positive_imbalance_-++",
        "+--": "negative_imbalance_+--",
        "-+-": "negative_imbalance_-+-",
        "--+": "negative_imbalance_--+",
        "+": "source_only_positive",
        "-": "source_only_negative",
    }
    return names[pattern]


def multiset_name(pattern: str) -> str:
    if pattern in {"+-", "-+"}:
        return "balanced"
    if pattern in {"++-", "+-+", "-++"}:
        return "positive_imbalance"
    if pattern in {"+--", "-+-", "--+"}:
        return "negative_imbalance"
    return "source_only"


def pattern_slug(pattern: str) -> str:
    return pattern.replace("+", "p").replace("-", "n")


def overlap_slug(value: str) -> str:
    return value.lower().replace("-", "_")


def polarity_short(polarity: Polarity) -> str:
    return "pos" if polarity == "positive" else "neg"


def normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped or stripped.lower() in {"none", "sample", "seed"}:
        return None
    return stripped
