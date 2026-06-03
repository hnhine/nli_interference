"""Dataset generation for the five interference experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from .base import (
    PERSONS,
    VERBS,
    Event,
    Polarity,
    VerbSpec,
    all_events,
    build_prompt,
    compact_polarity,
    event_metadata,
    format_assumptions,
    polarity_symbol,
    sentence,
)

EXPERIMENTS = (
    "exp1_phase_flip",
    "exp2_carrier_overlap",
    "exp3_clean_selection",
    "exp4_cancellation",
    "exp5_object_bound_phase",
)


def generate_suite(
    n_base_events: int = 20,
    seed: int = 0,
    experiments: Sequence[str] | None = None,
    include_exp3_sanity: bool = False,
    include_exp4_source_only: bool = True,
) -> list[dict[str, Any]]:
    """Generate the full experiment suite as CSV-ready row dictionaries."""

    selected = tuple(experiments or EXPERIMENTS)
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")

    rng = random.Random(seed)
    base_events = sample_base_events(n_base_events, rng)
    rows: list[dict[str, Any]] = []

    for base_index, claim_event in enumerate(base_events):
        base_id = f"base_{base_index:04d}"
        if "exp1_phase_flip" in selected:
            rows.extend(generate_exp1(base_id, claim_event))
        if "exp2_carrier_overlap" in selected:
            rows.extend(generate_exp2(base_id, claim_event, rng))
        if "exp3_clean_selection" in selected:
            rows.extend(generate_exp3(base_id, claim_event, rng, include_sanity=include_exp3_sanity))
        if "exp4_cancellation" in selected:
            rows.extend(generate_exp4(base_id, claim_event, include_source_only=include_exp4_source_only))
        if "exp5_object_bound_phase" in selected:
            rows.extend(generate_exp5(base_id, claim_event, rng))

    for idx, row in enumerate(rows):
        row["row_id"] = idx
    return rows


def sample_base_events(n_base_events: int, rng: random.Random) -> list[Event]:
    candidates = all_events()
    if n_base_events > len(candidates):
        raise ValueError(f"Requested {n_base_events} base events, but only {len(candidates)} are available")
    rng.shuffle(candidates)
    return candidates[:n_base_events]


def generate_exp1(base_id: str, event: Event) -> list[dict[str, Any]]:
    rows = []
    conditions: list[tuple[str, Polarity, Polarity, str, int]] = [
        ("A+ C+", "positive", "positive", "T", 1),
        ("A- C+", "negative", "positive", "F", -1),
        ("A+ C-", "positive", "negative", "F", -1),
        ("A- C-", "negative", "negative", "T", 1),
    ]

    for condition, assumption_polarity, claim_polarity, expected_label, expected_sign in conditions:
        assumption = sentence(event, assumption_polarity)
        row = make_sample(
            sample_id=f"exp1_{base_id}_{compact_polarity(assumption_polarity)}_{compact_polarity(claim_polarity)}",
            base_id=base_id,
            experiment="exp1_phase_flip",
            condition=condition,
            assumptions=[assumption],
            sources=[event],
            source_polarities=[assumption_polarity],
            claim_event=event,
            claim_polarity=claim_polarity,
            expected_label=expected_label,
            expected_R_sign=expected_sign,
            extra={
                "phase_relation": "same" if assumption_polarity == claim_polarity else "opposite",
                "phase_cos": 1 if assumption_polarity == claim_polarity else -1,
            },
        )
        rows.append(row)
    return rows


def generate_exp2(base_id: str, claim_event: Event, rng: random.Random) -> list[dict[str, Any]]:
    s_only_verb = choose_other_verb(claim_event.verb, rng, same_arg_type=True)
    source_by_overlap = {
        "SVO": claim_event,
        "SV": Event(claim_event.subject, claim_event.verb, choose_other(claim_event.verb.candidates, claim_event.obj, rng)),
        "VO": Event(choose_other(PERSONS, claim_event.subject, rng), claim_event.verb, claim_event.obj),
        "S-only": Event(
            claim_event.subject,
            s_only_verb,
            choose_other(s_only_verb.candidates, claim_event.obj, rng),
        ),
    }
    none_verb = choose_other_verb(claim_event.verb, rng, same_arg_type=True)
    source_by_overlap["none"] = Event(
        choose_other(PERSONS, claim_event.subject, rng),
        none_verb,
        choose_other(none_verb.candidates, claim_event.obj, rng),
    )

    rows = []
    for overlap_type in ("SVO", "SV", "VO", "S-only", "none"):
        source_event = source_by_overlap[overlap_type]
        for source_polarity in ("positive", "negative"):
            if overlap_type == "SVO":
                expected_label = "T" if source_polarity == "positive" else "F"
                expected_sign = 1 if source_polarity == "positive" else -1
            else:
                expected_label = "U"
                expected_sign = 0

            row = make_sample(
                sample_id=f"exp2_{base_id}_{slug(overlap_type)}_{compact_polarity(source_polarity)}",
                base_id=base_id,
                experiment="exp2_carrier_overlap",
                condition=f"{overlap_type}_{source_polarity}",
                assumptions=[sentence(source_event, source_polarity)],
                sources=[source_event],
                source_polarities=[source_polarity],
                claim_event=claim_event,
                claim_polarity="positive",
                expected_label=expected_label,
                expected_R_sign=expected_sign,
                extra={"phase_cos": 1 if source_polarity == "positive" else -1},
            )
            rows.append(row)
    return rows


def generate_exp3(base_id: str, claim_event: Event, rng: random.Random, include_sanity: bool = False) -> list[dict[str, Any]]:
    rows = []
    for match_idx in (1, 2, 3):
        for match_polarity in ("positive", "negative"):
            distractors = clean_distractors(claim_event, rng, count=2)
            distractor_polarities: list[Polarity] = ["positive", "positive"]
            rows.append(
                make_exp3_row(
                    base_id=base_id,
                    claim_event=claim_event,
                    match_idx=match_idx,
                    match_polarity=match_polarity,
                    distractors=distractors,
                    distractor_polarities=distractor_polarities,
                    sanity_type="",
                    sample_suffix=f"idx{match_idx}_{compact_polarity(match_polarity)}",
                )
            )

    if include_sanity:
        rows.extend(generate_exp3_sanity(base_id, claim_event, rng))
    return rows


def make_exp3_row(
    base_id: str,
    claim_event: Event,
    match_idx: int,
    match_polarity: Polarity,
    distractors: Sequence[Event],
    distractor_polarities: Sequence[Polarity],
    sanity_type: str,
    sample_suffix: str,
) -> dict[str, Any]:
    expected_label = "T" if match_polarity == "positive" else "F"
    expected_sign = 1 if match_polarity == "positive" else -1

    sources: list[Event] = []
    polarities: list[Polarity] = []
    distractor_iter = iter(zip(distractors, distractor_polarities))
    for idx in (1, 2, 3):
        if idx == match_idx:
            sources.append(claim_event)
            polarities.append(match_polarity)
        else:
            distractor_event, distractor_polarity = next(distractor_iter)
            sources.append(distractor_event)
            polarities.append(distractor_polarity)

    assumptions = [sentence(source, polarity) for source, polarity in zip(sources, polarities)]
    return make_sample(
        sample_id=f"exp3_{base_id}_{sample_suffix}",
        base_id=base_id,
        experiment="exp3_clean_selection",
        condition=f"match_idx_{match_idx}_{match_polarity}",
        assumptions=assumptions,
        sources=sources,
        source_polarities=polarities,
        claim_event=claim_event,
        claim_polarity="positive",
        expected_label=expected_label,
        expected_R_sign=expected_sign,
        primary_source_index=match_idx - 1,
        extra={
            "match_idx": match_idx,
            "match_polarity": match_polarity,
            "sanity_type": sanity_type,
            "phase_cos": 1 if match_polarity == "positive" else -1,
        },
    )


def generate_exp3_sanity(base_id: str, claim_event: Event, rng: random.Random) -> list[dict[str, Any]]:
    rows = []

    distractors = clean_distractors(claim_event, rng, count=2)
    rows.append(
        make_exp3_row(
            base_id=base_id,
            claim_event=claim_event,
            match_idx=1,
            match_polarity="positive",
            distractors=distractors,
            distractor_polarities=["negative", "negative"],
            sanity_type="majority_opposes_match",
            sample_suffix="sanity_majority_idx1_pos",
        )
    )

    distractors = clean_distractors(claim_event, rng, count=2)
    rows.append(
        make_exp3_row(
            base_id=base_id,
            claim_event=claim_event,
            match_idx=2,
            match_polarity="negative",
            distractors=distractors,
            distractor_polarities=["positive", "positive"],
            sanity_type="recency_not_last",
            sample_suffix="sanity_recency_idx2_neg",
        )
    )
    return rows


def generate_exp4(base_id: str, claim_event: Event, include_source_only: bool = True) -> list[dict[str, Any]]:
    rows = []
    patterns: list[tuple[str, list[Polarity], str, int, int]] = [
        ("balanced_+-", ["positive", "negative"], "U", 0, 0),
        ("positive_imbalance_++-", ["positive", "positive", "negative"], "T", 1, 1),
        ("negative_imbalance_+--", ["positive", "negative", "negative"], "F", -1, -1),
    ]

    for condition, polarities, expected_label, expected_sign, q_value in patterns:
        assumptions = [sentence(claim_event, polarity) for polarity in polarities]
        row = make_sample(
            sample_id=f"exp4_{base_id}_{pattern_slug(polarities)}",
            base_id=base_id,
            experiment="exp4_cancellation",
            condition=condition,
            assumptions=assumptions,
            sources=[claim_event for _ in polarities],
            source_polarities=polarities,
            claim_event=claim_event,
            claim_polarity="positive",
            expected_label=expected_label,
            expected_R_sign=expected_sign,
            extra=exp4_extra(polarities, q_value, source_only=False),
        )
        rows.append(row)

    if include_source_only:
        for polarity in ("positive", "negative"):
            expected_label = "T" if polarity == "positive" else "F"
            expected_sign = 1 if polarity == "positive" else -1
            row = make_sample(
                sample_id=f"exp4_{base_id}_source_only_{compact_polarity(polarity)}",
                base_id=base_id,
                experiment="exp4_cancellation",
                condition=f"source_only_{polarity}",
                assumptions=[sentence(claim_event, polarity)],
                sources=[claim_event],
                source_polarities=[polarity],
                claim_event=claim_event,
                claim_polarity="positive",
                expected_label=expected_label,
                expected_R_sign=expected_sign,
                extra=exp4_extra([polarity], 1 if polarity == "positive" else -1, source_only=True),
            )
            rows.append(row)

    return rows


def generate_exp5(base_id: str, positive_event: Event, rng: random.Random) -> list[dict[str, Any]]:
    negative_event = Event(
        positive_event.subject,
        positive_event.verb,
        choose_other(positive_event.verb.candidates, positive_event.obj, rng),
    )

    pos_first = (
        f"{positive_event.subject} {positive_event.verb.past} {positive_event.obj} "
        f"but did not {negative_event.verb.base} {negative_event.obj}."
    )
    neg_first = (
        f"{negative_event.subject} did not {negative_event.verb.base} {negative_event.obj} "
        f"but {positive_event.verb.past} {positive_event.obj}."
    )

    rows = []
    configs = [
        ("pos_then_neg", pos_first, [positive_event, negative_event], ["positive", "negative"]),
        ("neg_then_pos", neg_first, [negative_event, positive_event], ["negative", "positive"]),
    ]

    for order_pattern, assumption, sources, polarities in configs:
        rows.append(
            make_exp5_row(
                base_id=base_id,
                order_pattern=order_pattern,
                assumption=assumption,
                sources=sources,
                polarities=polarities,
                claim_event=positive_event,
                claim_object_role="positive_object",
                expected_label="T",
                expected_sign=1,
            )
        )
        rows.append(
            make_exp5_row(
                base_id=base_id,
                order_pattern=order_pattern,
                assumption=assumption,
                sources=sources,
                polarities=polarities,
                claim_event=negative_event,
                claim_object_role="negative_object",
                expected_label="F",
                expected_sign=-1,
            )
        )
    return rows


def make_exp5_row(
    base_id: str,
    order_pattern: str,
    assumption: str,
    sources: Sequence[Event],
    polarities: Sequence[Polarity],
    claim_event: Event,
    claim_object_role: str,
    expected_label: str,
    expected_sign: int,
) -> dict[str, Any]:
    primary_index = 0
    for idx, source in enumerate(sources):
        if source.key == claim_event.key:
            primary_index = idx
            break

    return make_sample(
        sample_id=f"exp5_{base_id}_{order_pattern}_{claim_object_role}",
        base_id=base_id,
        experiment="exp5_object_bound_phase",
        condition=f"{order_pattern}_{claim_object_role}",
        assumptions=[assumption],
        sources=sources,
        source_polarities=polarities,
        claim_event=claim_event,
        claim_polarity="positive",
        expected_label=expected_label,
        expected_R_sign=expected_sign,
        primary_source_index=primary_index,
        extra={
            "order_pattern": order_pattern,
            "claim_object_role": claim_object_role,
            "source_polarity": "mixed",
        },
    )


def make_sample(
    sample_id: str,
    base_id: str,
    experiment: str,
    condition: str,
    assumptions: Sequence[str],
    sources: Sequence[Event],
    source_polarities: Sequence[Polarity],
    claim_event: Event,
    claim_polarity: Polarity,
    expected_label: str,
    expected_R_sign: int,
    primary_source_index: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    claim = sentence(claim_event, claim_polarity)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "base_event_id": base_id,
        "experiment": experiment,
        "condition": condition,
        "assumption": format_assumptions(assumptions),
        "claim": claim,
        "prompt": build_prompt(assumptions, claim),
        "expected_label": expected_label,
        "expected_R_sign": expected_R_sign,
        "claim_polarity": claim_polarity,
        "n_assumptions": len(assumptions),
        "logit_T": None,
        "logit_F": None,
        "logit_U": None,
        "R": None,
        "U_gap": None,
        "pred_label": None,
        "is_correct": None,
    }
    row.update(event_metadata(claim_event, "claim"))

    if sources:
        primary = sources[primary_source_index]
        primary_polarity = source_polarities[primary_source_index]
        row.update(event_metadata(primary, "source"))
        row["source_polarity"] = primary_polarity
        row.update(overlap_metadata(primary, claim_event))

    for idx, (source, polarity) in enumerate(zip(sources, source_polarities), start=1):
        row.update(source_metadata(source, polarity, claim_event, idx))

    if extra:
        row.update(extra)
    return row


def source_metadata(source: Event, polarity: Polarity, claim_event: Event, idx: int) -> dict[str, Any]:
    overlap = overlap_metadata(source, claim_event)
    return {
        f"source{idx}_subject": source.subject,
        f"source{idx}_verb": source.verb.base,
        f"source{idx}_verb_base": source.verb.base,
        f"source{idx}_verb_past": source.verb.past,
        f"source{idx}_object": source.obj,
        f"source{idx}_arg_type": source.verb.arg_type,
        f"source{idx}_polarity": polarity,
        f"source{idx}_overlap_type": overlap["overlap_type"],
        f"source{idx}_overlap_count": overlap["overlap_count"],
    }


def overlap_metadata(source: Event, claim: Event) -> dict[str, Any]:
    same_subject = int(source.subject == claim.subject)
    same_verb = int(source.verb.base == claim.verb.base)
    same_object = int(source.obj == claim.obj)
    overlap_count = same_subject + same_verb + same_object

    if (same_subject, same_verb, same_object) == (1, 1, 1):
        overlap_type = "SVO"
    elif (same_subject, same_verb, same_object) == (1, 1, 0):
        overlap_type = "SV"
    elif (same_subject, same_verb, same_object) == (0, 1, 1):
        overlap_type = "VO"
    elif (same_subject, same_verb, same_object) == (1, 0, 0):
        overlap_type = "S-only"
    elif overlap_count == 0:
        overlap_type = "none"
    elif (same_subject, same_verb, same_object) == (1, 0, 1):
        overlap_type = "SO"
    elif (same_subject, same_verb, same_object) == (0, 1, 0):
        overlap_type = "V-only"
    elif (same_subject, same_verb, same_object) == (0, 0, 1):
        overlap_type = "O-only"
    else:
        overlap_type = "partial"

    return {
        "overlap_type": overlap_type,
        "overlap_count": overlap_count,
        "same_subject": same_subject,
        "same_verb": same_verb,
        "same_object": same_object,
    }


def exp4_extra(polarities: Sequence[Polarity], q_value: int, source_only: bool) -> dict[str, Any]:
    n_pos = sum(1 for polarity in polarities if polarity == "positive")
    n_neg = sum(1 for polarity in polarities if polarity == "negative")
    return {
        "pattern": "".join(polarity_symbol(polarity) for polarity in polarities),
        "q": q_value,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "source_only": int(source_only),
    }


def clean_distractors(claim_event: Event, rng: random.Random, count: int) -> list[Event]:
    candidates = [
        event
        for event in all_events()
        if event.subject != claim_event.subject
        and event.verb.base != claim_event.verb.base
        and event.obj != claim_event.obj
    ]
    rng.shuffle(candidates)
    unique: list[Event] = []
    seen: set[str] = set()
    for event in candidates:
        if event.key in seen:
            continue
        unique.append(event)
        seen.add(event.key)
        if len(unique) == count:
            return unique
    raise ValueError(f"Could not find {count} clean distractors for {claim_event.key}")


def choose_other(items: Sequence[str], current: str, rng: random.Random) -> str:
    choices = [item for item in items if item != current]
    if not choices:
        raise ValueError(f"No alternative found for {current}")
    return rng.choice(choices)


def choose_other_verb(current: VerbSpec, rng: random.Random, same_arg_type: bool) -> VerbSpec:
    choices = [verb for verb in VERBS if verb.base != current.base]
    if same_arg_type:
        choices = [verb for verb in choices if verb.arg_type == current.arg_type]
    if not choices:
        raise ValueError(f"No alternative verb found for {current.base}")
    return rng.choice(choices)



def slug(value: str) -> str:
    return value.lower().replace("+", "pos").replace("-", "_").replace(" ", "_")


def pattern_slug(polarities: Sequence[Polarity]) -> str:
    return "".join("p" if polarity == "positive" else "n" for polarity in polarities)
