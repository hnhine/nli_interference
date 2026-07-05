"""DAS pair generation for the atomic NLI causal model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from .base import PERSONS, VERBS, Event, Polarity, build_prompt, event_metadata, format_assumptions, polarity_symbol, sentence
from .das_spans import add_span_columns, build_prompt_with_spans
from .generation import clean_distractors, sample_base_events

DAS_TARGETS = ("pc", "pi", "m")
POLARITIES: tuple[Polarity, Polarity] = ("positive", "negative")
POLARITY_TO_SIGN = {"positive": 1, "negative": -1}
SIGN_TO_POLARITY: dict[int, Polarity] = {1: "positive", -1: "negative"}
MISMATCH_TYPES = ("object", "subject", "verb", "no_overlap")


@dataclass(frozen=True)
class DasExample:
    assumptions: tuple[str, ...]
    assumption_events: tuple[Event, ...]
    assumption_polarities: tuple[Polarity, ...]
    claim_event: Event
    claim_polarity: Polarity
    matched_idx: int
    m_i: int
    label: str

    @property
    def claim(self) -> str:
        return sentence(self.claim_event, self.claim_polarity)

    @property
    def p_i(self) -> int:
        return polarity_to_sign(self.assumption_polarities[self.matched_idx])

    @property
    def p_c(self) -> int:
        return polarity_to_sign(self.claim_polarity)


def high_level_label(m_i: int, p_i: int, p_c: int) -> str:
    if int(m_i) == 0:
        return "U"
    return "T" if int(p_i) == int(p_c) else "F"


def generate_das_pairs(
    n_base_events: int = 20,
    seed: int = 0,
    targets: Sequence[str] | None = None,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> list[dict[str, Any]]:
    selected = tuple(targets or DAS_TARGETS)
    unknown = sorted(set(selected) - set(DAS_TARGETS))
    if unknown:
        raise ValueError(f"Unknown DAS targets: {unknown}")

    rng = random.Random(seed)
    base_events = sample_base_events(n_base_events, rng)
    rows: list[dict[str, Any]] = []

    # Held-out claims must not be pre-exposed: train prompts may not contain
    # val/test base events, and val prompts may not contain test base events
    # (as distractors or mismatch events). The reverse direction is harmless,
    # and the event pool is too small for symmetric exclusion.
    splits = [split_for_index(idx, len(base_events), train_fraction, val_fraction) for idx in range(len(base_events))]
    keys_by_split: dict[str, set[str]] = {}
    for event, event_split in zip(base_events, splits):
        keys_by_split.setdefault(event_split, set()).add(event.key)
    split_order = {"train": 0, "val": 1, "test": 2}

    for base_index, claim_event in enumerate(base_events):
        base_id = f"base_{base_index:04d}"
        split = splits[base_index]
        excluded_keys = frozenset(
            key
            for later_split, keys in keys_by_split.items()
            if split_order[later_split] > split_order[split]
            for key in keys
        )
        for matched_idx in (0, 1, 2):
            if "pc" in selected:
                rows.extend(generate_pc_pairs(base_id, split, claim_event, matched_idx, rng, excluded_keys))
            if "pi" in selected:
                rows.extend(generate_pi_pairs(base_id, split, claim_event, matched_idx, rng, excluded_keys))
            if "m" in selected:
                rows.extend(generate_m_pairs(base_id, split, claim_event, matched_idx, rng, excluded_keys))

    for idx, row in enumerate(rows):
        row["row_id"] = idx
        row["run_family"] = "das_atomic"
    return rows


def generate_pc_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p_i in POLARITIES:
        for p_c_base in POLARITIES:
            p_c_src = flip_polarity(p_c_base)
            base = make_example(claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c_base, rng=rng, excluded_keys=excluded_keys)
            source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i,
                p_c=p_c_src,
                rng=rng,
                assumption_events=base.assumption_events,
                assumption_polarities=base.assumption_polarities,
            )
            target = high_level_label(base.m_i, base.p_i, source.p_c)
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pc", base_id, matched_idx, p_i, p_c_base, "main"),
                    base_id=base_id,
                    split=split,
                    target_var="pc",
                    control_type="main",
                    base=base,
                    source=source,
                    target_label=target,
                    base_site="claim_final",
                    source_site="claim_final",
                    extra={"p_c_src": source.p_c, "p_i_src": source.p_i, "m_src": source.m_i},
                )
            )

            no_match = make_example(claim_event, matched_idx, m_i=0, p_i=p_i, p_c=p_c_base, rng=rng, excluded_keys=excluded_keys)
            no_match_source = make_example(
                claim_event,
                matched_idx,
                m_i=0,
                p_i=p_i,
                p_c=p_c_src,
                rng=rng,
                assumption_events=no_match.assumption_events,
                assumption_polarities=no_match.assumption_polarities,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pc", base_id, matched_idx, p_i, p_c_base, "gate"),
                    base_id=base_id,
                    split=split,
                    target_var="pc",
                    control_type="gate_m0",
                    base=no_match,
                    source=no_match_source,
                    target_label="U",
                    base_site="claim_final",
                    source_site="claim_final",
                    extra={"p_c_src": no_match_source.p_c, "p_i_src": no_match_source.p_i, "m_src": no_match_source.m_i},
                )
            )

            label_copy_source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i,
                p_c=p_c_src,
                rng=rng,
                excluded_keys=excluded_keys,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pc", base_id, matched_idx, p_i, p_c_base, "label_copy_trap"),
                    base_id=base_id,
                    split=split,
                    target_var="pc",
                    control_type="label_copy_trap",
                    base=no_match,
                    source=label_copy_source,
                    target_label="U",
                    base_site="claim_final",
                    source_site="claim_final",
                    extra={"p_c_src": label_copy_source.p_c, "p_i_src": label_copy_source.p_i, "m_src": label_copy_source.m_i},
                )
            )
    return rows


def generate_pi_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p_i_base in POLARITIES:
        for p_c in POLARITIES:
            p_i_src = flip_polarity(p_i_base)
            base = make_example(claim_event, matched_idx, m_i=1, p_i=p_i_base, p_c=p_c, rng=rng, excluded_keys=excluded_keys)
            source_polarities = replace_tuple(base.assumption_polarities, matched_idx, p_i_src)
            source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_src,
                p_c=p_c,
                rng=rng,
                assumption_events=base.assumption_events,
                assumption_polarities=source_polarities,
            )
            target = high_level_label(base.m_i, source.p_i, base.p_c)
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pi", base_id, matched_idx, p_i_base, p_c, "main"),
                    base_id=base_id,
                    split=split,
                    target_var="pi",
                    control_type="main",
                    base=base,
                    source=source,
                    target_label=target,
                    base_site=f"a{matched_idx + 1}_final",
                    source_site=f"a{matched_idx + 1}_final",
                    extra={"p_i_src": source.p_i, "p_c_src": source.p_c, "m_src": source.m_i},
                )
            )

            distractor_idx = (matched_idx + 1) % 3
            distractor_src_polarity = flip_polarity(base.assumption_polarities[distractor_idx])
            distractor_polarities = replace_tuple(base.assumption_polarities, distractor_idx, distractor_src_polarity)
            distractor_source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_base,
                p_c=p_c,
                rng=rng,
                assumption_events=base.assumption_events,
                assumption_polarities=distractor_polarities,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pi", base_id, matched_idx, p_i_base, p_c, "distractor"),
                    base_id=base_id,
                    split=split,
                    target_var="pi",
                    control_type="distractor",
                    base=base,
                    source=distractor_source,
                    target_label=base.label,
                    base_site=f"a{distractor_idx + 1}_final",
                    source_site=f"a{distractor_idx + 1}_final",
                    extra={
                        "distractor_idx": distractor_idx,
                        "distractor_p_base": polarity_to_sign(base.assumption_polarities[distractor_idx]),
                        "distractor_p_src": polarity_to_sign(distractor_source.assumption_polarities[distractor_idx]),
                        "p_i_src": distractor_source.p_i,
                        "p_c_src": distractor_source.p_c,
                        "m_src": distractor_source.m_i,
                    },
                )
            )

            no_match_base = make_example(claim_event, matched_idx, m_i=0, p_i=p_i_base, p_c=p_c, rng=rng, excluded_keys=excluded_keys)
            label_copy_source = make_example(claim_event, matched_idx, m_i=1, p_i=p_i_src, p_c=p_c, rng=rng, excluded_keys=excluded_keys)
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pi", base_id, matched_idx, p_i_base, p_c, "label_copy_trap"),
                    base_id=base_id,
                    split=split,
                    target_var="pi",
                    control_type="label_copy_trap",
                    base=no_match_base,
                    source=label_copy_source,
                    target_label="U",
                    base_site=f"a{matched_idx + 1}_final",
                    source_site=f"a{matched_idx + 1}_final",
                    extra={"p_i_src": label_copy_source.p_i, "p_c_src": label_copy_source.p_c, "m_src": label_copy_source.m_i},
                )
            )
    return rows


def generate_m_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p_i in POLARITIES:
        for p_c in POLARITIES:
            for mismatch_type in MISMATCH_TYPES:
                matched_base = make_example(claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c, rng=rng, excluded_keys=excluded_keys)
                try:
                    mismatch_event = make_mismatch_event(claim_event, mismatch_type, rng, excluded_keys)
                    mismatch_exclusion_relaxed = 0
                except ValueError:
                    # The condition requires this mismatch type; keep it and flag
                    # the row so contaminated pairs can be audited or filtered.
                    mismatch_event = make_mismatch_event(claim_event, mismatch_type, rng)
                    mismatch_exclusion_relaxed = 1
                source_events = replace_tuple(matched_base.assumption_events, matched_idx, mismatch_event)
                no_match_source = make_example(
                    claim_event,
                    matched_idx,
                    m_i=0,
                    p_i=p_i,
                    p_c=p_c,
                    rng=rng,
                    assumption_events=source_events,
                    assumption_polarities=matched_base.assumption_polarities,
                )
                rows.append(
                    make_pair_row(
                        sample_id=sample_id("m", base_id, matched_idx, p_i, p_c, f"match_to_{mismatch_type}"),
                        base_id=base_id,
                        split=split,
                        target_var="m",
                        control_type="match_to_nomatch",
                        base=matched_base,
                        source=no_match_source,
                        target_label=high_level_label(no_match_source.m_i, matched_base.p_i, matched_base.p_c),
                        base_site="claim_final",
                        source_site="claim_final",
                        extra={
                            "mismatch_type": mismatch_type,
                            "mismatch_exclusion_relaxed": mismatch_exclusion_relaxed,
                            "m_src": no_match_source.m_i,
                            "p_i_src": no_match_source.p_i,
                            "p_c_src": no_match_source.p_c,
                        },
                    )
                )

                no_match_base = make_example(
                    claim_event,
                    matched_idx,
                    m_i=0,
                    p_i=p_i,
                    p_c=p_c,
                    rng=rng,
                    assumption_events=no_match_source.assumption_events,
                    assumption_polarities=no_match_source.assumption_polarities,
                )
                match_events = replace_tuple(no_match_base.assumption_events, matched_idx, claim_event)
                match_source = make_example(
                    claim_event,
                    matched_idx,
                    m_i=1,
                    p_i=p_i,
                    p_c=p_c,
                    rng=rng,
                    assumption_events=match_events,
                    assumption_polarities=no_match_base.assumption_polarities,
                )
                rows.append(
                    make_pair_row(
                        sample_id=sample_id("m", base_id, matched_idx, p_i, p_c, f"{mismatch_type}_to_match"),
                        base_id=base_id,
                        split=split,
                        target_var="m",
                        control_type="nomatch_to_match",
                        base=no_match_base,
                        source=match_source,
                        target_label=high_level_label(match_source.m_i, no_match_base.p_i, no_match_base.p_c),
                        base_site="claim_final",
                        source_site="claim_final",
                        extra={
                            "mismatch_type": mismatch_type,
                            "mismatch_exclusion_relaxed": mismatch_exclusion_relaxed,
                            "m_src": match_source.m_i,
                            "p_i_src": match_source.p_i,
                            "p_c_src": match_source.p_c,
                        },
                    )
                )

                trap_p_i = flip_polarity(p_i)
                trap_polarities = replace_tuple(no_match_base.assumption_polarities, matched_idx, trap_p_i)
                label_copy_source = make_example(
                    claim_event,
                    matched_idx,
                    m_i=1,
                    p_i=trap_p_i,
                    p_c=p_c,
                    rng=rng,
                    assumption_events=match_events,
                    assumption_polarities=trap_polarities,
                )
                rows.append(
                    make_pair_row(
                        sample_id=sample_id("m", base_id, matched_idx, p_i, p_c, f"label_copy_{mismatch_type}"),
                        base_id=base_id,
                        split=split,
                        target_var="m",
                        control_type="label_copy_trap",
                        base=no_match_base,
                        source=label_copy_source,
                        target_label=high_level_label(1, no_match_base.p_i, no_match_base.p_c),
                        base_site="claim_final",
                        source_site="claim_final",
                        extra={
                            "mismatch_type": mismatch_type,
                            "mismatch_exclusion_relaxed": mismatch_exclusion_relaxed,
                            "m_src": label_copy_source.m_i,
                            "p_i_src": label_copy_source.p_i,
                            "p_c_src": label_copy_source.p_c,
                        },
                    )
                )
    return rows


def make_example(
    claim_event: Event,
    matched_idx: int,
    m_i: int,
    p_i: Polarity,
    p_c: Polarity,
    rng: random.Random,
    assumption_events: Sequence[Event] | None = None,
    assumption_polarities: Sequence[Polarity] | None = None,
    excluded_keys: frozenset[str] = frozenset(),
) -> DasExample:
    if assumption_events is None:
        slot_event = claim_event if m_i == 1 else make_no_match_slot_event(claim_event, rng, excluded_keys)
        events = assumption_events_for_slot(claim_event, matched_idx, slot_event, rng, excluded_keys)
    else:
        events = tuple(assumption_events)

    if assumption_polarities is None:
        polarities = default_polarities(matched_idx, p_i)
    else:
        polarities = tuple(assumption_polarities)

    assumptions = tuple(sentence(event, polarity) for event, polarity in zip(events, polarities))
    label = high_level_label(m_i, polarity_to_sign(polarities[matched_idx]), polarity_to_sign(p_c))
    return DasExample(
        assumptions=assumptions,
        assumption_events=tuple(events),
        assumption_polarities=tuple(polarities),
        claim_event=claim_event,
        claim_polarity=p_c,
        matched_idx=matched_idx,
        m_i=int(m_i),
        label=label,
    )


def make_pair_row(
    sample_id: str,
    base_id: str,
    split: str,
    target_var: str,
    control_type: str,
    base: DasExample,
    source: DasExample,
    target_label: str,
    base_site: str,
    source_site: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_prompt = build_prompt_with_spans(base.assumptions, base.claim)
    source_prompt = build_prompt_with_spans(source.assumptions, source.claim)

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "base_event_id": base_id,
        "experiment": "das_atomic",
        "target_var": target_var,
        "control_type": control_type,
        "split": split,
        "matched_idx": base.matched_idx,
        "base_site": base_site,
        "source_site": source_site,
        "base_prompt": base_prompt.prompt,
        "source_prompt": source_prompt.prompt,
        "base_assumption": format_assumptions(base.assumptions),
        "source_assumption": format_assumptions(source.assumptions),
        "base_claim": base.claim,
        "source_claim": source.claim,
        "base_label": base.label,
        "source_label": source.label,
        "target_label": target_label,
        "expected_label": target_label,
        "m_base": base.m_i,
        "p_i_base": base.p_i,
        "p_c_base": base.p_c,
        "base_claim_polarity": base.claim_polarity,
        "source_claim_polarity": source.claim_polarity,
        "base_prompt_matches_standard": int(base_prompt.prompt == build_prompt(base.assumptions, base.claim)),
        "source_prompt_matches_standard": int(source_prompt.prompt == build_prompt(source.assumptions, source.claim)),
        "n_assumptions": len(base.assumptions),
    }
    row.update(event_metadata(base.claim_event, "claim"))
    for idx, (event, polarity) in enumerate(zip(base.assumption_events, base.assumption_polarities), start=1):
        row.update(event_metadata(event, f"base_source{idx}"))
        row[f"base_source{idx}_polarity"] = polarity
    for idx, (event, polarity) in enumerate(zip(source.assumption_events, source.assumption_polarities), start=1):
        row.update(event_metadata(event, f"source_source{idx}"))
        row[f"source_source{idx}_polarity"] = polarity
    add_span_columns(row, "base", base_prompt.spans)
    add_span_columns(row, "source", source_prompt.spans)
    if extra:
        row.update(extra)
    return row


def assumption_events_for_slot(
    claim_event: Event,
    matched_idx: int,
    slot_event: Event,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> tuple[Event, ...]:
    distractors = iter(clean_distractors(claim_event, rng, count=2, excluded_keys=excluded_keys))
    events: list[Event] = []
    for idx in range(3):
        events.append(slot_event if idx == matched_idx else next(distractors))
    return tuple(events)


def default_polarities(matched_idx: int, p_i: Polarity) -> tuple[Polarity, ...]:
    polarities: list[Polarity] = []
    for idx in range(3):
        if idx == matched_idx:
            polarities.append(p_i)
        else:
            polarities.append("positive")
    return tuple(polarities)


def make_no_match_slot_event(
    claim_event: Event,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> Event:
    """Pick a non-matching slot event, preferring near-miss mismatches.

    Any mismatch type is semantically valid for an m=0 slot; the fallback chain
    only matters when exclusions exhaust the tiny object-candidate pool.
    """

    for mismatch_type in ("object", "subject", "no_overlap"):
        try:
            return make_mismatch_event(claim_event, mismatch_type, rng, excluded_keys)
        except ValueError:
            continue
    raise ValueError(f"No no-match slot event available for {claim_event.key}")


def make_mismatch_event(
    claim_event: Event,
    mismatch_type: str,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> Event:
    if mismatch_type == "object":
        candidates = [
            Event(claim_event.subject, claim_event.verb, obj)
            for obj in claim_event.verb.candidates
            if obj != claim_event.obj
        ]
    elif mismatch_type == "subject":
        candidates = [
            Event(subject, claim_event.verb, claim_event.obj)
            for subject in PERSONS
            if subject != claim_event.subject
        ]
    elif mismatch_type == "verb":
        candidates = [
            Event(claim_event.subject, verb, claim_event.obj)
            for verb in VERBS
            if verb.base != claim_event.verb.base and verb.arg_type == claim_event.verb.arg_type
        ]
    elif mismatch_type == "no_overlap":
        return clean_distractors(claim_event, rng, count=1, excluded_keys=excluded_keys)[0]
    else:
        raise ValueError(f"Unknown mismatch type: {mismatch_type}")
    candidates = [event for event in candidates if event.key not in excluded_keys]
    if not candidates:
        raise ValueError(
            f"No {mismatch_type} mismatch candidate left for {claim_event.key} after excluding cross-split base events"
        )
    return rng.choice(candidates)


def replace_tuple(values: Sequence[Any], idx: int, value: Any) -> tuple[Any, ...]:
    out = list(values)
    out[idx] = value
    return tuple(out)


def flip_polarity(polarity: Polarity) -> Polarity:
    return "negative" if polarity == "positive" else "positive"


def polarity_to_sign(polarity: Polarity) -> int:
    return POLARITY_TO_SIGN[polarity]


def sample_id(target: str, base_id: str, matched_idx: int, p_i: Polarity, p_c: Polarity, suffix: str) -> str:
    return (
        f"das_{target}_{base_id}_idx{matched_idx + 1}_"
        f"ai{polarity_symbol(p_i)}_c{polarity_symbol(p_c)}_{suffix}"
    )


def split_for_index(index: int, total: int, train_fraction: float, val_fraction: float) -> str:
    if total <= 1:
        return "train"
    frac = index / total
    if frac < train_fraction:
        return "train"
    if frac < train_fraction + val_fraction:
        return "val"
    return "test"
