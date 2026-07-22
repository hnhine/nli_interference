"""DAS pair generation for the atomic NLI causal model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from .base import PERSONS, VERBS, Event, Polarity, VerbSpec, build_prompt, event_metadata, format_assumptions, polarity_symbol, sentence
from .das_spans import add_span_columns, build_prompt_with_spans
from .generation import clean_distractors, sample_base_events

DAS_TARGETS = ("pc", "pi", "m", "rho")
# Preserve the historical meaning of an omitted target list / ``--targets all``.
# Rho is opt-in so old generation commands remain exactly reproducible.
DEFAULT_DAS_TARGETS = ("pc", "pi", "m")
PC_VARIANTS = ("legacy", "v4")
PI_VARIANTS = ("legacy", "v4", "v5")
M_VARIANTS = ("legacy", "v4")
POLARITIES: tuple[Polarity, Polarity] = ("positive", "negative")
POLARITY_TO_SIGN = {"positive": 1, "negative": -1}
SIGN_TO_POLARITY: dict[int, Polarity] = {1: "positive", -1: "negative"}
MISMATCH_TYPES = ("object", "subject", "verb", "no_overlap")
M_VERB_POLICIES = ("legacy", "independent_v1")

# m-only vocabulary for a preregistered lexical-independence control. Verb
# mismatches are sampled across, never within, the paired semantic groups.
_M_INDEPENDENT_EXTRA_VERBS = (
    VerbSpec("mention", "mentioned", "location", ("PlaceA", "PlaceB", "PlaceC")),
    VerbSpec("recommend", "recommended", "location", ("PlaceA", "PlaceB", "PlaceC")),
    VerbSpec("describe", "described", "location", ("PlaceA", "PlaceB", "PlaceC")),
    VerbSpec("inspect", "inspected", "object", ("ObjectA", "ObjectB", "ObjectC")),
    VerbSpec("paint", "painted", "object", ("ObjectA", "ObjectB", "ObjectC")),
    VerbSpec("move", "moved", "object", ("ObjectA", "ObjectB", "ObjectC")),
)
M_INDEPENDENT_VERBS = VERBS + _M_INDEPENDENT_EXTRA_VERBS
_M_INDEPENDENT_GROUP_BY_VERB = {
    **{verb: "location_physical" for verb in ("visit", "reach", "explore", "enter")},
    **{verb: "location_discourse" for verb in ("like", "mention", "recommend", "describe")},
    **{verb: "object_original" for verb in ("open", "create", "clean", "collect", "close")},
    **{verb: "object_control" for verb in ("inspect", "paint", "move")},
}


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
    m_verb_policy: str = "legacy",
    pc_variant: str = "legacy",
    pi_variant: str = "legacy",
    m_variant: str = "legacy",
) -> list[dict[str, Any]]:
    selected = tuple(targets or DEFAULT_DAS_TARGETS)
    unknown = sorted(set(selected) - set(DAS_TARGETS))
    if unknown:
        raise ValueError(f"Unknown DAS targets: {unknown}")
    if m_verb_policy not in M_VERB_POLICIES:
        raise ValueError(f"Unknown m verb policy: {m_verb_policy!r}; choose from {M_VERB_POLICIES}")
    if m_verb_policy != "legacy" and set(selected) != {"m"}:
        raise ValueError("A non-legacy --m-verb-policy requires --targets m only")
    if pc_variant not in PC_VARIANTS:
        raise ValueError(f"Unknown pc variant: {pc_variant!r}; choose from {PC_VARIANTS}")
    if pc_variant != "legacy" and set(selected) != {"pc"}:
        raise ValueError("A non-legacy --pc-variant requires --targets pc only")
    if pi_variant not in PI_VARIANTS:
        raise ValueError(f"Unknown pi variant: {pi_variant!r}; choose from {PI_VARIANTS}")
    if pi_variant != "legacy" and set(selected) != {"pi"}:
        raise ValueError("A non-legacy --pi-variant requires --targets pi only")
    if m_variant not in M_VARIANTS:
        raise ValueError(f"Unknown m variant: {m_variant!r}; choose from {M_VARIANTS}")
    if m_variant != "legacy" and set(selected) != {"m"}:
        raise ValueError("A non-legacy --m-variant requires --targets m only")

    rng = random.Random(seed)
    base_verbs = M_INDEPENDENT_VERBS if m_verb_policy == "independent_v1" else VERBS
    base_events = sample_base_events(n_base_events, rng, verbs=base_verbs)
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
                pc_generator = generate_pc_v4_pairs if pc_variant == "v4" else generate_pc_pairs
                rows.extend(pc_generator(base_id, split, claim_event, matched_idx, rng, excluded_keys))
            if "pi" in selected:
                pi_generators = {
                    "legacy": generate_pi_pairs,
                    "v4": generate_pi_v4_pairs,
                    "v5": generate_pi_v5_pairs,
                }
                pi_generator = pi_generators[pi_variant]
                rows.extend(pi_generator(base_id, split, claim_event, matched_idx, rng, excluded_keys))
            if "m" in selected:
                m_generator = generate_m_v4_pairs if m_variant == "v4" else generate_m_pairs
                rows.extend(
                    m_generator(
                        base_id,
                        split,
                        claim_event,
                        matched_idx,
                        rng,
                        excluded_keys,
                        m_verb_policy=m_verb_policy,
                    )
                )
            if "rho" in selected:
                rows.extend(
                    generate_rho_pairs(
                        base_id,
                        split,
                        claim_event,
                        matched_idx,
                        rng,
                        excluded_keys,
                    )
                )

    for idx, row in enumerate(rows):
        row["row_id"] = idx
        if row.get("pc_variant") == "v4":
            row["run_family"] = "das_atomic_pc_v4"
        elif row.get("pi_variant") in {"v4", "v5"}:
            row["run_family"] = f"das_atomic_pi_{row['pi_variant']}"
        elif row.get("m_variant") == "v4":
            row["run_family"] = "das_atomic_m_v4"
        elif row.get("target_var") == "rho":
            row["run_family"] = "das_atomic_rho"
        else:
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



def generate_pc_v4_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Generate balanced raw-p_c identification and gating pairs.

    Per (matched_idx, p_i, p_c) cell, V4 creates one row for each of main,
    flip_both, flip_pi, gate_m0, and label_copy_trap. The five conditions are
    therefore equally represented; optimization-time weights can independently
    set the desired active/inactive sampling ratio.
    """
    rows: list[dict[str, Any]] = []

    def metadata(source: DasExample, group: str) -> dict[str, Any]:
        return {
            "p_c_src": source.p_c,
            "p_i_src": source.p_i,
            "m_src": source.m_i,
            "pc_variant": "v4",
            "pc_group": group,
        }

    for p_i in POLARITIES:
        for p_c_base in POLARITIES:
            p_c_src = flip_polarity(p_c_base)

            # Primary active transfer.
            for repeat in range(1):
                base = make_example(
                    claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c_base,
                    rng=rng, excluded_keys=excluded_keys,
                )
                source = make_example(
                    claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c_src,
                    rng=rng,
                    assumption_events=base.assumption_events,
                    assumption_polarities=base.assumption_polarities,
                )
                rows.append(
                    make_pair_row(
                        sample_id=sample_id("pcv4", base_id, matched_idx, p_i, p_c_base, f"main_r{repeat}"),
                        base_id=base_id,
                        split=split,
                        target_var="pc",
                        control_type="main",
                        base=base,
                        source=source,
                        target_label=high_level_label(base.m_i, base.p_i, source.p_c),
                        base_site="claim_final",
                        source_site="claim_final",
                        extra=metadata(source, "active") | {"pc_repeat": repeat},
                    )
                )

            # Identification: p_i and p_c both flip, while REL/source label stay fixed.
            both_base = make_example(
                claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c_base,
                rng=rng, excluded_keys=excluded_keys,
            )
            both_polarities = replace_tuple(
                both_base.assumption_polarities, matched_idx, flip_polarity(p_i)
            )
            both_source = make_example(
                claim_event, matched_idx, m_i=1, p_i=flip_polarity(p_i), p_c=p_c_src,
                rng=rng,
                assumption_events=both_base.assumption_events,
                assumption_polarities=both_polarities,
            )
            both_target = high_level_label(both_base.m_i, both_base.p_i, both_source.p_c)
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pcv4", base_id, matched_idx, p_i, p_c_base, "flip_both"),
                    base_id=base_id,
                    split=split,
                    target_var="pc",
                    control_type="probe_flip_both",
                    base=both_base,
                    source=both_source,
                    target_label=both_target,
                    base_site="claim_final",
                    source_site="claim_final",
                    extra=metadata(both_source, "active") | {
                        "pred_H_pc": both_target,
                        "pred_H_rel": both_base.label,
                        "pred_H_label": both_source.label,
                    },
                )
            )

            # Invariance: only p_i/REL/source label flip; p_c is unchanged.
            pi_base = make_example(
                claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c_base,
                rng=rng, excluded_keys=excluded_keys,
            )
            pi_polarities = replace_tuple(
                pi_base.assumption_polarities, matched_idx, flip_polarity(p_i)
            )
            pi_source = make_example(
                claim_event, matched_idx, m_i=1, p_i=flip_polarity(p_i), p_c=p_c_base,
                rng=rng,
                assumption_events=pi_base.assumption_events,
                assumption_polarities=pi_polarities,
            )
            pi_target = high_level_label(pi_base.m_i, pi_base.p_i, pi_source.p_c)
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pcv4", base_id, matched_idx, p_i, p_c_base, "flip_pi"),
                    base_id=base_id,
                    split=split,
                    target_var="pc",
                    control_type="probe_flip_pi",
                    base=pi_base,
                    source=pi_source,
                    target_label=pi_target,
                    base_site="claim_final",
                    source_site="claim_final",
                    extra=metadata(pi_source, "active") | {
                        "pred_H_pc": pi_target,
                        "pred_H_rel": pi_source.label,
                        "pred_H_label": pi_source.label,
                    },
                )
            )

            # Inactive gate: both base and source are unmatched.
            no_match = make_example(
                claim_event, matched_idx, m_i=0, p_i=p_i, p_c=p_c_base,
                rng=rng, excluded_keys=excluded_keys,
            )
            no_match_source = make_example(
                claim_event, matched_idx, m_i=0, p_i=p_i, p_c=p_c_src,
                rng=rng,
                assumption_events=no_match.assumption_events,
                assumption_polarities=no_match.assumption_polarities,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pcv4", base_id, matched_idx, p_i, p_c_base, "gate_m0"),
                    base_id=base_id,
                    split=split,
                    target_var="pc",
                    control_type="gate_m0",
                    base=no_match,
                    source=no_match_source,
                    target_label="U",
                    base_site="claim_final",
                    source_site="claim_final",
                    extra=metadata(no_match_source, "inactive"),
                )
            )

            # Anti-copy: source is matched/T-or-F, but the unmatched base must remain U.
            label_copy_source = make_example(
                claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c_src,
                rng=rng, excluded_keys=excluded_keys,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pcv4", base_id, matched_idx, p_i, p_c_base, "label_copy_trap"),
                    base_id=base_id,
                    split=split,
                    target_var="pc",
                    control_type="label_copy_trap",
                    base=no_match,
                    source=label_copy_source,
                    target_label="U",
                    base_site="claim_final",
                    source_site="claim_final",
                    extra=metadata(label_copy_source, "inactive"),
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
            no_match_polarities = replace_tuple(no_match_base.assumption_polarities, matched_idx, p_i_src)
            no_match_source = make_example(
                claim_event,
                matched_idx,
                m_i=0,
                p_i=p_i_src,
                p_c=p_c,
                rng=rng,
                assumption_events=no_match_base.assumption_events,
                assumption_polarities=no_match_polarities,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pi", base_id, matched_idx, p_i_base, p_c, "gate"),
                    base_id=base_id,
                    split=split,
                    target_var="pi",
                    control_type="gate_m0",
                    base=no_match_base,
                    source=no_match_source,
                    target_label="U",
                    base_site=f"a{matched_idx + 1}_final",
                    source_site=f"a{matched_idx + 1}_final",
                    extra={"p_i_src": no_match_source.p_i, "p_c_src": no_match_source.p_c, "m_src": no_match_source.m_i},
                )
            )

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


def generate_pi_v4_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Generate pi pairs covering both source-m values for either base gate.

    The intervention target follows only the base gate and source polarity:
    F(m_base, p_i_src, p_c_base). m_src is retained as an audit stratum, not
    as part of the target rule. The legacy generator remains untouched and
    is still the default.
    """

    rows = generate_pi_pairs(base_id, split, claim_event, matched_idx, rng, excluded_keys)

    # Complete the 2x2 (m_base, m_src) coverage. Legacy pi already contains
    # (1, 1) as main, (0, 0) as gate_m0, and (0, 1) as label_copy_trap.
    # This is the previously missing (1, 0) direction.
    for p_i_base in POLARITIES:
        for p_c in POLARITIES:
            p_i_src = flip_polarity(p_i_base)
            active_base = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_base,
                p_c=p_c,
                rng=rng,
                excluded_keys=excluded_keys,
            )
            inactive_source = make_example(
                claim_event,
                matched_idx,
                m_i=0,
                p_i=p_i_src,
                p_c=p_c,
                rng=rng,
                excluded_keys=excluded_keys,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pi_v4", base_id, matched_idx, p_i_base, p_c, "active_source_m0"),
                    base_id=base_id,
                    split=split,
                    target_var="pi",
                    control_type="active_source_m0",
                    base=active_base,
                    source=inactive_source,
                    target_label=high_level_label(active_base.m_i, inactive_source.p_i, active_base.p_c),
                    base_site=f"a{matched_idx + 1}_final",
                    source_site=f"a{matched_idx + 1}_final",
                    extra={
                        "p_i_src": inactive_source.p_i,
                        "p_c_src": inactive_source.p_c,
                        "m_src": inactive_source.m_i,
                    },
                )
            )

    regime_by_control = {
        "main": "active",
        "active_source_m0": "active",
        "gate_m0": "inactive",
        "label_copy_trap": "inactive",
        "distractor": "locality",
    }
    for row in rows:
        row["pi_variant"] = "v4"
        row["pi_regime"] = regime_by_control[str(row["control_type"])]
    return rows



def generate_pi_v5_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Add raw-p_i identification probes to the complete pi-v4 design."""

    rows = generate_pi_v4_pairs(base_id, split, claim_event, matched_idx, rng, excluded_keys)
    for row in rows:
        row["pi_variant"] = "v5"

    for p_i_base in POLARITIES:
        for p_c_base in POLARITIES:
            p_i_src = flip_polarity(p_i_base)
            p_c_src = flip_polarity(p_c_base)
            base = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_base,
                p_c=p_c_base,
                rng=rng,
                excluded_keys=excluded_keys,
            )

            flipped_polarities = replace_tuple(base.assumption_polarities, matched_idx, p_i_src)
            flip_both_source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_src,
                p_c=p_c_src,
                rng=rng,
                assumption_events=base.assumption_events,
                assumption_polarities=flipped_polarities,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pi_v5", base_id, matched_idx, p_i_base, p_c_base, "probe_flip_both"),
                    base_id=base_id,
                    split=split,
                    target_var="pi",
                    control_type="probe_flip_both",
                    base=base,
                    source=flip_both_source,
                    target_label=high_level_label(base.m_i, flip_both_source.p_i, base.p_c),
                    base_site=f"a{matched_idx + 1}_final",
                    source_site=f"a{matched_idx + 1}_final",
                    extra={
                        "p_i_src": flip_both_source.p_i,
                        "p_c_src": flip_both_source.p_c,
                        "m_src": flip_both_source.m_i,
                        "pi_identity_probe": "flip_both",
                    },
                )
            )

            flip_pc_source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_base,
                p_c=p_c_src,
                rng=rng,
                assumption_events=base.assumption_events,
                assumption_polarities=base.assumption_polarities,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("pi_v5", base_id, matched_idx, p_i_base, p_c_base, "probe_flip_pc"),
                    base_id=base_id,
                    split=split,
                    target_var="pi",
                    control_type="probe_flip_pc",
                    base=base,
                    source=flip_pc_source,
                    target_label=high_level_label(base.m_i, flip_pc_source.p_i, base.p_c),
                    base_site=f"a{matched_idx + 1}_final",
                    source_site=f"a{matched_idx + 1}_final",
                    extra={
                        "p_i_src": flip_pc_source.p_i,
                        "p_c_src": flip_pc_source.p_c,
                        "m_src": flip_pc_source.m_i,
                        "pi_identity_probe": "flip_pc",
                    },
                )
            )

    regime_by_control = {
        "main": "active",
        "active_source_m0": "active",
        "probe_flip_both": "active",
        "probe_flip_pc": "active",
        "gate_m0": "inactive",
        "label_copy_trap": "inactive",
        "distractor": "locality",
    }
    for row in rows:
        row["pi_variant"] = "v5"
        row["pi_regime"] = regime_by_control[str(row["control_type"])]
        row["pred_H_pi"] = high_level_label(row["m_base"], row["p_i_src"], row["p_c_base"])
        row["pred_H_pc"] = high_level_label(row["m_base"], row["p_i_base"], row["p_c_src"])
        source_rel = int(row["p_i_src"]) * int(row["p_c_src"])
        row["pred_H_rel"] = "U" if int(row["m_base"]) == 0 else ("T" if source_rel == 1 else "F")
        row["pred_H_label"] = row["source_label"]
    return rows


def generate_rho_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Generate identified pre-gate rho = p_i * p_c intervention pairs.

    Rho is defined for the designated assumption slot even when that event does
    not match the claim. The base gate alone determines whether transferred rho
    controls T/F (m_base=1) or is suppressed to U (m_base=0).

    Default DAS training uses all six controls; source_m0 complements
    label_copy_trap by distinguishing pre-gate rho from source label/gated REL.
    """

    rows: list[dict[str, Any]] = []

    def add_pair(
        *,
        control_type: str,
        base: DasExample,
        source: DasExample,
        target_label: str,
    ) -> None:
        rho_base = int(base.p_i) * int(base.p_c)
        rho_src = int(source.p_i) * int(source.p_c)
        rows.append(
            make_pair_row(
                sample_id=sample_id(
                    "rho",
                    base_id,
                    matched_idx,
                    base.assumption_polarities[matched_idx],
                    base.claim_polarity,
                    control_type,
                ),
                base_id=base_id,
                split=split,
                target_var="rho",
                control_type=control_type,
                base=base,
                source=source,
                target_label=target_label,
                base_site="claim_final",
                source_site="claim_final",
                extra={
                    "m_src": source.m_i,
                    "p_i_src": source.p_i,
                    "p_c_src": source.p_c,
                    "rho_base": rho_base,
                    "rho_src": rho_src,
                    "rho_regime": "active" if base.m_i == 1 else "inactive",
                    "rho_direction": f"{rho_base:+d}_to_{rho_src:+d}",
                    "rho_identity": int(rho_base == rho_src),
                    "rho_default_train": 1,
                    "pred_H_rho": target_label,
                    "pred_H_pi": high_level_label(base.m_i, source.p_i, base.p_c),
                    "pred_H_pc": high_level_label(base.m_i, base.p_i, source.p_c),
                    "pred_H_gated_rel": source.label,
                    "pred_H_label": source.label,
                },
            )
        )

    for p_i_base in POLARITIES:
        for p_c_base in POLARITIES:
            p_i_src = flip_polarity(p_i_base)
            p_c_src = flip_polarity(p_c_base)

            active_base = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_base,
                p_c=p_c_base,
                rng=rng,
                excluded_keys=excluded_keys,
            )

            flip_pi_polarities = replace_tuple(
                active_base.assumption_polarities,
                matched_idx,
                p_i_src,
            )
            flip_pi_source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_src,
                p_c=p_c_base,
                rng=rng,
                assumption_events=active_base.assumption_events,
                assumption_polarities=flip_pi_polarities,
            )
            add_pair(
                control_type="flip_pi",
                base=active_base,
                source=flip_pi_source,
                target_label=high_level_label(
                    active_base.m_i,
                    flip_pi_source.p_i,
                    flip_pi_source.p_c,
                ),
            )

            flip_pc_source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_base,
                p_c=p_c_src,
                rng=rng,
                assumption_events=active_base.assumption_events,
                assumption_polarities=active_base.assumption_polarities,
            )
            add_pair(
                control_type="flip_pc",
                base=active_base,
                source=flip_pc_source,
                target_label=high_level_label(
                    active_base.m_i,
                    flip_pc_source.p_i,
                    flip_pc_source.p_c,
                ),
            )

            hold_both_source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_src,
                p_c=p_c_src,
                rng=rng,
                assumption_events=active_base.assumption_events,
                assumption_polarities=flip_pi_polarities,
            )
            add_pair(
                control_type="hold_both",
                base=active_base,
                source=hold_both_source,
                target_label=high_level_label(
                    active_base.m_i,
                    hold_both_source.p_i,
                    hold_both_source.p_c,
                ),
            )

            inactive_source = make_example(
                claim_event,
                matched_idx,
                m_i=0,
                p_i=p_i_src,
                p_c=p_c_base,
                rng=rng,
                excluded_keys=excluded_keys,
            )
            add_pair(
                control_type="source_m0",
                base=active_base,
                source=inactive_source,
                target_label=high_level_label(
                    active_base.m_i,
                    inactive_source.p_i,
                    inactive_source.p_c,
                ),
            )

            inactive_base = make_example(
                claim_event,
                matched_idx,
                m_i=0,
                p_i=p_i_base,
                p_c=p_c_base,
                rng=rng,
                excluded_keys=excluded_keys,
            )
            gate_polarities = replace_tuple(
                inactive_base.assumption_polarities,
                matched_idx,
                p_i_src,
            )
            gate_source = make_example(
                claim_event,
                matched_idx,
                m_i=0,
                p_i=p_i_src,
                p_c=p_c_base,
                rng=rng,
                assumption_events=inactive_base.assumption_events,
                assumption_polarities=gate_polarities,
            )
            add_pair(
                control_type="gate_m0",
                base=inactive_base,
                source=gate_source,
                target_label="U",
            )

            label_copy_source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i_src,
                p_c=p_c_base,
                rng=rng,
                excluded_keys=excluded_keys,
            )
            add_pair(
                control_type="label_copy_trap",
                base=inactive_base,
                source=label_copy_source,
                target_label="U",
            )

    return rows


def generate_m_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
    m_verb_policy: str = "legacy",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p_i in POLARITIES:
        for p_c in POLARITIES:
            for mismatch_type in MISMATCH_TYPES:
                matched_base = make_example(claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c, rng=rng, excluded_keys=excluded_keys)
                try:
                    mismatch_event = make_mismatch_event(
                        claim_event, mismatch_type, rng, excluded_keys, verb_policy=m_verb_policy
                    )
                    mismatch_exclusion_relaxed = 0
                except ValueError:
                    # The condition requires this mismatch type; keep it and flag
                    # the row so contaminated pairs can be audited or filtered.
                    mismatch_event = make_mismatch_event(claim_event, mismatch_type, rng, verb_policy=m_verb_policy)
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
                            "m_verb_policy": m_verb_policy,
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
                            "m_verb_policy": m_verb_policy,
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
                            "m_verb_policy": m_verb_policy,
                            "mismatch_type": mismatch_type,
                            "mismatch_exclusion_relaxed": mismatch_exclusion_relaxed,
                            "m_src": label_copy_source.m_i,
                            "p_i_src": label_copy_source.p_i,
                            "p_c_src": label_copy_source.p_c,
                        },
                    )
                )
    return rows


def generate_m_v4_pairs(
    base_id: str,
    split: str,
    claim_event: Event,
    matched_idx: int,
    rng: random.Random,
    excluded_keys: frozenset[str] = frozenset(),
    m_verb_policy: str = "legacy",
) -> list[dict[str, Any]]:
    """Add an m=1 -> m=1 anti-label-copy control to legacy m pairs."""

    rows = generate_m_pairs(
        base_id,
        split,
        claim_event,
        matched_idx,
        rng,
        excluded_keys,
        m_verb_policy=m_verb_policy,
    )
    for row in rows:
        row["m_variant"] = "v4"
        if row["control_type"] == "label_copy_trap":
            row["m_label_copy_trap_type"] = "activate_match"

    for p_i in POLARITIES:
        for p_c in POLARITIES:
            base = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=p_i,
                p_c=p_c,
                rng=rng,
                excluded_keys=excluded_keys,
            )
            source_p_i = flip_polarity(p_i)
            source_polarities = replace_tuple(base.assumption_polarities, matched_idx, source_p_i)
            source = make_example(
                claim_event,
                matched_idx,
                m_i=1,
                p_i=source_p_i,
                p_c=p_c,
                rng=rng,
                assumption_events=base.assumption_events,
                assumption_polarities=source_polarities,
            )
            rows.append(
                make_pair_row(
                    sample_id=sample_id("m_v4", base_id, matched_idx, p_i, p_c, "label_copy_same_m1"),
                    base_id=base_id,
                    split=split,
                    target_var="m",
                    control_type="label_copy_trap_same_m1",
                    base=base,
                    source=source,
                    target_label=base.label,
                    base_site="claim_final",
                    source_site="claim_final",
                    extra={
                        "m_variant": "v4",
                        "m_verb_policy": m_verb_policy,
                        "m_label_copy_trap_type": "same_m1",
                        "m_src": source.m_i,
                        "p_i_src": source.p_i,
                        "p_c_src": source.p_c,
                    },
                )
            )
    if split in {"val", "test"}:
        rows = balance_m_v4_eval_controls(rows, base_id=base_id, matched_idx=matched_idx)
    return rows


def balance_m_v4_eval_controls(
    rows: list[dict[str, Any]],
    *,
    base_id: str,
    matched_idx: int,
) -> list[dict[str, Any]]:
    """Keep one row per polarity cell and control for balanced v4 eval."""

    base_index = int(base_id.rsplit("_", 1)[-1])
    balanced: list[dict[str, Any]] = []
    for row in rows:
        if row["control_type"] == "label_copy_trap_same_m1":
            row["m_eval_balanced"] = 1
            balanced.append(row)
            continue
        p_i_offset = 0 if int(row["p_i_base"]) == 1 else 2
        p_c_offset = 0 if int(row["p_c_base"]) == 1 else 1
        mismatch_index = (base_index + matched_idx + p_i_offset + p_c_offset) % len(MISMATCH_TYPES)
        if row.get("mismatch_type") == MISMATCH_TYPES[mismatch_index]:
            row["m_eval_balanced"] = 1
            balanced.append(row)
    return balanced


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
    verb_policy: str = "legacy",
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
        if verb_policy == "legacy":
            verb_pool = VERBS
        elif verb_policy == "independent_v1":
            verb_pool = allowed_independent_mismatch_verbs(claim_event.verb)
        else:
            raise ValueError(f"Unknown m verb policy: {verb_policy!r}; choose from {M_VERB_POLICIES}")
        candidates = [
            Event(claim_event.subject, verb, claim_event.obj)
            for verb in verb_pool
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


def allowed_independent_mismatch_verbs(claim_verb: VerbSpec) -> tuple[VerbSpec, ...]:
    """Return only verbs in the preregistered opposite semantic group."""

    claim_group = _M_INDEPENDENT_GROUP_BY_VERB.get(claim_verb.base)
    if claim_group is None:
        raise ValueError(f"Verb {claim_verb.base!r} is not covered by independent_v1")
    if claim_group == "location_physical":
        target_group = "location_discourse"
    elif claim_group == "location_discourse":
        target_group = "location_physical"
    elif claim_group == "object_original":
        target_group = "object_control"
    else:
        target_group = "object_original"
    return tuple(
        verb
        for verb in M_INDEPENDENT_VERBS
        if verb.arg_type == claim_verb.arg_type and _M_INDEPENDENT_GROUP_BY_VERB[verb.base] == target_group
    )


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
