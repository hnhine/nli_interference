"""Evaluation-only triples for joint causal control by ``m`` and ``rho``.

Each row contains one base prompt, an m donor, a rho donor, and two purity
donors whose target causal values equal the base values.  No rotations are
trained on these rows; they are intended only for composition evaluation with
previously learned m and rho subspaces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from .base import VERBS, Event, build_prompt, event_metadata, format_assumptions
from .das_data import DasExample, high_level_label, make_example
from .das_spans import add_span_columns, build_prompt_with_spans
from .generation import sample_base_events
from .io_utils import read_rows_csv


@dataclass(frozen=True)
class JointGateCell:
    name: str
    m_base: int
    rho_base: int
    m_donor: int
    rho_donor: int
    family: str


# The two open cells are the strict synergy tests: rho_donor is forced to be
# the opposite of rho_base, so neither single patch can produce the joint
# target.  Close and hold-open cells test gate dominance and rho replacement.
JOINT_GATE_CELLS = (
    JointGateCell("open_to_F", 0, +1, 1, -1, "open_synergy"),
    JointGateCell("open_to_T", 0, -1, 1, +1, "open_synergy"),
    JointGateCell("close_from_T", 1, +1, 0, -1, "close_gate"),
    JointGateCell("close_from_F", 1, -1, 0, +1, "close_gate"),
    JointGateCell("rho_flip_T_to_F", 1, +1, 1, -1, "rho_replace"),
    JointGateCell("rho_flip_F_to_T", 1, -1, 1, +1, "rho_replace"),
)


def sign_to_polarity(sign: int) -> str:
    if int(sign) == 1:
        return "positive"
    if int(sign) == -1:
        return "negative"
    raise ValueError(f"Expected polarity sign +/-1, got {sign!r}")


def label_from_m_rho(m_value: int, rho_value: int) -> str:
    # Fix p_c=+1 without loss of generality; high_level_label only depends on
    # the product p_i*p_c once m is specified.
    return high_level_label(int(m_value), int(rho_value), 1)


def make_state(
    claim_event: Event,
    matched_idx: int,
    *,
    m_value: int,
    rho_value: int,
    p_c_sign: int,
    rng: random.Random,
) -> DasExample:
    p_i_sign = int(rho_value) * int(p_c_sign)
    return make_example(
        claim_event,
        matched_idx,
        int(m_value),
        sign_to_polarity(p_i_sign),
        sign_to_polarity(p_c_sign),
        rng,
    )


def add_example_fields(row: dict[str, Any], prefix: str, example: DasExample) -> None:
    prompt = build_prompt_with_spans(example.assumptions, example.claim)
    row[f"{prefix}_prompt"] = prompt.prompt
    row[f"{prefix}_assumption"] = format_assumptions(example.assumptions)
    row[f"{prefix}_claim"] = example.claim
    row[f"{prefix}_label"] = example.label
    row[f"{prefix}_m"] = example.m_i
    row[f"{prefix}_p_i"] = example.p_i
    row[f"{prefix}_p_c"] = example.p_c
    row[f"{prefix}_rho"] = example.p_i * example.p_c
    row[f"{prefix}_site"] = "claim_final"
    row[f"{prefix}_prompt_matches_standard"] = int(
        prompt.prompt == build_prompt(example.assumptions, example.claim)
    )
    add_span_columns(row, prefix, prompt.spans)


def load_reference_claim_events(
    samples_path: str,
    *,
    split: str = "test",
) -> list[Event]:
    """Load unique claim events from an existing held-out DAS split."""

    verb_by_base = {verb.base: verb for verb in VERBS}
    events: dict[str, Event] = {}
    for row in read_rows_csv(samples_path):
        if split and row.get("split") != split:
            continue
        verb_name = str(row.get("claim_verb_base", ""))
        if verb_name not in verb_by_base:
            continue
        event = Event(
            str(row["claim_subject"]),
            verb_by_base[verb_name],
            str(row["claim_object"]),
        )
        events[event.key] = event
    if not events:
        raise ValueError(f"No claim events found in {samples_path!r} for split={split!r}")
    return [events[key] for key in sorted(events)]


def generate_joint_gate_rows(
    *,
    n_base_events: int = 40,
    seed: int = 1729,
    reference_samples: str | None = None,
    reference_split: str = "test",
    rho_source_regimes: Sequence[int] = (0, 1),
) -> list[dict[str, Any]]:
    """Generate balanced, evaluation-only joint-gate rows.

    With 40 events, three designated premise positions, two claim polarities,
    and both rho-source regimes, each transition cell contains 480 rows (240
    per rho-source regime), comfortably above the requested 200 rows/cell.
    """

    rng = random.Random(seed)
    if reference_samples:
        candidates = load_reference_claim_events(reference_samples, split=reference_split)
        rng.shuffle(candidates)
        if n_base_events > len(candidates):
            raise ValueError(
                f"Requested {n_base_events} events, but reference split only has {len(candidates)}"
            )
        claim_events = candidates[:n_base_events]
    else:
        claim_events = sample_base_events(n_base_events, rng)

    regimes = tuple(int(value) for value in rho_source_regimes)
    if not regimes or any(value not in (0, 1) for value in regimes):
        raise ValueError("rho_source_regimes must contain only 0 and/or 1")

    rows: list[dict[str, Any]] = []
    for event_index, claim_event in enumerate(claim_events):
        base_event_id = f"joint_base_{event_index:04d}"
        for matched_idx in (0, 1, 2):
            for p_c_sign in (-1, +1):
                for cell in JOINT_GATE_CELLS:
                    if cell.family == "open_synergy" and cell.rho_donor != -cell.rho_base:
                        raise AssertionError(f"Broken synergy specification: {cell}")
                    for rho_source_m in regimes:
                        base = make_state(
                            claim_event,
                            matched_idx,
                            m_value=cell.m_base,
                            rho_value=cell.rho_base,
                            p_c_sign=p_c_sign,
                            rng=rng,
                        )
                        # The m donor changes only the intended high-level m
                        # value; its rho is kept at the base value so an m-only
                        # patch has a fully specified counterfactual prediction.
                        m_source = make_state(
                            claim_event,
                            matched_idx,
                            m_value=cell.m_donor,
                            rho_value=cell.rho_base,
                            p_c_sign=p_c_sign,
                            rng=rng,
                        )
                        rho_source = make_state(
                            claim_event,
                            matched_idx,
                            m_value=rho_source_m,
                            rho_value=cell.rho_donor,
                            p_c_sign=p_c_sign,
                            rng=rng,
                        )

                        # Purity donors keep the patched causal value fixed but
                        # deliberately change the other variable and context.
                        m_same_source = make_state(
                            claim_event,
                            matched_idx,
                            m_value=cell.m_base,
                            rho_value=-cell.rho_base,
                            p_c_sign=p_c_sign,
                            rng=rng,
                        )
                        rho_same_source = make_state(
                            claim_event,
                            matched_idx,
                            m_value=1 - cell.m_base,
                            rho_value=cell.rho_base,
                            p_c_sign=p_c_sign,
                            rng=rng,
                        )

                        expected_none = label_from_m_rho(cell.m_base, cell.rho_base)
                        expected_m_only = label_from_m_rho(cell.m_donor, cell.rho_base)
                        expected_rho_only = label_from_m_rho(cell.m_base, cell.rho_donor)
                        expected_joint = label_from_m_rho(cell.m_donor, cell.rho_donor)
                        sample_id = (
                            f"joint_gate_{base_event_id}_idx{matched_idx + 1}_"
                            f"pc{p_c_sign:+d}_{cell.name}_rhoSrcM{rho_source_m}"
                        )
                        row: dict[str, Any] = {
                            "row_id": len(rows),
                            "sample_id": sample_id,
                            "base_event_id": base_event_id,
                            "experiment": "das_joint_gate",
                            "split": "test",
                            "target_var": "joint_gate",
                            "cell_type": cell.name,
                            "cell_family": cell.family,
                            "matched_idx": matched_idx,
                            "m_base": cell.m_base,
                            "rho_base": cell.rho_base,
                            "m_donor": cell.m_donor,
                            "rho_donor": cell.rho_donor,
                            "rho_source_m": rho_source_m,
                            "p_c_sign": p_c_sign,
                            "expected_none": expected_none,
                            "expected_m_only": expected_m_only,
                            "expected_rho_only": expected_rho_only,
                            "expected_joint": expected_joint,
                            "expected_same_value": expected_none,
                            "rho_opposite_constraint": int(cell.rho_donor == -cell.rho_base),
                            "is_synergy_cell": int(cell.family == "open_synergy"),
                        }
                        add_example_fields(row, "base", base)
                        add_example_fields(row, "m_source", m_source)
                        add_example_fields(row, "rho_source", rho_source)
                        add_example_fields(row, "m_same_source", m_same_source)
                        add_example_fields(row, "rho_same_source", rho_same_source)
                        row["strict_assembly"] = int(
                            expected_joint
                            not in {
                                base.label,
                                m_source.label,
                                rho_source.label,
                            }
                        )
                        row.update(event_metadata(claim_event, "claim"))
                        rows.append(row)

    validate_joint_gate_rows(rows)
    return rows


def validate_joint_gate_rows(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Joint-gate dataset is empty")
    expected_cells = {cell.name for cell in JOINT_GATE_CELLS}
    observed_cells = {str(row["cell_type"]) for row in rows}
    if observed_cells != expected_cells:
        raise ValueError(f"Cell mismatch: expected={expected_cells}, observed={observed_cells}")

    for row in rows:
        m_base = int(row["m_base"])
        rho_base = int(row["rho_base"])
        m_donor = int(row["m_donor"])
        rho_donor = int(row["rho_donor"])
        expected = {
            "expected_none": label_from_m_rho(m_base, rho_base),
            "expected_m_only": label_from_m_rho(m_donor, rho_base),
            "expected_rho_only": label_from_m_rho(m_base, rho_donor),
            "expected_joint": label_from_m_rho(m_donor, rho_donor),
        }
        for key, value in expected.items():
            if row[key] != value:
                raise ValueError(f"{row['sample_id']}: {key}={row[key]!r}, expected {value!r}")
        if int(row["is_synergy_cell"]):
            if rho_donor != -rho_base:
                raise ValueError(f"{row['sample_id']}: synergy row violates rho_donor=-rho_base")
            if row["expected_m_only"] == row["expected_joint"]:
                raise ValueError(f"{row['sample_id']}: m-only already equals joint target")
            if row["expected_rho_only"] == row["expected_joint"]:
                raise ValueError(f"{row['sample_id']}: rho-only already equals joint target")

