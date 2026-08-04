"""Dataset derivation for the Section 6.1 lexical-negation experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .base import Event, VerbSpec, build_prompt, format_assumptions
from .das_spans import add_span_columns, build_prompt_with_spans
from .negation_forms import (
    CANONICAL_FORM,
    TIER1_FORM_KEYS,
    render_double_negation_family,
    render_polarity,
)

PairRenderer = Callable[[Event, int], str]
POLARITY_CELLS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
RHO_CONTROLS = (
    "flip_pi",
    "flip_pc",
    "hold_both",
    "source_m0",
    "gate_m0",
    "label_copy_trap",
)


def event_from_columns(row: dict[str, Any], prefix: str) -> Event:
    return Event(
        subject=str(row[f"{prefix}_subject"]),
        verb=VerbSpec(
            base=str(row[f"{prefix}_verb_base"]),
            past=str(row[f"{prefix}_verb_past"]),
            arg_type=str(row.get(f"{prefix}_arg_type", "generic")),
            candidates=(),
        ),
        obj=str(row[f"{prefix}_object"]),
    )


def sign(value: Any) -> int:
    raw = str(value).strip().lower()
    if raw in {"1", "+1", "positive"}:
        return 1
    if raw in {"-1", "negative"}:
        return -1
    raise ValueError(f"Cannot parse polarity sign from {value!r}")


def tier1_renderer(form: str) -> PairRenderer:
    return lambda event, polarity: render_polarity(event, polarity, form)


def render_side(
    row: dict[str, Any],
    side: str,
    premise_renderer: PairRenderer,
    claim_renderer: PairRenderer,
) -> tuple[list[str], str]:
    if side not in {"base", "source"}:
        raise ValueError(f"Unknown DAS side: {side!r}")
    matched_idx = int(row["matched_idx"])
    assumptions: list[str] = []
    for slot in range(int(row.get("n_assumptions", 3))):
        prefix = f"{side}_source{slot + 1}"
        event = event_from_columns(row, prefix)
        polarity = sign(row[f"{prefix}_polarity"])
        renderer = premise_renderer if slot == matched_idx else tier1_renderer(CANONICAL_FORM)
        assumptions.append(renderer(event, polarity))

    claim_event = event_from_columns(row, "claim")
    claim_polarity = sign(
        row["base_claim_polarity"] if side == "base" else row["source_claim_polarity"]
    )
    return assumptions, claim_renderer(claim_event, claim_polarity)


def derive_pair_row(
    row: dict[str, Any],
    *,
    experiment: str,
    base_premise_renderer: PairRenderer,
    base_claim_renderer: PairRenderer,
    source_premise_renderer: PairRenderer,
    source_claim_renderer: PairRenderer,
    base_form: str,
    source_form: str,
    form: str,
    direction: str = "",
) -> dict[str, Any]:
    out = dict(row)
    base_assumptions, base_claim = render_side(
        row, "base", base_premise_renderer, base_claim_renderer
    )
    source_assumptions, source_claim = render_side(
        row, "source", source_premise_renderer, source_claim_renderer
    )
    base_prompt = build_prompt_with_spans(base_assumptions, base_claim)
    source_prompt = build_prompt_with_spans(source_assumptions, source_claim)

    out.update(
        {
            "sample_id": f"{row['sample_id']}__{experiment.lower()}__{form}__{direction or 'main'}",
            "section61_experiment": experiment,
            "form": form,
            "base_form": base_form,
            "source_form": source_form,
            "direction": direction,
            "base_prompt": base_prompt.prompt,
            "source_prompt": source_prompt.prompt,
            "base_assumption": format_assumptions(base_assumptions),
            "source_assumption": format_assumptions(source_assumptions),
            "base_claim": base_claim,
            "source_claim": source_claim,
        }
    )
    add_span_columns(out, "base", base_prompt.spans)
    add_span_columns(out, "source", source_prompt.spans)
    return out


def derive_rho_within(rows: Iterable[dict[str, Any]], form: str) -> list[dict[str, Any]]:
    renderer = tier1_renderer(form)
    return [
        derive_pair_row(
            row,
            experiment="E2",
            base_premise_renderer=renderer,
            base_claim_renderer=renderer,
            source_premise_renderer=renderer,
            source_claim_renderer=renderer,
            base_form=form,
            source_form=form,
            form=form,
        )
        for row in rows
    ]


def derive_rho_cross(rows: Iterable[dict[str, Any]], form: str) -> list[dict[str, Any]]:
    base_renderer = tier1_renderer(form)
    source_renderer = tier1_renderer(CANONICAL_FORM)
    return [
        derive_pair_row(
            row,
            experiment="E3",
            base_premise_renderer=base_renderer,
            base_claim_renderer=base_renderer,
            source_premise_renderer=source_renderer,
            source_claim_renderer=source_renderer,
            base_form=form,
            source_form=CANONICAL_FORM,
            form=form,
        )
        for row in rows
    ]


def derive_local_form(
    rows: Iterable[dict[str, Any]], target_var: str, form: str
) -> list[dict[str, Any]]:
    if target_var not in {"pi", "pc"}:
        raise ValueError("E4 only supports pi or pc")
    canonical = tier1_renderer(CANONICAL_FORM)
    varied = tier1_renderer(form)
    premise_renderer = varied if target_var == "pi" else canonical
    claim_renderer = varied if target_var == "pc" else canonical
    return [
        derive_pair_row(
            row,
            experiment="E4",
            base_premise_renderer=premise_renderer,
            base_claim_renderer=claim_renderer,
            source_premise_renderer=premise_renderer,
            source_claim_renderer=claim_renderer,
            base_form=form,
            source_form=form,
            form=form,
            direction=target_var,
        )
        for row in rows
    ]


def derive_m_form(
    rows: Iterable[dict[str, Any]], form: str
) -> list[dict[str, Any]]:
    renderer = tier1_renderer(form)
    return [
        derive_pair_row(
            row,
            experiment="E5",
            base_premise_renderer=renderer,
            base_claim_renderer=renderer,
            source_premise_renderer=renderer,
            source_claim_renderer=renderer,
            base_form=form,
            source_form=form,
            form=form,
        )
        for row in rows
    ]


def derive_m_never(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible helper for the preregistered E5 arm."""
    return derive_m_form(rows, "never")


def derive_rho_mixed(
    rows: Iterable[dict[str, Any]], premise_form: str, claim_form: str
) -> list[dict[str, Any]]:
    premise_renderer = tier1_renderer(premise_form)
    claim_renderer = tier1_renderer(claim_form)
    direction = f"{premise_form}_premise__{claim_form}_claim"
    return [
        derive_pair_row(
            row,
            experiment="E6",
            base_premise_renderer=premise_renderer,
            base_claim_renderer=claim_renderer,
            source_premise_renderer=premise_renderer,
            source_claim_renderer=claim_renderer,
            base_form=direction,
            source_form=direction,
            form="mixed",
            direction=direction,
        )
        for row in rows
    ]


def derive_double_negation(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        derive_pair_row(
            row,
            experiment="E7",
            base_premise_renderer=render_double_negation_family,
            base_claim_renderer=render_double_negation_family,
            source_premise_renderer=render_double_negation_family,
            source_claim_renderer=render_double_negation_family,
            base_form="double_negation",
            source_form="double_negation",
            form="double_negation",
        )
        for row in rows
    ]


def unique_test_events(rows: Iterable[dict[str, Any]], n_events: int = 150) -> list[tuple[str, Event]]:
    by_id: dict[str, Event] = {}
    for row in rows:
        if str(row.get("split")) != "test":
            continue
        base_id = str(row["base_event_id"])
        by_id.setdefault(base_id, event_from_columns(row, "claim"))
    events = sorted(by_id.items())
    if len(events) < n_events:
        raise ValueError(f"Need {n_events} held-out events, found {len(events)}")
    return events[:n_events]


def generate_behavioral_rows(
    events: Iterable[tuple[str, Event]],
    forms: Iterable[str] = TIER1_FORM_KEYS,
    *,
    double_negation: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_id, event in events:
        for form in forms:
            renderer = (
                render_double_negation_family
                if double_negation
                else tier1_renderer(form)
            )
            for p_i, p_c in POLARITY_CELLS:
                assumption = renderer(event, p_i)
                claim = renderer(event, p_c)
                expected = "T" if p_i == p_c else "F"
                cell = f"{'+' if p_i == 1 else '-'}{'+' if p_c == 1 else '-'}"
                rows.append(
                    {
                        "sample_id": f"section61_behavior_{base_id}_{form}_{cell}",
                        "base_event_id": base_id,
                        "experiment": "section61_behavior",
                        "section61_experiment": "E7" if double_negation else "E1",
                        "form": form,
                        "cell": cell,
                        "p_i": p_i,
                        "p_c": p_c,
                        "rho": p_i * p_c,
                        "prompt": build_prompt([assumption], claim),
                        "assumption": assumption,
                        "claim": claim,
                        "expected_label": expected,
                        "expected_R_sign": p_i * p_c,
                        "claim_axis_sign": 1,
                        "n_assumptions": 1,
                    }
                )
    return rows


def filter_das_rows(
    rows: Iterable[dict[str, Any]],
    target_var: str,
    controls: Iterable[str] | None = None,
    split: str = "test",
) -> list[dict[str, Any]]:
    allowed = set(controls) if controls is not None else None
    selected = [
        row
        for row in rows
        if str(row.get("target_var")) == target_var
        and str(row.get("split")) == split
        and (allowed is None or str(row.get("control_type")) in allowed)
    ]
    if not selected:
        raise ValueError(f"No {target_var} DAS rows matched split={split!r}")
    return selected


def anchor_mismatches(original: Iterable[dict[str, Any]], derived: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for before, after in zip(original, derived)
        if str(before["base_prompt"]) != str(after["base_prompt"])
        or str(before["source_prompt"]) != str(after["source_prompt"])
    )


def source_as_base(row: dict[str, Any]) -> dict[str, Any]:
    """Expose a derived pair's source prompt to base-side hidden-state helpers."""

    out = dict(row)
    out["base_prompt"] = row["source_prompt"]
    out["base_site"] = row.get("source_site", row.get("base_site", "claim_final"))
    out["base_label"] = row.get("source_label", "")
    for key, value in row.items():
        if not key.startswith("source_") or "_span_" not in key:
            continue
        out[f"base_{key.removeprefix('source_')}"] = value
    return out
