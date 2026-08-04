"""Truth-functional surface forms used by the Section 6.1 experiments.

The canonical representation of a polarity stays ``+1``/``-1``. This module
only changes how that value is rendered. Keeping the semantic value separate
from the string builder is important for the double-negation stress test, where
the lexically negative-looking sentence realizes ``+1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .base import Event, VerbSpec, coerce_event, did_not_ever, neg, never, pos

SentenceBuilder = Callable[[str | Event, VerbSpec | dict[str, object] | None, str | None], str]


def didnt(subject_or_event: str | Event, verb: VerbSpec | dict[str, object] | None = None,
           obj: str | None = None) -> str:
    event = coerce_event(subject_or_event, verb, obj)
    return f"{event.subject} didn't {event.verb.base} {event.obj}."


def failed_to(subject_or_event: str | Event, verb: VerbSpec | dict[str, object] | None = None,
              obj: str | None = None) -> str:
    event = coerce_event(subject_or_event, verb, obj)
    return f"{event.subject} failed to {event.verb.base} {event.obj}."


def not_the_case(subject_or_event: str | Event, verb: VerbSpec | dict[str, object] | None = None,
                 obj: str | None = None) -> str:
    event = coerce_event(subject_or_event, verb, obj)
    return f"It is not the case that {event.subject} {event.verb.past} {event.obj}."


def did_not_fail_to(subject_or_event: str | Event, verb: VerbSpec | dict[str, object] | None = None,
                    obj: str | None = None) -> str:
    event = coerce_event(subject_or_event, verb, obj)
    return f"{event.subject} did not fail to {event.verb.base} {event.obj}."


@dataclass(frozen=True)
class NegationForm:
    key: str
    display: str
    negative_builder: SentenceBuilder
    tier: str = "tier1"

    def render(self, event: Event, polarity: int) -> str:
        if int(polarity) == 1:
            return pos(event)
        if int(polarity) == -1:
            return self.negative_builder(event, None, None)
        raise ValueError(f"Polarity must be +1 or -1, got {polarity!r}")


TIER1_FORMS: dict[str, NegationForm] = {
    "did_not": NegationForm("did_not", "did not", neg),
    "didnt": NegationForm("didnt", "didn't", didnt),
    "never": NegationForm("never", "never", never),
    "did_not_ever": NegationForm("did_not_ever", "did not ever", did_not_ever),
    "failed_to": NegationForm("failed_to", "failed to", failed_to),
    "not_the_case": NegationForm("not_the_case", "it is not the case that", not_the_case),
}

TIER1_FORM_KEYS = tuple(TIER1_FORMS)
CANONICAL_FORM = "did_not"
NEW_FORM_KEYS = tuple(key for key in TIER1_FORM_KEYS if key != CANONICAL_FORM)


def render_polarity(event: Event, polarity: int, form: str = CANONICAL_FORM) -> str:
    try:
        spec = TIER1_FORMS[form]
    except KeyError as exc:
        raise ValueError(f"Unknown Tier-1 negation form: {form!r}") from exc
    return spec.render(event, int(polarity))


def render_double_negation_family(event: Event, polarity: int) -> str:
    """Render E7: ``failed to`` is -1 and outer ``did not`` makes it +1."""

    if int(polarity) == 1:
        return did_not_fail_to(event)
    if int(polarity) == -1:
        return failed_to(event)
    raise ValueError(f"Polarity must be +1 or -1, got {polarity!r}")
