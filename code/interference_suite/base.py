"""Base vocabulary, sentence builders, and prompt formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

Polarity = Literal["positive", "negative"]

z = {
    "persons": [
        "Jack", "Luke", "Noah", "Liam", "Owen", "Ryan", "Adam", "Evan", "Leo", "Ian",
        "Eric", "John", "Paul", "Dean", "Martin", "Cody", "Brian", "Mike", "Thomas",
        "Chris", "Mark", "Tyler", "Jake", "Kyle", "Zane", "Emma", "Ella", "Ava", "Mia",
        "Lily", "Zoe", "Chloe", "Grace", "Lucy", "Anna", "Clara", "Nora", "Ruby", "Ivy",
        "Alice", "Eva", "Jane", "Rose", "Lisa", "Sara", "Kate", "Jean", "Claire",
        "Amber", "Sophie",
    ],
    "verbs": [
        {"base": "visit", "past": "visited", "arg_type": "location", "candidates": ["PlaceA", "PlaceB", "PlaceC"]},
        {"base": "reach", "past": "reached", "arg_type": "location", "candidates": ["PlaceA", "PlaceB", "PlaceC"]},
        {"base": "explore", "past": "explored", "arg_type": "location", "candidates": ["PlaceA", "PlaceB", "PlaceC"]},
        {"base": "enter", "past": "entered", "arg_type": "location", "candidates": ["PlaceA", "PlaceB", "PlaceC"]},
        {"base": "like", "past": "liked", "arg_type": "location", "candidates": ["PlaceA", "PlaceB", "PlaceC"]},
        {"base": "open", "past": "opened", "arg_type": "object", "candidates": ["ObjectA", "ObjectB", "ObjectC"]},
        {"base": "create", "past": "created", "arg_type": "object", "candidates": ["ObjectA", "ObjectB", "ObjectC"]},
        {"base": "clean", "past": "cleaned", "arg_type": "object", "candidates": ["ObjectA", "ObjectB", "ObjectC"]},
        {"base": "collect", "past": "collected", "arg_type": "object", "candidates": ["ObjectA", "ObjectB", "ObjectC"]},
        {"base": "close", "past": "closed", "arg_type": "object", "candidates": ["ObjectA", "ObjectB", "ObjectC"]},
    ],
}


@dataclass(frozen=True)
class VerbSpec:
    """Verb metadata used by all probes."""

    base: str
    past: str
    arg_type: str
    candidates: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "VerbSpec":
        return cls(
            base=str(value["base"]),
            past=str(value["past"]),
            arg_type=str(value["arg_type"]),
            candidates=tuple(str(item) for item in value["candidates"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "base": self.base,
            "past": self.past,
            "arg_type": self.arg_type,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True)
class Event:
    """An event carrier e = (S, V, O)."""

    subject: str
    verb: VerbSpec
    obj: str

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.verb.base}|{self.obj}"


PERSONS = tuple(z["persons"])
VERBS = tuple(VerbSpec.from_dict(item) for item in z["verbs"])
LABELS = ("T", "F", "U")


def as_verb_spec(value: VerbSpec | dict[str, object]) -> VerbSpec:
    if isinstance(value, VerbSpec):
        return value
    return VerbSpec.from_dict(value)


def coerce_event(subject_or_event: str | Event, verb: VerbSpec | dict[str, object] | None = None, obj: str | None = None) -> Event:
    if isinstance(subject_or_event, Event):
        return subject_or_event
    if verb is None or obj is None:
        raise ValueError("verb and obj are required when subject_or_event is not an Event")
    return Event(str(subject_or_event), as_verb_spec(verb), str(obj))


def pos(subject_or_event: str | Event, verb: VerbSpec | dict[str, object] | None = None, obj: str | None = None) -> str:
    event = coerce_event(subject_or_event, verb, obj)
    return f"{event.subject} {event.verb.past} {event.obj}."


def neg(subject_or_event: str | Event, verb: VerbSpec | dict[str, object] | None = None, obj: str | None = None) -> str:
    event = coerce_event(subject_or_event, verb, obj)
    return f"{event.subject} did not {event.verb.base} {event.obj}."


def sentence(event: Event, polarity: Polarity) -> str:
    if polarity == "positive":
        return pos(event)
    if polarity == "negative":
        return neg(event)
    raise ValueError(f"Unknown polarity: {polarity}")


def polarity_symbol(polarity: Polarity) -> str:
    if polarity == "positive":
        return "+"
    if polarity == "negative":
        return "-"
    raise ValueError(f"Unknown polarity: {polarity}")


def polarity_from_symbol(symbol: str) -> Polarity:
    if symbol == "+":
        return "positive"
    if symbol == "-":
        return "negative"
    raise ValueError(f"Unknown polarity symbol: {symbol}")


def compact_polarity(polarity: Polarity) -> str:
    return "pos" if polarity == "positive" else "neg"


def format_assumptions(assumptions: Sequence[str]) -> str:
    if len(assumptions) == 1:
        return assumptions[0]
    return "\n".join(f"{idx}. {text}" for idx, text in enumerate(assumptions, start=1))


def build_prompt(assumptions: Sequence[str], claim: str) -> str:
    if len(assumptions) == 1:
        assumption_block = f"Assumption: {assumptions[0]}"
    else:
        numbered = "\n".join(f"{idx}. {text}" for idx, text in enumerate(assumptions, start=1))
        assumption_block = f"Assumption:\n{numbered}"
    return (
        f"{assumption_block}\n\n"
        f"Claim: {claim}\n\n"
        "Choose exactly one:\n"
        "T = must be true\n"
        "F = must be false\n"
        "U = cannot be determined\n\n"
        "Answer:\n"
    )


def event_metadata(event: Event, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_subject": event.subject,
        f"{prefix}_verb_base": event.verb.base,
        f"{prefix}_verb_past": event.verb.past,
        f"{prefix}_object": event.obj,
        f"{prefix}_arg_type": event.verb.arg_type,
    }


def all_events(persons: Iterable[str] = PERSONS, verbs: Iterable[VerbSpec] = VERBS) -> list[Event]:
    return [Event(person, verb, obj) for person in persons for verb in verbs for obj in verb.candidates]
