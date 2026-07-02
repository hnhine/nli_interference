"""Prompt span and token-site helpers for DAS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .base import build_prompt


@dataclass(frozen=True)
class PromptWithSpans:
    prompt: str
    spans: dict[str, tuple[int, int]]


def build_prompt_with_spans(assumptions: Sequence[str], claim: str) -> PromptWithSpans:
    """Build the standard prompt and return char spans for DAS token sites.

    The returned prompt is asserted to match ``base.build_prompt`` exactly so the
    behavioral and DAS pipelines share the same surface form.
    """

    text = ""
    spans: dict[str, tuple[int, int]] = {}

    def add(value: str) -> tuple[int, int]:
        nonlocal text
        start = len(text)
        text += value
        return start, len(text)

    if len(assumptions) == 1:
        add("Assumption: ")
        spans["a1"] = add(assumptions[0])
    else:
        add("Assumption:\n")
        for idx, assumption in enumerate(assumptions, start=1):
            add(f"{idx}. ")
            spans[f"a{idx}"] = add(assumption)
            if idx != len(assumptions):
                add("\n")

    add("\n\nClaim: ")
    spans["claim"] = add(claim)
    add(
        "\n\n"
        "Choose exactly one:\n"
        "T = must be true\n"
        "F = must be false\n"
        "U = cannot be determined\n\n"
    )
    spans["answer"] = add("Answer:")
    add("\n")

    expected = build_prompt(assumptions, claim)
    if text != expected:
        raise AssertionError("DAS prompt builder diverged from base.build_prompt")
    return PromptWithSpans(prompt=text, spans=spans)


def add_span_columns(row: dict[str, Any], prefix: str, spans: dict[str, tuple[int, int]]) -> None:
    for name, (start, end) in spans.items():
        row[f"{prefix}_{name}_span_start"] = start
        row[f"{prefix}_{name}_span_end"] = end


def resolve_token_site(tokenizer: Any, text: str, row: dict[str, Any], prefix: str, site: str) -> int:
    """Resolve a row-level DAS site to a token index.

    Sites are char-span based except ``answer_token``, which is the last
    non-whitespace token in the prompt.
    """

    if site == "answer_token":
        return last_non_whitespace_token(tokenizer, text)
    if site == "claim_final":
        return last_token_in_named_span(tokenizer, text, row, prefix, "claim")
    if site == "matched_assumption_final":
        matched_idx = int(row["matched_idx"])
        return last_token_in_named_span(tokenizer, text, row, prefix, f"a{matched_idx + 1}")
    if site.startswith("a") and site.endswith("_final"):
        return last_token_in_named_span(tokenizer, text, row, prefix, site.removesuffix("_final"))
    raise ValueError(f"Unknown DAS token site: {site}")


def last_token_in_named_span(tokenizer: Any, text: str, row: dict[str, Any], prefix: str, span_name: str) -> int:
    start_key = f"{prefix}_{span_name}_span_start"
    end_key = f"{prefix}_{span_name}_span_end"
    if start_key not in row or end_key not in row:
        raise ValueError(f"Missing span columns for {prefix}:{span_name}")
    return last_non_whitespace_token_in_char_span(
        tokenizer,
        text,
        (int(row[start_key]), int(row[end_key])),
    )


def last_non_whitespace_token(tokenizer: Any, text: str) -> int:
    stripped_end = len(text.rstrip())
    if stripped_end == 0:
        raise ValueError("Cannot resolve a token site in an empty prompt")
    return last_non_whitespace_token_in_char_span(tokenizer, text, (0, stripped_end))


def last_non_whitespace_token_in_char_span(tokenizer: Any, text: str, char_span: tuple[int, int]) -> int:
    start, end = char_span
    if start >= end:
        raise ValueError(f"Empty char span: {char_span}")

    try:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded["offset_mapping"]
    except Exception:
        return fallback_last_token_by_prefix(tokenizer, text, start, end)

    hits: list[int] = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if tok_start < end and tok_end > start and text[tok_start:tok_end].strip():
            hits.append(idx)
    if hits:
        return hits[-1]
    return fallback_last_token_by_prefix(tokenizer, text, start, end)


def fallback_last_token_by_prefix(tokenizer: Any, text: str, start: int, end: int) -> int:
    """Fallback for slow tokenizers without offset mappings."""

    span_text = text[start:end].rstrip()
    if not span_text:
        raise ValueError(f"No non-whitespace text in char span {(start, end)}")
    prefix = text[:start]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    span_ids = tokenizer(span_text, add_special_tokens=False)["input_ids"]
    if not span_ids:
        raise ValueError(f"No token found for char span {(start, end)}")
    return len(prefix_ids) + len(span_ids) - 1
