from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_spans import (  # noqa: E402
    add_span_columns,
    build_prompt_with_spans,
    last_lexical_token_in_char_span,
    last_non_whitespace_token_in_char_span,
    resolve_token_site,
)
from interference_suite.base import Event, VERBS  # noqa: E402
from interference_suite.negation_forms import TIER1_FORM_KEYS, render_polarity  # noqa: E402


class OffsetTokenizer:
    """Minimal tokenizer stub exposing offsets for a single fixed string."""

    def __init__(self, text: str, offsets: list[tuple[int, int]]):
        self.text = text
        self.offsets = offsets

    def __call__(self, text: str, **kwargs):
        if text != self.text:
            raise AssertionError(f"Unexpected text: {text!r}")
        return {"offset_mapping": self.offsets}


class CharacterTokenizer:
    """Offset tokenizer with one token per non-whitespace character."""

    def __call__(self, text: str, **kwargs):
        return {
            "offset_mapping": [
                (idx, idx + 1)
                for idx, char in enumerate(text)
                if not char.isspace()
            ]
        }


class LexicalFinalSiteTests(unittest.TestCase):
    def test_skips_period_newline_boundary_token(self):
        text = "ObjectB.\n\n"
        tokenizer = OffsetTokenizer(text, [(0, 7), (7, 10)])

        self.assertEqual(last_non_whitespace_token_in_char_span(tokenizer, text, (0, 8)), 1)
        self.assertEqual(last_lexical_token_in_char_span(tokenizer, text, (0, 8)), 0)

    def test_resolves_matched_assumption_lexical_final(self):
        text = "First.\nSecond.\nThird.\n\n"
        tokenizer = OffsetTokenizer(
            text,
            [(0, 5), (5, 7), (7, 13), (13, 15), (15, 20), (20, 23)],
        )
        row = {
            "matched_idx": 2,
            "base_a3_span_start": 15,
            "base_a3_span_end": 21,
        }

        position = resolve_token_site(
            tokenizer,
            text,
            row,
            "base",
            "matched_assumption_lexical_final",
        )

        self.assertEqual(position, 4)

    def test_row_lexical_final_preserves_row_specific_site(self):
        text = "First.\nSecond.\nThird.\n\n"
        tokenizer = OffsetTokenizer(
            text,
            [(0, 5), (5, 7), (7, 13), (13, 15), (15, 20), (20, 23)],
        )
        row = {
            "base_site": "a3_final",
            "base_a3_span_start": 15,
            "base_a3_span_end": 21,
        }

        position = resolve_token_site(tokenizer, text, row, "base", "row_lexical_final")

        self.assertEqual(position, 4)

    def test_rejects_span_without_lexical_character(self):
        text = ".\n"
        tokenizer = OffsetTokenizer(text, [(0, 2)])
        with self.assertRaisesRegex(ValueError, "No alphanumeric character"):
            last_lexical_token_in_char_span(tokenizer, text, (0, 1))


class Tier1NegationSpanTests(unittest.TestCase):
    def test_all_six_forms_resolve_premise_and_claim_sites(self):
        event = Event("Jack", VERBS[0], "PlaceA")
        tokenizer = CharacterTokenizer()
        self.assertEqual(len(TIER1_FORM_KEYS), 6)
        for form in TIER1_FORM_KEYS:
            with self.subTest(form=form):
                sentence = render_polarity(event, -1, form)
                built = build_prompt_with_spans([sentence], sentence)
                row = {}
                add_span_columns(row, "base", built.spans)
                premise_pos = resolve_token_site(
                    tokenizer, built.prompt, row, "base", "premise_final"
                )
                claim_pos = resolve_token_site(
                    tokenizer, built.prompt, row, "base", "claim_final"
                )
                lexical_pos = resolve_token_site(
                    tokenizer, built.prompt, row, "base", "premise_lexical_final"
                )
                offsets = tokenizer(built.prompt)["offset_mapping"]
                premise_offset = offsets[premise_pos]
                claim_offset = offsets[claim_pos]
                lexical_offset = offsets[lexical_pos]
                self.assertTrue(
                    built.spans["a1"][0] <= premise_offset[0] < built.spans["a1"][1]
                )
                self.assertTrue(
                    built.spans["claim"][0] <= claim_offset[0] < built.spans["claim"][1]
                )
                self.assertEqual(built.prompt[premise_offset[0]:premise_offset[1]], ".")
                self.assertEqual(built.prompt[claim_offset[0]:claim_offset[1]], ".")
                self.assertEqual(built.prompt[lexical_offset[0]:lexical_offset[1]], "A")


if __name__ == "__main__":
    unittest.main()
