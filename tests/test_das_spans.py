from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_spans import (  # noqa: E402
    last_lexical_token_in_char_span,
    last_non_whitespace_token_in_char_span,
    resolve_token_site,
)


class OffsetTokenizer:
    """Minimal tokenizer stub exposing offsets for a single fixed string."""

    def __init__(self, text: str, offsets: list[tuple[int, int]]):
        self.text = text
        self.offsets = offsets

    def __call__(self, text: str, **kwargs):
        if text != self.text:
            raise AssertionError(f"Unexpected text: {text!r}")
        return {"offset_mapping": self.offsets}


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


if __name__ == "__main__":
    unittest.main()
