"""E0 integration check for Section 6.1 span resolution on real tokenizers."""

from __future__ import annotations

import argparse

from interference_suite.base import Event, VERBS
from interference_suite.das_spans import (
    add_span_columns,
    build_prompt_with_spans,
    resolve_token_site,
)
from interference_suite.model import DEFAULT_CACHE_DIR
from interference_suite.negation_forms import TIER1_FORM_KEYS, render_polarity


def assert_inside(
    tokenizer,
    prompt: str,
    row: dict,
    prefix: str,
    site: str,
    span_name: str,
) -> None:
    position = resolve_token_site(tokenizer, prompt, row, prefix, site)
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    start, end = encoded["offset_mapping"][position]
    span_start = int(row[f"{prefix}_{span_name}_span_start"])
    span_end = int(row[f"{prefix}_{span_name}_span_end"])
    if not (start < span_end and end > span_start):
        raise AssertionError(
            f"{site} token offset={(start, end)} is outside {prefix}:{span_name} "
            f"span={(span_start, span_end)}"
        )


def main() -> int:
    args = build_parser().parse_args()
    from transformers import AutoTokenizer

    event = Event("Jack", VERBS[0], "PlaceA")
    for model_name in args.model_names:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        for form in TIER1_FORM_KEYS:
            negative = render_polarity(event, -1, form)
            prompt = build_prompt_with_spans([negative], negative)
            row: dict[str, object] = {}
            for prefix in ("base", "source"):
                add_span_columns(row, prefix, prompt.spans)
                assert_inside(
                    tokenizer, prompt.prompt, row, prefix, "premise_final", "a1"
                )
                assert_inside(
                    tokenizer, prompt.prompt, row, prefix, "claim_final", "claim"
                )
        print(f"PASS {model_name}: {len(TIER1_FORM_KEYS)} forms")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["Qwen/Qwen3-8B", "microsoft/Phi-4-mini-instruct"],
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
