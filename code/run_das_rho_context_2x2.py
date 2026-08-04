"""Evaluate a frozen rho DAS subspace in a paired 2x2 source audit.

The audit addresses a specific ambiguity between the ``source_m0`` identity
control and cross-matching-state transfer.  Every retained row has an open
base (m=1) from one of the two rho-replacement cells in
``joint_gate_test150/triples.csv``.  The *same base prompt* is patched from
four source prompts that cross:

    source m:   same vs opposite the base
    source rho: same vs opposite the base

By default, the rho contrast is made as a true within-context minimal pair.
The two same-rho anchors are taken from the existing row:

    same m     -> m_source
    opposite m -> rho_same_source

For each anchor, the opposite-rho source is synthesized by flipping only the
polarity of the designated premise.  ``--source-pairing existing`` retains
the independently generated four-source quartet for comparison.

Only the frozen rho coordinates are transferred.  Thus all four conditions
are scored against the same intervention target g(m_base, rho_source), as
well as against preservation and reversal of the base label.  No rotation is
trained or modified by this script.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from interference_suite.das_pyvene import (
    encode_to_device,
    import_runtime,
    load_hf_model,
    to_jsonable,
)
from interference_suite.base import VERBS, Event, format_assumptions, sentence
from interference_suite.das_spans import (
    add_span_columns,
    build_prompt_with_spans,
    resolve_token_site,
)
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.joint_gate_intervention import (
    constrained_patch,
    orthonormalize_basis,
    random_orthonormal_basis,
)
from interference_suite.model import (
    DEFAULT_CACHE_DIR,
    progress_iter,
    resolve_label_tokens,
)
from run_das_ablation import get_decoder_layers
from run_das_joint_gate import batches, capture_hidden_at_layers, load_rotation


SourceCondition = tuple[str, str, str, str]
SourceConditions = tuple[SourceCondition, ...]

EXISTING_SOURCE_CONDITIONS: SourceConditions = (
    ("m_same_rho_same", "m_source", "same", "same"),
    ("m_same_rho_opposite", "m_same_source", "same", "opposite"),
    ("m_opposite_rho_same", "rho_same_source", "opposite", "same"),
    ("m_opposite_rho_opposite", "rho_source", "opposite", "opposite"),
)

MINIMAL_SOURCE_CONDITIONS: SourceConditions = (
    ("m_same_rho_same", "m_source", "same", "same"),
    ("m_same_rho_opposite", "m_source_rho_flip", "same", "opposite"),
    ("m_opposite_rho_same", "rho_same_source", "opposite", "same"),
    (
        "m_opposite_rho_opposite",
        "rho_same_source_rho_flip",
        "opposite",
        "opposite",
    ),
)

MINIMAL_PAIR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("m_source", "m_source_rho_flip"),
    ("rho_same_source", "rho_same_source_rho_flip"),
)

RHO_REPLACE_CELLS = {"rho_flip_T_to_F", "rho_flip_F_to_T"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired 2x2 source-m x source-rho audit of one frozen rho "
            "DAS rotation."
        )
    )
    parser.add_argument(
        "--samples",
        default="data/das/joint_gate_test150/triples.csv",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--rho-rotation", required=True)
    parser.add_argument("--site", default="claim_final")
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-token-style", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--source-pairing",
        choices=("minimal", "existing"),
        default="minimal",
        help=(
            "Use within-context premise-polarity minimal pairs (default), or "
            "the four independently generated source prompts already in the "
            "joint-gate rows."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and count the paired audit rows without loading a model.",
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Also evaluate equal-rank random orthonormal subspaces using these "
            "seeds. The exact same base/source rows are reused for every seed."
        ),
    )
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Evaluate only --random-seeds, omitting the learned rho subspace.",
    )
    return parser


def high_level_label(m_value: int | str, rho_value: int | str) -> str:
    m_int = int(m_value)
    rho_int = int(rho_value)
    if m_int not in (0, 1):
        raise ValueError(f"Expected m in {{0,1}}, got {m_value!r}")
    if rho_int not in (-1, 1):
        raise ValueError(f"Expected rho in {{-1,+1}}, got {rho_value!r}")
    if m_int == 0:
        return "U"
    return "T" if rho_int == 1 else "F"


def reverse_tf(label: str) -> str:
    if label == "T":
        return "F"
    if label == "F":
        return "T"
    raise ValueError(f"Cannot define a reversed T/F target for label {label!r}")


def source_conditions_for(source_pairing: str) -> SourceConditions:
    if source_pairing == "minimal":
        return MINIMAL_SOURCE_CONDITIONS
    if source_pairing == "existing":
        return EXISTING_SOURCE_CONDITIONS
    raise ValueError(f"Unknown source pairing: {source_pairing!r}")


def prompt_span_text(
    row: dict[str, Any],
    prefix: str,
    span_name: str,
) -> str:
    """Extract one named source component from its recorded char span."""

    prompt_key = f"{prefix}_prompt"
    start_key = f"{prefix}_{span_name}_span_start"
    end_key = f"{prefix}_{span_name}_span_end"
    for key in (prompt_key, start_key, end_key):
        if key not in row:
            raise ValueError(
                f"{row.get('sample_id', '<unknown>')}: missing required column {key!r}"
            )
    prompt = str(row[prompt_key])
    start = int(row[start_key])
    end = int(row[end_key])
    if not 0 <= start < end <= len(prompt):
        raise ValueError(
            f"{row.get('sample_id', '<unknown>')}: invalid {prefix}:{span_name} "
            f"span ({start}, {end}) for prompt length {len(prompt)}"
        )
    return prompt[start:end]


def parse_canonical_sentence(text: str) -> tuple[Event, str]:
    """Parse a canonical positive/negative event sentence using ``VERBS``."""

    if not text.endswith("."):
        raise ValueError(f"Not a canonical event sentence: {text!r}")
    body = text[:-1]
    for verb in VERBS:
        candidates = (
            ("negative", f" did not {verb.base} "),
            ("positive", f" {verb.past} "),
        )
        for polarity, marker in candidates:
            if marker not in body:
                continue
            pieces = body.split(marker)
            if len(pieces) != 2 or not all(pieces):
                continue
            event = Event(pieces[0], verb, pieces[1])
            if sentence(event, polarity) == text:
                return event, polarity
    raise ValueError(f"Not a canonical VERBS event sentence: {text!r}")


def flip_canonical_sentence(text: str) -> str:
    event, polarity = parse_canonical_sentence(text)
    flipped = "negative" if polarity == "positive" else "positive"
    return sentence(event, flipped)


def polarity_sign(polarity: str) -> int:
    if polarity == "positive":
        return 1
    if polarity == "negative":
        return -1
    raise ValueError(f"Unknown polarity: {polarity!r}")


def synthesize_rho_flip_source(
    row: dict[str, Any],
    *,
    anchor_prefix: str,
    output_prefix: str,
) -> None:
    """Copy an anchor and flip only its designated premise polarity."""

    sample_id = str(row["sample_id"])
    matched_idx = int(row["matched_idx"])
    if matched_idx not in (0, 1, 2):
        raise ValueError(f"{sample_id}: expected matched_idx in {{0,1,2}}")

    assumptions = [
        prompt_span_text(row, anchor_prefix, f"a{index}")
        for index in range(1, 4)
    ]
    claim = prompt_span_text(row, anchor_prefix, "claim")
    if claim != str(row[f"{anchor_prefix}_claim"]):
        raise ValueError(f"{sample_id}: {anchor_prefix} claim span disagrees with metadata")

    assumptions[matched_idx] = flip_canonical_sentence(assumptions[matched_idx])
    rebuilt = build_prompt_with_spans(assumptions, claim)

    anchor_stem = f"{anchor_prefix}_"
    output_stem = f"{output_prefix}_"
    for key, value in tuple(row.items()):
        if key.startswith(anchor_stem) and not key.startswith(output_stem):
            row[f"{output_stem}{key[len(anchor_stem):]}"] = value

    anchor_m = int(row[f"{anchor_prefix}_m"])
    anchor_pc = int(row[f"{anchor_prefix}_p_c"])
    anchor_pi = int(row[f"{anchor_prefix}_p_i"])
    anchor_rho = int(row[f"{anchor_prefix}_rho"])
    if anchor_pi * anchor_pc != anchor_rho:
        raise ValueError(f"{sample_id}: {anchor_prefix} has inconsistent p_i, p_c, rho")

    flipped_pi = -anchor_pi
    flipped_rho = -anchor_rho
    if flipped_pi * anchor_pc != flipped_rho:
        raise AssertionError("Synthetic polarity flip produced inconsistent rho")

    row[f"{output_prefix}_prompt"] = rebuilt.prompt
    row[f"{output_prefix}_assumption"] = format_assumptions(assumptions)
    row[f"{output_prefix}_claim"] = claim
    row[f"{output_prefix}_label"] = high_level_label(anchor_m, flipped_rho)
    row[f"{output_prefix}_m"] = anchor_m
    row[f"{output_prefix}_p_c"] = anchor_pc
    row[f"{output_prefix}_p_i"] = flipped_pi
    row[f"{output_prefix}_rho"] = flipped_rho
    row[f"{output_prefix}_prompt_matches_standard"] = 1
    add_span_columns(row, output_prefix, rebuilt.spans)


def prepare_source_rows(
    rows: list[dict[str, Any]],
    *,
    source_pairing: str,
) -> list[dict[str, Any]]:
    """Return private row copies with synthetic sources added when requested."""

    prepared = [dict(row) for row in rows]
    if source_pairing == "existing":
        return prepared
    if source_pairing != "minimal":
        raise ValueError(f"Unknown source pairing: {source_pairing!r}")
    for row in prepared:
        for anchor_prefix, output_prefix in MINIMAL_PAIR_PREFIXES:
            synthesize_rho_flip_source(
                row,
                anchor_prefix=anchor_prefix,
                output_prefix=output_prefix,
            )
    return prepared


def validate_minimal_pair(
    row: dict[str, Any],
    *,
    anchor_prefix: str,
    flipped_prefix: str,
) -> None:
    """Assert that a synthesized pair changes only the target polarity."""

    sample_id = str(row["sample_id"])
    matched_idx = int(row["matched_idx"])
    anchor_assumptions = [
        prompt_span_text(row, anchor_prefix, f"a{index}")
        for index in range(1, 4)
    ]
    flipped_assumptions = [
        prompt_span_text(row, flipped_prefix, f"a{index}")
        for index in range(1, 4)
    ]
    anchor_parsed = [parse_canonical_sentence(text) for text in anchor_assumptions]
    flipped_parsed = [parse_canonical_sentence(text) for text in flipped_assumptions]

    anchor_event_signature = tuple(event.key for event, _ in anchor_parsed)
    flipped_event_signature = tuple(event.key for event, _ in flipped_parsed)
    if anchor_event_signature != flipped_event_signature:
        raise ValueError(
            f"{sample_id}: {anchor_prefix}/{flipped_prefix} event signatures differ"
        )
    for index, (anchor_text, flipped_text) in enumerate(
        zip(anchor_assumptions, flipped_assumptions)
    ):
        if index == matched_idx:
            if flipped_text != flip_canonical_sentence(anchor_text):
                raise ValueError(
                    f"{sample_id}: {flipped_prefix} does not canonically flip a{index + 1}"
                )
        elif flipped_text != anchor_text:
            raise ValueError(
                f"{sample_id}: distractor a{index + 1} changes in minimal pair"
            )

    anchor_claim = prompt_span_text(row, anchor_prefix, "claim")
    flipped_claim = prompt_span_text(row, flipped_prefix, "claim")
    if anchor_claim != flipped_claim:
        raise ValueError(f"{sample_id}: claim changes in minimal source pair")
    claim_event, claim_polarity = parse_canonical_sentence(anchor_claim)
    anchor_event, anchor_polarity = anchor_parsed[matched_idx]
    _, flipped_polarity = flipped_parsed[matched_idx]

    expected_m = int(anchor_event.key == claim_event.key)
    if int(row[f"{anchor_prefix}_m"]) != expected_m:
        raise ValueError(f"{sample_id}: {anchor_prefix}_m disagrees with event match")
    if int(row[f"{flipped_prefix}_m"]) != expected_m:
        raise ValueError(f"{sample_id}: polarity flip changes m metadata")
    if polarity_sign(anchor_polarity) != int(row[f"{anchor_prefix}_p_i"]):
        raise ValueError(f"{sample_id}: {anchor_prefix}_p_i disagrees with a{matched_idx + 1}")
    if polarity_sign(flipped_polarity) != int(row[f"{flipped_prefix}_p_i"]):
        raise ValueError(f"{sample_id}: {flipped_prefix}_p_i disagrees with a{matched_idx + 1}")
    if polarity_sign(claim_polarity) != int(row[f"{anchor_prefix}_p_c"]):
        raise ValueError(f"{sample_id}: {anchor_prefix}_p_c disagrees with claim")

    for suffix in ("claim", "site"):
        if str(row[f"{anchor_prefix}_{suffix}"]) != str(
            row[f"{flipped_prefix}_{suffix}"]
        ):
            raise ValueError(f"{sample_id}: minimal pair changes {suffix}")
    for suffix in ("p_c", "m"):
        if int(row[f"{anchor_prefix}_{suffix}"]) != int(
            row[f"{flipped_prefix}_{suffix}"]
        ):
            raise ValueError(f"{sample_id}: minimal pair changes {suffix}")
    if int(row[f"{flipped_prefix}_p_i"]) != -int(row[f"{anchor_prefix}_p_i"]):
        raise ValueError(f"{sample_id}: minimal pair does not invert p_i")
    if int(row[f"{flipped_prefix}_rho"]) != -int(row[f"{anchor_prefix}_rho"]):
        raise ValueError(f"{sample_id}: minimal pair does not invert rho")


def select_primary_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    """Select the paired open-base audit without changing row direction.

    ``rho_source_m=0`` is important: on these open-base rho-replacement rows,
    ``rho_source`` is exactly the existing opposite-m/opposite-rho corner.
    All anchors needed by either source-pairing mode are stored on that same
    row, so no source or base needs to be joined from another example.
    """

    selected = [
        row
        for row in rows
        if row.get("split") == split
        and str(row.get("cell_type")) in RHO_REPLACE_CELLS
        and int(row["base_m"]) == 1
        and int(row["rho_source_m"]) == 0
    ]
    selected.sort(key=lambda row: int(row["row_id"]))
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        selected = selected[:max_rows]
    if not selected:
        raise ValueError(
            "No open-base rho-replacement rows with rho_source_m=0 matched "
            f"split={split!r}"
        )
    return selected


def validate_source_quartet(
    rows: list[dict[str, Any]],
    *,
    source_conditions: SourceConditions,
    source_pairing: str,
) -> dict[str, Any]:
    """Check that each row implements the intended paired 2x2 design."""

    seen_ids: set[str] = set()
    condition_counts: Counter[str] = Counter()
    base_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate selected sample_id: {sample_id}")
        seen_ids.add(sample_id)

        base_m = int(row["base_m"])
        base_rho = int(row["base_rho"])
        base_label = str(row["base_label"])
        if base_m != 1:
            raise ValueError(f"{sample_id}: primary audit requires base_m=1")
        if base_label != high_level_label(base_m, base_rho):
            raise ValueError(
                f"{sample_id}: base_label={base_label!r} disagrees with "
                f"g({base_m},{base_rho})"
            )
        if int(row["rho_source_m"]) != 0:
            raise ValueError(f"{sample_id}: primary audit requires rho_source_m=0")

        base_claim = str(row["base_claim"])
        base_pc = int(row["base_p_c"])
        if source_pairing == "minimal":
            for anchor_prefix, flipped_prefix in MINIMAL_PAIR_PREFIXES:
                validate_minimal_pair(
                    row,
                    anchor_prefix=anchor_prefix,
                    flipped_prefix=flipped_prefix,
                )

        for condition, prefix, m_relation, rho_relation in source_conditions:
            source_m = int(row[f"{prefix}_m"])
            source_rho = int(row[f"{prefix}_rho"])
            expected_m = base_m if m_relation == "same" else 1 - base_m
            expected_rho = base_rho if rho_relation == "same" else -base_rho
            if source_m != expected_m or source_rho != expected_rho:
                raise ValueError(
                    f"{sample_id}: {condition} via {prefix} has "
                    f"(m,rho)=({source_m},{source_rho}), expected "
                    f"({expected_m},{expected_rho})"
                )
            if int(row[f"{prefix}_p_c"]) != base_pc:
                raise ValueError(f"{sample_id}: {prefix} changes p_c")
            if str(row[f"{prefix}_claim"]) != base_claim:
                raise ValueError(f"{sample_id}: {prefix} changes the claim text")
            natural_label = high_level_label(source_m, source_rho)
            if str(row[f"{prefix}_label"]) != natural_label:
                raise ValueError(
                    f"{sample_id}: {prefix}_label={row[f'{prefix}_label']!r}, "
                    f"expected {natural_label!r}"
                )
            condition_counts[condition] += 1
        base_counts[f"m{base_m}_rho{base_rho:+d}"] += 1

        # These fields make accidental use of another joint-gate cell obvious.
        expected_cell = "rho_flip_T_to_F" if base_rho == 1 else "rho_flip_F_to_T"
        if str(row["cell_type"]) != expected_cell:
            raise ValueError(
                f"{sample_id}: cell_type={row['cell_type']!r}, expected "
                f"{expected_cell!r} for base_rho={base_rho}"
            )
        if row_index == 0:
            # Fail early with a readable error if the dataset predates the
            # source fields required for this audit.
            for _, prefix, _, _ in source_conditions:
                for suffix in (
                    "prompt",
                    "claim_span_start",
                    "claim_span_end",
                    "answer_span_start",
                    "answer_span_end",
                ):
                    key = f"{prefix}_{suffix}"
                    if key not in row:
                        raise ValueError(f"Dataset is missing required column {key!r}")

    return {
        "source_pairing": source_pairing,
        "n_rows": len(rows),
        "n_unique_sample_ids": len(seen_ids),
        "base_counts": dict(sorted(base_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
    }


def score_logits(
    torch: Any,
    logits: Any,
    enc: dict[str, Any],
    label_ids: dict[str, int],
) -> list[dict[str, Any]]:
    final = enc["attention_mask"].sum(dim=1) - 1
    index = torch.arange(logits.shape[0], device=logits.device)
    next_logits = logits[index, final]
    outputs: list[dict[str, Any]] = []
    for row_logits in next_logits:
        values = {
            label: float(row_logits[token_id].detach().cpu())
            for label, token_id in label_ids.items()
        }
        outputs.append(
            {
                **{f"logit_{label}": value for label, value in values.items()},
                "pred_label": max(values, key=values.get),
                "R": values["T"] - values["F"],
            }
        )
    return outputs


def collect_rho_source_states(
    *,
    model: Any,
    layers: Any,
    torch: Any,
    tokenizer: Any,
    device: Any,
    batch_rows: list[dict[str, Any]],
    rho_layer: int,
    site: str,
    source_conditions: SourceConditions,
) -> dict[str, Any]:
    texts: list[str] = []
    expanded_rows: list[dict[str, Any]] = []
    expanded_prefixes: list[str] = []
    for _, prefix, _, _ in source_conditions:
        texts.extend(str(row[f"{prefix}_prompt"]) for row in batch_rows)
        expanded_rows.extend(batch_rows)
        expanded_prefixes.extend([prefix] * len(batch_rows))
    captured = capture_hidden_at_layers(
        model,
        layers,
        torch,
        tokenizer,
        device,
        texts,
        expanded_rows,
        expanded_prefixes,
        {rho_layer},
        site,
    )[rho_layer]
    n = len(batch_rows)
    return {
        condition: captured[index * n : (index + 1) * n]
        for index, (condition, _, _, _) in enumerate(source_conditions)
    }


def run_rho_patch(
    *,
    model: Any,
    layers: Any,
    torch: Any,
    enc: dict[str, Any],
    positions: Any,
    rho_layer: int,
    u_rho: Any,
    source_hidden: Any,
    label_ids: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    source_coords = source_hidden.to(u_rho.device) @ u_rho
    diagnostics: dict[str, float] = {}

    def hook(module: Any, inputs: Any, output: Any):
        hs = output[0] if isinstance(output, tuple) else output
        pos = positions.to(hs.device)
        index = torch.arange(hs.shape[0], device=hs.device)
        base_hidden = hs[index, pos].to(torch.float32)
        patched, diag = constrained_patch(
            torch,
            base_hidden,
            [u_rho.to(hs.device)],
            [source_coords.to(hs.device)],
        )
        hs[index, pos] = patched.to(hs.dtype)
        for key in (
            "coordinate_residual_mean",
            "coordinate_residual_max",
            "update_norm_mean",
        ):
            diagnostics[key] = float(diag[key])
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    handle = layers[rho_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits
    finally:
        handle.remove()
    return score_logits(torch, logits, enc, label_ids), diagnostics


def summarize(
    scored: list[dict[str, Any]],
    *,
    source_conditions: SourceConditions,
) -> list[dict[str, Any]]:
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("overall", scored)]
    for base_m in (0, 1):
        for base_rho in (-1, 1):
            scoped = [
                row
                for row in scored
                if int(row["base_m"]) == base_m
                and int(row["base_rho"]) == base_rho
            ]
            if scoped:
                scopes.append((f"m{base_m}_rho{base_rho:+d}", scoped))

    # Preserve the execution order while allowing learned and random runs to
    # share the same source-condition labels without being averaged together.
    condition_order = list(
        dict.fromkeys(str(row["condition"]) for row in scored)
    )
    summary: list[dict[str, Any]] = []
    for scope, scope_rows in scopes:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scope_rows:
            groups[str(row["condition"])].append(row)
        for condition in condition_order:
            rows = groups.get(condition, [])
            if not rows:
                continue
            summary.append(
                {
                    "scope": scope,
                    "condition": condition,
                    "base_condition": rows[0]["base_condition"],
                    "intervention_kind": rows[0]["intervention_kind"],
                    "random_seed": rows[0]["random_seed"],
                    "source_m_relation": rows[0]["source_m_relation"],
                    "source_rho_relation": rows[0]["source_rho_relation"],
                    "n": len(rows),
                    "n_events": len({str(row["base_event_id"]) for row in rows}),
                    "source_target_accuracy": sum(
                        int(row["source_target_correct"]) for row in rows
                    )
                    / len(rows),
                    "base_preservation": sum(
                        int(row["base_preserved"]) for row in rows
                    )
                    / len(rows),
                    "reversed_base_accuracy": sum(
                        int(row["reversed_base_correct"]) for row in rows
                    )
                    / len(rows),
                    "mean_R": sum(float(row["R"]) for row in rows) / len(rows),
                    "mean_delta_R": sum(float(row["delta_R"]) for row in rows)
                    / len(rows),
                    "mean_base_aligned_delta_R": sum(
                        float(row["base_aligned_delta_R"]) for row in rows
                    )
                    / len(rows),
                    "mean_abs_delta_R": sum(
                        abs(float(row["delta_R"])) for row in rows
                    )
                    / len(rows),
                    "mean_coordinate_residual": sum(
                        float(row["coordinate_residual_mean"]) for row in rows
                    )
                    / len(rows),
                    "mean_update_norm": sum(
                        float(row["update_norm_mean"]) for row in rows
                    )
                    / len(rows),
                }
            )
    return summary

def main() -> int:
    args = build_parser().parse_args()
    if args.random_only and not args.random_seeds:
        raise ValueError("--random-only requires at least one --random-seeds value")
    if len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError("--random-seeds must not contain duplicates")

    source_conditions = source_conditions_for(args.source_pairing)
    all_rows = read_rows_csv(args.samples)
    selected_rows = select_primary_rows(
        all_rows,
        split=args.split,
        max_rows=args.max_rows,
    )
    rows = prepare_source_rows(
        selected_rows,
        source_pairing=args.source_pairing,
    )
    validation = validate_source_quartet(
        rows,
        source_conditions=source_conditions,
        source_pairing=args.source_pairing,
    )
    print(
        f"Validated {args.source_pairing} paired open-base audit: "
        f"{validation['n_rows']} rows; {validation['base_counts']}"
    )
    if args.validate_only:
        return 0

    torch, _, auto_model_cls, auto_tokenizer_cls = import_runtime()
    tokenizer, model = load_hf_model(
        torch=torch,
        auto_model_cls=auto_model_cls,
        auto_tokenizer_cls=auto_tokenizer_cls,
        model_name=args.model_name,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    device = next(model.parameters()).device
    layers = get_decoder_layers(model)
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    hidden_size = int(model.config.hidden_size)

    import numpy as np

    rho_raw, rho_meta = load_rotation(Path(args.rho_rotation), np)
    rho_layer = int(rho_meta["layer"])
    if str(rho_meta.get("target_var", "rho")) != "rho":
        raise ValueError(
            f"Expected a rho rotation, metadata target_var={rho_meta.get('target_var')!r}"
        )
    if int(rho_raw.shape[0]) != hidden_size:
        raise ValueError(
            f"Rotation hidden size {rho_raw.shape[0]} does not match model "
            f"hidden_size={hidden_size}"
        )
    rho_rank = int(rho_raw.shape[1])

    basis_runs: list[tuple[str, int | None, Any]] = []
    if not args.random_only:
        basis_runs.append(
            (
                "learned",
                None,
                orthonormalize_basis(
                    torch,
                    torch.tensor(rho_raw, device=device),
                ),
            )
        )
    for random_seed in args.random_seeds:
        basis_runs.append(
            (
                "random",
                int(random_seed),
                random_orthonormal_basis(
                    torch,
                    hidden_size,
                    rho_rank,
                    device=device,
                    seed=20_000 + int(random_seed),
                ),
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "rho_context_2x2_scored.partial.csv"

    scored: list[dict[str, Any]] = []
    batch_list = list(batches(rows, args.eval_batch_size))
    for batch_index, batch_rows in enumerate(
        progress_iter(batch_list, total=len(batch_list), desc="rho context 2x2"),
        start=1,
    ):
        source_states = collect_rho_source_states(
            model=model,
            layers=layers,
            torch=torch,
            tokenizer=tokenizer,
            device=device,
            batch_rows=batch_rows,
            rho_layer=rho_layer,
            site=args.site,
            source_conditions=source_conditions,
        )
        texts = [str(row["base_prompt"]) for row in batch_rows]
        enc = encode_to_device(tokenizer, texts, device)
        positions = torch.tensor(
            [
                resolve_token_site(tokenizer, text, row, "base", args.site)
                for text, row in zip(texts, batch_rows)
            ],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            base_logits = model(**enc, use_cache=False).logits
        base_outputs = score_logits(
            torch,
            base_logits,
            enc,
            label_tokens.token_ids,
        )

        for intervention_kind, random_seed, basis in basis_runs:
            run_tag = (
                "learned"
                if intervention_kind == "learned"
                else f"random_s{random_seed}"
            )
            condition_outputs: dict[
                str, tuple[list[dict[str, Any]], dict[str, float]]
            ] = {}
            for condition, _, _, _ in source_conditions:
                condition_outputs[condition] = run_rho_patch(
                    model=model,
                    layers=layers,
                    torch=torch,
                    enc=enc,
                    positions=positions,
                    rho_layer=rho_layer,
                    u_rho=basis,
                    source_hidden=source_states[condition],
                    label_ids=label_tokens.token_ids,
                )

            stored_none = (
                "none" if intervention_kind == "learned" else f"{run_tag}:none"
            )
            for row, base_output in zip(batch_rows, base_outputs):
                base_label = str(row["base_label"])
                reversed_label = reverse_tf(base_label)
                scored.append(
                    {
                        "sample_id": row["sample_id"],
                        "base_event_id": row["base_event_id"],
                        "cell_type": row["cell_type"],
                        "base_m": row["base_m"],
                        "base_rho": row["base_rho"],
                        "base_label": base_label,
                        "source_pairing": args.source_pairing,
                        "condition": stored_none,
                        "base_condition": "none",
                        "intervention_kind": intervention_kind,
                        "random_seed": "" if random_seed is None else random_seed,
                        "source_prefix": "none",
                        "source_m_relation": "none",
                        "source_rho_relation": "none",
                        "source_m": row["base_m"],
                        "source_rho": row["base_rho"],
                        "source_natural_label": base_label,
                        "source_target_label": base_label,
                        "reversed_base_label": reversed_label,
                        "pred_label": base_output["pred_label"],
                        "source_target_correct": int(
                            base_output["pred_label"] == base_label
                        ),
                        "base_preserved": int(
                            base_output["pred_label"] == base_label
                        ),
                        "reversed_base_correct": int(
                            base_output["pred_label"] == reversed_label
                        ),
                        "logit_T": base_output["logit_T"],
                        "logit_F": base_output["logit_F"],
                        "logit_U": base_output["logit_U"],
                        "R": base_output["R"],
                        "base_R": base_output["R"],
                        "delta_R": 0.0,
                        "base_aligned_delta_R": 0.0,
                        "coordinate_residual_mean": 0.0,
                        "update_norm_mean": 0.0,
                    }
                )

            for condition, prefix, m_relation, rho_relation in source_conditions:
                outputs, diagnostics = condition_outputs[condition]
                stored_condition = (
                    condition
                    if intervention_kind == "learned"
                    else f"{run_tag}:{condition}"
                )
                for row, base_output, output in zip(
                    batch_rows, base_outputs, outputs
                ):
                    base_label = str(row["base_label"])
                    reversed_label = reverse_tf(base_label)
                    source_m = int(row[f"{prefix}_m"])
                    source_rho = int(row[f"{prefix}_rho"])
                    source_target = high_level_label(row["base_m"], source_rho)
                    delta_r = float(output["R"]) - float(base_output["R"])
                    base_sign = 1.0 if base_label == "T" else -1.0
                    scored.append(
                        {
                            "sample_id": row["sample_id"],
                            "base_event_id": row["base_event_id"],
                            "cell_type": row["cell_type"],
                            "base_m": row["base_m"],
                            "base_rho": row["base_rho"],
                            "base_label": base_label,
                            "source_pairing": args.source_pairing,
                            "condition": stored_condition,
                            "base_condition": condition,
                            "intervention_kind": intervention_kind,
                            "random_seed": "" if random_seed is None else random_seed,
                            "source_prefix": prefix,
                            "source_m_relation": m_relation,
                            "source_rho_relation": rho_relation,
                            "source_m": source_m,
                            "source_rho": source_rho,
                            "source_natural_label": row[f"{prefix}_label"],
                            "source_target_label": source_target,
                            "reversed_base_label": reversed_label,
                            "pred_label": output["pred_label"],
                            "source_target_correct": int(
                                output["pred_label"] == source_target
                            ),
                            "base_preserved": int(
                                output["pred_label"] == base_label
                            ),
                            "reversed_base_correct": int(
                                output["pred_label"] == reversed_label
                            ),
                            "logit_T": output["logit_T"],
                            "logit_F": output["logit_F"],
                            "logit_U": output["logit_U"],
                            "R": output["R"],
                            "base_R": base_output["R"],
                            "delta_R": delta_r,
                            "base_aligned_delta_R": base_sign * delta_r,
                            "coordinate_residual_mean": diagnostics.get(
                                "coordinate_residual_mean", 0.0
                            ),
                            "update_norm_mean": diagnostics.get(
                                "update_norm_mean", 0.0
                            ),
                        }
                    )

        if args.checkpoint_every and batch_index % args.checkpoint_every == 0:
            write_rows_csv(scored, partial_path)

    scored_path = output_dir / "rho_context_2x2_scored.csv"
    summary_path = output_dir / "rho_context_2x2_summary.csv"
    summary_json_path = output_dir / "rho_context_2x2_summary.json"
    write_rows_csv(scored, scored_path)
    summary = summarize(scored, source_conditions=source_conditions)
    write_rows_csv(summary, summary_path)

    metadata = {
        "model_name": args.model_name,
        "samples": args.samples,
        "split": args.split,
        "site": args.site,
        "source_pairing": args.source_pairing,
        "rho_rotation": args.rho_rotation,
        "rho_layer": rho_layer,
        "rho_rank": rho_rank,
        "random_only": bool(args.random_only),
        "random_seeds": [int(seed) for seed in args.random_seeds],
        "random_basis_seed_offset": 20_000,
        "audit_scope": (
            "open bases; rho-replacement cells; rho_source_m=0; identical base "
            "rows across all four source conditions"
        ),
        "conditions": [
            {
                "name": condition,
                "source_prefix": prefix,
                "source_m_relation": m_relation,
                "source_rho_relation": rho_relation,
            }
            for condition, prefix, m_relation, rho_relation in source_conditions
        ],
        "validation": validation,
    }
    summary_json_path.write_text(
        json.dumps(to_jsonable({"metadata": metadata, "summary": summary}), indent=2)
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(to_jsonable(metadata), indent=2)
    )

    print(
        "\ncondition                              source-target   preserve   "
        "reverse   aligned dR   upd-norm"
    )
    for entry in summary:
        if entry["scope"] != "overall":
            continue
        print(
            f"{entry['condition']:<38} "
            f"{entry['source_target_accuracy']:>8.3f}      "
            f"{entry['base_preservation']:>8.3f}  "
            f"{entry['reversed_base_accuracy']:>8.3f}  "
            f"{entry['mean_base_aligned_delta_R']:>10.3f}  "
            f"{entry['mean_update_norm']:>9.3f}"
        )
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
