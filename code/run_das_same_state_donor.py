"""Disentangle the joint-gate purity control: same-state vs cross-state donors.

The ``joint_same_value`` control in run_das_joint_gate.py sources each channel
from a donor that matches the base on that channel but differs on the other one
(m_same_source has the opposite rho, rho_same_source the opposite m).  A failure
there is therefore ambiguous: the transferred coordinate may be entangled with
the other causal variable, or merely with the donor's surface content.

This script adds the missing donor: another *base* prompt with the same m and
the same rho but a different event.  Running both donor kinds over the same rows
separates the two explanations.  All conditions rewrite values the base already
holds, so every expected label is the base label.  No subspace is trained.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from interference_suite.das_pyvene import encode_to_device, import_runtime, load_hf_model, to_jsonable
from interference_suite.das_spans import resolve_token_site
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.joint_gate_intervention import constrained_patch, orthonormalize_basis
from interference_suite.model import DEFAULT_CACHE_DIR, progress_iter, resolve_label_tokens
from run_das_ablation import get_decoder_layers
from run_das_joint_gate import batches, capture_hidden_at_layers, load_rotation

# (condition, donor kind, channels written)
CONDITIONS = (
    ("none", None, ()),
    ("same_state_joint", "same_state", ("m", "rho")),
    ("same_state_m_only", "same_state", ("m",)),
    ("same_state_rho_only", "same_state", ("rho",)),
    ("cross_state_joint", "cross_state", ("m", "rho")),
    ("cross_state_m_only", "cross_state", ("m",)),
    ("cross_state_rho_only", "cross_state", ("rho",)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare same-state and cross-state donors for the joint-gate purity control."
    )
    parser.add_argument("--samples", default="data/das/joint_gate_test150/triples.csv")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--m-rotation", required=True)
    parser.add_argument("--rho-rotation", required=True)
    parser.add_argument("--site", default="claim_final")
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-token-style", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def attach_same_state_donors(
    rows: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pair every row with another row whose base holds the same (m, rho).

    The donor must come from a different event so the only shared property is
    the pair of causal values.
    """

    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[(str(row["base_m"]), str(row["base_rho"]))].append(row)

    rng = random.Random(seed)
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        pool = pools[(str(row["base_m"]), str(row["base_rho"]))]
        candidates = [
            other
            for other in pool
            if str(other["base_event_id"]) != str(row["base_event_id"])
        ]
        if not candidates:
            dropped += 1
            continue
        donor = candidates[rng.randrange(len(candidates))]
        enriched = dict(row)
        for field in ("prompt", "event_id", "m", "rho", "label"):
            enriched[f"same_state_donor_{field}"] = donor[f"base_{field}"]
        for span in ("claim_span_start", "claim_span_end", "answer_span_start", "answer_span_end"):
            enriched[f"same_state_donor_{span}"] = donor[f"base_{span}"]
        kept.append(enriched)
    return kept, {"rows_dropped_without_donor": dropped}


def donor_view(row: dict[str, Any]) -> dict[str, Any]:
    """Re-key a same-state donor so resolve_token_site can read it as a base."""

    view = dict(row)
    for field in ("prompt", "claim_span_start", "claim_span_end", "answer_span_start", "answer_span_end"):
        view[f"base_{field}"] = row[f"same_state_donor_{field}"]
    return view


def score_logits(torch: Any, logits: Any, enc: dict[str, Any], label_ids: dict[str, int]):
    final = enc["attention_mask"].sum(dim=1) - 1
    index = torch.arange(logits.shape[0], device=logits.device)
    next_logits = logits[index, final]
    outputs = []
    for row_logits in next_logits:
        values = {label: float(row_logits[token_id].detach().cpu()) for label, token_id in label_ids.items()}
        outputs.append({
            **{f"logit_{label}": value for label, value in values.items()},
            "pred_label": max(values, key=values.get),
        })
    return outputs


def collect_donor_states(
    *,
    model: Any,
    layers: Any,
    torch: Any,
    tokenizer: Any,
    device: Any,
    batch_rows: list[dict[str, Any]],
    m_layer: int,
    rho_layer: int,
    site: str,
) -> dict[str, Any]:
    same_views = [donor_view(row) for row in batch_rows]
    specs = [
        ("same_state", [str(view["base_prompt"]) for view in same_views], same_views, "base"),
        ("cross_m", [str(row["m_same_source_prompt"]) for row in batch_rows], batch_rows, "m_same_source"),
        ("cross_rho", [str(row["rho_same_source_prompt"]) for row in batch_rows], batch_rows, "rho_same_source"),
    ]
    texts: list[str] = []
    expanded_rows: list[dict[str, Any]] = []
    expanded_prefixes: list[str] = []
    for _, group_texts, group_rows, prefix in specs:
        texts.extend(group_texts)
        expanded_rows.extend(group_rows)
        expanded_prefixes.extend([prefix] * len(group_rows))
    captured = capture_hidden_at_layers(
        model,
        layers,
        torch,
        tokenizer,
        device,
        texts,
        expanded_rows,
        expanded_prefixes,
        {m_layer, rho_layer},
        site,
    )
    n = len(batch_rows)
    slices = {name: slice(i * n, (i + 1) * n) for i, (name, *_rest) in enumerate(specs)}
    return {
        ("same_state", "m"): captured[m_layer][slices["same_state"]],
        ("same_state", "rho"): captured[rho_layer][slices["same_state"]],
        ("cross_state", "m"): captured[m_layer][slices["cross_m"]],
        ("cross_state", "rho"): captured[rho_layer][slices["cross_rho"]],
    }


def run_condition(
    *,
    model: Any,
    layers: Any,
    torch: Any,
    enc: dict[str, Any],
    positions: Any,
    specs: dict[int, dict[str, Any]],
    label_ids: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    diagnostics: dict[str, list[float]] = defaultdict(list)
    handles = []
    for layer_id, spec in sorted(specs.items()):
        def hook(module, inputs, output, *, current=spec):
            hs = output[0] if isinstance(output, tuple) else output
            pos = positions.to(hs.device)
            index = torch.arange(hs.shape[0], device=hs.device)
            h = hs[index, pos].to(torch.float32)
            bases = [basis.to(hs.device) for basis in current["bases"]]
            coords = [coord.to(hs.device) for coord in current["coords"]]
            patched, diag = constrained_patch(torch, h, bases, coords)
            hs[index, pos] = patched.to(hs.dtype)
            for key in ("coordinate_residual_mean", "coordinate_residual_max", "update_norm_mean"):
                diagnostics[key].append(float(diag[key]))
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        handles.append(layers[layer_id].register_forward_hook(hook))
    try:
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    merged = {key: sum(values) / len(values) for key, values in diagnostics.items() if values}
    return score_logits(torch, logits, enc, label_ids), merged


def prepare_specs(
    *,
    donor_kind: str | None,
    channels: tuple[str, ...],
    m_layer: int,
    rho_layer: int,
    u_m: Any,
    u_rho: Any,
    donors: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    if donor_kind is None or not channels:
        return {}
    by_layer: dict[int, dict[str, list[Any]]] = defaultdict(lambda: {"bases": [], "coords": []})
    for channel in channels:
        layer = m_layer if channel == "m" else rho_layer
        basis = u_m if channel == "m" else u_rho
        hidden = donors[(donor_kind, channel)]
        by_layer[layer]["bases"].append(basis)
        by_layer[layer]["coords"].append(hidden.to(basis.device) @ basis)
    return {layer: dict(spec) for layer, spec in by_layer.items()}


def cluster_interval(rows: list[dict[str, Any]], key: str, *, samples: int, seed: int) -> tuple[float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_event_id"])].append(int(row[key]))
    if samples <= 0:
        value = sum(int(row[key]) for row in rows) / len(rows)
        return value, value
    ids = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        drawn = [ids[rng.randrange(len(ids))] for _ in ids]
        values = [value for cluster in drawn for value in grouped[cluster]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def summarize(scored: list[dict[str, Any]], *, bootstrap_samples: int, seed: int):
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", scored)]
    for m_value in ("0", "1"):
        for rho_value in ("-1", "1"):
            scopes.append((
                f"m{m_value}_rho{rho_value}",
                [row for row in scored if str(row["base_m"]) == m_value and str(row["base_rho"]) == rho_value],
            ))
    output = []
    for scope_index, (scope, scoped) in enumerate(scopes):
        if not scoped:
            continue
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scoped:
            groups[str(row["condition"])].append(row)
        for group_index, (condition, rows) in enumerate(sorted(groups.items())):
            low, high = cluster_interval(
                rows, "unchanged", samples=bootstrap_samples, seed=seed + scope_index * 1000 + group_index
            )
            output.append({
                "scope": scope,
                "condition": condition,
                "donor_kind": rows[0]["donor_kind"],
                "channels": rows[0]["channels"],
                "n": len(rows),
                "n_clusters": len({row["base_event_id"] for row in rows}),
                "unchanged_rate": sum(int(row["unchanged"]) for row in rows) / len(rows),
                "unchanged_ci_low": low,
                "unchanged_ci_high": high,
                "matches_rho_flip": sum(int(row["matches_rho_flip"]) for row in rows) / len(rows),
                "mean_coordinate_residual": sum(float(row["coordinate_residual_mean"]) for row in rows) / len(rows),
                "mean_update_norm": sum(float(row["update_norm_mean"]) for row in rows) / len(rows),
            })
    return output


def flipped_rho_label(m_base: str, rho_base: str) -> str:
    if int(m_base) == 0:
        return "U"
    return "T" if int(rho_base) < 0 else "F"


def main() -> int:
    args = build_parser().parse_args()
    rows = [row for row in read_rows_csv(args.samples) if row.get("split") == args.split]
    if not rows:
        raise ValueError("No rows matched the requested split")
    rows, donor_stats = attach_same_state_donors(rows, seed=args.seed)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"Loaded {len(rows)} rows ({donor_stats['rows_dropped_without_donor']} lacked a same-state donor)")

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

    import numpy as np

    m_raw, m_meta = load_rotation(Path(args.m_rotation), np)
    rho_raw, rho_meta = load_rotation(Path(args.rho_rotation), np)
    m_layer, rho_layer = int(m_meta["layer"]), int(rho_meta["layer"])
    u_m = orthonormalize_basis(torch, torch.tensor(m_raw, device=device))
    u_rho = orthonormalize_basis(torch, torch.tensor(rho_raw, device=device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "same_state_donor_scored.partial.csv"

    scored: list[dict[str, Any]] = []
    batch_list = list(batches(rows, args.eval_batch_size))
    for batch_index, batch_rows in enumerate(
        progress_iter(batch_list, total=len(batch_list), desc="same-state donor"), start=1
    ):
        donors = collect_donor_states(
            model=model,
            layers=layers,
            torch=torch,
            tokenizer=tokenizer,
            device=device,
            batch_rows=batch_rows,
            m_layer=m_layer,
            rho_layer=rho_layer,
            site=args.site,
        )
        texts = [str(row["base_prompt"]) for row in batch_rows]
        enc = encode_to_device(tokenizer, texts, device)
        positions = torch.tensor(
            [resolve_token_site(tokenizer, text, row, "base", args.site) for text, row in zip(texts, batch_rows)],
            dtype=torch.long,
            device=device,
        )
        for condition, donor_kind, channels in CONDITIONS:
            specs = prepare_specs(
                donor_kind=donor_kind,
                channels=channels,
                m_layer=m_layer,
                rho_layer=rho_layer,
                u_m=u_m,
                u_rho=u_rho,
                donors=donors,
            )
            if specs:
                outputs, diagnostics = run_condition(
                    model=model,
                    layers=layers,
                    torch=torch,
                    enc=enc,
                    positions=positions,
                    specs=specs,
                    label_ids=label_tokens.token_ids,
                )
            else:
                with torch.no_grad():
                    logits = model(**enc, use_cache=False).logits
                outputs = score_logits(torch, logits, enc, label_tokens.token_ids)
                diagnostics = {}
            for row, output in zip(batch_rows, outputs):
                expected = str(row["base_label"])
                scored.append({
                    "sample_id": row["sample_id"],
                    "base_event_id": row["base_event_id"],
                    "same_state_donor_event_id": row["same_state_donor_event_id"],
                    "cell_type": row["cell_type"],
                    "condition": condition,
                    "donor_kind": donor_kind or "none",
                    "channels": "+".join(channels) if channels else "none",
                    "base_m": row["base_m"],
                    "base_rho": row["base_rho"],
                    "expected_label": expected,
                    "pred_label": output["pred_label"],
                    "unchanged": int(output["pred_label"] == expected),
                    "matches_rho_flip": int(
                        output["pred_label"] == flipped_rho_label(str(row["base_m"]), str(row["base_rho"]))
                    ),
                    "logit_T": output["logit_T"],
                    "logit_F": output["logit_F"],
                    "logit_U": output["logit_U"],
                    "coordinate_residual_mean": diagnostics.get("coordinate_residual_mean", 0.0),
                    "update_norm_mean": diagnostics.get("update_norm_mean", 0.0),
                })
        if args.checkpoint_every and batch_index % args.checkpoint_every == 0:
            write_rows_csv(scored, partial_path)

    write_rows_csv(scored, output_dir / "same_state_donor_scored.csv")
    summary = summarize(scored, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    write_rows_csv(summary, output_dir / "same_state_donor_summary.csv")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            to_jsonable({
                "model_name": args.model_name,
                "samples": args.samples,
                "site": args.site,
                "m_rotation": args.m_rotation,
                "rho_rotation": args.rho_rotation,
                "m_layer": m_layer,
                "rho_layer": rho_layer,
                "m_rank": int(u_m.shape[1]),
                "rho_rank": int(u_rho.shape[1]),
                "n_rows": len(rows),
                "conditions": [name for name, _, _ in CONDITIONS],
                **donor_stats,
            }),
            indent=2,
        )
    )

    print("\ncondition                 unchanged   matches_rho_flip")
    for entry in summary:
        if entry["scope"] != "all":
            continue
        print(
            f"{entry['condition']:<24} {entry['unchanged_rate']:>8.3f}   {entry['matches_rho_flip']:>8.3f}"
        )
    print(f"\nWrote {output_dir}/same_state_donor_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
