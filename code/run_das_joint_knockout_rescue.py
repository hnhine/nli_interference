"""Test natural use of fixed ``m``/``rho`` subspaces by knockout and downstream rescue.

The evaluator corrupts one or both causal coordinates at a claim-final layer,
then optionally restores clean-base coordinates at a later answer-token layer.
No subspace is trained by this script.
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
from interference_suite.joint_gate_intervention import (
    constrained_patch,
    orthonormalize_basis,
    random_orthonormal_basis,
)
from interference_suite.joint_gate_rescue import RESCUE_CONDITIONS, RescueCondition, expected_label
from interference_suite.model import DEFAULT_CACHE_DIR, progress_iter, resolve_label_tokens
from run_das_ablation import get_decoder_layers
from run_das_joint_gate import load_rotation


def batches(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def score_logits(torch: Any, logits: Any, enc: dict[str, Any], label_ids: dict[str, int]):
    final = enc["attention_mask"].sum(dim=1) - 1
    b = torch.arange(logits.shape[0], device=logits.device)
    next_logits = logits[b, final]
    outputs = []
    for row_logits in next_logits:
        values = {label: float(row_logits[token_id].detach().cpu()) for label, token_id in label_ids.items()}
        outputs.append({
            **{f"logit_{label}": value for label, value in values.items()},
            "pred_label": max(values, key=values.get),
        })
    return outputs


def capture_claim_donors(
    *,
    model: Any,
    layers: Any,
    torch: Any,
    tokenizer: Any,
    device: Any,
    rows: list[dict[str, Any]],
    claim_layer: int,
) -> dict[str, Any]:
    prefixes = ("rho_same_source", "m_same_source")
    texts: list[str] = []
    expanded_rows: list[dict[str, Any]] = []
    expanded_prefixes: list[str] = []
    for prefix in prefixes:
        texts.extend(str(row[f"{prefix}_prompt"]) for row in rows)
        expanded_rows.extend(rows)
        expanded_prefixes.extend([prefix] * len(rows))
    enc = encode_to_device(tokenizer, texts, device)
    positions = torch.tensor(
        [
            resolve_token_site(tokenizer, text, row, prefix, "claim_final")
            for text, row, prefix in zip(texts, expanded_rows, expanded_prefixes)
        ],
        dtype=torch.long,
        device=device,
    )
    captured: dict[str, Any] = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        b = torch.arange(hs.shape[0], device=hs.device)
        captured["hidden"] = hs[b, positions.to(hs.device)].detach().to(torch.float32)
        return output

    handle = layers[claim_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        handle.remove()
    hidden = captured["hidden"]
    n = len(rows)
    return {
        "m_opposite": hidden[:n],       # rho_same_source: opposite m, same rho
        "rho_same": hidden[:n],        # same source, projected through U_rho
        "rho_opposite": hidden[n:],    # m_same_source: same m, opposite rho
        "m_same": hidden[n:],          # same source, projected through U_m
    }


def run_clean(
    *,
    model: Any,
    layers: Any,
    torch: Any,
    tokenizer: Any,
    device: Any,
    rows: list[dict[str, Any]],
    claim_layer: int,
    answer_layer: int,
    label_ids: dict[str, int],
) -> tuple[dict[str, Any], Any, Any, Any, Any, list[dict[str, Any]]]:
    texts = [str(row["base_prompt"]) for row in rows]
    enc = encode_to_device(tokenizer, texts, device)
    claim_positions = torch.tensor(
        [resolve_token_site(tokenizer, text, row, "base", "claim_final") for text, row in zip(texts, rows)],
        dtype=torch.long,
        device=device,
    )
    answer_positions = torch.tensor(
        [resolve_token_site(tokenizer, text, row, "base", "answer_token") for text, row in zip(texts, rows)],
        dtype=torch.long,
        device=device,
    )
    captured: dict[str, Any] = {}

    def claim_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        b = torch.arange(hs.shape[0], device=hs.device)
        captured["claim"] = hs[b, claim_positions.to(hs.device)].detach().to(torch.float32)
        return output

    def answer_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        b = torch.arange(hs.shape[0], device=hs.device)
        captured["answer"] = hs[b, answer_positions.to(hs.device)].detach().to(torch.float32)
        return output

    handles = [layers[claim_layer].register_forward_hook(claim_hook)]
    handles.append(layers[answer_layer].register_forward_hook(answer_hook))
    try:
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    return (
        enc,
        claim_positions,
        answer_positions,
        captured["claim"],
        captured["answer"],
        score_logits(torch, logits, enc, label_ids),
    )


def condition_specs(
    *,
    condition: RescueCondition,
    claim_layer: int,
    answer_layer: int,
    u_m_claim: Any,
    u_rho_claim: Any,
    u_m_answer: Any,
    u_rho_answer: Any,
    random_m_claim: Any,
    random_rho_claim: Any,
    random_m_answer: Any,
    random_rho_answer: Any,
    donors: dict[str, Any],
    clean_claim: Any,
    clean_answer: Any,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    clean_m_claim = clean_claim @ u_m_claim
    clean_rho_claim = clean_claim @ u_rho_claim
    claim_map = {
        # A single-channel intervention explicitly preserves the other learned
        # channel because independently trained DAS subspaces need not be orthogonal.
        "m_flip": [
            (u_m_claim, donors["m_opposite"] @ u_m_claim),
            (u_rho_claim, clean_rho_claim),
        ],
        "rho_flip": [
            (u_m_claim, clean_m_claim),
            (u_rho_claim, donors["rho_opposite"] @ u_rho_claim),
        ],
        "both_flip": [
            (u_m_claim, donors["m_opposite"] @ u_m_claim),
            (u_rho_claim, donors["rho_opposite"] @ u_rho_claim),
        ],
        "m_same": [
            (u_m_claim, donors["m_same"] @ u_m_claim),
            (u_rho_claim, clean_rho_claim),
        ],
        "rho_same": [
            (u_m_claim, clean_m_claim),
            (u_rho_claim, donors["rho_same"] @ u_rho_claim),
        ],
        "both_same": [
            (u_m_claim, donors["m_same"] @ u_m_claim),
            (u_rho_claim, donors["rho_same"] @ u_rho_claim),
        ],
        "random_m_flip": [
            (random_m_claim, donors["m_opposite"] @ random_m_claim),
            (u_m_claim, clean_m_claim),
            (u_rho_claim, clean_rho_claim),
        ],
        "random_rho_flip": [
            (random_rho_claim, donors["rho_opposite"] @ random_rho_claim),
            (u_m_claim, clean_m_claim),
            (u_rho_claim, clean_rho_claim),
        ],
        "random_both_flip": [
            (random_m_claim, donors["m_opposite"] @ random_m_claim),
            (random_rho_claim, donors["rho_opposite"] @ random_rho_claim),
            (u_m_claim, clean_m_claim),
            (u_rho_claim, clean_rho_claim),
        ],
    }
    if condition.claim_patch != "none":
        pairs = claim_map[condition.claim_patch]
        specs.append({
            "layer": claim_layer,
            "site": "claim",
            "bases": [pair[0] for pair in pairs],
            "coords": [pair[1] for pair in pairs],
        })

    answer_map = {
        # None means preserve the coordinate present immediately before the
        # downstream patch, so selective restore changes exactly one channel.
        "m": [
            (u_m_answer, clean_answer @ u_m_answer),
            (u_rho_answer, None),
        ],
        "rho": [
            (u_m_answer, None),
            (u_rho_answer, clean_answer @ u_rho_answer),
        ],
        "both": [
            (u_m_answer, clean_answer @ u_m_answer),
            (u_rho_answer, clean_answer @ u_rho_answer),
        ],
        "random_both": [
            (random_m_answer, clean_answer @ random_m_answer),
            (random_rho_answer, clean_answer @ random_rho_answer),
            (u_m_answer, None),
            (u_rho_answer, None),
        ],
    }
    if condition.answer_restore != "none":
        pairs = answer_map[condition.answer_restore]
        specs.append({
            "layer": answer_layer,
            "site": "answer",
            "bases": [pair[0] for pair in pairs],
            "coords": [pair[1] for pair in pairs],
        })
    return specs


def run_condition(
    *,
    model: Any,
    layers: Any,
    torch: Any,
    enc: dict[str, Any],
    claim_positions: Any,
    answer_positions: Any,
    specs: list[dict[str, Any]],
    label_ids: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        by_layer[int(spec["layer"])].append(spec)
    diagnostics: dict[str, list[float]] = defaultdict(list)
    handles = []
    for layer_id, layer_specs in sorted(by_layer.items()):
        def hook(module, inputs, output, *, current_specs=layer_specs):
            hs = output[0] if isinstance(output, tuple) else output
            for spec in current_specs:
                positions = claim_positions if spec["site"] == "claim" else answer_positions
                pos = positions.to(hs.device)
                b = torch.arange(hs.shape[0], device=hs.device)
                h = hs[b, pos].to(torch.float32)
                bases = [basis.to(hs.device) for basis in spec["bases"]]
                coords = [
                    h @ basis if coord is None else coord.to(hs.device)
                    for basis, coord in zip(bases, spec["coords"])
                ]
                patched, diag = constrained_patch(torch, h, bases, coords)
                hs[b, pos] = patched.to(hs.dtype)
                for key in ("coordinate_residual_mean", "coordinate_residual_max", "update_norm_mean"):
                    diagnostics[f"{spec['site']}_{key}"].append(float(diag[key]))
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


def cluster_interval(
    rows: list[dict[str, Any]],
    key: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
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
    output = []
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", scored)]
    for m_value in (0, 1):
        for rho_value in (-1, 1):
            scopes.append((
                f"m{m_value}_rho{rho_value:+d}",
                [row for row in scored if int(row["m_base"]) == m_value and int(row["rho_base"]) == rho_value],
            ))
    for scope_index, (scope, scoped) in enumerate(scopes):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scoped:
            groups[str(row["condition"])].append(row)
        for group_index, (condition, rows) in enumerate(sorted(groups.items())):
            lo, hi = cluster_interval(
                rows,
                "correct_expected",
                samples=bootstrap_samples,
                seed=seed + scope_index * 1000 + group_index,
            )
            output.append({
                "scope": scope,
                "condition": condition,
                "condition_family": rows[0]["condition_family"],
                "n": len(rows),
                "n_clusters": len({row["base_event_id"] for row in rows}),
                "IIA": sum(int(row["correct_expected"]) for row in rows) / len(rows),
                "IIA_ci_low": lo,
                "IIA_ci_high": hi,
                "base_accuracy": sum(int(row["correct_base"]) for row in rows) / len(rows),
                "mean_expected_margin": sum(float(row["expected_margin"]) for row in rows) / len(rows),
                "mean_claim_residual": sum(float(row["claim_coordinate_residual_mean"]) for row in rows) / len(rows),
                "mean_answer_residual": sum(float(row["answer_coordinate_residual_mean"]) for row in rows) / len(rows),
            })
    return output


def mediation_metrics(summary: list[dict[str, Any]]):
    output = []
    for scope in sorted({row["scope"] for row in summary}):
        lookup = {row["condition"]: row for row in summary if row["scope"] == scope}
        none = lookup["none"]["base_accuracy"]
        row = {"scope": scope, "none_base_accuracy": none}
        for prefix, corrupt, rescue in (
            ("m", "claim_m_flip", "claim_m_flip_answer_m_restore"),
            ("rho", "claim_rho_flip", "claim_rho_flip_answer_rho_restore"),
            ("both", "claim_both_flip", "claim_both_flip_answer_both_restore"),
        ):
            corrupt_base = float(lookup[corrupt]["base_accuracy"])
            rescue_base = float(lookup[rescue]["base_accuracy"])
            denominator = none - corrupt_base
            row[f"{prefix}_corrupt_IIA"] = lookup[corrupt]["IIA"]
            row[f"{prefix}_necessity_drop"] = none - corrupt_base
            row[f"{prefix}_rescue_gain"] = rescue_base - corrupt_base
            row[f"{prefix}_rescue_fraction"] = (
                (rescue_base - corrupt_base) / denominator if abs(denominator) > 1e-12 else float("nan")
            )
        both_base = float(lookup["claim_both_flip"]["base_accuracy"])
        random_base = float(lookup["claim_both_flip_answer_random_restore"]["base_accuracy"])
        row["random_rescue_gain"] = random_base - both_base
        row["rescue_specificity"] = row["both_rescue_gain"] - row["random_rescue_gain"]
        row["selective_m_restore_IIA"] = lookup["claim_both_flip_answer_m_restore"]["IIA"]
        row["selective_rho_restore_IIA"] = lookup["claim_both_flip_answer_rho_restore"]["IIA"]
        row["same_value_both_IIA"] = lookup["claim_both_same"]["IIA"]
        row["answer_restore_only_IIA"] = lookup["answer_both_restore_only"]["IIA"]
        output.append(row)
    return output


def main() -> int:
    args = build_parser().parse_args()
    rows = [row for row in read_rows_csv(args.samples) if row.get("split") == args.split]
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No rows matched")
    print(f"Loaded {len(rows)} knockout-rescue rows")

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

    rotations = {}
    metadata = {}
    for key, path in {
        "m_claim": args.m_claim_rotation,
        "rho_claim": args.rho_claim_rotation,
        "m_answer": args.m_answer_rotation,
        "rho_answer": args.rho_answer_rotation,
    }.items():
        raw, meta = load_rotation(Path(path), np)
        if int(raw.shape[0]) != hidden_size:
            raise ValueError(f"{key} hidden size mismatch: {raw.shape[0]} != {hidden_size}")
        rotations[key] = orthonormalize_basis(torch, torch.tensor(raw, device=device))
        metadata[key] = meta
    claim_layers = {int(metadata[key]["layer"]) for key in ("m_claim", "rho_claim")}
    answer_layers = {int(metadata[key]["layer"]) for key in ("m_answer", "rho_answer")}
    if len(claim_layers) != 1 or len(answer_layers) != 1:
        raise ValueError("m and rho rotations must share a layer within each site")
    claim_layer = claim_layers.pop()
    answer_layer = answer_layers.pop()
    if answer_layer < claim_layer:
        raise ValueError("answer rescue layer must not precede claim corruption layer")

    random_m_claim = random_orthonormal_basis(
        torch, hidden_size, rotations["m_claim"].shape[1], device=device, seed=10_000 + args.random_seed
    )
    random_rho_claim = random_orthonormal_basis(
        torch, hidden_size, rotations["rho_claim"].shape[1], device=device, seed=20_000 + args.random_seed
    )
    random_m_answer = random_orthonormal_basis(
        torch, hidden_size, rotations["m_answer"].shape[1], device=device, seed=30_000 + args.random_seed
    )
    random_rho_answer = random_orthonormal_basis(
        torch, hidden_size, rotations["rho_answer"].shape[1], device=device, seed=40_000 + args.random_seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "knockout_rescue_scored.partial.csv"
    scored: list[dict[str, Any]] = []
    batch_list = list(batches(rows, args.eval_batch_size))
    for batch_index, batch_rows in enumerate(
        progress_iter(batch_list, total=len(batch_list), desc="joint knockout-rescue"), start=1
    ):
        donors = capture_claim_donors(
            model=model,
            layers=layers,
            torch=torch,
            tokenizer=tokenizer,
            device=device,
            rows=batch_rows,
            claim_layer=claim_layer,
        )
        enc, claim_positions, answer_positions, clean_claim, clean_answer, clean_predictions = run_clean(
            model=model,
            layers=layers,
            torch=torch,
            tokenizer=tokenizer,
            device=device,
            rows=batch_rows,
            claim_layer=claim_layer,
            answer_layer=answer_layer,
            label_ids=label_tokens.token_ids,
        )
        for condition in RESCUE_CONDITIONS:
            if condition.name == "none":
                predictions = clean_predictions
                diagnostics = {}
            else:
                specs = condition_specs(
                    condition=condition,
                    claim_layer=claim_layer,
                    answer_layer=answer_layer,
                    u_m_claim=rotations["m_claim"],
                    u_rho_claim=rotations["rho_claim"],
                    u_m_answer=rotations["m_answer"],
                    u_rho_answer=rotations["rho_answer"],
                    random_m_claim=random_m_claim,
                    random_rho_claim=random_rho_claim,
                    random_m_answer=random_m_answer,
                    random_rho_answer=random_rho_answer,
                    donors=donors,
                    clean_claim=clean_claim,
                    clean_answer=clean_answer,
                )
                predictions, diagnostics = run_condition(
                    model=model,
                    layers=layers,
                    torch=torch,
                    enc=enc,
                    claim_positions=claim_positions,
                    answer_positions=answer_positions,
                    specs=specs,
                    label_ids=label_tokens.token_ids,
                )
            for source_row, prediction in zip(batch_rows, predictions):
                target = expected_label(condition, int(source_row["m_base"]), int(source_row["rho_base"]))
                base_label = str(source_row["expected_none"])
                logits = {label: float(prediction[f"logit_{label}"]) for label in ("T", "F", "U")}
                pred_label = str(prediction["pred_label"])
                scored.append({
                    "sample_id": source_row["sample_id"],
                    "base_event_id": source_row["base_event_id"],
                    "cell_type": source_row["cell_type"],
                    "m_base": source_row["m_base"],
                    "rho_base": source_row["rho_base"],
                    "condition": condition.name,
                    "condition_family": condition.family,
                    "expected_label": target,
                    "base_label": base_label,
                    "pred_label": pred_label,
                    "correct_expected": int(pred_label == target),
                    "correct_base": int(pred_label == base_label),
                    "expected_margin": logits[target] - max(value for label, value in logits.items() if label != target),
                    **logits,
                    "claim_coordinate_residual_mean": diagnostics.get("claim_coordinate_residual_mean", 0.0),
                    "answer_coordinate_residual_mean": diagnostics.get("answer_coordinate_residual_mean", 0.0),
                    "claim_update_norm_mean": diagnostics.get("claim_update_norm_mean", 0.0),
                    "answer_update_norm_mean": diagnostics.get("answer_update_norm_mean", 0.0),
                })
        if args.checkpoint_every > 0 and (
            batch_index % args.checkpoint_every == 0 or batch_index == len(batch_list)
        ):
            write_rows_csv(scored, partial_path)

    scored_path = write_rows_csv(scored, output_dir / "knockout_rescue_scored.csv")
    summary = summarize(scored, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    summary_path = write_rows_csv(summary, output_dir / "knockout_rescue_summary.csv")
    metrics_path = write_rows_csv(mediation_metrics(summary), output_dir / "mediation_metrics.csv")
    run_metadata = {
        "model_name": args.model_name,
        "samples": args.samples,
        "n_rows": len(rows),
        "claim_layer": claim_layer,
        "answer_layer": answer_layer,
        "m_claim_rotation": args.m_claim_rotation,
        "rho_claim_rotation": args.rho_claim_rotation,
        "m_answer_rotation": args.m_answer_rotation,
        "rho_answer_rotation": args.rho_answer_rotation,
        "random_seed": args.random_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "base_event_id",
        "overlap_preservation": True,
        "conditions": [condition.name for condition in RESCUE_CONDITIONS],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(to_jsonable(run_metadata), indent=2))
    print(f"Wrote {scored_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/das/joint_gate_test150/triples.csv")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--m-claim-rotation", required=True)
    parser.add_argument("--rho-claim-rotation", required=True)
    parser.add_argument("--m-answer-rotation", required=True)
    parser.add_argument("--rho-answer-rotation", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5)
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


if __name__ == "__main__":
    raise SystemExit(main())
