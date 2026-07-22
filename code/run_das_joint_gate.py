"""Evaluate compositional control from fixed m and rho DAS subspaces.

This script never trains a rotation.  It loads one learned m basis and one
learned rho basis, collects donor coordinates from independent prompts, and
patches a base prompt under single, simultaneous, purity, random, and
sequential-order conditions.
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
    sequential_patch,
)
from interference_suite.model import DEFAULT_CACHE_DIR, progress_iter, resolve_label_tokens
from run_das_ablation import get_decoder_layers


CONCEPTUAL_CONDITIONS = (
    "none",
    "m_only",
    "rho_only",
    "joint_constrained",
    "joint_same_value",
    "joint_random_m",
    "joint_random_rho",
    "m_then_rho",
    "rho_then_m",
)


def batches(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def load_rotation(path: Path, np: Any) -> tuple[Any, dict[str, Any]]:
    metadata_path = path / "rotation_weight_metadata.json"
    weight_path = path / "rotation_weight.npy"
    if not metadata_path.exists() or not weight_path.exists():
        raise FileNotFoundError(f"Rotation directory is incomplete: {path}")
    metadata = json.loads(metadata_path.read_text())
    return np.load(weight_path), metadata


def stratified_limit(
    rows: list[dict[str, Any]],
    max_rows_per_cell_regime: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if not max_rows_per_cell_regime or max_rows_per_cell_regime <= 0:
        return rows
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["cell_type"]), str(row["rho_source_m"]))].append(row)
    kept: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        kept.extend(group[:max_rows_per_cell_regime])
    return sorted(kept, key=lambda row: int(row["row_id"]))


def capture_hidden_at_layers(
    model: Any,
    layers: Any,
    torch: Any,
    tokenizer: Any,
    device: Any,
    texts: list[str],
    rows: list[dict[str, Any]],
    prefixes: list[str],
    layer_ids: set[int],
    site: str,
) -> dict[int, Any]:
    """Capture only requested block outputs, avoiding all-layer hidden-state storage."""

    enc = encode_to_device(tokenizer, texts, device)
    positions = torch.tensor(
        [
            resolve_token_site(tokenizer, text, row, prefix, site)
            for text, row, prefix in zip(texts, rows, prefixes)
        ],
        dtype=torch.long,
        device=device,
    )
    captured: dict[int, Any] = {}
    handles = []
    for layer_id in sorted(layer_ids):
        def hook(module, inputs, output, *, selected_layer=layer_id):
            hs = output[0] if isinstance(output, tuple) else output
            pos = positions.to(hs.device)
            b = torch.arange(hs.shape[0], device=hs.device)
            captured[selected_layer] = hs[b, pos].detach().to(torch.float32)
            return output

        handles.append(layers[layer_id].register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = layer_ids - set(captured)
    if missing:
        raise RuntimeError(f"Failed to capture layers: {sorted(missing)}")
    return captured


def collect_donor_states(
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
    prefixes = ("m_source", "rho_source", "m_same_source", "rho_same_source")
    texts: list[str] = []
    expanded_rows: list[dict[str, Any]] = []
    expanded_prefixes: list[str] = []
    for prefix in prefixes:
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
        {m_layer, rho_layer},
        site,
    )
    n = len(batch_rows)
    slices = {prefix: slice(index * n, (index + 1) * n) for index, prefix in enumerate(prefixes)}
    return {
        "m": captured[m_layer][slices["m_source"]],
        "rho": captured[rho_layer][slices["rho_source"]],
        "m_same": captured[m_layer][slices["m_same_source"]],
        "rho_same": captured[rho_layer][slices["rho_same_source"]],
    }


def prepare_interventions(
    *,
    condition: str,
    m_layer: int,
    rho_layer: int,
    u_m: Any,
    u_rho: Any,
    donor_h: dict[str, Any],
    random_m: Any | None = None,
    random_rho: Any | None = None,
) -> dict[int, dict[str, Any]]:
    z_m = donor_h["m"].to(u_m.device) @ u_m
    z_rho = donor_h["rho"].to(u_rho.device) @ u_rho
    z_m_same = donor_h["m_same"].to(u_m.device) @ u_m
    z_rho_same = donor_h["rho_same"].to(u_rho.device) @ u_rho

    specs: list[tuple[int, Any, Any]]
    method = "constrained"
    if condition == "m_only":
        specs = [(m_layer, u_m, z_m)]
    elif condition == "rho_only":
        specs = [(rho_layer, u_rho, z_rho)]
    elif condition == "joint_constrained":
        specs = [(m_layer, u_m, z_m), (rho_layer, u_rho, z_rho)]
    elif condition == "joint_same_value":
        specs = [(m_layer, u_m, z_m_same), (rho_layer, u_rho, z_rho_same)]
    elif condition == "joint_random_m":
        if random_m is None:
            raise ValueError("joint_random_m requires a random m basis")
        z_random_m = donor_h["m"].to(random_m.device) @ random_m
        specs = [(m_layer, random_m, z_random_m), (rho_layer, u_rho, z_rho)]
    elif condition == "joint_random_rho":
        if random_rho is None:
            raise ValueError("joint_random_rho requires a random rho basis")
        z_random_rho = donor_h["rho"].to(random_rho.device) @ random_rho
        specs = [(m_layer, u_m, z_m), (rho_layer, random_rho, z_random_rho)]
    elif condition == "m_then_rho":
        specs = [(m_layer, u_m, z_m), (rho_layer, u_rho, z_rho)]
        method = "sequential"
    elif condition == "rho_then_m":
        specs = [(rho_layer, u_rho, z_rho), (m_layer, u_m, z_m)]
        method = "sequential"
    else:
        raise ValueError(f"Unknown intervention condition: {condition}")

    by_layer: dict[int, dict[str, Any]] = {}
    for layer, basis, coords in specs:
        if layer not in by_layer:
            by_layer[layer] = {"method": method, "bases": [], "coords": []}
        elif by_layer[layer]["method"] != method:
            raise ValueError("Mixed patch methods at one layer")
        by_layer[layer]["bases"].append(basis)
        by_layer[layer]["coords"].append(coords)
    return by_layer


def run_base_condition(
    *,
    model: Any,
    layers: Any,
    torch: Any,
    enc: dict[str, Any],
    positions: Any,
    interventions: dict[int, dict[str, Any]],
    label_ids: dict[str, int],
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    diagnostics: list[dict[str, Any]] = []
    handles = []
    for layer_id, spec in sorted(interventions.items()):
        def hook(module, inputs, output, *, patch_spec=spec):
            hs = output[0] if isinstance(output, tuple) else output
            pos = positions.to(hs.device)
            b = torch.arange(hs.shape[0], device=hs.device)
            h = hs[b, pos].to(torch.float32)
            bases = [basis.to(hs.device) for basis in patch_spec["bases"]]
            coords = [coords.to(hs.device) for coords in patch_spec["coords"]]
            if patch_spec["method"] == "constrained":
                patched, diag = constrained_patch(torch, h, bases, coords)
            else:
                patched, diag = sequential_patch(torch, h, bases, coords)
            diagnostics.append(diag)
            hs[b, pos] = patched.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        handles.append(layers[layer_id].register_forward_hook(hook))
    try:
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()

    final = enc["attention_mask"].sum(dim=1) - 1
    b = torch.arange(logits.shape[0], device=logits.device)
    next_logits = logits[b, final]
    outputs: list[dict[str, float | str]] = []
    for row_logits in next_logits:
        values = {label: float(row_logits[token_id].detach().cpu()) for label, token_id in label_ids.items()}
        outputs.append({**{f"logit_{key}": value for key, value in values.items()}, "pred_label": max(values, key=values.get)})

    if diagnostics:
        merged = {
            key: sum(float(diag.get(key, 0.0)) for diag in diagnostics) / len(diagnostics)
            for key in (
                "coordinate_residual_mean",
                "coordinate_residual_max",
                "update_norm_mean",
                "update_norm_max",
                "gram_condition_number",
            )
        }
    else:
        merged = {key: 0.0 for key in (
            "coordinate_residual_mean",
            "coordinate_residual_max",
            "update_norm_mean",
            "update_norm_max",
            "gram_condition_number",
        )}
    return outputs, merged


def expected_key(condition: str) -> str:
    if condition == "none":
        return "expected_none"
    if condition == "m_only":
        return "expected_m_only"
    if condition == "rho_only":
        return "expected_rho_only"
    if condition == "joint_same_value":
        return "expected_same_value"
    return "expected_joint"


def bootstrap_interval(values: list[int], *, samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if samples <= 0:
        value = sum(values) / len(values)
        return value, value
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def summarize_scored(scored: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for scope in ("all", "strict"):
        scoped = scored if scope == "all" else [row for row in scored if int(row["strict_assembly"]) == 1]
        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in scoped:
            groups[(str(row["condition_family"]), str(row["condition"]), str(row["cell_type"]), str(row["rho_source_m"]))].append(row)
        for group_index, (key, rows) in enumerate(sorted(groups.items())):
            family, condition, cell, rho_source_m = key
            expected_values = [int(row["correct_expected"]) for row in rows]
            joint_values = [int(row["correct_joint_target"]) for row in rows]
            lo, hi = bootstrap_interval(expected_values, samples=bootstrap_samples, seed=seed + group_index)
            summary.append({
                "scope": scope,
                "condition_family": family,
                "condition": condition,
                "cell_type": cell,
                "rho_source_m": rho_source_m,
                "n": len(rows),
                "IIA": sum(expected_values) / len(expected_values),
                "IIA_ci_low": lo,
                "IIA_ci_high": hi,
                "joint_target_accuracy": sum(joint_values) / len(joint_values),
                "mean_expected_margin": sum(float(row["expected_margin"]) for row in rows) / len(rows),
                "mean_coordinate_residual": sum(float(row["coordinate_residual_mean"]) for row in rows) / len(rows),
                "mean_update_norm": sum(float(row["update_norm_mean"]) for row in rows) / len(rows),
            })
    return summary


def synergy_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["scope"], row["condition_family"], row["cell_type"], row["rho_source_m"]): row
        for row in summary
    }
    out: list[dict[str, Any]] = []
    cells = sorted({row["cell_type"] for row in summary})
    regimes = sorted({row["rho_source_m"] for row in summary})
    for scope in ("all", "strict"):
        for cell in cells:
            for regime in regimes:
                joint = lookup.get((scope, "joint_constrained", cell, regime))
                m_only = lookup.get((scope, "m_only", cell, regime))
                rho_only = lookup.get((scope, "rho_only", cell, regime))
                if not joint or not m_only or not rho_only:
                    continue
                best_single = max(float(m_only["joint_target_accuracy"]), float(rho_only["joint_target_accuracy"]))
                out.append({
                    "scope": scope,
                    "cell_type": cell,
                    "rho_source_m": regime,
                    "joint_IIA": joint["IIA"],
                    "m_only_joint_target_accuracy": m_only["joint_target_accuracy"],
                    "rho_only_joint_target_accuracy": rho_only["joint_target_accuracy"],
                    "best_single_joint_target_accuracy": best_single,
                    "joint_gain": float(joint["IIA"]) - best_single,
                })
    return out


def main() -> int:
    args = build_parser().parse_args()
    rows = [row for row in read_rows_csv(args.samples) if row.get("split") == args.split]
    rows = stratified_limit(rows, args.max_rows_per_cell_regime, seed=args.seed)
    if not rows:
        raise ValueError("No joint-gate rows matched")
    print(f"Loaded {len(rows)} joint-gate rows")

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

    m_raw, m_meta = load_rotation(Path(args.m_rotation), np)
    rho_raw, rho_meta = load_rotation(Path(args.rho_rotation), np)
    if int(m_raw.shape[0]) != hidden_size or int(rho_raw.shape[0]) != hidden_size:
        raise ValueError(f"Rotation hidden sizes do not match model hidden_size={hidden_size}")
    m_layer = int(m_meta["layer"])
    rho_layer = int(rho_meta["layer"])
    if args.composition_mode == "common" and m_layer != rho_layer:
        raise ValueError(f"common mode requires equal layers, got m=L{m_layer}, rho=L{rho_layer}")
    u_m = orthonormalize_basis(torch, torch.tensor(m_raw, device=device))
    u_rho = orthonormalize_basis(torch, torch.tensor(rho_raw, device=device))
    m_rank, rho_rank = int(u_m.shape[1]), int(u_rho.shape[1])

    random_bases: dict[int, tuple[Any, Any]] = {}
    for random_seed in args.random_seeds:
        random_bases[int(random_seed)] = (
            random_orthonormal_basis(torch, hidden_size, m_rank, device=device, seed=10_000 + int(random_seed)),
            random_orthonormal_basis(torch, hidden_size, rho_rank, device=device, seed=20_000 + int(random_seed)),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "joint_gate_scored.partial.csv"
    scored: list[dict[str, Any]] = []
    batch_list = list(batches(rows, args.eval_batch_size))
    for batch_index, batch_rows in enumerate(
        progress_iter(batch_list, total=len(batch_list), desc="joint gate"),
        start=1,
    ):
        donor_h = collect_donor_states(
            model,
            layers,
            torch,
            tokenizer,
            device,
            batch_rows,
            m_layer,
            rho_layer,
            args.site,
        )
        base_texts = [str(row["base_prompt"]) for row in batch_rows]
        base_enc = encode_to_device(tokenizer, base_texts, device)
        base_positions = torch.tensor(
            [resolve_token_site(tokenizer, text, row, "base", args.site) for text, row in zip(base_texts, batch_rows)],
            dtype=torch.long,
            device=device,
        )

        condition_runs: list[tuple[str, str, int | None, dict[int, dict[str, Any]]]] = [
            ("none", "none", None, {}),
        ]
        for condition in ("m_only", "rho_only", "joint_constrained", "joint_same_value"):
            condition_runs.append((condition, condition, None, prepare_interventions(
                condition=condition,
                m_layer=m_layer,
                rho_layer=rho_layer,
                u_m=u_m,
                u_rho=u_rho,
                donor_h=donor_h,
            )))
        if m_layer == rho_layer:
            for condition in ("m_then_rho", "rho_then_m"):
                condition_runs.append((condition, condition, None, prepare_interventions(
                    condition=condition,
                    m_layer=m_layer,
                    rho_layer=rho_layer,
                    u_m=u_m,
                    u_rho=u_rho,
                    donor_h=donor_h,
                )))
        for random_seed, (random_m, random_rho) in random_bases.items():
            for family in ("joint_random_m", "joint_random_rho"):
                condition_runs.append((f"{family}_s{random_seed}", family, random_seed, prepare_interventions(
                    condition=family,
                    m_layer=m_layer,
                    rho_layer=rho_layer,
                    u_m=u_m,
                    u_rho=u_rho,
                    donor_h=donor_h,
                    random_m=random_m,
                    random_rho=random_rho,
                )))

        for condition, family, random_seed, interventions in condition_runs:
            predictions, diagnostics = run_base_condition(
                model=model,
                layers=layers,
                torch=torch,
                enc=base_enc,
                positions=base_positions,
                interventions=interventions,
                label_ids=label_tokens.token_ids,
            )
            target_key = expected_key(family)
            for source_row, prediction in zip(batch_rows, predictions):
                expected = str(source_row[target_key])
                joint_target = str(source_row["expected_joint"])
                logits = {label: float(prediction[f"logit_{label}"]) for label in ("T", "F", "U")}
                pred_label = str(prediction["pred_label"])
                scored.append({
                    "sample_id": source_row["sample_id"],
                    "cell_type": source_row["cell_type"],
                    "cell_family": source_row["cell_family"],
                    "rho_source_m": source_row["rho_source_m"],
                    "strict_assembly": source_row["strict_assembly"],
                    "m_base": source_row["m_base"],
                    "rho_base": source_row["rho_base"],
                    "m_donor": source_row["m_donor"],
                    "rho_donor": source_row["rho_donor"],
                    "base_label": source_row["base_label"],
                    "m_source_label": source_row["m_source_label"],
                    "rho_source_label": source_row["rho_source_label"],
                    "condition": condition,
                    "condition_family": family,
                    "random_seed": "" if random_seed is None else random_seed,
                    "expected_label": expected,
                    "joint_target_label": joint_target,
                    "pred_label": pred_label,
                    "correct_expected": int(pred_label == expected),
                    "correct_joint_target": int(pred_label == joint_target),
                    "logit_T": logits["T"],
                    "logit_F": logits["F"],
                    "logit_U": logits["U"],
                    "expected_margin": logits[expected] - max(value for label, value in logits.items() if label != expected),
                    **diagnostics,
                })

        if args.checkpoint_every > 0 and (
            batch_index % args.checkpoint_every == 0 or batch_index == len(batch_list)
        ):
            write_rows_csv(scored, partial_path)

    scored_path = write_rows_csv(scored, output_dir / "joint_gate_scored.csv")
    summary = summarize_scored(scored, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    summary_path = write_rows_csv(summary, output_dir / "joint_gate_summary.csv")
    gains = synergy_rows(summary)
    gain_path = write_rows_csv(gains, output_dir / "joint_gate_synergy.csv")
    run_metadata = {
        "model_name": args.model_name,
        "samples": args.samples,
        "n_rows": len(rows),
        "m_rotation": args.m_rotation,
        "rho_rotation": args.rho_rotation,
        "m_layer": m_layer,
        "rho_layer": rho_layer,
        "m_rank": m_rank,
        "rho_rank": rho_rank,
        "site": args.site,
        "composition_mode": args.composition_mode,
        "random_seeds": args.random_seeds,
        "conceptual_conditions": CONCEPTUAL_CONDITIONS,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(to_jsonable(run_metadata), indent=2))
    print(f"Wrote {scored_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {gain_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/das/joint_gate_v1/triples.csv")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--m-rotation", required=True)
    parser.add_argument("--rho-rotation", required=True)
    parser.add_argument("--composition-mode", choices=["common", "two_peak"], default="common")
    parser.add_argument("--site", default="claim_final")
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-rows-per-cell-regime", type=int, default=None)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
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


if __name__ == "__main__":
    raise SystemExit(main())
