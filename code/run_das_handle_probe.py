"""Probe what the trained DAS subspace actually carries: p_c, REL, or label.

Eval-only: loads a trained rotation and runs two new interchange conditions
whose predictions separate the hypotheses (the training distribution cannot,
because flipping p_c there also flips REL = p_i*p_c and the source label).

  probe_flip_both : source flips BOTH p_i and p_c  -> REL and source label unchanged
      H_pc    predicts the label FLIPS  (injected p_c changed)
      H_REL   predicts no change        (injected REL unchanged)
      H_label predicts no change        (source label unchanged)

  probe_flip_pi   : source flips ONLY p_i          -> REL and source label flip
      H_pc    predicts no change        (injected p_c unchanged)
      H_REL   predicts the label FLIPS
      H_label predicts the label FLIPS

target_label is set to the H_pc prediction, so the reported IIA is
"agreement with H_pc"; per-hypothesis match rates are also reported.

A sanity block re-evaluates the original main/gate/trap test pairs with the
loaded rotation to confirm the weights were restored correctly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from interference_suite.das_data import (
    POLARITIES,
    flip_polarity,
    high_level_label,
    make_example,
    make_pair_row,
    replace_tuple,
    split_for_index,
)
from interference_suite.das_pyvene import (
    build_intervenable,
    collect_rotation_weights,
    evaluate_pyvene_das,
    get_input_device,
    import_runtime,
    load_hf_model,
    set_intervenable_device,
    to_jsonable,
)
from interference_suite.generation import sample_base_events
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, resolve_label_tokens


def generate_probe_rows(n_base_events: int, seed: int, split: str, train_fraction: float, val_fraction: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base_events = sample_base_events(n_base_events, rng)
    rows: list[dict[str, Any]] = []
    for base_index, claim_event in enumerate(base_events):
        event_split = split_for_index(base_index, len(base_events), train_fraction, val_fraction)
        if split != "all" and event_split != split:
            continue
        base_id = f"base_{base_index:04d}"
        for matched_idx in (0, 1, 2):
            for p_i in POLARITIES:
                for p_c in POLARITIES:
                    base = make_example(claim_event, matched_idx, m_i=1, p_i=p_i, p_c=p_c, rng=rng)
                    source_polarities = replace_tuple(base.assumption_polarities, matched_idx, flip_polarity(p_i))

                    flip_both = make_example(
                        claim_event,
                        matched_idx,
                        m_i=1,
                        p_i=flip_polarity(p_i),
                        p_c=flip_polarity(p_c),
                        rng=rng,
                        assumption_events=base.assumption_events,
                        assumption_polarities=source_polarities,
                    )
                    rows.append(
                        make_pair_row(
                            sample_id=f"probe_both_{base_id}_idx{matched_idx + 1}_{p_i[:3]}_{p_c[:3]}",
                            base_id=base_id,
                            split=event_split,
                            target_var="pc",
                            control_type="probe_flip_both",
                            base=base,
                            source=flip_both,
                            target_label=high_level_label(1, base.p_i, flip_both.p_c),
                            base_site="claim_final",
                            source_site="claim_final",
                            extra={
                                "pred_H_pc": high_level_label(1, base.p_i, flip_both.p_c),
                                "pred_H_rel": base.label,
                                "pred_H_label": flip_both.label,
                            },
                        )
                    )

                    flip_pi = make_example(
                        claim_event,
                        matched_idx,
                        m_i=1,
                        p_i=flip_polarity(p_i),
                        p_c=p_c,
                        rng=rng,
                        assumption_events=base.assumption_events,
                        assumption_polarities=source_polarities,
                    )
                    rows.append(
                        make_pair_row(
                            sample_id=f"probe_pi_{base_id}_idx{matched_idx + 1}_{p_i[:3]}_{p_c[:3]}",
                            base_id=base_id,
                            split=event_split,
                            target_var="pc",
                            control_type="probe_flip_pi",
                            base=base,
                            source=flip_pi,
                            target_label=high_level_label(1, base.p_i, flip_pi.p_c),
                            base_site="claim_final",
                            source_site="claim_final",
                            extra={
                                "pred_H_pc": high_level_label(1, base.p_i, flip_pi.p_c),
                                "pred_H_rel": flip_pi.label,
                                "pred_H_label": flip_pi.label,
                            },
                        )
                    )
    for idx, row in enumerate(rows):
        row["row_id"] = idx
    return rows


def load_rotation(intervenable: Any, torch: Any, rotation_dir: Path) -> None:
    saved = torch.load(rotation_dir / "rotation_weight.pt", map_location="cpu")
    weights = collect_rotation_weights(intervenable)
    if len(weights) != 1:
        raise RuntimeError(f"Expected exactly one rotation weight, found {list(weights)}")
    weight = next(iter(weights.values()))
    if tuple(weight.shape) != tuple(saved.shape):
        raise RuntimeError(f"Shape mismatch: intervention {tuple(weight.shape)} vs saved {tuple(saved.shape)}")
    with torch.no_grad():
        weight.data.copy_(saved.to(dtype=weight.dtype, device=weight.device))


def hypothesis_match_rates(scored: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for control in sorted({row["control_type"] for row in scored}):
        control_rows = [row for row in scored if row["control_type"] == control]
        n = len(control_rows)
        pred_counts: dict[str, int] = {}
        for row in control_rows:
            pred_counts[row["pred_label"]] = pred_counts.get(row["pred_label"], 0) + 1
        out[control] = {
            "n": n,
            "match_H_pc": sum(row["pred_label"] == row["pred_H_pc"] for row in control_rows) / n,
            "match_H_rel": sum(row["pred_label"] == row["pred_H_rel"] for row in control_rows) / n,
            "match_H_label": sum(row["pred_label"] == row["pred_H_label"] for row in control_rows) / n,
            "pred_counts": pred_counts,
            "global_top_in_TFU_rate": sum(int(row.get("global_top_in_TFU", 1)) for row in control_rows) / n,
        }
    return out


def main() -> int:
    args = build_parser().parse_args()
    rotation_dir = Path(args.rotation_dir)
    metadata = json.loads((rotation_dir / "rotation_weight_metadata.json").read_text(encoding="utf-8"))
    layer = int(metadata["layer"])
    rank = int(metadata["rank"])
    component = str(metadata.get("component", "block_output"))
    site = str(metadata.get("site", "claim_final"))
    print(f"Loaded rotation metadata: layer={layer} rank={rank} component={component} site={site}")

    probe_rows = generate_probe_rows(args.n_base_events, args.seed, args.split, args.train_fraction, args.val_fraction)
    print(f"Generated {len(probe_rows)} probe rows (split={args.split})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, pv, auto_model_cls, auto_tokenizer_cls = import_runtime()
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
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    intervenable = build_intervenable(pv, model, layer=layer, rank=rank, component=component)
    input_device = get_input_device(model, torch, args.device)
    set_intervenable_device(intervenable, input_device)
    load_rotation(intervenable, torch, rotation_dir)
    print("Rotation weights restored.")

    summary: dict[str, Any] = {
        "rotation_dir": str(rotation_dir),
        "layer": layer,
        "rank": rank,
        "site": site,
        "split": args.split,
        "n_probe_rows": len(probe_rows),
    }

    if args.sanity_samples:
        sanity_rows = [
            row
            for row in read_rows_csv(args.sanity_samples)
            if row.get("target_var") == "pc" and row.get("split") == args.split
        ]
        sanity_metrics, _ = evaluate_pyvene_das(
            intervenable, sanity_rows, tokenizer, torch, input_device,
            label_tokens.token_ids, args.eval_batch_size, site, "Sanity (original pairs)",
        )
        summary["sanity"] = sanity_metrics
        main_iia = (sanity_metrics.get("by_control", {}).get("main", {}) or {}).get("IIA")
        print(f"Sanity main IIA with restored rotation: {main_iia}")

    probe_metrics, probe_scored = evaluate_pyvene_das(
        intervenable, probe_rows, tokenizer, torch, input_device,
        label_tokens.token_ids, args.eval_batch_size, site, "Handle probes",
    )
    summary["probe_metrics"] = probe_metrics
    summary["hypothesis_match"] = hypothesis_match_rates(probe_scored)

    write_rows_csv(probe_scored, output_dir / "probe_scored.csv")
    (output_dir / "handle_probe_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")

    print("\n=== Hypothesis match rates ===")
    for control, stats in summary["hypothesis_match"].items():
        print(
            f"{control:18s} n={stats['n']:5d}  H_pc={stats['match_H_pc']:.4f}  "
            f"H_rel={stats['match_H_rel']:.4f}  H_label={stats['match_H_label']:.4f}  preds={stats['pred_counts']}"
        )
    print(f"\nWrote {output_dir / 'handle_probe_summary.json'} and probe_scored.csv")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe whether a trained DAS subspace carries p_c, REL, or label.")
    parser.add_argument("--rotation-dir", required=True, help="Dir with rotation_weight.pt + rotation_weight_metadata.json")
    parser.add_argument("--sanity-samples", default=None, help="Original pairs CSV; re-evaluated to verify the restored rotation.")
    parser.add_argument("--n-base-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--label-token-style", default="auto")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", default="data/das/handle_probe")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
