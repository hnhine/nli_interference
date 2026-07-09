"""Ablate a trained DAS subspace in a normal forward pass to test NECESSITY.

Interchange proves sufficiency (patching the subspace flips the label). This
asks the complementary question: if we DESTROY the subspace during an ordinary
run of the base prompt (no counterfactual injected), can the model still solve
the task, or does a dormant backup path recover p_c?

Corruption modes (applied only to the R-subspace at the intervention site):
  zero     : h <- h - (R^T h) R          (project the subspace out)
  resample : h <- h - (R^T h)R + (R^T h')R, h' = a random other example
             (in-distribution values, destroys THIS row's p_c; the honest mode)

Subspaces:
  das    : the trained rotation (necessity of the found direction)
  random : a fresh random orthonormal 32-dim subspace (damage baseline)

The reported effect is the DIFFERENCE das-vs-random: ablating a random subspace
also perturbs the state, so only the excess drop is attributable to p_c content.

Predictions if the subspace is necessary (no backup path):
  main (m=1)  accuracy collapses toward chance / drifts to U (p_c lost)
  gate (m=0)  accuracy stays high (label is U regardless of p_c) -- specificity
  random subspace ablation: little effect anywhere

Example:
    python code/run_das_ablation.py \
        --samples data/das/pc_1000_v2/pairs.csv \
        --rotation-dir data/das/qwen3_8_pc_1000_v2_l18_r32_claim_b32_allcontrols \
        --model-name Qwen/Qwen3-8B --split test --local-files-only \
        --output-dir data/das/ablation_qwen3_L18
"""

from __future__ import annotations

import argparse
import json
import random
from math import ceil
from pathlib import Path
from typing import Any

from interference_suite.das_pyvene import encode_to_device, import_runtime, load_hf_model, mean, to_jsonable
from interference_suite.das_spans import resolve_token_site
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, progress_iter, resolve_label_tokens


def get_decoder_layers(model: Any) -> Any:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers (model.model.layers)")
    return layers


def make_rotation(kind: str, saved: Any, hidden: int, rank: int, torch: Any, device: Any, dtype: Any) -> Any:
    if kind == "das":
        R = saved
    else:  # random orthonormal
        g = torch.randn(hidden, rank, dtype=torch.float32)
        R, _ = torch.linalg.qr(g)
    return R.to(device=device, dtype=torch.float32)


def run_condition(model, layers, layer, site, R, mode, rows, tokenizer, torch, device, label_ids, batch_size, rng):
    """Return scored rows for one (subspace, mode) condition, or mode='none'."""
    hook_state: dict[str, Any] = {"positions": None, "R": R, "mode": mode}

    def hook(module, inputs, output):
        if hook_state["positions"] is None or hook_state["mode"] == "none":
            return output
        hs = output[0] if isinstance(output, tuple) else output
        pos = hook_state["positions"]
        b = torch.arange(hs.shape[0], device=hs.device)
        Rmat = hook_state["R"]
        h = hs[b, pos].to(torch.float32)                 # [B,H]
        z = h @ Rmat                                     # [B,rank]
        proj = z @ Rmat.T                                # [B,H]
        if hook_state["mode"] == "zero":
            new = h - proj
        else:  # resample: donor = permuted rows within batch
            perm = torch.randperm(h.shape[0], device=hs.device)
            donor = (z[perm] @ Rmat.T)
            new = h - proj + donor
        hs[b, pos] = new.to(hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    handle = layers[layer].register_forward_hook(hook)
    scored: list[dict[str, Any]] = []
    try:
        batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
        for batch_rows in progress_iter(batches, total=len(batches), desc=f"{mode}"):
            texts = [str(r["base_prompt"]) for r in batch_rows]
            enc = encode_to_device(tokenizer, texts, device)
            positions = torch.tensor(
                [resolve_token_site(tokenizer, t, r, "base", site) for t, r in zip(texts, batch_rows)],
                dtype=torch.long, device=device,
            )
            hook_state["positions"] = positions
            with torch.no_grad():
                logits = model(**enc).logits
            final_idx = enc["attention_mask"].sum(dim=1) - 1
            bidx = torch.arange(logits.shape[0], device=logits.device)
            nl = logits[bidx, final_idx]
            for r, row_logits in zip(batch_rows, nl):
                vals = {lab: float(row_logits[tid].detach().cpu()) for lab, tid in label_ids.items()}
                pred = max(vals, key=vals.get)
                scored.append({
                    "sample_id": r.get("sample_id"), "control_type": r.get("control_type"),
                    "true_label": r.get("base_label"), "pred_label": pred,
                    "is_correct": int(pred == r.get("base_label")),
                    "R": vals["T"] - vals["F"], "U_gap": vals["U"] - max(vals["T"], vals["F"]),
                })
    finally:
        handle.remove()
        hook_state["positions"] = None
    return scored


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(scored), "accuracy": mean(float(r["is_correct"]) for r in scored)}
    by: dict[str, Any] = {}
    for c in sorted({r["control_type"] for r in scored}):
        cr = [r for r in scored if r["control_type"] == c]
        preds = {lab: sum(r["pred_label"] == lab for r in cr) / len(cr) for lab in ("T", "F", "U")}
        by[c] = {"n": len(cr), "accuracy": mean(float(r["is_correct"]) for r in cr),
                 "mean_R": mean(float(r["R"]) for r in cr), "pred_dist": preds}
    out["by_control"] = by
    return out


CONDITION_NAMES = ["none", "das_zero", "das_resample", "rand_zero", "rand_resample"]


def ablate_one(model, layers, tokenizer, torch, device, hidden, rows, rotation_dir, label_ids, batch_size, seed):
    """Run all 5 ablation conditions for one trained rotation. Returns (summary, all_scored)."""
    import numpy as np

    meta = json.loads((Path(rotation_dir) / "rotation_weight_metadata.json").read_text())
    layer, rank = int(meta["layer"]), int(meta["rank"])
    site = str(meta.get("site", "claim_final"))
    saved = torch.tensor(np.load(Path(rotation_dir) / "rotation_weight.npy"))
    if tuple(saved.shape) != (hidden, rank):
        raise RuntimeError(f"rotation shape {tuple(saved.shape)} != ({hidden},{rank})")

    R_das = make_rotation("das", saved, hidden, rank, torch, device, None)
    torch.manual_seed(seed)
    R_rand = make_rotation("random", saved, hidden, rank, torch, device, None)
    conditions = [("none", None), ("zero", R_das), ("resample", R_das), ("zero", R_rand), ("resample", R_rand)]

    rng = random.Random(seed)
    summary: dict[str, Any] = {"layer": layer, "rank": rank, "site": site}
    all_scored: dict[str, Any] = {}
    for (mode, R), name in zip(conditions, CONDITION_NAMES):
        print(f"  --- {name} (L{layer}/{site}) ---")
        scored = run_condition(model, layers, layer, site, R, mode, rows, tokenizer, torch, device,
                               label_ids, batch_size, rng)
        summary[name] = summarize(scored)
        all_scored[name] = scored
    return summary, all_scored


def print_ablation_table(summary, all_scored):
    controls = sorted({r["control_type"] for r in all_scored["none"]})
    hdr = f"{'condition':16s} {'overall':>8s} " + " ".join(f"{c:>16s}" for c in controls)
    print(hdr); print("-" * len(hdr))
    for name in CONDITION_NAMES:
        s = summary[name]
        cells = " ".join(f"{s['by_control'][c]['accuracy']:>16.4f}" for c in controls)
        print(f"{name:16s} {s['accuracy']:>8.4f} {cells}")
    def macc(name): return summary[name]["by_control"].get("main", {}).get("accuracy")
    if macc("none") is not None:
        print(f"main drop DAS resample={macc('none')-macc('das_resample'):+.4f}"
              f" random resample={macc('none')-macc('rand_resample'):+.4f}"
              f" excess(necessity)={macc('rand_resample')-macc('das_resample'):+.4f}")


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    rows = [r for r in read_rows_csv(args.samples)
            if r.get("target_var") == args.target_var and (args.split == "all" or r.get("split") == args.split)]
    rng.shuffle(rows)  # global shuffle so within-batch resample donors are random
    if not rows:
        raise ValueError("No rows matched filters")

    torch, _, amc, atc = import_runtime()
    tokenizer, model = load_hf_model(torch=torch, auto_model_cls=amc, auto_tokenizer_cls=atc,
        model_name=args.model_name, device=args.device, device_map=args.device_map,
        torch_dtype=args.torch_dtype, trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    device = next(model.parameters()).device
    layers = get_decoder_layers(model)
    hidden = model.config.hidden_size

    summary, all_scored = ablate_one(model, layers, tokenizer, torch, device, hidden, rows,
                                     args.rotation_dir, label_tokens.token_ids, args.eval_batch_size, args.seed)
    summary["model"] = args.model_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    write_rows_csv([{**r, "condition": n} for n, s in all_scored.items() for r in s], output_dir / "ablation_scored.csv")
    print("\n================  ABLATION RESULTS (accuracy on base prompts)  ================")
    print_ablation_table(summary, all_scored)
    print(f"\nWrote {output_dir / 'ablation_summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Necessity ablation of a trained DAS subspace.")
    p.add_argument("--samples", required=True)
    p.add_argument("--rotation-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--target-var", default="pc")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--label-token-style", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/das/ablation")
    p.add_argument("--device", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
