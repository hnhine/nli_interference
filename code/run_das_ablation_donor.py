"""Class-conditioned donor resample-ablation (necessity, clean version).

Unconditional resample mixes donors of both classes, so its accuracy drop is a
composition-dependent lower bound. Here donors are conditioned on the base
row's true label (main rows only):

  resample_same     : donor has the SAME label  -> predicts baseline accuracy;
                      any drop = information in the subspace beyond the target
                      variable (purity, causal-scrubbing style)
  resample_opposite : donor has the OPPOSITE label -> predicts accuracy ~ 0 under
                      full mediation; anything above 0 = rescue by backup paths
                      (direct necessity metric)

Two passes per cell: (1) collect z = R^T h for every row at the cell's site
with no intervention; (2) re-run with each row's subspace coordinates replaced
by a sampled donor's z.

Example (Qwen pc cells):
    python code/run_das_ablation_donor.py --samples data/das/pc_1000_v2/pairs.csv \
        --model-name Qwen/Qwen3-8B --split test --local-files-only \
        --output-dir data/das/donor_ablation_qwen \
        --rotation-dirs data/das/qwen3_8_pc_1000_v2_l18_r32_claim_b32_allcontrols \
                        data/das/pc_forced_sweep_full/L14 data/das/pc_forced_sweep_full/L26
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from math import ceil
from pathlib import Path
from typing import Any

from interference_suite.das_pyvene import encode_to_device, import_runtime, load_hf_model, mean, to_jsonable
from interference_suite.das_spans import resolve_token_site
from interference_suite.io_utils import read_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, progress_iter, resolve_label_tokens
from run_das_ablation import get_decoder_layers


def row_site(row: dict[str, Any], site: str) -> str:
    return str(row["base_site"]) if site == "row" else site


def batches(rows: list, size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def collect_z(model, torch, tokenizer, device, rows, layer, site, R, batch_size):
    """Pass 1: z = R^T h at the site for every row, natural forward."""
    zs = []
    n_batches = ceil(len(rows) / batch_size)
    for batch_rows in progress_iter(list(batches(rows, batch_size)), total=n_batches, desc="collect z"):
        texts = [str(r["base_prompt"]) for r in batch_rows]
        enc = encode_to_device(tokenizer, texts, device)
        pos = torch.tensor([resolve_token_site(tokenizer, t, r, "base", row_site(r, site))
                            for t, r in zip(texts, batch_rows)], dtype=torch.long, device=device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[layer + 1]
        b = torch.arange(hs.shape[0], device=device)
        zs.append((hs[b, pos].to(torch.float32) @ R).cpu())
    return torch.cat(zs, dim=0)  # [n, rank]


def score_pass(model, torch, tokenizer, device, rows, layer, site, R, donor_z, label_ids, batch_size, desc):
    """Pass 2: forward with per-row subspace coords replaced by donor_z (None = no hook)."""
    layers = get_decoder_layers(model)
    state: dict[str, Any] = {"pos": None, "z": None}

    def hook(module, inputs, output):
        if state["pos"] is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        b = torch.arange(hs.shape[0], device=hs.device)
        h = hs[b, state["pos"]].to(torch.float32)
        new = h - (h @ R) @ R.T + state["z"].to(hs.device) @ R.T
        hs[b, state["pos"]] = new.to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    handle = layers[layer].register_forward_hook(hook) if donor_z is not None else None
    scored = []
    try:
        idx = 0
        n_batches = ceil(len(rows) / batch_size)
        for batch_rows in progress_iter(list(batches(rows, batch_size)), total=n_batches, desc=desc):
            texts = [str(r["base_prompt"]) for r in batch_rows]
            enc = encode_to_device(tokenizer, texts, device)
            if donor_z is not None:
                state["pos"] = torch.tensor([resolve_token_site(tokenizer, t, r, "base", row_site(r, site))
                                             for t, r in zip(texts, batch_rows)], dtype=torch.long, device=device)
                state["z"] = donor_z[idx : idx + len(batch_rows)]
            idx += len(batch_rows)
            with torch.no_grad():
                logits = model(**enc).logits
            final = enc["attention_mask"].sum(dim=1) - 1
            b = torch.arange(logits.shape[0], device=logits.device)
            nl = logits[b, final]
            for r, rl in zip(batch_rows, nl):
                vals = {lab: float(rl[tid]) for lab, tid in label_ids.items()}
                pred = max(vals, key=vals.get)
                true = str(r["base_label"])
                scored.append({
                    "pred": pred, "true": true, "correct": int(pred == true),
                    "absR": abs(vals["T"] - vals["F"]),
                    "margin_true": vals[true] - max(v for k, v in vals.items() if k != true),
                    "is_U": int(pred == "U"),
                })
    finally:
        if handle is not None:
            handle.remove()
        state["pos"] = None
    return scored


def stats(scored):
    return {
        "n": len(scored),
        "accuracy": mean(s["correct"] for s in scored),
        "mean_absR": mean(s["absR"] for s in scored),
        "mean_margin_true": mean(s["margin_true"] for s in scored),
        "U_rate": mean(s["is_U"] for s in scored),
    }


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    rows = [r for r in read_rows_csv(args.samples)
            if r.get("target_var") == args.target_var and r.get("split") == args.split
            and r.get("control_type") == "main"]
    if not rows:
        raise ValueError("No main rows matched")
    print(f"{len(rows)} main rows (split={args.split})")

    torch, _, amc, atc = import_runtime()
    tokenizer, model = load_hf_model(torch=torch, auto_model_cls=amc, auto_tokenizer_cls=atc,
        model_name=args.model_name, device=args.device, device_map=args.device_map,
        torch_dtype=args.torch_dtype, trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    device = next(model.parameters()).device
    hidden = model.config.hidden_size

    import numpy as np
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rdir in args.rotation_dirs:
        meta = json.loads((Path(rdir) / "rotation_weight_metadata.json").read_text())
        layer, rank, site = int(meta["layer"]), int(meta["rank"]), str(meta.get("site", "claim_final"))
        R = torch.tensor(np.load(Path(rdir) / "rotation_weight.npy"), dtype=torch.float32, device=device)
        assert tuple(R.shape) == (hidden, rank)
        name = Path(rdir).name if Path(rdir).name.startswith("L") else f"L{layer:02d}_{site}"
        print(f"\n===== {name} ({rdir}) =====")

        z = collect_z(model, torch, tokenizer, device, rows, layer, site, R, args.eval_batch_size)

        labels = [str(r[args.condition_on]) for r in rows]
        pools: dict[str, list[int]] = {}
        for i, lab in enumerate(labels):
            pools.setdefault(lab, []).append(i)
        classes = sorted(pools)
        donor_same = torch.stack([z[rng.choice([j for j in pools[labels[i]] if j != i])] for i in range(len(rows))])
        donor_opp = torch.stack([z[rng.choice(pools[classes[1] if labels[i] == classes[0] else classes[0]])]
                                 for i in range(len(rows))])

        cell = {"cell": name, "layer": layer, "site": site, "rank": rank}
        for cond, dz in (("none", None), ("resample_same", donor_same), ("resample_opposite", donor_opp)):
            st = stats(score_pass(model, torch, tokenizer, device, rows, layer, site, R, dz,
                                  label_tokens.token_ids, args.eval_batch_size, cond))
            for k, v in st.items():
                cell[f"{cond}_{k}"] = v
            print(f"  {cond:18s} acc={st['accuracy']:.4f} |R|={st['mean_absR']:.2f} U={st['U_rate']:.2f}")
        cell["purity_drop"] = cell["none_accuracy"] - cell["resample_same_accuracy"]
        cell["backup_rescue"] = cell["resample_opposite_accuracy"]
        records.append(cell)
        with (output_dir / "donor_sweep.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
        (output_dir / f"{name}.json").write_text(json.dumps(to_jsonable(cell), indent=2))

    print("\n========  DONOR-CONDITIONED NECESSITY  ========")
    print(f"{'cell':24s} {'none':>6s} {'same':>6s} {'opp':>6s} {'purity_drop':>12s} {'backup_rescue':>14s}")
    for c in records:
        print(f"{c['cell']:24s} {c['none_accuracy']:>6.3f} {c['resample_same_accuracy']:>6.3f} "
              f"{c['resample_opposite_accuracy']:>6.3f} {c['purity_drop']:>12.3f} {c['backup_rescue']:>14.3f}")
    print(f"\nWrote {output_dir / 'donor_sweep.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Class-conditioned donor resample ablation (main rows).")
    p.add_argument("--rotation-dirs", nargs="+", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--target-var", default="pc")
    p.add_argument("--condition-on", default="base_label",
                   help="Row column defining donor classes: base_label for REL/label subspaces, p_c_base or p_i_base for raw-polarity subspaces.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--label-token-style", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/das/donor_ablation")
    p.add_argument("--device", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
