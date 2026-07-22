"""Ablate a trained DAS subspace in a normal forward pass to test NECESSITY.

Interchange proves sufficiency (patching the subspace flips the label). This
asks the complementary question: if we DESTROY the subspace during an ordinary
run of the base prompt (no counterfactual injected), can the model still solve
the task, or does a dormant backup path recover p_c?

Corruption modes (applied only to the R-subspace at the intervention site):
  zero     : h <- h - (R^T h) R          (project the subspace out)
  resample : h <- h - (R^T h)R + (R^T h')R, h' = a random other example
             (in-distribution values, destroys THIS row's p_c; the honest mode)
  resample_same:
             h' has the same target-variable value, preferably from a different
             event; tests whether the subspace also carries nuisance content
  resample_opposite:
             h' has the opposite target-variable value; directly tests whether
             downstream computation follows that value or a backup path

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
from typing import Any, Sequence

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


SCORED_METADATA_COLUMNS = (
    "base_event_id",
    "target_var",
    "split",
    "mismatch_type",
    "matched_idx",
    "m_base",
    "m_src",
    "p_i_base",
    "p_i_src",
    "p_c_base",
    "p_c_src",
    "target_label",
    "eval_side",
    "base_mismatch_type",
    "donor_mismatch_type",
    "base_slot",
    "donor_slot",
    "matrix_m",
)


def run_condition(model, layers, layer, site, R, mode, rows, tokenizer, torch, device, label_ids,
                  batch_size, rng, donor_z=None):
    """Return scored rows for one (subspace, mode) condition, or mode='none'."""
    hook_state: dict[str, Any] = {"positions": None, "R": R, "mode": mode, "donor_z": None}

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
        elif hook_state["mode"] in ("resample_same", "resample_opposite"):
            new = h - proj + hook_state["donor_z"].to(hs.device) @ Rmat.T
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
        offset = 0
        for batch_rows in progress_iter(batches, total=len(batches), desc=f"{mode}"):
            texts = [str(r["base_prompt"]) for r in batch_rows]
            enc = encode_to_device(tokenizer, texts, device)
            positions = torch.tensor(
                [
                    resolve_token_site(tokenizer, t, r, "base", site if site != "row" else str(r["base_site"]))
                    for t, r in zip(texts, batch_rows)
                ],
                dtype=torch.long, device=device,
            )
            hook_state["positions"] = positions
            if donor_z is not None:
                hook_state["donor_z"] = donor_z[offset : offset + len(batch_rows)]
            offset += len(batch_rows)
            with torch.no_grad():
                logits = model(**enc).logits
            final_idx = enc["attention_mask"].sum(dim=1) - 1
            bidx = torch.arange(logits.shape[0], device=logits.device)
            nl = logits[bidx, final_idx]
            for r, row_logits in zip(batch_rows, nl):
                vals = {lab: float(row_logits[tid].detach().cpu()) for lab, tid in label_ids.items()}
                pred = max(vals, key=vals.get)
                target_label = r.get("target_label")
                record = {
                    "sample_id": r.get("sample_id"), "control_type": r.get("control_type"),
                    "true_label": r.get("base_label"), "pred_label": pred,
                    "is_correct": int(pred == r.get("base_label")),
                    "counterfactual_label": target_label,
                    "is_counterfactual_correct": int(target_label not in (None, "") and pred == target_label),
                    "R": vals["T"] - vals["F"], "U_gap": vals["U"] - max(vals["T"], vals["F"]),
                }
                record.update({key: r.get(key) for key in SCORED_METADATA_COLUMNS if key in r})
                scored.append(record)
    finally:
        handle.remove()
        hook_state["positions"] = None
        hook_state["donor_z"] = None
    return scored


def collect_subspace_coordinates(model, layer, site, R, rows, tokenizer, torch, device, batch_size):
    """Collect natural-run z=hR in row order for conditioned donor resampling."""
    zs = []
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    for batch_rows in progress_iter(batches, total=len(batches), desc="collect z"):
        texts = [str(r["base_prompt"]) for r in batch_rows]
        enc = encode_to_device(tokenizer, texts, device)
        positions = torch.tensor(
            [
                resolve_token_site(tokenizer, t, r, "base", site if site != "row" else str(r["base_site"]))
                for t, r in zip(texts, batch_rows)
            ],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[layer + 1]
        b = torch.arange(hs.shape[0], device=hs.device)
        zs.append((hs[b, positions].to(torch.float32) @ R).cpu())
    return torch.cat(zs, dim=0)


def resolve_condition_column(target_var: str, condition_on: str) -> str:
    """Resolve the row column whose value defines same/opposite donor classes."""
    if condition_on != "auto":
        return condition_on
    return {"pc": "p_c_base", "pi": "p_i_base", "m": "m_base"}.get(target_var, "base_label")


M_MAIN_CONTROLS = frozenset({"match_to_nomatch", "nomatch_to_match"})


def parse_column_list(value: str | Sequence[str] | None) -> list[str]:
    """Parse comma-separated/repeated donor hold columns, preserving order."""
    if value is None:
        return []
    raw = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    for item in raw:
        for column in str(item).split(","):
            column = column.strip()
            if column and column.lower() != "none" and column not in out:
                out.append(column)
    return out


def resolve_condition_hold_columns(target_var: str, condition_hold: str | Sequence[str] | None) -> list[str]:
    """Resolve nuisance columns that must stay fixed while the donor class flips."""
    if condition_hold == "auto" or condition_hold == ["auto"]:
        if target_var == "m":
            return ["p_i_base", "p_c_base", "mismatch_type", "matched_idx"]
        return []
    return parse_column_list(condition_hold)


def resolve_donor_pool(target_var: str, donor_pool: str) -> str:
    if donor_pool == "auto":
        return "m_main" if target_var == "m" else "within_control"
    return donor_pool


def resolve_donor_event_policy(target_var: str, donor_event_policy: str) -> str:
    if donor_event_policy == "auto":
        return "require" if target_var == "m" else "prefer"
    return donor_event_policy


def make_donor_spec(condition_on: str, condition_hold: Sequence[str], donor_pool: str,
                    donor_event_policy: str) -> dict[str, Any]:
    return {
        "condition_on": condition_on,
        "condition_hold": list(condition_hold),
        "donor_pool": donor_pool,
        "donor_event_policy": donor_event_policy,
        "candidate_controls": sorted(M_MAIN_CONTROLS) if donor_pool == "m_main" else None,
    }


def build_donor_indices(
    rows: list[dict[str, Any]],
    condition_on: str,
    seed: int,
    *,
    condition_hold: str | Sequence[str] | None = None,
    donor_pool: str = "within_control",
    donor_event_policy: str = "prefer",
) -> dict[str, list[int]]:
    """Build fixed same/opposite donor mappings.

    ``condition_on`` is the binary variable to preserve/flip. Every
    ``condition_hold`` column is fixed. Legacy pc/pi behavior uses
    ``donor_pool='within_control'``. For m, ``m_main`` pools candidates across
    the two causal directions because each direction contains only one base-m
    class; label-copy rows remain recipients but never become donors.
    """
    hold_columns = parse_column_list(condition_hold)
    if donor_pool not in {"within_control", "m_main", "all"}:
        raise ValueError(f"Unknown donor_pool {donor_pool!r}")
    if donor_event_policy not in {"prefer", "require", "ignore"}:
        raise ValueError(f"Unknown donor_event_policy {donor_event_policy!r}")

    rng = random.Random(seed)

    def row_value(
        row: dict[str, Any], column: str, index: int, *, allow_missing: bool = False
    ) -> str | None:
        value = row.get(column, "")
        if str(value) == "":
            if allow_missing:
                return None
            raise ValueError(f"Missing donor column {column!r} on row {index}")
        return str(value)

    def pool_key(row: dict[str, Any]) -> str:
        if donor_pool == "within_control":
            return str(row.get("control_type", ""))
        return donor_pool

    # (pool, held nuisance values) -> condition class -> candidate row indices
    candidate_pools: dict[tuple[str, tuple[str, ...]], dict[str, list[int]]] = {}
    for i, row in enumerate(rows):
        # M-v4 contains an evaluation-only same-m=1 trap without a physical
        # mismatch event. It is a recipient but never a donor under m_main, so
        # skip non-main candidates before requiring mismatch metadata.
        if donor_pool == "m_main" and str(row.get("control_type", "")) not in M_MAIN_CONTROLS:
            continue
        lab = row_value(row, condition_on, i)
        held = tuple(row_value(row, column, i) for column in hold_columns)
        candidate_pools.setdefault((pool_key(row), held), {}).setdefault(lab, []).append(i)

    same = [-1] * len(rows)
    opposite = [-1] * len(rows)
    for i, row in enumerate(rows):
        lab = row_value(row, condition_on, i)
        held = tuple(
            row_value(row, column, i, allow_missing=True) for column in hold_columns
        )
        stratum = (pool_key(row), held)
        if any(value is None for value in held):
            # A missing nuisance value means that nuisance is inapplicable to
            # this recipient, not that all nuisance matching should be dropped.
            # Pool exact donor strata only across the missing dimensions.
            pools: dict[str, list[int]] = {}
            for (candidate_pool, candidate_held), candidate_classes in candidate_pools.items():
                if candidate_pool != stratum[0]:
                    continue
                if not all(
                    recipient is None or recipient == candidate
                    for recipient, candidate in zip(held, candidate_held)
                ):
                    continue
                for candidate_class, indices in candidate_classes.items():
                    pools.setdefault(candidate_class, []).extend(indices)
        else:
            pools = candidate_pools.get(stratum, {})
        classes = sorted(pools)
        if len(classes) != 2:
            raise ValueError(
                f"Condition {condition_on!r} must have exactly two donor classes in "
                f"stratum pool={stratum[0]!r}, holds={dict(zip(hold_columns, held))}; found {classes}"
            )
        if lab not in pools:
            raise ValueError(f"Recipient class {lab!r} has no candidate pool in stratum {stratum!r}")
        other = classes[1] if lab == classes[0] else classes[0]

        def candidates(indices: Sequence[int], kind: str) -> list[int]:
            nonself = [j for j in indices if j != i]
            if not nonself:
                raise ValueError(f"No non-self {kind} donor for row {i} in stratum {stratum!r}")
            if donor_event_policy == "ignore":
                return nonself
            event = str(row.get("base_event_id", ""))
            different = [j for j in nonself if str(rows[j].get("base_event_id", "")) != event]
            if different:
                return different
            if donor_event_policy == "require":
                raise ValueError(f"No different-event {kind} donor for row {i} in stratum {stratum!r}")
            return nonself

        same[i] = rng.choice(candidates(pools[lab], "same-class"))
        opposite[i] = rng.choice(candidates(pools[other], "opposite-class"))
    return {"same": same, "opposite": opposite}


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(scored),
        "accuracy": mean(float(r["is_correct"]) for r in scored),
        "counterfactual_accuracy": mean(float(r["is_counterfactual_correct"]) for r in scored),
    }
    by: dict[str, Any] = {}
    for c in sorted({r["control_type"] for r in scored}):
        cr = [r for r in scored if r["control_type"] == c]
        preds = {lab: sum(r["pred_label"] == lab for r in cr) / len(cr) for lab in ("T", "F", "U")}
        by[c] = {"n": len(cr), "accuracy": mean(float(r["is_correct"]) for r in cr),
                 "counterfactual_accuracy": mean(float(r["is_counterfactual_correct"]) for r in cr),
                 "mean_R": mean(float(r["R"]) for r in cr), "pred_dist": preds}
    out["by_control"] = by
    return out


CONDITION_NAMES = [
    "none", "das_zero", "das_resample", "das_resample_same",
    "das_resample_opposite", "rand_zero", "rand_resample",
]


def ablate_one(model, layers, tokenizer, torch, device, hidden, rows, rotation_dir, label_ids,
               batch_size, seed, donor_indices=None, condition_on=None, donor_spec=None):
    """Run unified necessity conditions for one trained rotation."""
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
    conditions = [("none", None, None), ("zero", R_das, None), ("resample", R_das, None)]
    if donor_indices is not None:
        z = collect_subspace_coordinates(model, layer, site, R_das, rows, tokenizer, torch, device, batch_size)
        conditions.extend([
            ("resample_same", R_das, z[donor_indices["same"]]),
            ("resample_opposite", R_das, z[donor_indices["opposite"]]),
        ])
    conditions.extend([("zero", R_rand, None), ("resample", R_rand, None)])

    rng = random.Random(seed)
    summary: dict[str, Any] = {"layer": layer, "rank": rank, "site": site}
    if condition_on is not None:
        summary["condition_on"] = condition_on
    if donor_spec is not None:
        summary["donor_spec"] = donor_spec
        summary["condition_hold"] = donor_spec.get("condition_hold", [])
        summary["donor_pool"] = donor_spec.get("donor_pool")
        summary["donor_event_policy"] = donor_spec.get("donor_event_policy")
    all_scored: dict[str, Any] = {}
    names = CONDITION_NAMES if donor_indices is not None else [
        "none", "das_zero", "das_resample", "rand_zero", "rand_resample"
    ]
    for (mode, R, donor_z), name in zip(conditions, names):
        print(f"  --- {name} (L{layer}/{site}) ---")
        scored = run_condition(model, layers, layer, site, R, mode, rows, tokenizer, torch, device,
                               label_ids, batch_size, rng, donor_z=donor_z)
        summary[name] = summarize(scored)
        all_scored[name] = scored
    return summary, all_scored


def print_ablation_table(summary, all_scored):
    controls = sorted({r["control_type"] for r in all_scored["none"]})
    hdr = f"{'condition':16s} {'overall':>8s} " + " ".join(f"{c:>16s}" for c in controls)
    print(hdr); print("-" * len(hdr))
    for name in (n for n in CONDITION_NAMES if n in summary):
        s = summary[name]
        cells = " ".join(f"{s['by_control'][c]['accuracy']:>16.4f}" for c in controls)
        print(f"{name:16s} {s['accuracy']:>8.4f} {cells}")
    def macc(name):
        return (summary.get(name) or {}).get("by_control", {}).get("main", {}).get("accuracy")
    if macc("none") is not None:
        print(f"main drop DAS resample={macc('none')-macc('das_resample'):+.4f}"
              f" random resample={macc('none')-macc('rand_resample'):+.4f}"
              f" excess(necessity)={macc('rand_resample')-macc('das_resample'):+.4f}")
        if macc("das_resample_same") is not None:
            print(f"main purity_drop={macc('none')-macc('das_resample_same'):+.4f}"
                  f" backup_rescue={macc('das_resample_opposite'):.4f}")
    elif "das_resample_opposite" in summary:
        print("\nconditioned donor diagnostics by control:")
        for ctrl in controls:
            base = summary["none"]["by_control"][ctrl]["accuracy"]
            same = summary["das_resample_same"]["by_control"][ctrl]["accuracy"]
            opposite = summary["das_resample_opposite"]["by_control"][ctrl]
            print(
                f"  {ctrl:18s} same_ret={same:.4f} purity_drop={base-same:+.4f} "
                f"opp_orig={opposite['accuracy']:.4f} opp_cf={opposite['counterfactual_accuracy']:.4f} "
                f"opp_U={opposite['pred_dist']['U']:.4f}"
            )


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    rows = [r for r in read_rows_csv(args.samples)
            if r.get("target_var") == args.target_var and (args.split == "all" or r.get("split") == args.split)]
    rng.shuffle(rows)  # global shuffle so within-batch resample donors are random
    if not rows:
        raise ValueError("No rows matched filters")
    condition_on = resolve_condition_column(args.target_var, args.condition_on)
    condition_hold = resolve_condition_hold_columns(args.target_var, args.condition_hold)
    if args.split == "all" and "split" not in condition_hold:
        condition_hold.append("split")
    donor_pool = resolve_donor_pool(args.target_var, args.donor_pool)
    donor_event_policy = resolve_donor_event_policy(args.target_var, args.donor_event_policy)
    donor_spec = make_donor_spec(condition_on, condition_hold, donor_pool, donor_event_policy)
    donor_spec["seed"] = args.seed
    donor_indices = None if args.skip_conditioned_donors else build_donor_indices(
        rows,
        condition_on,
        args.seed,
        condition_hold=condition_hold,
        donor_pool=donor_pool,
        donor_event_policy=donor_event_policy,
    )

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
                                     args.rotation_dir, label_tokens.token_ids, args.eval_batch_size, args.seed,
                                     donor_indices=donor_indices, condition_on=condition_on,
                                     donor_spec=None if donor_indices is None else donor_spec)
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
    p.add_argument("--condition-on", default="auto",
                   help="Column defining same/opposite donor classes; auto maps pc/pi/m to p_c_base/p_i_base/m_base.")
    p.add_argument(
        "--condition-hold",
        default="auto",
        help=("Comma-separated nuisance columns fixed for same/opposite donors. "
              "auto uses none for pc/pi and p_i_base,p_c_base,mismatch_type,matched_idx for m."),
    )
    p.add_argument(
        "--donor-pool",
        default="auto",
        choices=["auto", "within_control", "m_main", "all"],
        help="Candidate pooling; auto keeps pc/pi within-control and pools m across its two main directions.",
    )
    p.add_argument(
        "--donor-event-policy",
        default="auto",
        choices=["auto", "prefer", "require", "ignore"],
        help="Whether conditioned donors should come from a different base event; auto requires this for m.",
    )
    p.add_argument("--skip-conditioned-donors", action="store_true",
                   help="Run only the legacy zero/random-resample conditions.")
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
