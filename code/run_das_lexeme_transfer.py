"""Lexeme-transfer test: is the subspace a `not`-template feature or a polarity variable?

Forms ladder in interference_suite/das_lexeme_data.py. Four tests per cell:
  A  behavioral gate (once per run; negative-varied rows per form)
  B  geometry: canonical polarity axis coordinate + orthogonal drift + frozen NN
  C  TRUE minimal-pair interchange: base = canonical positive row; source = the
     pair's own stored source (shared events/distractors) re-rendered in form F
  D  cross-event same-polarity donors (unambiguous forms; conditioning column
     configurable via --d-condition-on, e.g. base_label for REL cells)

Verdict per cell (descriptive, thresholds recorded):
  anchor valid iff departure >= --anchor-departure-min and cf_acc >= --anchor-min
  transfer ratio = form_cf_acc / anchor_cf_acc
  labels: lexeme_invariant_polarity | not_template | partial_transfer |
          inconclusive (gate) | invalid_anchor
Graded forms (rarely/seldom) never enter the label; they are reported as
descriptive graded-axis metrics only.

Cells are cached: an existing {cell}.json in --output-dir is loaded, not rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from interference_suite.das_lexeme_data import (
    GRADED_FORMS,
    NEGATION_FORMS,
    UNAMBIGUOUS_FORMS,
    count_anchor_mismatches,
    count_source_anchor_mismatches,
    derive_form_rows,
    derive_source_form_rows,
    is_negative_varied,
    varied_polarity_column,
)
from interference_suite.das_pyvene import import_runtime, load_hf_model, mean, to_jsonable
from interference_suite.io_utils import read_rows_csv
from interference_suite.model import DEFAULT_CACHE_DIR, resolve_label_tokens
from run_das_ablation import collect_subspace_coordinates, get_decoder_layers, run_condition


def margin_to(label: str, R: float, U_gap: float) -> float:
    if label == "U":
        return U_gap
    if label == "T":
        return min(R, -U_gap if R >= 0 else R - U_gap)
    return min(-R, -U_gap if R <= 0 else -R - U_gap)


def flipped(label: str) -> str:
    return {"T": "F", "F": "T"}[label]


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    vary = args.vary
    target_var = args.target_var or ("pc" if vary == "claim" else "pi")

    source_rows = [r for r in read_rows_csv(args.samples)
                   if r.get("target_var") == target_var and r.get("split") == args.split
                   and r.get("control_type") == "main"]
    if args.n_events:
        keep = set(sorted({r["base_event_id"] for r in source_rows})[: args.n_events])
        source_rows = [r for r in source_rows if r["base_event_id"] in keep]
    if not source_rows:
        raise ValueError("No main rows matched")
    bad_b = count_anchor_mismatches(source_rows, vary)
    bad_s = count_source_anchor_mismatches(source_rows, vary)
    if bad_b or bad_s:
        raise RuntimeError(f"Anchor regression failed: base={bad_b}, source={bad_s} rows differ from originals")
    print(f"{len(source_rows)} canonical main rows | base & source reconstruction exact")

    forms = args.forms
    form_rows = {f: derive_form_rows(source_rows, f, vary) for f in forms}
    # C uses positive-varied bases + their own stored sources re-rendered per form
    pos_idx = [i for i, r in enumerate(source_rows) if not is_negative_varied(r, vary)]
    c_base_rows = [form_rows["did_not"][i] for i in pos_idx]
    c_source_rows = {f: derive_source_form_rows([source_rows[i] for i in pos_idx], f, vary) for f in forms}

    torch, _, amc, atc = import_runtime()
    tokenizer, model = load_hf_model(torch=torch, auto_model_cls=amc, auto_tokenizer_cls=atc,
        model_name=args.model_name, device=args.device, device_map=args.device_map,
        torch_dtype=args.torch_dtype, trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    label_tokens = resolve_label_tokens(tokenizer, args.label_token_style)
    label_ids = label_tokens.token_ids
    device = next(model.parameters()).device
    layers = get_decoder_layers(model)
    hidden = model.config.hidden_size

    import numpy as np
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    summary: dict[str, Any] = {"model": args.model_name, "vary": vary, "target_var": target_var,
                               "forms": list(forms), "d_condition_on": args.d_condition_on,
                               "thresholds": {k: getattr(args, k) for k in
                                              ("gate_min", "anchor_min", "anchor_departure_min",
                                               "transfer_hi", "fail_lo")},
                               "cells": {}}

    # ---------- Test A (rotation-independent, run once) ----------
    print("\n===== Test A: behavioral gate =====")
    gate: dict[str, dict] = {}
    for f in forms:
        neg_rows = [r for r in form_rows[f] if is_negative_varied(r, vary)]
        scored = run_condition(model, layers, 0, "row", None, "none", neg_rows,
                               tokenizer, torch, device, label_ids, args.eval_batch_size, rng)
        if f in UNAMBIGUOUS_FORMS:
            st = {"n": len(scored),
                  "acc": mean(s["is_correct"] for s in scored),
                  "U_rate": mean(s["pred_label"] == "U" for s in scored),
                  "mean_margin": mean(margin_to(s["true_label"], float(s["R"]), float(s["U_gap"])) for s in scored)}
        else:
            st = {"n": len(scored),
                  "pred_dist": {lab: mean(s["pred_label"] == lab for s in scored) for lab in ("T", "F", "U")}}
        gate[f] = st
        print(f"  {f:14s} {st}")
    summary["behavioral_gate"] = gate

    # ---------- per-rotation cells ----------
    for rdir in args.rotation_dirs:
        cell = Path(rdir).name
        cell_json = output_dir / f"{cell}.json"
        if cell_json.exists():
            cached = json.loads(cell_json.read_text())
            summary["cells"][cell] = cached["cell_out"]
            records.extend(cached["records"])
            print(f"\n===== skip {cell}: cached =====")
            write_outputs(records, summary, output_dir)
            continue
        if not (Path(rdir) / "rotation_weight_metadata.json").exists():
            print(f"\n===== SKIP {cell}: no rotation =====")
            continue

        meta = json.loads((Path(rdir) / "rotation_weight_metadata.json").read_text())
        layer, rank = int(meta["layer"]), int(meta["rank"])
        site = str(meta.get("site", "claim_final"))
        R = torch.tensor(np.load(Path(rdir) / "rotation_weight.npy"), dtype=torch.float32, device=device)
        assert tuple(R.shape) == (hidden, rank)
        print(f"\n===== cell {cell} (L{layer}/{site}) =====")
        cell_records: list[dict] = []
        cell_out: dict[str, Any] = {"layer": layer, "site": site, "rank": rank}

        z_base = {f: collect_subspace_coordinates(model, layer, site, R, form_rows[f],
                                                  tokenizer, torch, device, args.eval_batch_size)
                  for f in forms}

        # ---------- Test B ----------
        dn_rows = form_rows["did_not"]
        neg_mask = torch.tensor([is_negative_varied(r, vary) for r in dn_rows])
        mu_pos = z_base["did_not"][~neg_mask].mean(0)
        mu_neg = z_base["did_not"][neg_mask].mean(0)
        u = mu_neg - mu_pos
        axis_len = float(u.norm())
        u = u / u.norm()
        within = z_base["did_not"][neg_mask] - mu_neg
        within_orth = float((within - torch.outer(within @ u, u)).norm(dim=1).mean())
        for f in forms:
            zn = z_base[f][neg_mask]
            a = float(((zn - mu_pos) @ u).mean()) / max(axis_len, 1e-8)
            drift = zn.mean(0) - mu_neg
            drift_orth = float((drift - (drift @ u) * u).norm())
            frac_neg = float((((zn - mu_neg) ** 2).sum(1) < ((zn - mu_pos) ** 2).sum(1)).float().mean())
            rec = {"cell": cell, "test": "B_geometry", "form": f, "n": int(neg_mask.sum()),
                   "axis_coord_rel": round(a, 3),
                   "orth_drift_over_within": round(drift_orth / max(within_orth, 1e-8), 3),
                   "centroid_classified_neg": round(frac_neg, 4)}
            cell_records.append(rec)
            print(f"  B {f:14s} axis={a:+.2f} drift/within={rec['orth_drift_over_within']:.2f} NN-neg={frac_neg:.3f}")
        cell_out["geometry"] = {r["form"]: r for r in cell_records if r["test"] == "B_geometry"}

        # ---------- Test C: true minimal pairs ----------
        c_stats: dict[str, dict] = {}
        for f in forms:
            z_src = collect_subspace_coordinates(model, layer, site, R, c_source_rows[f],
                                                 tokenizer, torch, device, args.eval_batch_size)
            scored = run_condition(model, layers, layer, site, R, "resample_same", c_base_rows,
                                   tokenizer, torch, device, label_ids, args.eval_batch_size, rng,
                                   donor_z=z_src)
            cf = [flipped(str(br["base_label"])) for br in c_base_rows]
            orig = [str(br["base_label"]) for br in c_base_rows]
            st = {"n": len(scored),
                  "counterfactual_acc": mean(s["pred_label"] == c for s, c in zip(scored, cf)),
                  "original_retention": mean(s["pred_label"] == o for s, o in zip(scored, orig)),
                  "U_rate": mean(s["pred_label"] == "U" for s in scored),
                  "descriptive_only": f in GRADED_FORMS}
            st["departure_rate"] = 1.0 - st["original_retention"]
            c_stats[f] = st
            cell_records.append({"cell": cell, "test": "C_interchange", "form": f,
                                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}})
            print(f"  C {f:14s} cf={st['counterfactual_acc']:.3f} ret={st['original_retention']:.3f} U={st['U_rate']:.3f}")
        cell_out["interchange"] = c_stats

        # ---------- Test D: cross-event same-value donors ----------
        d_col = args.d_condition_on or varied_polarity_column(vary)
        neg_indices = [i for i, r in enumerate(dn_rows) if is_negative_varied(r, vary)]
        d_stats: dict[str, dict] = {}
        for f in [f for f in forms if f in UNAMBIGUOUS_FORMS]:
            donors = []
            for i in neg_indices:
                row = dn_rows[i]
                cands = [j for j in neg_indices
                         if dn_rows[j]["base_event_id"] != row["base_event_id"]
                         and str(dn_rows[j].get(d_col)) == str(row.get(d_col))]
                donors.append(z_base[f][rng.choice(cands or [j for j in neg_indices if j != i])])
            scored = run_condition(model, layers, layer, site, R, "resample_same",
                                   [dn_rows[i] for i in neg_indices], tokenizer, torch, device,
                                   label_ids, args.eval_batch_size, rng, donor_z=torch.stack(donors))
            st = {"n": len(scored), "acc": mean(s["is_correct"] for s in scored),
                  "U_rate": mean(s["pred_label"] == "U" for s in scored), "condition_on": d_col}
            d_stats[f] = st
            cell_records.append({"cell": cell, "test": "D_donor_same", "form": f,
                                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}})
            print(f"  D {f:14s} acc={st['acc']:.3f} U={st['U_rate']:.3f} (cond={d_col})")
        cell_out["donor_same"] = d_stats

        cell_out["verdict"] = verdict(c_stats, gate, args)
        print(f"  VERDICT: {json.dumps(cell_out['verdict'])}")
        summary["cells"][cell] = cell_out
        records.extend(cell_records)
        cell_json.write_text(json.dumps(to_jsonable({"cell_out": cell_out, "records": cell_records}), indent=2))
        write_outputs(records, summary, output_dir)

    print(f"\nWrote {output_dir / 'lexeme_transfer.csv'} and lexeme_transfer_summary.json")
    return 0


def verdict(c_stats: dict, gate: dict, args) -> dict:
    anchor = c_stats.get("did_not")
    if not anchor:
        return {"label": "invalid_anchor", "reason": "no did_not anchor"}
    cf, dep = anchor["counterfactual_acc"], anchor["departure_rate"]
    out: dict[str, Any] = {"anchor_cf": round(cf, 3), "anchor_departure": round(dep, 3)}
    if dep < args.anchor_departure_min or cf < args.anchor_min:
        out.update(label="invalid_anchor",
                   reason=f"departure {dep:.2f} < {args.anchor_departure_min} or cf {cf:.2f} < {args.anchor_min}")
        return out
    ratios = {f: c_stats[f]["counterfactual_acc"] / cf
              for f in UNAMBIGUOUS_FORMS if f != "did_not" and f in c_stats}
    out["transfer_ratios"] = {f: round(r, 3) for f, r in ratios.items()}
    out["U_rates"] = {f: round(c_stats[f]["U_rate"], 3) for f in c_stats}
    gate_fail = [f for f in ratios if gate.get(f, {}).get("acc", 1.0) < args.gate_min]
    if gate_fail:
        out.update(label="inconclusive", reason=f"behavioral gate failed for {gate_fail}")
        return out
    never_r, ever_r = ratios.get("never"), ratios.get("did_not_ever")
    if never_r is not None and never_r >= args.transfer_hi and (ever_r is None or ever_r >= args.transfer_hi):
        out["label"] = "lexeme_invariant_polarity"
    elif never_r is not None and never_r <= args.fail_lo and ever_r is not None and ever_r >= args.transfer_hi:
        out["label"] = "not_template"
    else:
        out["label"] = "partial_transfer"
    return out


def write_outputs(records: list[dict], summary: dict, output_dir: Path) -> None:
    if records:
        fieldnames: list[str] = []
        for rec in records:
            for k in rec:
                if k not in fieldnames:
                    fieldnames.append(k)
        with (output_dir / "lexeme_transfer.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader(); w.writerows(records)
    (output_dir / "lexeme_transfer_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lexeme transfer test for DAS subspaces.")
    p.add_argument("--rotation-dirs", nargs="+", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--vary", choices=["claim", "assumption"], default="claim")
    p.add_argument("--target-var", default=None, help="Defaults: claim->pc, assumption->pi")
    p.add_argument("--forms", nargs="+", default=list(NEGATION_FORMS), choices=list(NEGATION_FORMS))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-events", type=int, default=60, help="Subsample base events; 0 = all")
    p.add_argument("--d-condition-on", default=None,
                   help="Donor class column for Test D; default = varied polarity column. Use base_label for REL cells.")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--label-token-style", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate-min", type=float, default=0.9)
    p.add_argument("--anchor-min", type=float, default=0.5)
    p.add_argument("--anchor-departure-min", type=float, default=0.8)
    p.add_argument("--transfer-hi", type=float, default=0.8)
    p.add_argument("--fail-lo", type=float, default=0.4)
    p.add_argument("--output-dir", default="data/das/lexeme_transfer")
    p.add_argument("--device", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
