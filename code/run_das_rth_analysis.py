"""Analyze R^T h coordinate distributions: natural runs vs post-swap values.

Input: hidden_states.pt from `das-dump-hidden` (base/source hidden vectors at
the intervention site) + a trained rotation_weight. Because the interchange
sets the base's subspace coordinates exactly to the source's natural values
(z_swap = R^T h_src for orthonormal R), comparing z_src against the natural
per-class distribution answers whether hyper-decisive readouts come from
extreme coordinates (they should NOT, by construction) or from the hybrid
combination of base-rest + source-subspace.

Reported per class (T/F label of the prompt):
  - separation of natural clusters along the top discriminating direction
  - swap displacement ||z_src - z_base|| vs natural within-class spread
  - percentile of each z_src inside its own class's natural distribution

Example:
    python code/run_das_rth_analysis.py \
        --hidden data/das/hidden_L18_main_test/hidden_states.pt \
        --rotation data/das/qwen3_8_pc_1000_v2_l18_r32_claim_b32_allcontrols/rotation_weight.npy \
        --output-dir data/das/rth_analysis_L18
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    args = build_parser().parse_args()
    import numpy as np
    import torch

    blob = torch.load(args.hidden, map_location="cpu", weights_only=False)
    base_h = blob["base_hidden"].float().numpy()
    source_h = blob["source_hidden"].float().numpy()
    meta = blob["metadata_rows"]
    rotation = np.load(args.rotation)  # (hidden, rank)
    ortho_err = float(np.abs(rotation.T @ rotation - np.eye(rotation.shape[1])).max())

    z_base = base_h @ rotation      # (n, rank)
    z_src = source_h @ rotation
    base_labels = np.array([str(m["base_label"]) for m in meta])
    src_labels = np.array([str(m["source_label"]) for m in meta])

    # Natural coordinate pool per class: base and source rows are all natural runs.
    z_nat = np.concatenate([z_base, z_src], axis=0)
    nat_labels = np.concatenate([base_labels, src_labels], axis=0)
    classes = sorted(set(nat_labels))

    # Top discriminating direction inside the subspace (difference of class means).
    stats: dict = {
        "n_pairs": int(z_base.shape[0]),
        "rank": int(rotation.shape[1]),
        "rotation_orthonormal_max_err": ortho_err,
        "classes": {},
    }
    if len(classes) == 2:
        mu_a = z_nat[nat_labels == classes[0]].mean(axis=0)
        mu_b = z_nat[nat_labels == classes[1]].mean(axis=0)
        w = mu_b - mu_a
        w /= np.linalg.norm(w)
        proj = z_nat @ w
        pa, pb = proj[nat_labels == classes[0]], proj[nat_labels == classes[1]]
        pooled_sd = float(np.sqrt((pa.var() + pb.var()) / 2))
        stats["top_direction"] = {
            "classes": classes,
            "mean_gap": float(pb.mean() - pa.mean()),
            "pooled_sd": pooled_sd,
            "separation_d": float((pb.mean() - pa.mean()) / pooled_sd),
            "overlap_rate": float(((pa > (pa.mean() + pb.mean()) / 2).mean() + (pb < (pa.mean() + pb.mean()) / 2).mean()) / 2),
        }

    # Swap displacement vs natural within-class spread.
    disp = np.linalg.norm(z_src - z_base, axis=1)
    within = []
    for c in classes:
        zc = z_nat[nat_labels == c]
        within.append(np.linalg.norm(zc - zc.mean(axis=0), axis=1))
    within = np.concatenate(within)
    stats["swap_displacement"] = {
        "mean": float(disp.mean()),
        "median": float(np.median(disp)),
        "natural_within_class_norm_mean": float(within.mean()),
        "ratio_mean": float(disp.mean() / within.mean()),
    }

    # Are z_src values extreme within their own class's natural distribution?
    for c in classes:
        zc = z_nat[nat_labels == c]
        mu, sd = zc.mean(axis=0), zc.std(axis=0) + 1e-8
        zs = z_src[src_labels == c]
        m_dist = np.sqrt((((zs - mu) / sd) ** 2).mean(axis=1))
        m_nat = np.sqrt((((zc - mu) / sd) ** 2).mean(axis=1))
        stats["classes"][c] = {
            "n_src": int(len(zs)),
            "src_norm_z_mean": float(m_dist.mean()),
            "natural_norm_z_mean": float(m_nat.mean()),
            "src_beyond_natural_p95_rate": float((m_dist > np.percentile(m_nat, 95)).mean()),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rth_analysis.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"n={stats['n_pairs']} rank={stats['rank']} | rotation orthonormal err={ortho_err:.2e}")
    if "top_direction" in stats:
        td = stats["top_direction"]
        print(f"Top direction {td['classes']}: gap={td['mean_gap']:+.3f}  d(Cohen)={td['separation_d']:.2f}  overlap={td['overlap_rate']:.3%}")
    sd = stats["swap_displacement"]
    print(f"Swap displacement: mean={sd['mean']:.3f} vs natural within-class norm {sd['natural_within_class_norm_mean']:.3f} (ratio {sd['ratio_mean']:.2f}x)")
    for c, cs in stats["classes"].items():
        print(f"class {c}: src normalized-z mean={cs['src_norm_z_mean']:.2f} vs natural {cs['natural_norm_z_mean']:.2f} | vuot p95 tu nhien: {cs['src_beyond_natural_p95_rate']:.1%}")
    print(f"Wrote {output_dir / 'rth_analysis.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R^T h coordinate analysis for DAS hidden dumps.")
    parser.add_argument("--hidden", required=True, help="hidden_states.pt from das-dump-hidden")
    parser.add_argument("--rotation", required=True, help="rotation_weight.npy from a DAS run")
    parser.add_argument("--output-dir", default="data/das/rth_analysis")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
