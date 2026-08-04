"""Subspace agreement across DAS training seeds.

Given several seed run-dirs that each contain a trained cell exporting
rotation_weight.npy (shape [hidden, rank]), report how close the recovered
rank-r subspaces are across seeds via principal angles.

For two orthonormal bases U, V (columns span the subspace), the singular values
of U^T V are cos(theta_i) of the principal angles. mean_cos ~= 1 and
max_angle ~= 0 deg mean "the seeds recover the *same* subspace", not merely
"each seed finds *a* good subspace".

Usage:
    python code/analyze_subspace_agreement.py \
        data/das/seed_rho_qwen_L18/seed0/L18_claim_final \
        data/das/seed_rho_qwen_L18/seed1/L18_claim_final \
        ... (one cell dir per seed)
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def load_basis(cell_dir: Path) -> np.ndarray:
    w = np.load(cell_dir / "rotation_weight.npy")  # [hidden, rank]
    # Orthonormalize columns (training re-orthonormalizes, but be safe).
    q, _ = np.linalg.qr(w)
    return q[:, : w.shape[1]]


def principal_angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(u.T @ v, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))  # angles in degrees, ascending


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("need >=2 seed cell dirs", file=sys.stderr)
        return 2
    dirs = [Path(a) for a in argv[1:]]
    bases = [load_basis(d) for d in dirs]
    rank = bases[0].shape[1]
    print(f"{len(bases)} seeds, rank {rank}\n")

    mean_cos_all = []
    for (i, a), (j, b) in combinations(enumerate(bases), 2):
        ang = principal_angles(a, b)
        mean_cos = float(np.cos(np.radians(ang)).mean())
        mean_cos_all.append(mean_cos)
        print(f"seed{i} vs seed{j}: mean_cos={mean_cos:.4f}  "
              f"max_angle={ang.max():6.2f} deg  median_angle={np.median(ang):6.2f} deg")

    print(f"\nOVERALL mean principal-angle cosine over "
          f"{len(mean_cos_all)} pairs: {np.mean(mean_cos_all):.4f} "
          f"(1.0 = identical subspace)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
