"""Measure alignment between DAS subspaces and the output logit-contrast plane.

For label tokens T/F/U the decision space at the unembedding is the plane
L = span{w_T - w_U, w_F - w_U}, pulled back through the final RMSNorm by the
elementwise gain (the per-example 1/rms scale does not change directions).
For every stored rotation this script reports the principal cosines between
the DAS subspace and L, plus the fraction of each contrast direction captured
inside the subspace, against a random-subspace baseline of matched rank.

CPU only; reads lm_head rows directly from safetensors shards.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from interference_suite.model import DEFAULT_CACHE_DIR, resolve_label_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Principal angles between DAS rotations and the logit-contrast plane."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--rotation-dirs",
        nargs="+",
        required=True,
        help="Sweep dirs containing L<layer>_<site>/rotation_weight.npy",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE_DIR))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, default=20)
    return parser.parse_args()


def snapshot_dir(cache_dir: Path, model_name: str) -> Path:
    repo = f"models--{model_name.replace('/', '--')}"
    snapshots = sorted((cache_dir / repo / "snapshots").iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshot under {cache_dir / repo}")
    return snapshots[-1]


def load_rows(snapshot: Path, tensor_name: str, row_ids: list[int] | None) -> np.ndarray:
    from safetensors import safe_open

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if tensor_name not in weight_map:
            raise KeyError(tensor_name)
        shard = snapshot / weight_map[tensor_name]
    else:
        shard = snapshot / "model.safetensors"
    # framework="pt" handles bfloat16 shards; convert to float32 numpy.
    with safe_open(shard, framework="pt") as handle:
        tensor = handle.get_slice(tensor_name)
        if row_ids is None:
            return tensor[:].float().numpy()
        return np.stack([tensor[i].float().numpy() for i in row_ids])


def logit_plane(model_name: str, cache_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(cache_dir), local_files_only=True
    )
    label_ids = resolve_label_tokens(tokenizer).token_ids
    snapshot = snapshot_dir(cache_dir, model_name)
    ordered = [label_ids[label] for label in ("T", "F", "U")]
    try:
        rows = load_rows(snapshot, "lm_head.weight", ordered)
    except KeyError:
        rows = load_rows(snapshot, "model.embed_tokens.weight", ordered)
    gain = load_rows(snapshot, "model.norm.weight", None)
    w_t, w_f, w_u = (gain * row for row in rows)
    contrasts = {
        "TU": w_t - w_u,
        "FU": w_f - w_u,
        "TF": w_t - w_f,
    }
    plane = np.linalg.qr(np.stack([contrasts["TU"], contrasts["FU"]], axis=1))[0]
    return plane, contrasts


def captured_fraction(basis: np.ndarray, direction: np.ndarray) -> float:
    unit = direction / np.linalg.norm(direction)
    return float(np.linalg.norm(basis.T @ unit))


def principal_cosines(basis: np.ndarray, plane: np.ndarray) -> tuple[float, float]:
    singular = np.linalg.svd(basis.T @ plane, compute_uv=False)
    return float(singular[0]), float(singular[1])


def random_baseline(
    dim: int, rank: int, plane: np.ndarray, contrasts: dict[str, np.ndarray], seeds: int
) -> dict[str, float]:
    cos1_values, cap_values = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        basis = np.linalg.qr(rng.standard_normal((dim, rank)))[0]
        cos1_values.append(principal_cosines(basis, plane)[0])
        cap_values.append(captured_fraction(basis, contrasts["TU"]))
    return {
        "rand_cos1_mean": float(np.mean(cos1_values)),
        "rand_cos1_max": float(np.max(cos1_values)),
        "rand_cap_TU_mean": float(np.mean(cap_values)),
    }


def main() -> int:
    args = parse_args()
    plane, contrasts = logit_plane(args.model_name, args.cache_dir)
    dim = plane.shape[0]
    baselines: dict[int, dict[str, float]] = {}
    rows = []
    for sweep in args.rotation_dirs:
        sweep_path = Path(sweep)
        for weight_path in sorted(sweep_path.glob("L*/rotation_weight.npy")):
            cell = weight_path.parent.name
            layer, site = cell.split("_", 1)
            basis = np.linalg.qr(np.load(weight_path).astype(np.float32))[0]
            rank = basis.shape[1]
            if rank not in baselines:
                baselines[rank] = random_baseline(
                    dim, rank, plane, contrasts, args.random_seeds
                )
            cos1, cos2 = principal_cosines(basis, plane)
            rows.append(
                {
                    "model": args.model_name,
                    "sweep_dir": sweep_path.name,
                    "cell": cell,
                    "layer": int(layer.lstrip("L")),
                    "site": site,
                    "rank": rank,
                    "cos1": round(cos1, 4),
                    "cos2": round(cos2, 4),
                    "cap_TU": round(captured_fraction(basis, contrasts["TU"]), 4),
                    "cap_FU": round(captured_fraction(basis, contrasts["FU"]), 4),
                    "cap_TF": round(captured_fraction(basis, contrasts["TF"]), 4),
                    **{key: round(value, 4) for key, value in baselines[rank].items()},
                }
            )
    if not rows:
        raise SystemExit("No rotation_weight.npy found under the given dirs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['sweep_dir']:<44} {row['cell']:<20} cos1={row['cos1']:.3f} "
            f"cap_TF={row['cap_TF']:.3f} cap_TU={row['cap_TU']:.3f} "
            f"(rand cos1 {row['rand_cos1_mean']:.3f})"
        )
    print(f"Wrote {args.output} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
