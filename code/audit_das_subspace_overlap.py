"""Audit overlap and principal angles between same-layer m and rho DAS bases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rotation(directory: str | Path, expected_target: str) -> tuple[np.ndarray, dict[str, Any]]:
    root = Path(directory).resolve()
    metadata_path = root / "rotation_weight_metadata.json"
    weight_path = root / "rotation_weight.npy"
    if not metadata_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Incomplete rotation directory: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("target_var")) != expected_target:
        raise ValueError(
            f"{root}: target_var={metadata.get('target_var')!r}, expected {expected_target!r}"
        )
    raw = np.load(weight_path)
    if raw.ndim != 2:
        raise ValueError(f"{weight_path}: expected a matrix, found shape {raw.shape}")
    rank = int(metadata["rank"])
    if raw.shape[1] != rank:
        raise ValueError(f"{weight_path}: shape {raw.shape} conflicts with rank={rank}")
    if np.linalg.matrix_rank(raw) != rank:
        raise ValueError(f"{weight_path}: matrix is not full column rank")
    basis, _ = np.linalg.qr(raw.astype(np.float64), mode="reduced")
    provenance = {
        "directory": str(root),
        "metadata": metadata,
        "metadata_sha256": sha256_file(metadata_path),
        "weight_sha256": sha256_file(weight_path),
        "raw_shape": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "qr_gram_max_abs_error": float(
            np.max(np.abs(basis.T @ basis - np.eye(rank)))
        ),
    }
    return basis, provenance


def validate_pair(m_info: dict[str, Any], rho_info: dict[str, Any]) -> dict[str, Any]:
    m_meta = m_info["metadata"]
    rho_meta = rho_info["metadata"]
    for field in ("model_name", "layer", "site", "rank", "component"):
        if str(m_meta.get(field)) != str(rho_meta.get(field)):
            raise ValueError(
                f"m/rho metadata mismatch for {field}: "
                f"{m_meta.get(field)!r} != {rho_meta.get(field)!r}"
            )
    if m_info["raw_shape"] != rho_info["raw_shape"]:
        raise ValueError(
            f"m/rho basis shapes differ: {m_info['raw_shape']} != {rho_info['raw_shape']}"
        )
    return {
        "model_name": str(m_meta["model_name"]),
        "layer": int(m_meta["layer"]),
        "site": str(m_meta["site"]),
        "rank": int(m_meta["rank"]),
        "component": str(m_meta["component"]),
        "hidden_size": int(m_info["raw_shape"][0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-rotation-dir", required=True)
    parser.add_argument("--rho-rotation-dir", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    m_basis, m_info = load_rotation(args.m_rotation_dir, "m")
    rho_basis, rho_info = load_rotation(args.rho_rotation_dir, "rho")
    contract = validate_pair(m_info, rho_info)
    hidden = contract["hidden_size"]
    rank = contract["rank"]

    cross = m_basis.T @ rho_basis
    singular_values = np.linalg.svd(cross, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    cos2 = singular_values**2
    principal_angles_radians = np.arccos(singular_values)
    principal_angles_degrees = np.degrees(principal_angles_radians)
    frobenius_squared = float(np.sum(cross**2))
    normalized_overlap = frobenius_squared / rank
    random_baseline = rank / hidden

    spectrum = []
    cumulative = 0.0
    for index, (cosine, squared, radians, degrees) in enumerate(
        zip(
            singular_values,
            cos2,
            principal_angles_radians,
            principal_angles_degrees,
        ),
        start=1,
    ):
        cumulative += float(squared)
        spectrum.append(
            {
                "principal_index": index,
                "principal_cosine": float(cosine),
                "principal_cosine_squared": float(squared),
                "principal_angle_radians": float(radians),
                "principal_angle_degrees": float(degrees),
                "cumulative_frobenius_squared": cumulative,
                "cumulative_normalized_overlap": cumulative / rank,
            }
        )

    summary = {
        "schema_version": 1,
        "run_type": "das_same_layer_m_rho_overlap_audit",
        **contract,
        "training_seed": args.training_seed,
        "metric_definition": "||B_m^T B_rho||_F^2 / k",
        "frobenius_squared": frobenius_squared,
        "normalized_frobenius_squared": normalized_overlap,
        "random_subspace_expectation_k_over_d": random_baseline,
        "overlap_to_random_expectation_ratio": normalized_overlap / random_baseline,
        "principal_angle_summary": {
            "min_degrees": float(np.min(principal_angles_degrees)),
            "median_degrees": float(np.median(principal_angles_degrees)),
            "mean_degrees": float(np.mean(principal_angles_degrees)),
            "max_degrees": float(np.max(principal_angles_degrees)),
            "max_principal_cosine": float(np.max(singular_values)),
            "mean_principal_cosine_squared": float(np.mean(cos2)),
            "n_cosine_gt_0_7": int(np.sum(singular_values > 0.7)),
            "n_cosine_gt_0_5": int(np.sum(singular_values > 0.5)),
        },
        "interpretation_scope": (
            "Low overlap only tests direct linear leakage between m and rho bases; "
            "it does not identify or rule out a shared downstream T/F decision axis."
        ),
        "m_rotation": m_info,
        "rho_rotation": rho_info,
    }

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "overlap_summary.json"
    spectrum_path = output / "principal_angles.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with spectrum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spectrum[0]))
        writer.writeheader()
        writer.writerows(spectrum)

    print(f"Wrote {summary_path}")
    print(f"Wrote {spectrum_path}")
    print(
        f"k={rank}, d={hidden}, ||Bm^T Brho||_F^2/k={normalized_overlap:.6f}, "
        f"random k/d={random_baseline:.6f}, ratio={normalized_overlap/random_baseline:.3f}"
    )
    print(
        "principal angles (deg): "
        f"min={np.min(principal_angles_degrees):.2f}, "
        f"median={np.median(principal_angles_degrees):.2f}, "
        f"mean={np.mean(principal_angles_degrees):.2f}, "
        f"max={np.max(principal_angles_degrees):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
