"""Generate the exact 42k-prompt Section 4 behavioral screening protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from interference_suite.generation import generate_suite, generate_supplements
from interference_suite.io_utils import write_rows_csv


EXPECTED_COUNTS = {
    "exp1_phase_flip": 4_000,
    "exp2_counterbalanced_overlap": 20_000,
    "exp3_clean_selection": 18_000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-base-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="data/section4_screening/protocol_seed0",
    )
    args = parser.parse_args()
    if args.n_base_events != 1000:
        raise ValueError("The paper screening protocol requires exactly 1000 base events")

    original = generate_suite(
        n_base_events=args.n_base_events,
        seed=args.seed,
        experiments=["exp1_phase_flip", "exp3_clean_selection"],
        include_exp3_sanity=False,
    )
    gate = generate_supplements(
        n_base_events=args.n_base_events,
        seed=args.seed,
        base_events_from_csv="none",
        sections=["exp2_counterbalanced_overlap"],
    )
    blocks = {
        "polarity_composition": [
            row for row in original if row["experiment"] == "exp1_phase_flip"
        ],
        "matching_gate": gate,
        "selective_retrieval": [
            row for row in original if row["experiment"] == "exp3_clean_selection"
        ],
    }
    rows = []
    for block, block_rows in blocks.items():
        for row in block_rows:
            row["run_family"] = "section4_screening"
            row["screening_block"] = block
            rows.append(row)
    for index, row in enumerate(rows):
        row["row_id"] = index

    counts = Counter(str(row["experiment"]) for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"Protocol count mismatch: {dict(counts)} != {EXPECTED_COUNTS}")
    if len(rows) != 42_000:
        raise RuntimeError(f"Expected 42,000 rows, found {len(rows)}")

    output = Path(args.output_dir)
    samples_path = write_rows_csv(rows, output / "samples.csv")
    manifest = {
        "schema_version": 1,
        "run_type": "section4_behavioral_screening_protocol",
        "generator_seed": args.seed,
        "n_base_events": args.n_base_events,
        "n_rows": len(rows),
        "counts_by_experiment": dict(counts),
        "samples": {
            "path": str(samples_path.resolve()),
            "sha256": sha256(samples_path),
            "bytes": samples_path.stat().st_size,
        },
        "contract": (
            "All screened models must score this exact samples.csv; do not regenerate "
            "per model or substitute older full-suite outputs."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} canonical Section 4 prompts to {samples_path}")
    print(dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
