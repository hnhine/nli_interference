"""Generate evaluation-only triples for the joint m/rho gate test."""

from __future__ import annotations

import argparse
from collections import Counter

from interference_suite.io_utils import write_rows_csv, write_rows_jsonl
from interference_suite.joint_gate_data import generate_joint_gate_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-base-events", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--reference-samples",
        default="data/das/rho_v1/pairs.csv",
        help="Use unique claim events from this held-out DAS dataset.",
    )
    parser.add_argument("--reference-split", default="test")
    parser.add_argument(
        "--rho-source-regimes",
        type=int,
        nargs="+",
        default=[0, 1],
        choices=[0, 1],
    )
    parser.add_argument("--output", default="data/das/joint_gate_v1/triples.csv")
    parser.add_argument("--jsonl", default="data/das/joint_gate_v1/triples.jsonl")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = generate_joint_gate_rows(
        n_base_events=args.n_base_events,
        seed=args.seed,
        reference_samples=args.reference_samples,
        reference_split=args.reference_split,
        rho_source_regimes=args.rho_source_regimes,
    )
    write_rows_csv(rows, args.output)
    if args.jsonl:
        write_rows_jsonl(rows, args.jsonl)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print("By cell:", dict(sorted(Counter(row["cell_type"] for row in rows).items())))
    print("By rho source m:", dict(sorted(Counter(row["rho_source_m"] for row in rows).items())))
    print("Strict assembly:", dict(sorted(Counter(row["strict_assembly"] for row in rows).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
