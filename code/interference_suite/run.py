"""Command line entry point for the interference experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generation import EXPERIMENTS, generate_exp6, generate_suite, generate_supplements, supplemental_sections_for_experiments
from .io_utils import read_rows_csv, write_rows_csv, write_rows_jsonl
from .metrics import load_results, write_summary_outputs
from .model import DEFAULT_CACHE_DIR, evaluate_rows
from .plots import plot_all


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        rows = build_rows_from_args(args)
        output_dir = Path(args.output_dir)
        csv_path = write_rows_csv(rows, output_dir / args.csv_name)
        if args.jsonl:
            write_rows_jsonl(rows, output_dir / args.jsonl_name)
        print(f"Wrote {len(rows)} generated samples to {csv_path}")
        return 0

    if args.command == "run":
        rows = build_rows_from_args(args)
        if args.limit is not None:
            rows = rows[: args.limit]
        rows = score_rows_from_args(rows, args)
        output_dir = Path(args.output_dir)
        csv_path = write_rows_csv(rows, output_dir / args.csv_name)
        if args.jsonl:
            write_rows_jsonl(rows, output_dir / args.jsonl_name)
        df = load_results(csv_path)
        write_main_pipeline_summaries(df, output_dir)
        if args.plots:
            plot_paths = plot_all(df, output_dir / "plots")
            print(f"Wrote {len(plot_paths)} plots to {output_dir / 'plots'}")
        print(f"Wrote {len(rows)} scored samples to {csv_path}")
        print(f"Wrote summary to {output_dir / 'summary' / 'summary_metrics.json'}")
        return 0

    if args.command == "exp6-generate":
        rows = build_exp6_rows_from_args(args)
        output_dir = Path(args.output_dir)
        csv_path = write_rows_csv(rows, output_dir / args.csv_name)
        if args.jsonl:
            write_rows_jsonl(rows, output_dir / args.jsonl_name)
        print(f"Wrote {len(rows)} Exp6 generated samples to {csv_path}")
        return 0

    if args.command == "exp6-run":
        rows = read_rows_csv(args.samples)
        if args.limit is not None:
            rows = rows[: args.limit]
        rows = score_rows_from_args(rows, args)
        output_dir = Path(args.output_dir)
        csv_path = write_rows_csv(rows, output_dir / args.csv_name)
        if args.jsonl:
            write_rows_jsonl(rows, output_dir / args.jsonl_name)
        df = load_results(csv_path)
        write_main_pipeline_summaries(df, output_dir)
        if args.plots:
            plot_paths = plot_all(df, output_dir / "plots")
            print(f"Wrote {len(plot_paths)} plots to {output_dir / 'plots'}")
        print(f"Wrote {len(rows)} Exp6 scored samples to {csv_path}")
        print(f"Wrote summary to {output_dir / 'summary' / 'summary_metrics.json'}")
        return 0

    if args.command == "exp6-summarize":
        df = load_results(args.scored)
        output_dir = Path(args.output_dir)
        write_summary_outputs(df, output_dir)
        if args.plots:
            plot_all(df, output_dir / "plots")
        print(f"Wrote Exp6 summary outputs to {output_dir}")
        return 0

    if args.command == "summarize":
        df = load_results(args.input_csv)
        output_dir = Path(args.output_dir)
        write_summary_outputs(df, output_dir)
        if args.plots:
            plot_all(df, output_dir / "plots")
        print(f"Wrote summary outputs to {output_dir}")
        return 0

    parser.print_help()
    return 1


def write_main_pipeline_summaries(df, output_dir: Path) -> None:
    write_summary_outputs(df, output_dir / "summary")


def score_rows_from_args(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    print(f"Using Hugging Face cache directory: {args.cache_dir}")
    if args.local_files_only:
        print("local_files_only=True: missing model files will fail instead of downloading.")
    return evaluate_rows(
        rows,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        label_token_style=args.label_token_style,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )


def build_exp6_rows_from_args(args: argparse.Namespace) -> list[dict[str, object]]:
    include_exp6a = bool(args.include_exp6a or not args.include_exp6b)
    rows = generate_exp6(
        n_base_events=args.n_base_events,
        seed=args.seed,
        base_events_from_csv=args.base_events_from_csv,
        include_exp6a=include_exp6a,
        include_exp6b=args.include_exp6b,
        exp6b_allowed_verbs=parse_exp6b_verbs(args.exp6b_strict_verbs),
    )
    for row in rows:
        row["run_family"] = "exp6_suite"
    for idx, row in enumerate(rows):
        row["row_id"] = idx
    return rows


def parse_exp6b_verbs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_rows_from_args(args: argparse.Namespace) -> list[dict[str, object]]:
    experiments = args.experiments
    if experiments == ["all"]:
        experiments = list(EXPERIMENTS)
    original_rows = generate_suite(
        n_base_events=args.n_base_events,
        seed=args.seed,
        experiments=experiments,
        include_exp3_sanity=args.include_exp3_sanity,
        include_exp4_source_only=not args.no_exp4_source_only,
    )
    for row in original_rows:
        row["run_family"] = "original_suite"

    supplement_sections = supplemental_sections_for_experiments(experiments)
    supplemental_rows = generate_supplements(
        n_base_events=args.n_base_events,
        seed=args.seed,
        base_events_from_csv="none",
        sections=supplement_sections,
    )
    for row in supplemental_rows:
        row["run_family"] = "supplemental_suite"
    rows = original_rows + supplemental_rows

    for idx, row in enumerate(rows):
        row["row_id"] = idx
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, score, and summarize the full interference NLI suite.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate the full merged sample suite without running a model.")
    add_generation_args(generate)
    generate.add_argument("--jsonl", action="store_true", help="Also write JSONL output.")

    run = subparsers.add_parser("run", help="Generate and score the full merged suite with a causal LM.")
    add_generation_args(run)
    add_model_args(run)
    run.add_argument("--limit", type=int, default=None, help="Score only the first N generated rows.")
    run.add_argument("--plots", action="store_true", help="Write standard plots after scoring.")
    run.add_argument("--jsonl", action="store_true", help="Also write JSONL output.")

    exp6_generate = subparsers.add_parser("exp6-generate", help="Generate Exp6 samples without running a model.")
    add_exp6_generation_args(exp6_generate)
    exp6_generate.add_argument("--jsonl", action="store_true", help="Also write JSONL output.")

    exp6_run = subparsers.add_parser("exp6-run", help="Score a generated Exp6 CSV with a causal LM.")
    exp6_run.add_argument("--samples", required=True, help="Generated Exp6 samples CSV.")
    exp6_run.add_argument("--output-dir", default="data/exp6")
    exp6_run.add_argument("--csv-name", default="scored.csv")
    exp6_run.add_argument("--jsonl-name", default="scored.jsonl")
    exp6_run.add_argument("--jsonl", action="store_true", help="Also write JSONL output.")
    exp6_run.add_argument("--limit", type=int, default=None, help="Score only the first N rows from --samples.")
    add_model_args(exp6_run)
    exp6_run.add_argument("--plots", action="store_true", help="Write Exp6 plots after scoring.")

    exp6_summarize = subparsers.add_parser("exp6-summarize", help="Summarize an existing Exp6 scored CSV.")
    exp6_summarize.add_argument("--scored", required=True)
    exp6_summarize.add_argument("--output-dir", default="data/exp6/summary")
    exp6_summarize.add_argument("--plots", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Summarize an existing full-suite scored CSV.")
    summarize.add_argument("--input-csv", required=True)
    summarize.add_argument("--output-dir", default="interference/data/summary")
    summarize.add_argument("--plots", action="store_true")

    return parser


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="Explicit device when --device-map none is used.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map. Use 'none' to disable.")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "none", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--label-token-style", default="auto", choices=["auto", "bare", "space", "newline"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face hub cache directory. Use default/none to let Transformers choose.")
    parser.add_argument("--local-files-only", action="store_true", help="Load only from --cache-dir and fail if any model file is missing.")


def add_exp6_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-base-events", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-events-from-csv", default="none")
    parser.add_argument("--include-exp6a", action="store_true", help="Include Exp6A absolute negation rows. Default when --include-exp6b is not set.")
    parser.add_argument("--include-exp6b", action="store_true", help="Also include the strict-verb Exp6B frequency-negation scaffold.")
    parser.add_argument("--exp6b-strict-verbs", default="visit,explore,enter")
    parser.add_argument("--output-dir", default="data/exp6")
    parser.add_argument("--csv-name", default="samples.csv")
    parser.add_argument("--jsonl-name", default="samples.jsonl")


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-base-events", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="interference/data/generated")
    parser.add_argument("--csv-name", default="samples.csv")
    parser.add_argument("--jsonl-name", default="samples.jsonl")
    parser.add_argument("--experiments", nargs="+", default=["all"], choices=["all", *EXPERIMENTS], help=argparse.SUPPRESS)
    parser.add_argument("--include-exp3-sanity", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-exp4-source-only", action="store_true", help=argparse.SUPPRESS)


if __name__ == "__main__":
    raise SystemExit(main())
