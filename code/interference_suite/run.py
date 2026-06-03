"""Command line entry point for the interference experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generation import EXPERIMENTS, generate_suite
from .io_utils import write_rows_csv, write_rows_jsonl
from .metrics import load_results, write_summary_outputs
from .model import DEFAULT_CACHE_DIR, evaluate_rows
from .next_generation import NEXT_SECTIONS, generate_next_run
from .next_metrics import write_next_run_summary
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
        print(f"Using Hugging Face cache directory: {args.cache_dir}")
        if args.local_files_only:
            print("local_files_only=True: missing model files will fail instead of downloading.")
        rows = evaluate_rows(
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
        output_dir = Path(args.output_dir)
        csv_path = write_rows_csv(rows, output_dir / args.csv_name)
        if args.jsonl:
            write_rows_jsonl(rows, output_dir / args.jsonl_name)
        df = load_results(csv_path)
        summary = write_summary_outputs(df, output_dir / "summary")
        if args.plots:
            plot_paths = plot_all(df, output_dir / "plots")
            print(f"Wrote {len(plot_paths)} plots to {output_dir / 'plots'}")
        print(f"Wrote {len(rows)} scored samples to {csv_path}")
        print(f"Wrote summary metrics to {output_dir / 'summary' / 'summary_metrics.json'}")
        return 0 if summary else 1

    if args.command == "summarize":
        df = load_results(args.input_csv)
        output_dir = Path(args.output_dir)
        write_summary_outputs(df, output_dir)
        if args.plots:
            plot_all(df, output_dir / "plots")
        print(f"Wrote summary outputs to {output_dir}")
        return 0

    if args.command == "next-generate":
        rows = build_next_rows_from_args(args)
        output_dir = Path(args.output_dir)
        csv_path = write_rows_csv(rows, output_dir / args.csv_name)
        if args.jsonl:
            write_rows_jsonl(rows, output_dir / args.jsonl_name)
        print(f"Wrote {len(rows)} next-run generated samples to {csv_path}")
        return 0

    if args.command == "next-run":
        rows = build_next_rows_from_args(args)
        if args.limit is not None:
            rows = rows[: args.limit]
        print(f"Using Hugging Face cache directory: {args.cache_dir}")
        if args.local_files_only:
            print("local_files_only=True: missing model files will fail instead of downloading.")
        rows = evaluate_rows(
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
        output_dir = Path(args.output_dir)
        csv_path = write_rows_csv(rows, output_dir / args.csv_name)
        if args.jsonl:
            write_rows_jsonl(rows, output_dir / args.jsonl_name)
        df = load_results(csv_path)
        write_next_run_summary(df, output_dir / "summary")
        print(f"Wrote {len(rows)} scored next-run samples to {csv_path}")
        print(f"Wrote next-run summary metrics to {output_dir / 'summary' / 'next_run_summary_metrics.json'}")
        return 0

    if args.command == "next-summarize":
        df = load_results(args.input_csv)
        output_dir = Path(args.output_dir)
        write_next_run_summary(df, output_dir)
        print(f"Wrote next-run summary outputs to {output_dir}")
        return 0

    parser.print_help()
    return 1


def build_next_rows_from_args(args: argparse.Namespace) -> list[dict[str, object]]:
    sections = args.sections
    if sections == ["all"]:
        sections = list(NEXT_SECTIONS)
    return generate_next_run(
        n_base_events=args.n_base_events,
        seed=args.seed,
        base_events_from_csv=args.base_events_from_csv,
        sections=sections,
    )


def build_rows_from_args(args: argparse.Namespace) -> list[dict[str, object]]:
    experiments = args.experiments
    if experiments == ["all"]:
        experiments = list(EXPERIMENTS)
    return generate_suite(
        n_base_events=args.n_base_events,
        seed=args.seed,
        experiments=experiments,
        include_exp3_sanity=args.include_exp3_sanity,
        include_exp4_source_only=not args.no_exp4_source_only,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, score, and summarize interference NLI probes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate samples without running a model.")
    add_generation_args(generate)
    generate.add_argument("--jsonl", action="store_true", help="Also write JSONL output.")

    run = subparsers.add_parser("run", help="Generate samples and score them with a causal LM.")
    add_generation_args(run)
    run.add_argument("--model-name", required=True, help="Hugging Face model id or local model path.")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--limit", type=int, default=None, help="Score only the first N generated rows.")
    run.add_argument("--device", default=None, help="Explicit device when --device-map none is used.")
    run.add_argument("--device-map", default="auto", help="Transformers device_map. Use 'none' to disable.")
    run.add_argument("--torch-dtype", default="auto", choices=["auto", "none", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    run.add_argument("--label-token-style", default="auto", choices=["auto", "bare", "space", "newline"])
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face hub cache directory. Use default/none to let Transformers choose.")
    run.add_argument("--local-files-only", action="store_true", help="Load only from --cache-dir and fail if any model file is missing.")
    run.add_argument("--plots", action="store_true", help="Write standard plots after scoring.")
    run.add_argument("--jsonl", action="store_true", help="Also write JSONL output.")

    summarize = subparsers.add_parser("summarize", help="Summarize an existing scored CSV.")
    summarize.add_argument("--input-csv", required=True)
    summarize.add_argument("--output-dir", default="interference/data/summary")
    summarize.add_argument("--plots", action="store_true")

    next_generate = subparsers.add_parser("next-generate", help="Generate focused next-run diagnostic samples without scoring.")
    add_next_generation_args(next_generate)
    next_generate.add_argument("--jsonl", action="store_true", help="Also write JSONL output.")

    next_run = subparsers.add_parser("next-run", help="Generate and score focused next-run diagnostics.")
    add_next_generation_args(next_run)
    add_model_args(next_run)
    next_run.add_argument("--jsonl", action="store_true", help="Also write JSONL output.")

    next_summarize = subparsers.add_parser("next-summarize", help="Summarize an existing scored next-run CSV.")
    next_summarize.add_argument("--input-csv", required=True)
    next_summarize.add_argument("--output-dir", default="data/qwen3_8_next/summary")
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N generated rows.")
    parser.add_argument("--device", default=None, help="Explicit device when --device-map none is used.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map. Use 'none' to disable.")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "none", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--label-token-style", default="auto", choices=["auto", "bare", "space", "newline"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face hub cache directory. Use default/none to let Transformers choose.")
    parser.add_argument("--local-files-only", action="store_true", help="Load only from --cache-dir and fail if any model file is missing.")


def add_next_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-base-events", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-events-from-csv", default="data/qwen3_8_pilot/samples.csv", help="Existing pilot samples.csv to reuse base events. Use 'none' to sample by seed.")
    parser.add_argument("--sections", nargs="+", default=["all"], choices=["all", *NEXT_SECTIONS])
    parser.add_argument("--output-dir", default="data/qwen3_8_next")
    parser.add_argument("--csv-name", default="samples.csv")
    parser.add_argument("--jsonl-name", default="samples.jsonl")


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-base-events", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="interference/data/generated")
    parser.add_argument("--csv-name", default="samples.csv")
    parser.add_argument("--jsonl-name", default="samples.jsonl")
    parser.add_argument("--experiments", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    parser.add_argument("--include-exp3-sanity", action="store_true")
    parser.add_argument("--no-exp4-source-only", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
