# Interference Experiment Suite

This package implements the zero-shot T/F/U NLI probes from the interference spec plus the focused diagnostics. The public CLI path is one command family: `generate`, `run`, and `summarize` operate on the full merged suite by default.

## Generate all samples

```bash
python code/run_interference_suite.py generate \
  --n-base-events 20 \
  --output-dir data/generated
```

This writes one `samples.csv` containing Exp 1 through Exp 5 and the focused diagnostics, with prompts, expected labels, event metadata, overlap metadata, and empty logit/result columns.

## Score All Experiments

```bash
python code/run_interference_suite.py run \
  --model-name Qwen/Qwen3-8B \
  --n-base-events 20 \
  --batch-size 8 \
  --output-dir data/qwen3_8_full \
  --plots
```

By default, model files are cached under `/workspace/huggingface/hub`. The first run downloads
missing files there; later runs reuse that cache. To forbid downloads after the cache is complete,
add `--local-files-only`.

The runner reads next-token logits for `T`, `F`, and `U`, then writes:

- `samples.csv`
- `summary/summary_metrics.json` with Exp 1 through Exp 5 and `next_diagnostics`
- aggregate CSVs for all summarized experiments and diagnostics
- optional original-suite plots under `plots/`

## Summarize an existing scored CSV

```bash
python code/run_interference_suite.py summarize \
  --input-csv data/qwen3_8_full/samples.csv \
  --output-dir data/qwen3_8_full \
  --plots
```

## Module map

- `base.py`: base `z`, `Event`, sentence builders, and T/F/U prompt formatting.
- `generation.py`: Exp 1 to Exp 5 sample generation plus the focused diagnostics previously kept in `next_generation.py`.
- `model.py`: Hugging Face causal LM scorer for T/F/U logits.
- `metrics.py`: phase, carrier proxy, selection, cancellation, object-binding, and focused diagnostic metrics.
- `plots.py`: standard plot plan.
- `run.py`: CLI orchestration.

## Full Run

The main `run` command scores the original suite and focused diagnostics in one model load. For a large run such as 1000 base events:

```bash
python code/run_interference_suite.py run \
  --model-name Qwen/Qwen3-8B \
  --n-base-events 1000 \
  --batch-size 8 \
  --output-dir data/qwen3_8_1000 \
  --local-files-only \
  --plots
```

This produces about `63 * n_base_events` rows by default: original suite rows plus Exp4 v2, unrelated-conflict, Exp2b, and duplicate-control diagnostics. For 1000 base events, that is about 63,000 rows. The single merged summary lives at `summary/summary_metrics.json`.

