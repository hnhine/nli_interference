# Interference Experiment Suite

This package implements the five zero-shot T/F/U NLI probes from the interference spec.

## Generate pilot samples

```bash
python interference/code/run_interference_suite.py generate \
  --n-base-events 20 \
  --output-dir interference/data/pilot
```

This writes `samples.csv` with prompts, expected labels, event metadata, overlap metadata,
and empty logit/result columns.

## Score a model

```bash
python interference/code/run_interference_suite.py run \
  --model-name Qwen/Qwen2.5-1.5B \
  --n-base-events 20 \
  --batch-size 8 \
  --output-dir interference/data/qwen_pilot \
  --plots
```

By default, model files are cached under `/workspace/huggingface/hub`. The first run downloads
missing files there; later runs reuse that cache. To forbid downloads after the cache is complete,
add `--local-files-only`.

The runner reads next-token logits for `T`, `F`, and `U`, then writes:

- `samples.csv`
- `summary/summary_metrics.json`
- aggregate CSVs for Exp 1 through Exp 5
- optional plots under `plots/`

## Summarize an existing scored CSV

```bash
python interference/code/run_interference_suite.py summarize \
  --input-csv interference/data/qwen_pilot/samples.csv \
  --output-dir interference/data/qwen_pilot/summary \
  --plots
```

## Module map

- `base.py`: base `z`, `Event`, sentence builders, and T/F/U prompt formatting.
- `generation.py`: Exp 1 to Exp 5 sample generation.
- `model.py`: Hugging Face causal LM scorer for T/F/U logits.
- `metrics.py`: phase, carrier proxy, selection, cancellation, and object-binding metrics.
- `plots.py`: standard plot plan.
- `run.py`: CLI orchestration.

## Focused next-run diagnostics

After the Qwen3-8B pilot, run the focused diagnostic package without rerunning the original suite:

```bash
python code/run_interference_suite.py next-run \
  --model-name Qwen/Qwen3-8B \
  --base-events-from-csv data/qwen3_8_pilot/samples.csv \
  --batch-size 8 \
  --output-dir data/qwen3_8_next
```

This generates and scores:

- Exp4 v2 source-order permutations: `+-`, `-+`, `++-`, `+-+`, `-++`, `+--`, `-+-`, `--+`, plus `+` and `-` controls.
- Unrelated-conflict controls: same-event conflict in the assumptions, unrelated claim.
- Exp2b claim-polarity-counterbalanced carrier overlap: `SVO`, `SV`, `VO`, `S-only`, `none` with `A+C+`, `A-C+`, `A+C-`, `A-C-`.
- Duplicate amplification controls: `++` and `--`.

The full pilot-scale next run has 680 rows for 20 base events. Summary files are written under `data/qwen3_8_next/summary`, including order-permutation diagnostics, unrelated-conflict rates, Exp2b phase slopes, and duplicate deltas.

To generate only the CSV without scoring:

```bash
python code/run_interference_suite.py next-generate \
  --base-events-from-csv data/qwen3_8_pilot/samples.csv \
  --output-dir data/qwen3_8_next
```

## Main Pipeline Includes Next Diagnostics

The main `run` command now scores the original suite and the focused next-run diagnostics in one model load. For a large run such as 1000 base events:

```bash
python code/run_interference_suite.py run \
  --model-name Qwen/Qwen3-8B \
  --n-base-events 1000 \
  --batch-size 8 \
  --output-dir data/qwen3_8_1000 \
  --local-files-only \
  --plots
```

This produces about `63 * n_base_events` rows by default: original suite rows plus Exp4 v2, unrelated-conflict, Exp2b, and duplicate-control diagnostics. For 1000 base events, that is about 63,000 rows. The output summaries are split into:

- `summary/summary_metrics.json` for the original suite.
- `summary_next/next_run_summary_metrics.json` for the next diagnostics.

To run only the original suite, add:

```bash
--no-next-diagnostics
```

To include only selected next diagnostics, use for example:

```bash
--next-sections exp4_v2 unrelated_conflict
```

