# Interference Experiment Suite

This package implements the zero-shot T/F/U NLI probes from the interference spec plus the supplemental sections. The public CLI path is one command family: `generate`, `run`, and `summarize` operate on the full merged suite by default. Exp6A is included in the full suite; Exp6 also has separate commands for focused pilots.

## Generate all samples

```bash
python code/run_interference_suite.py generate \
  --n-base-events 20 \
  --output-dir data/generated
```

This writes one `samples.csv` containing Exp 1 through Exp 6A and the supplemental sections, with prompts, expected labels, event metadata, overlap metadata, and empty logit/result columns.

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

The runner reads label logits for `T`, `F`, and `U`, then writes:

- `samples.csv`
- `summary/summary_metrics.json` with Exp 1 through Exp 6 and `supplements`
- aggregate CSVs for all summarized experiments and supplements
- optional full-suite plots under `plots/`

## Summarize an existing scored CSV

```bash
python code/run_interference_suite.py summarize \
  --input-csv data/qwen3_8_full/samples.csv \
  --output-dir data/qwen3_8_full \
  --plots
```

## Focused Exp6 Pilot

Generate the settled Exp6A absolute-negation rows (`did not`, `did not ever`, `never`) without the rest of the suite:

```bash
python code/run_interference_suite.py exp6-generate \
  --n-base-events 20 \
  --include-exp6a \
  --output-dir data/qwen3_8_exp6
```

Score an existing Exp6 sample CSV:

```bash
python code/run_interference_suite.py exp6-run \
  --model-name Qwen/Qwen3-8B \
  --samples data/qwen3_8_exp6/samples.csv \
  --batch-size 8 \
  --output-dir data/qwen3_8_exp6 \
  --local-files-only \
  --plots
```

Summarize an already scored Exp6 CSV:

```bash
python code/run_interference_suite.py exp6-summarize \
  --scored data/qwen3_8_exp6/scored.csv \
  --output-dir data/qwen3_8_exp6/summary \
  --plots
```

Exp6B is available as a scaffold for small pilots only, restricted by default to `visit`, `explore`, and `enter`:

```bash
python code/run_interference_suite.py exp6-generate \
  --n-base-events 20 \
  --include-exp6b \
  --output-dir data/qwen3_8_exp6b
```

## Module map

- `base.py`: base `z`, `Event`, sentence builders, and T/F/U prompt formatting.
- `generation.py`: Exp 1 to Exp 6A sample generation, the Exp6B strict-verb scaffold, and the supplemental sections that travel with their parent experiments.
- `model.py`: Hugging Face causal LM scorer for T/F/U logits.
- `metrics.py`: phase, carrier proxy, selection, cancellation, object-binding, Exp6 negation-form, and supplemental metrics.
- `plots.py`: standard plot plan, including Exp6A diagnostics when present.
- `run.py`: CLI orchestration.

## Full Run

The main `run` command scores the original suite and supplemental sections in one model load. For a large run such as 1000 base events:

```bash
python code/run_interference_suite.py run \
  --model-name Qwen/Qwen3-8B \
  --n-base-events 1000 \
  --batch-size 8 \
  --output-dir data/qwen3_8_1000 \
  --local-files-only \
  --plots
```

This produces about `75 * n_base_events` rows by default: original suite rows including Exp6A, plus Exp2 counterbalanced overlap plus Exp4 order-permutation, unrelated-conflict, and duplicate-control supplements. For 1000 base events, that is about 75,000 rows. The single merged summary lives at `summary/summary_metrics.json`.

## DAS With pyvene

Generate base/source counterfactual pairs for the atomic causal model (`p_c`, `p_i`, `m_i`):

```bash
python code/run_interference_suite.py das-generate \
  --n-base-events 20 \
  --output-dir data/das/generated
```

Train one DAS intervention with `pyvene`:

```bash
python code/run_interference_suite.py das-run \
  --samples data/das/generated/pairs.csv \
  --model-name Qwen/Qwen3-8B \
  --target-var pc \
  --layer 24 \
  --rank 1 \
  --batch-size 16 \
  --steps 1000 \
  --output-dir data/das/qwen3_8_pc_l24_r1 \
  --local-files-only
```

`das-run` keeps the base model frozen and trains only the pyvene low-rank rotated intervention. The default row sites are `claim_final` for `p_c` and `m_i`, and the matched assumption final token for `p_i`; pass `--site answer_token` or another site to run controls.
