# Interference Experiment Suite

This package implements the zero-shot T/F/U NLI probes from the interference spec plus the supplemental sections. The public CLI path is one command family: `generate`, `run`, and `summarize` operate on the full merged suite by default. Exp6A is included in the full suite; Exp6 also has separate commands for focused pilots.

## Generate all samples

```bash
python code/run_interference_suite.py generate \
  --n-base-events 20 \
  --output-dir data/generated
```

This writes one `samples.csv` containing Exp 1 through Exp 6A and the supplemental sections, with prompts, expected labels, event metadata, overlap metadata, and empty logit/result columns.

Exp3 uses exact minimal pairs. For every base event and target position it fixes the same three
event identities, samples an anchor polarity independently for each distractor, and crosses target
polarity (`positive`/`negative`) with three distractor configurations (`anchor`, first distractor
flipped, second distractor flipped). Its summary writes direct target- and distractor-flip effects to
`summary/exp3_intervention_pairs.csv` and reports their mean-absolute-effect ratio in
`summary_metrics.json`.

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

This produces about `87 * n_base_events` rows by default: original suite rows including Exp6A, plus Exp2 counterbalanced overlap plus Exp4 order-permutation, unrelated-conflict, and duplicate-control supplements. For 1000 base events, that is about 87,000 rows. The single merged summary lives at `summary/summary_metrics.json`.

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

`das-run` keeps the base model frozen and trains only the pyvene low-rank rotated intervention. The default row sites are `claim_final` for `p_c` and `m_i`, and the matched assumption final token for `p_i`; pass `--site answer_token` or another site to run controls. Rows flagged `mismatch_exclusion_relaxed=1` (m pairs whose mismatch event relaxed the cross-split exclusion) are dropped from train/val/test by default; pass `--include-relaxed` to keep them. `summary_metrics.json` records `n_relaxed_excluded` plus a `test_core` block restricted to the interchange controls, alongside the all-controls `test` metrics.

## Standardized DAS relay sweeps with all training controls

Run these commands from the repository root. Every sweep trains on all control
types with --train-control-types all. The pi_1000_v2 and pc_1000_v2 datasets
contain equally many main, distractor, gate_m0, and label_copy_trap rows, so
without an explicit --train-control-proportions argument the training pool is
balanced 25/25/25/25. The m_1000_v2 dataset is likewise balanced across its
three controls.

The standard p_i site below is row, i.e. the row-specific sentence-boundary
token selected by each pair. The row_lexical_final site is a separate diagnostic
and must not be mixed into the primary cross-model comparison. The p_c and m
maps evaluate both claim_final and answer_token. Use a fresh output directory
for every run because relay_map.json and relay_map.csv are rewritten after each
completed cell.

### p_i: assumption polarity

Qwen3-8B:

    python code/run_das_relay_map.py --samples data/das/pi_1000_v2/pairs.csv --model-name Qwen/Qwen3-8B --target-var pi --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 --sites row --rank 16 --steps 500 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/qwen3_8_pi_allcontrols_r16_stride2

Phi-4 Mini Instruct:

    python code/run_das_relay_map.py --samples data/das/pi_1000_v2/pairs.csv --model-name microsoft/Phi-4-mini-instruct --target-var pi --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 --sites row --rank 16 --steps 500 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/phi4_pi_allcontrols_r16_stride2

Granite 4.1 8B:

    python code/run_das_relay_map.py --samples data/das/pi_1000_v2/pairs.csv --model-name ibm-granite/granite-4.1-8b --target-var pi --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 --sites row --rank 16 --steps 500 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/granite41_pi_allcontrols_r16_stride2

Gemma 4 12B:

    python code/run_das_relay_map.py --samples data/das/pi_1000_v2/pairs.csv --model-name google/gemma-4-12B --target-var pi --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 --sites row --rank 16 --steps 500 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/gemma4_12b_pi_allcontrols_r16_stride2

### p_c: claim polarity

Qwen3-8B:

    python code/run_das_relay_map.py --samples data/das/pc_1000_v2/pairs.csv --model-name Qwen/Qwen3-8B --target-var pc --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 --sites claim_final answer_token --rank 16 --steps 500 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/qwen3_8_pc_allcontrols_r16_stride2

Phi-4 Mini Instruct:

    python code/run_das_relay_map.py --samples data/das/pc_1000_v2/pairs.csv --model-name microsoft/Phi-4-mini-instruct --target-var pc --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 --sites claim_final answer_token --rank 16 --steps 500 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/phi4_pc_allcontrols_r16_stride2

Granite 4.1 8B:

    python code/run_das_relay_map.py --samples data/das/pc_1000_v2/pairs.csv --model-name ibm-granite/granite-4.1-8b --target-var pc --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 --sites claim_final answer_token --rank 16 --steps 500 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/granite41_pc_allcontrols_r16_stride2

Gemma 4 12B:

    python code/run_das_relay_map.py --samples data/das/pc_1000_v2/pairs.csv --model-name google/gemma-4-12B --target-var pc --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 --sites claim_final answer_token --rank 16 --steps 500 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/gemma4_12b_pc_allcontrols_r16_stride2

### m: event-claim match

These m runs use 750 optimization steps, matching the existing Qwen m protocol.
Because --train-control-types all includes label_copy_trap in optimization, its
reported IIA is an in-training anti-copy constraint, not held-out anti-copy
evidence.

Qwen3-8B:

    python code/run_das_relay_map.py --samples data/das/m_1000_v2/pairs.csv --model-name Qwen/Qwen3-8B --target-var m --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 --sites claim_final answer_token --rank 16 --steps 750 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/qwen3_8_m_allcontrols_r16_stride2

Phi-4 Mini Instruct:

    python code/run_das_relay_map.py --samples data/das/m_1000_v2/pairs.csv --model-name microsoft/Phi-4-mini-instruct --target-var m --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 --sites claim_final answer_token --rank 16 --steps 750 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/phi4_m_allcontrols_r16_stride2

Granite 4.1 8B:

    python code/run_das_relay_map.py --samples data/das/m_1000_v2/pairs.csv --model-name ibm-granite/granite-4.1-8b --target-var m --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 --sites claim_final answer_token --rank 16 --steps 750 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/granite41_m_allcontrols_r16_stride2

Gemma 4 12B:

    python code/run_das_relay_map.py --samples data/das/m_1000_v2/pairs.csv --model-name google/gemma-4-12B --target-var m --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 --sites claim_final answer_token --rank 16 --steps 750 --batch-size 32 --eval-batch-size 64 --learning-rate 0.002 --eval-interval 0 --seed 0 --train-control-types all --torch-dtype bfloat16 --local-files-only --output-dir data/das/gemma4_12b_m_allcontrols_r16_stride2
