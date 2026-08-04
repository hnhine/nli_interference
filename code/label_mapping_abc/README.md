# Frozen DAS validation with A/B/C labels

This directory contains an auxiliary, evaluation-only diagnostic. It reuses
the frozen rank-64 DAS rotations at the selected claim-final and answer-token
peaks, but changes the output symbols in the validation prompts as follows:

```text
A = must be true
B = must be false
C = cannot be determined
```

Thus the semantic mapping is fixed as `T -> A`, `F -> B`, and `U -> C`. The
natural-language definitions and every assumption and claim remain unchanged.
No DAS rotation is retrained.

The code is isolated from the main pipeline. It only reads the existing DAS
pair CSVs and frozen rotations, then writes new files below
`data/das/label_mapping_abc/`. It does not modify the source datasets, training
code, rotations, or their existing summaries.

## Peak cells

| Model | Variable | Claim final | Answer token |
| --- | --- | ---: | ---: |
| Phi-4 | `pc` | L10 | L16 |
| Phi-4 | `pi` | L10 | L16 |
| Phi-4 | `rho` | L12 | L16 |
| Phi-4 | `m` | L12 | L16 |
| Qwen | `pc` | L14 | L22 |
| Qwen | `pi` | L16 | L22 |
| Qwen | `rho` | L18 | L24 |
| Qwen | `m` | L16 | L22 |

Each cell is evaluated for rotation seeds 0, 1, and 2 by default.

## Run

Preflight all expected files without loading a model:

```bash
DRY_RUN=1 bash code/label_mapping_abc/run_all.sh phi4
DRY_RUN=1 bash code/label_mapping_abc/run_all.sh qwen
```

Small smoke test on one frozen cell:

```bash
GPU=0 LOCAL_FILES_ONLY=1 MAX_ROWS=50 TARGETS="rho" \
SITES="claim_final" SEEDS="0" \
bash code/label_mapping_abc/run_all.sh phi4
```

Full validation evaluation (one model per GPU):

```bash
GPU=0 LOCAL_FILES_ONLY=1 bash code/label_mapping_abc/run_all.sh phi4
GPU=1 LOCAL_FILES_ONLY=1 bash code/label_mapping_abc/run_all.sh qwen
```

The two full commands can run concurrently in separate tmux panes. To select a
subset, override the space-separated `TARGETS`, `SITES`, or `SEEDS` variables.

## Outputs

For each model and variable, the runner writes:

- `val_abc_baseline_scored.csv`: ordinary A/B/C predictions before patching;
- `seed*/L*_<site>/val_abc_intervention_scored.csv`: row-level frozen-DAS
  intervention results;
- `seed*/L*_<site>/summary.json`: A/B/C IIA and the corresponding original
  T/F/U validation metrics read from the frozen cell;
- `aggregate.csv` and `aggregate.json`: three-seed means and confirmatory
  scores.

`confirmatory_ABC_IIA` follows the paper's aggregation rule: first average each
identifying control across training seeds, then take the minimum across those
control means. `confirmatory_delta_ABC_minus_TFU` directly measures how much
performance changes under label remapping at the same target, layer, site, and

The identifying-control sets are unchanged from the current DAS protocol:
`pc` uses `main`, `probe_flip_both`, and `probe_flip_pi`; `pi` uses `main`,
`active_source_m0`, `probe_flip_both`, and `probe_flip_pc`; `rho` uses the full
six-control relation audit (`flip_pi`, `flip_pc`, `hold_both`, `source_m0`,
`gate_m0`, and `label_copy_trap`); and `m` uses both transfer directions plus
`label_copy_trap` and `label_copy_trap_same_m1`.
