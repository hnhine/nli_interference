#!/usr/bin/env bash
set -euo pipefail

# Auxiliary evaluation only: frozen peak DAS rotations, validation split, A/B/C labels.
# Nothing in the original datasets, rotations, or summaries is modified.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:?Usage: GPU=0 bash code/label_mapping_abc/run_all.sh phi4-or-qwen}"
GPU="${GPU:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

read -r -a TARGET_ARGS <<< "${TARGETS:-pc pi rho m}"
read -r -a SITE_ARGS <<< "${SITES:-claim_final answer_token}"
read -r -a SEED_ARGS <<< "${SEEDS:-0 1 2}"

EXTRA_ARGS=()
if [[ -n "${MAX_ROWS:-}" ]]; then
  EXTRA_ARGS+=(--max-rows "${MAX_ROWS}")
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--local-files-only)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"

python code/label_mapping_abc/eval_peak_subspaces.py \
  --model "${MODEL}" \
  --targets "${TARGET_ARGS[@]}" \
  --sites "${SITE_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}" \
  --split val \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --device-map none \
  --device cuda \
  --torch-dtype bfloat16 \
  --output-dir data/das/label_mapping_abc \
  "${EXTRA_ARGS[@]}"
