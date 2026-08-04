#!/usr/bin/env bash
# Evaluate the Section 5 joint (m, rho) patch for Qwen r64.
# Each variable uses its own independently selected peak:
#   claim_final: m=L16, rho=L18
#   answer_token: m=L22, rho=L24
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash code/run_qwen_joint_gate_r64_3seed_peaks.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SAMPLES="${SAMPLES:-data/das/joint_gate_test150/triples.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/das/joint_gate_test150_qwen_r64_peaks}"
START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
DTYPE="${DTYPE:-bfloat16}"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "ERROR: required file is missing or empty: $1" >&2
    exit 2
  fi
}

run_cell() {
  local seed="$1"
  local site="$2"
  local m_layer="$3"
  local rho_layer="$4"
  local site_label="$5"

  local m_rotation="data/das/seed_sweep_qwen_m_r64/seed${seed}/L${m_layer}_${site}"
  local rho_rotation="data/das/seed_sweep_qwen_rho_r64/seed${seed}/L${rho_layer}_${site}"
  local output_dir="$OUTPUT_ROOT/seed${seed}/${site_label}"

  require_file "$m_rotation/rotation_weight.pt"
  require_file "$m_rotation/rotation_weight_metadata.json"
  require_file "$rho_rotation/rotation_weight.pt"
  require_file "$rho_rotation/rotation_weight_metadata.json"

  if [[ -s "$output_dir/joint_gate_scored.csv" &&
        -s "$output_dir/joint_gate_summary.csv" &&
        -s "$output_dir/run_metadata.json" ]]; then
    echo "=== skip completed joint gate: seed$seed $site_label ==="
    return
  fi

  echo
  echo "=== Qwen joint gate seed$seed $site: m=L$m_layer rho=L$rho_layer ==="
  python3 code/run_das_joint_gate.py \
    --samples "$SAMPLES" \
    --model-name "$MODEL" \
    --m-rotation "$m_rotation" \
    --rho-rotation "$rho_rotation" \
    --composition-mode two_peak \
    --site "$site" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --random-seeds 0 1 2 \
    --bootstrap-samples 1000 \
    --checkpoint-every 5 \
    --seed "$seed" \
    --device-map none \
    --device cuda \
    --torch-dtype "$DTYPE" \
    --local-files-only \
    --output-dir "$output_dir"
}

require_file "$SAMPLES"

last_seed=$((START_SEED + NUM_SEEDS - 1))
for seed in $(seq "$START_SEED" "$last_seed"); do
  run_cell "$seed" claim_final 16 18 claim
  run_cell "$seed" answer_token 22 24 answer
done

echo
echo "DONE. Qwen r64 joint-gate results: $OUTPUT_ROOT/seed{${START_SEED}..${last_seed}}/{claim,answer}/"
