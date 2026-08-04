#!/usr/bin/env bash
# Train Phi-4 p_i DAS rotations at the final token of the SVO-matching
# assumption.  This is a one-site, full stride-2, rank-64 sweep.
#
# Defaults:
#   CUDA_VISIBLE_DEVICES=0
#   START_SEED=0
#   NUM_SEEDS=3
#
# Example:
#   CUDA_VISIBLE_DEVICES=1 bash code/run_phi4_pi_matched_assumption_seed_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."

START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-3}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/das/seed_sweep_phi4_pi_r64_matched_assumption_final}"
LAYERS=(0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30)

if ! [[ "$START_SEED" =~ ^[0-9]+$ ]]; then
  echo "START_SEED must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$NUM_SEEDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SEEDS must be a positive integer" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Phi-4 p_i matched-assumption sweep: layers=${LAYERS[*]} rank=64"
echo "Seeds: $START_SEED..$((START_SEED + NUM_SEEDS - 1))"
echo "Output root: $OUTPUT_ROOT"

for seed in $(seq "$START_SEED" $((START_SEED + NUM_SEEDS - 1))); do
  echo
  echo "=== phi4_pi matched_assumption_final seed $seed ==="

  python3 code/run_das_relay_map.py \
    --samples data/das/pi_v5/pairs.csv \
    --model-name microsoft/Phi-4-mini-instruct \
    --target-var pi \
    --layers "${LAYERS[@]}" \
    --sites matched_assumption_final \
    --rank 64 \
    --epochs 1 \
    --batch-size 20 \
    --eval-batch-size 60 \
    --learning-rate 0.002 \
    --eval-interval "$EVAL_INTERVAL" \
    --seed "$seed" \
    --train-control-types all \
    --train-control-proportions \
      main=0.3 \
      active_source_m0=0.3 \
      probe_flip_both=0.1 \
      probe_flip_pc=0.1 \
      gate_m0=0.05 \
      label_copy_trap=0.05 \
      distractor=0.1 \
    --torch-dtype "$DTYPE" \
    --device-map none \
    --device cuda \
    --local-files-only \
    --resume \
    --output-dir "$OUTPUT_ROOT/seed$seed"
done

echo
echo "DONE. Results under $OUTPUT_ROOT/seed*/L{layer}_matched_assumption_final/"
