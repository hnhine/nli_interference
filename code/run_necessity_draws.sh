#!/usr/bin/env bash
# Necessity: many random-subspace draws at each peak cell (eval-only, no retrain).
# NEx = mean(rand_* acc over draws) - das_* acc. das_* is identical across seeds
# (fixed trained subspace); only rand_* varies with --seed.
#
# Usage:
#   bash code/run_necessity_draws.sh                 # all 8 combos, seeds 0..19
#   NUM_DRAWS=10 bash code/run_necessity_draws.sh    # fewer draws
#   bash code/run_necessity_draws.sh qwen_rho phi4_pi # only listed combos
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_DRAWS="${NUM_DRAWS:-20}"
DTYPE="${DTYPE:-bfloat16}"

# name          target dataset model                          rotation-dir (peak cell, already trained)
NEC=(
  "qwen_rho rho rho_v1 Qwen/Qwen3-8B                 qwen_rho_v1_r16_stride2_1ep_b32/L18_claim_final"
  "qwen_pc  pc  pc_v4  Qwen/Qwen3-8B                 qwen_pc_v4_r16_stride2_1ep/L14_claim_final"
  "qwen_m   m   m_v4   Qwen/Qwen3-8B                 qwen_m_v4_r16_stride2_1ep_b32/L28_answer_token"
  "qwen_pi  pi  pi_v5  Qwen/Qwen3-8B                 qwen_pi_v5_rawid_r16_stride2_1ep_b20/L16_claim_final"
  "phi4_rho rho rho_v1 microsoft/Phi-4-mini-instruct phi4_rho_r64_stride2_1ep/L12_row"
  "phi4_pc  pc  pc_v4  microsoft/Phi-4-mini-instruct phi4_pc_v4_r64_stride2_1ep/L10_claim_final"
  "phi4_m   m   m_v4   microsoft/Phi-4-mini-instruct phi4_m_v4_r64_stride2/L12_claim_final"
  "phi4_pi  pi  pi_v5  microsoft/Phi-4-mini-instruct phi4_pi_v5_rawid_r64_stride2_1ep_b20/L12_claim_final"
)

WANT=("$@")  # optional subset of combo names; empty = all
want() { [ ${#WANT[@]} -eq 0 ] && return 0; for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

for row in "${NEC[@]}"; do
  set -- $row; NAME=$1; VAR=$2; DS=$3; MODEL=$4; ROT=$5
  want "$NAME" || continue
  for S in $(seq 0 $((NUM_DRAWS - 1))); do
    echo "=== necessity $NAME seed $S ($((S + 1))/$NUM_DRAWS) ==="
    python code/run_das_ablation.py \
      --samples data/das/$DS/pairs.csv \
      --rotation-dir data/das/$ROT \
      --model-name $MODEL \
      --target-var $VAR \
      --split test --local-files-only --torch-dtype "$DTYPE" \
      --seed $S \
      --output-dir data/das/necessity_draws/${NAME}/seed$S
  done
done
echo "DONE. Results under data/das/necessity_draws/<combo>/seed*/"
