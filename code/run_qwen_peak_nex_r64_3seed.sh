#!/usr/bin/env bash
# Evaluate peak-cell necessity (NEx) for all four Qwen r64 variables and all
# three trained rotation seeds. No DAS rotation is trained by this script.
#
# Peak cells:
#   pi:  L16 claim_final, L22 answer_token
#   pc:  L14 claim_final, L22 answer_token
#   rho: L18 claim_final, L24 answer_token
#   m:   L16 claim_final, L22 answer_token
#
# The evaluator seed is intentionally fixed at 0 for every rotation seed, as
# in the Phi-4 peak NEx runs. Thus variation across seed0/1/2 measures trained
# rotation variation rather than donor/random-control resampling variation.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash code/run_qwen_peak_nex_r64_3seed.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
ABLATION_SEED="${ABLATION_SEED:-0}"

# Keep m last so a separately running qwen_m seed0 L22 cell has several hours
# to complete before this queue needs it.
SPECS=(
  "pi pi_v5 16 claim_final"
  "pi pi_v5 22 answer_token"
  "pc pc_v4 14 claim_final"
  "pc pc_v4 22 answer_token"
  "rho rho_v1 18 claim_final"
  "rho rho_v1 24 answer_token"
  "m m_v4 16 claim_final"
  "m m_v4 22 answer_token"
)

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Qwen r64 peak NEx: 4 variables x 2 sites x 3 rotation seeds = 24 evaluations"
echo "Fixed evaluator/donor seed: $ABLATION_SEED"

for spec in "${SPECS[@]}"; do
  read -r var dataset layer site <<< "$spec"

  for rotation_seed in 0 1 2; do
    rotation_dir="data/das/seed_sweep_qwen_${var}_r64/seed${rotation_seed}/L$(printf '%02d' "$layer")_${site}"
    output_dir="data/das/ablation_qwen_seed${rotation_seed}_peak/${var}/L$(printf '%02d' "$layer")_${site}"

    if [[ -s "$output_dir/ablation_summary.json" && -s "$output_dir/ablation_scored.csv" ]]; then
      echo "=== skip completed NEx: $output_dir ==="
      continue
    fi

    if [[ ! -s "$rotation_dir/rotation_weight.pt" || ! -s "$rotation_dir/summary_metrics.json" ]]; then
      echo "ERROR: incomplete rotation cell: $rotation_dir" >&2
      exit 2
    fi

    echo
    echo "=== Qwen NEx var=$var seed=$rotation_seed L${layer}_${site} ==="
    python3 code/run_das_ablation.py \
      --samples "data/das/${dataset}/pairs.csv" \
      --rotation-dir "$rotation_dir" \
      --model-name "$MODEL" \
      --target-var "$var" \
      --split test \
      --eval-batch-size "$EVAL_BATCH_SIZE" \
      --seed "$ABLATION_SEED" \
      --device-map none \
      --device cuda \
      --torch-dtype "$DTYPE" \
      --local-files-only \
      --output-dir "$output_dir"
  done
done

echo
echo "DONE. Results under data/das/ablation_qwen_seed{0,1,2}_peak/<var>/L<layer>_<site>/"
