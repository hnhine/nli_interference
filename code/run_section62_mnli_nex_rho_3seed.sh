#!/usr/bin/env bash
set -euo pipefail

# MNLI resample necessity for frozen rho rotations at the claim-final peak.
#
# Examples (run the two models on separate GPUs):
#   CUDA_VISIBLE_DEVICES=0 bash code/run_section62_mnli_nex_rho_3seed.sh phi4
#   CUDA_VISIBLE_DEVICES=1 bash code/run_section62_mnli_nex_rho_3seed.sh qwen
#
# Optional overrides:
#   START_SEED=0 NUM_SEEDS=1 BATCH_SIZE=64 RANDOM_SEEDS="0 1 2 3 4"

cd /workspace/nhi/nli_interference

START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DONOR_SEED="${DONOR_SEED:-0}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2 3 4}"

if (( NUM_SEEDS < 1 )); then
  echo "NUM_SEEDS must be positive" >&2
  exit 2
fi

if (( $# == 0 )); then
  targets=(phi4 qwen)
else
  targets=("$@")
fi

for target in "${targets[@]}"; do
  case "$target" in
    phi4)
      behavioral="data/section62/phi4/behavioral/behavioral_summary.json"
      rotation_root="data/das/seed_sweep_phi4_rho"
      layer=12
      output_root="data/section62/mnli_nex/phi4"
      ;;
    qwen)
      behavioral="data/section62/qwen/behavioral/behavioral_summary.json"
      rotation_root="data/das/seed_sweep_qwen_rho_r64"
      layer=18
      output_root="data/section62/mnli_nex/qwen"
      ;;
    *)
      echo "Unknown target '$target'; use phi4 and/or qwen" >&2
      exit 2
      ;;
  esac

  last_seed=$((START_SEED + NUM_SEEDS - 1))
  for ((rotation_seed = START_SEED; rotation_seed <= last_seed; rotation_seed++)); do
    rotation_dir="${rotation_root}/seed${rotation_seed}/L${layer}_claim_final"
    output_dir="${output_root}/seed${rotation_seed}/L${layer}_claim_final"
    if [[ ! -s "${rotation_dir}/rotation_weight.npy" || \
          ! -s "${rotation_dir}/rotation_weight_metadata.json" ]]; then
      echo "Missing frozen rotation: ${rotation_dir}" >&2
      exit 1
    fi

    echo "=== MNLI NEx target=${target} rotation_seed=${rotation_seed} L${layer}/claim_final ==="
    python3 code/run_section62_mnli_nex.py \
      --behavioral-summary "$behavioral" \
      --rho-rotation-dir "$rotation_dir" \
      --rotation-seed "$rotation_seed" \
      --layer "$layer" \
      --site claim_final \
      --rank 64 \
      --donor-seed "$DONOR_SEED" \
      --random-seeds $RANDOM_SEEDS \
      --batch-size "$BATCH_SIZE" \
      --device-map none \
      --device cuda \
      --local-files-only \
      --resume \
      --output-dir "$output_dir"
  done

  python3 code/aggregate_section62_mnli_nex.py \
    --input-root "$output_root" \
    --output-dir "$output_root/aggregate"
done

echo "DONE. Results under data/section62/mnli_nex/{phi4,qwen}/seed*/"
