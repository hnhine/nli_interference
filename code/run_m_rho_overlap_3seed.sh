#!/usr/bin/env bash
set -euo pipefail

# Same-layer overlap audit for learned m and rho rank-64 subspaces.
# CPU-only: no CUDA_VISIBLE_DEVICES is required.

cd /workspace/nhi/nli_interference

OUTPUT_ROOT="${OUTPUT_ROOT:-data/das/subspace_overlap_m_rho_r64}"

run_cell() {
  local model="$1"
  local seed="$2"
  local layer="$3"
  local site="$4"
  local m_root rho_root

  case "$model" in
    phi4)
      m_root="data/das/seed_sweep_phi4_m"
      rho_root="data/das/seed_sweep_phi4_rho"
      ;;
    qwen)
      m_root="data/das/seed_sweep_qwen_m_r64"
      rho_root="data/das/seed_sweep_qwen_rho_r64"
      ;;
    *)
      echo "Unknown model '$model'" >&2
      exit 2
      ;;
  esac

  local cell="L${layer}_${site}"
  local output_dir="${OUTPUT_ROOT}/${model}/seed${seed}/${cell}"
  echo "=== overlap ${model} seed${seed} ${cell} ==="
  python3 code/audit_das_subspace_overlap.py \
    --m-rotation-dir "${m_root}/seed${seed}/${cell}" \
    --rho-rotation-dir "${rho_root}/seed${seed}/${cell}" \
    --training-seed "$seed" \
    --output-dir "$output_dir"
}

for seed in 0 1 2; do
  # Phi: m and rho have the same selected layer at both sites.
  run_cell phi4 "$seed" 12 claim_final
  run_cell phi4 "$seed" 16 answer_token

  # Qwen: audit both the m-favored and rho-favored same-layer candidates.
  run_cell qwen "$seed" 16 claim_final
  run_cell qwen "$seed" 18 claim_final
  run_cell qwen "$seed" 22 answer_token
  run_cell qwen "$seed" 24 answer_token
done

python3 code/aggregate_das_subspace_overlap.py \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/aggregate"

echo "DONE. Results under ${OUTPUT_ROOT}/"
