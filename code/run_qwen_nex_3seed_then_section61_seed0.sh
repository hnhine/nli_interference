#!/usr/bin/env bash
# One-GPU queue:
#   1. Qwen r64 peak NEx for rotation seeds 0, 1, and 2.
#   2. Qwen Section 6.1 E1-E7 evaluation for rotation seed 0.
#
# Expected wall time on one RTX A6000: roughly 11.5-12.5 hours.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#     bash code/run_qwen_nex_3seed_then_section61_seed0.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
GENERATED_DIR="${GENERATED_DIR:-data/section61/generated}"
SECTION61_OUTPUT_ROOT="${SECTION61_OUTPUT_ROOT:-data/section61/qwen_seeds/seed0}"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "ERROR: required file is missing or empty: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: required directory is missing: $1" >&2
    exit 2
  fi
}

run_behavioral() {
  local samples="$1"
  local output_dir="$2"

  if [[ -s "$output_dir/behavioral_summary.json" && -s "$output_dir/scored.csv" ]]; then
    echo "=== skip completed behavioral stage: $output_dir ==="
    return
  fi

  echo "=== behavioral: $output_dir ==="
  python3 code/run_section61.py behavioral \
    --samples "$samples" \
    --model-name "$MODEL" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --device-map none \
    --device cuda \
    --torch-dtype "$DTYPE" \
    --local-files-only \
    --output-dir "$output_dir"
}

run_das() {
  local rotation_dir="$1"
  local output_dir="$2"
  shift 2
  local samples=("$@")

  if [[ -s "$output_dir/das_summary.json" && -s "$output_dir/scored.csv" ]]; then
    echo "=== skip completed DAS stage: $output_dir ==="
    return
  fi

  require_dir "$rotation_dir"
  require_file "$rotation_dir/rotation_weight.pt"
  require_file "$rotation_dir/summary_metrics.json"

  echo "=== DAS: $output_dir ==="
  python3 code/run_section61.py das \
    --samples "${samples[@]}" \
    --rotation-dir "$rotation_dir" \
    --seed 0 \
    --model-name "$MODEL" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --device-map none \
    --device cuda \
    --torch-dtype "$DTYPE" \
    --local-files-only \
    --output-dir "$output_dir"
}

for required in \
  "$GENERATED_DIR/e1_behavioral.csv" \
  "$GENERATED_DIR/e2_rho_within.csv" \
  "$GENERATED_DIR/e3_rho_cross.csv" \
  "$GENERATED_DIR/e4_pi_within.csv" \
  "$GENERATED_DIR/e4_pc_within.csv" \
  "$GENERATED_DIR/e5_m_did_not.csv" \
  "$GENERATED_DIR/e5_m_did_not_ever.csv" \
  "$GENERATED_DIR/e5_m_didnt.csv" \
  "$GENERATED_DIR/e5_m_failed_to.csv" \
  "$GENERATED_DIR/e5_m_never.csv" \
  "$GENERATED_DIR/e5_m_not_the_case.csv" \
  "$GENERATED_DIR/e6_rho_mixed.csv" \
  "$GENERATED_DIR/e7_behavioral.csv"
do
  require_file "$required"
done

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo
echo "######## Part 1/2: Qwen r64 peak NEx, rotation seeds 0-2 ########"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MODEL="$MODEL" DTYPE="$DTYPE" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
bash code/run_qwen_peak_nex_r64_3seed.sh

echo
echo "######## Part 2/2: Qwen Section 6.1, rotation seed 0 ########"

output_root="$SECTION61_OUTPUT_ROOT"
rho_root="data/das/seed_sweep_qwen_rho_r64/seed0"
pi_root="data/das/seed_sweep_qwen_pi_r64/seed0"
pc_root="data/das/seed_sweep_qwen_pc_r64/seed0"
m_root="data/das/seed_sweep_qwen_m_r64/seed0"

# E1 and E7 are behavioral gates. Qwen E7 double negation currently fails its
# behavioral gate, so—matching the existing pipeline—there is no E7 DAS run.
run_behavioral "$GENERATED_DIR/e1_behavioral.csv" "$output_root/e1"
run_behavioral "$GENERATED_DIR/e7_behavioral.csv" "$output_root/e7_behavior"

# E2: within-form rho transfer.
run_das "$rho_root/L18_claim_final" "$output_root/e2_claim" \
  "$GENERATED_DIR/e2_rho_within.csv"
run_das "$rho_root/L24_answer_token" "$output_root/e2_answer" \
  "$GENERATED_DIR/e2_rho_within.csv"

# E3: cross-form rho transfer.
run_das "$rho_root/L18_claim_final" "$output_root/e3_cross_claim" \
  "$GENERATED_DIR/e3_rho_cross.csv"
run_das "$rho_root/L24_answer_token" "$output_root/e3_cross_answer" \
  "$GENERATED_DIR/e3_rho_cross.csv"

# E4: raw p_i and p_c identification.
run_das "$pi_root/L16_claim_final" "$output_root/e4_pi_claim" \
  "$GENERATED_DIR/e4_pi_within.csv"
run_das "$pi_root/L22_answer_token" "$output_root/e4_pi_answer" \
  "$GENERATED_DIR/e4_pi_within.csv"
run_das "$pc_root/L14_claim_final" "$output_root/e4_pc_claim" \
  "$GENERATED_DIR/e4_pc_within.csv"
run_das "$pc_root/L22_answer_token" "$output_root/e4_pc_answer" \
  "$GENERATED_DIR/e4_pc_within.csv"

# E5: matching gate m across all six lexical forms.
e5_samples=(
  "$GENERATED_DIR/e5_m_did_not.csv"
  "$GENERATED_DIR/e5_m_did_not_ever.csv"
  "$GENERATED_DIR/e5_m_didnt.csv"
  "$GENERATED_DIR/e5_m_failed_to.csv"
  "$GENERATED_DIR/e5_m_never.csv"
  "$GENERATED_DIR/e5_m_not_the_case.csv"
)
run_das "$m_root/L16_claim_final" "$output_root/e5_m_claim" \
  "${e5_samples[@]}"
run_das "$m_root/L22_answer_token" "$output_root/e5_m_answer" \
  "${e5_samples[@]}"

# E6: mixed-form rho transfer.
run_das "$rho_root/L18_claim_final" "$output_root/e6_mixed_claim" \
  "$GENERATED_DIR/e6_rho_mixed.csv"
run_das "$rho_root/L24_answer_token" "$output_root/e6_mixed_answer" \
  "$GENERATED_DIR/e6_rho_mixed.csv"

echo "=== summarize Section 6.1 Qwen seed0 ==="
python3 code/run_section61.py summarize \
  --behavioral-summaries \
    "$output_root/e1/behavioral_summary.json" \
    "$output_root/e7_behavior/behavioral_summary.json" \
  --das-summaries \
    "$output_root/e2_claim/das_summary.json" \
    "$output_root/e2_answer/das_summary.json" \
    "$output_root/e3_cross_claim/das_summary.json" \
    "$output_root/e3_cross_answer/das_summary.json" \
    "$output_root/e4_pi_claim/das_summary.json" \
    "$output_root/e4_pi_answer/das_summary.json" \
    "$output_root/e4_pc_claim/das_summary.json" \
    "$output_root/e4_pc_answer/das_summary.json" \
    "$output_root/e5_m_claim/das_summary.json" \
    "$output_root/e5_m_answer/das_summary.json" \
    "$output_root/e6_mixed_claim/das_summary.json" \
    "$output_root/e6_mixed_answer/das_summary.json" \
  --output-dir "$output_root/summary"

echo
echo "DONE."
echo "NEx: data/das/ablation_qwen_seed{0,1,2}_peak/"
echo "Section 6.1 seed0: $output_root/"
