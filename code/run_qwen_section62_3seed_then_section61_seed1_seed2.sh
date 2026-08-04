#!/usr/bin/env bash
# One-GPU queue:
#   1. Ensure the two missing Qwen m r64 seed-0 execution cells exist.
#   2. Qwen Section 6.2 (MNLI naturalistic transfer), rotation seeds 0, 1, 2,
#      at both common execution sites:
#        - L18 claim_final
#        - L24 answer_token
#   3. Qwen Section 6.1 (negation-form robustness), rotation seeds 1 and 2.
#
# Section 6.2 is NOT the Section 5 joint rho+m patch.  It evaluates:
#   - zero/necessity effects for frozen rho and m, with rank-matched controls;
#   - opposite-rho coordinate transfer within each MNLI polarity square;
#   - m transfer between polarized and all-U strata, with rho/random controls.
# The rho and m rotations are always evaluated at the same execution cell, so
# the wrong-variable control is layer/site matched. L18 claim_final is the rho
# claim peak and L24 answer_token is the rho answer peak. Missing seed-0 m
# rotations at these cells are trained as explicit prerequisites.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#     bash code/run_qwen_section62_3seed_then_section61_seed1_seed2.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DTYPE="${DTYPE:-bfloat16}"
SECTION61_BATCH_SIZE="${SECTION61_BATCH_SIZE:-64}"
SECTION62_BATCH_SIZE="${SECTION62_BATCH_SIZE:-16}"

SECTION61_GENERATED="${SECTION61_GENERATED:-data/section61/generated}"
SECTION61_OUTPUT_ROOT="${SECTION61_OUTPUT_ROOT:-data/section61/qwen_seeds}"

SECTION62_GENERATED="${SECTION62_GENERATED:-data/section62/generated}"
SECTION62_BEHAVIORAL="${SECTION62_BEHAVIORAL:-data/section62/qwen/behavioral}"
SECTION62_OUTPUT_ROOT="${SECTION62_OUTPUT_ROOT:-data/section62/qwen/r2_rank64_seed_sweep}"

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

rotation_complete() {
  local rotation_dir="$1"
  [[ -s "$rotation_dir/rotation_weight.pt" &&
     -s "$rotation_dir/rotation_weight_metadata.json" &&
     -s "$rotation_dir/summary_metrics.json" ]]
}

run_section61_behavioral() {
  local samples="$1"
  local output_dir="$2"

  if [[ -s "$output_dir/behavioral_summary.json" &&
        -s "$output_dir/scored.csv" ]]; then
    echo "=== skip completed Section 6.1 behavioral: $output_dir ==="
    return
  fi

  echo "=== Section 6.1 behavioral: $output_dir ==="
  python3 code/run_section61.py behavioral \
    --samples "$samples" \
    --model-name "$MODEL" \
    --batch-size "$SECTION61_BATCH_SIZE" \
    --eval-batch-size "$SECTION61_BATCH_SIZE" \
    --device-map none \
    --device cuda \
    --torch-dtype "$DTYPE" \
    --local-files-only \
    --output-dir "$output_dir"
}

run_section61_das() {
  local rotation_dir="$1"
  local output_dir="$2"
  shift 2
  local samples=("$@")

  if [[ -s "$output_dir/das_summary.json" &&
        -s "$output_dir/scored.csv" ]]; then
    echo "=== skip completed Section 6.1 DAS: $output_dir ==="
    return
  fi

  require_dir "$rotation_dir"
  require_file "$rotation_dir/rotation_weight.pt"
  require_file "$rotation_dir/summary_metrics.json"

  echo "=== Section 6.1 DAS: $output_dir ==="
  python3 code/run_section61.py das \
    --samples "${samples[@]}" \
    --rotation-dir "$rotation_dir" \
    --seed 0 \
    --model-name "$MODEL" \
    --eval-batch-size "$SECTION61_BATCH_SIZE" \
    --device-map none \
    --device cuda \
    --torch-dtype "$DTYPE" \
    --local-files-only \
    --output-dir "$output_dir"
}

run_section61_seed() {
  local seed="$1"
  local output_root="$SECTION61_OUTPUT_ROOT/seed${seed}"
  local rho_root="data/das/seed_sweep_qwen_rho_r64/seed${seed}"
  local pi_root="data/das/seed_sweep_qwen_pi_r64/seed${seed}"
  local pc_root="data/das/seed_sweep_qwen_pc_r64/seed${seed}"
  local m_root="data/das/seed_sweep_qwen_m_r64/seed${seed}"

  echo
  echo "######## Section 6.1: Qwen rotation seed $seed ########"

  # E1 and E7 are behavioral gates.  E7 DAS is intentionally absent because
  # Qwen's double-negation behavioral gate does not pass.
  run_section61_behavioral \
    "$SECTION61_GENERATED/e1_behavioral.csv" \
    "$output_root/e1"
  run_section61_behavioral \
    "$SECTION61_GENERATED/e7_behavioral.csv" \
    "$output_root/e7_behavior"

  # E2: within-form rho transfer.
  run_section61_das \
    "$rho_root/L18_claim_final" "$output_root/e2_claim" \
    "$SECTION61_GENERATED/e2_rho_within.csv"
  run_section61_das \
    "$rho_root/L24_answer_token" "$output_root/e2_answer" \
    "$SECTION61_GENERATED/e2_rho_within.csv"

  # E3: cross-form rho transfer.
  run_section61_das \
    "$rho_root/L18_claim_final" "$output_root/e3_cross_claim" \
    "$SECTION61_GENERATED/e3_rho_cross.csv"
  run_section61_das \
    "$rho_root/L24_answer_token" "$output_root/e3_cross_answer" \
    "$SECTION61_GENERATED/e3_rho_cross.csv"

  # E4: raw p_i and p_c identification.
  run_section61_das \
    "$pi_root/L16_claim_final" "$output_root/e4_pi_claim" \
    "$SECTION61_GENERATED/e4_pi_within.csv"
  run_section61_das \
    "$pi_root/L22_answer_token" "$output_root/e4_pi_answer" \
    "$SECTION61_GENERATED/e4_pi_within.csv"
  run_section61_das \
    "$pc_root/L14_claim_final" "$output_root/e4_pc_claim" \
    "$SECTION61_GENERATED/e4_pc_within.csv"
  run_section61_das \
    "$pc_root/L22_answer_token" "$output_root/e4_pc_answer" \
    "$SECTION61_GENERATED/e4_pc_within.csv"

  # E5: matching gate m across all six lexical forms.
  local e5_samples=(
    "$SECTION61_GENERATED/e5_m_did_not.csv"
    "$SECTION61_GENERATED/e5_m_did_not_ever.csv"
    "$SECTION61_GENERATED/e5_m_didnt.csv"
    "$SECTION61_GENERATED/e5_m_failed_to.csv"
    "$SECTION61_GENERATED/e5_m_never.csv"
    "$SECTION61_GENERATED/e5_m_not_the_case.csv"
  )
  run_section61_das \
    "$m_root/L16_claim_final" "$output_root/e5_m_claim" \
    "${e5_samples[@]}"
  run_section61_das \
    "$m_root/L22_answer_token" "$output_root/e5_m_answer" \
    "${e5_samples[@]}"

  # E6: mixed-form rho transfer.
  run_section61_das \
    "$rho_root/L18_claim_final" "$output_root/e6_mixed_claim" \
    "$SECTION61_GENERATED/e6_rho_mixed.csv"
  run_section61_das \
    "$rho_root/L24_answer_token" "$output_root/e6_mixed_answer" \
    "$SECTION61_GENERATED/e6_rho_mixed.csv"

  echo "=== summarize Section 6.1 Qwen seed $seed ==="
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
}

ensure_section62_behavioral() {
  if [[ -s "$SECTION62_BEHAVIORAL/behavioral_summary.json" &&
        -s "$SECTION62_BEHAVIORAL/scored.csv" ]]; then
    echo "=== reuse completed Section 6.2 behavioral handoff ==="
    return
  fi

  echo "=== Section 6.2 behavioral handoff ==="
  python3 code/run_section62.py behavioral \
    --r0-manifest "$SECTION62_GENERATED/manifest.json" \
    --square-samples "$SECTION62_GENERATED/mnli_square_valid.csv" \
    --all-u-samples "$SECTION62_GENERATED/mnli_all_u.csv" \
    --model-name "$MODEL" \
    --batch-size "$SECTION62_BATCH_SIZE" \
    --device-map none \
    --device cuda \
    --torch-dtype "$DTYPE" \
    --local-files-only \
    --output-dir "$SECTION62_BEHAVIORAL"
}

ensure_qwen_m_seed0_cell() {
  local layer="$1"
  local site="$2"
  local cell
  cell="L$(printf '%02d' "$layer")_${site}"
  local rotation_dir="data/das/seed_sweep_qwen_m_r64/seed0/${cell}"
  if rotation_complete "$rotation_dir"; then
    echo "=== reuse Qwen m r64 seed0 $cell ==="
    return
  fi

  echo
  echo "######## prerequisite: train Qwen m r64 seed0 $cell ########"
  python3 code/run_das_relay_map.py \
    --samples data/das/m_v4/pairs.csv \
    --model-name "$MODEL" \
    --target-var m \
    --layers "$layer" \
    --sites "$site" \
    --rank 64 \
    --epochs 1 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --learning-rate 0.002 \
    --eval-interval 0 \
    --seed 0 \
    --train-control-types all \
    --train-control-proportions \
      match_to_nomatch=0.4 \
      nomatch_to_match=0.4 \
      label_copy_trap=0.1 \
      label_copy_trap_same_m1=0.1 \
    --device-map none \
    --device cuda \
    --torch-dtype "$DTYPE" \
    --local-files-only \
    --output-dir data/das/seed_sweep_qwen_m_r64/seed0 \
    --resume

  if ! rotation_complete "$rotation_dir"; then
    echo "ERROR: training returned without a complete rotation: $rotation_dir" >&2
    exit 2
  fi
}

run_section62_profile() {
  local seed="$1"
  local layer="$2"
  local site="$3"
  local cell
  cell="L$(printf '%02d' "$layer")_${site}"
  local rho_rotation="data/das/seed_sweep_qwen_rho_r64/seed${seed}/${cell}"
  local m_rotation="data/das/seed_sweep_qwen_m_r64/seed${seed}/${cell}"
  local seed_root="$SECTION62_OUTPUT_ROOT/seed${seed}"
  local r2_dir="$seed_root/$site"
  local opposite_dir="$r2_dir/opposite_rho"
  local r3_dir="$r2_dir/r3_u_gate_transfer"
  local behavioral_summary="$SECTION62_BEHAVIORAL/behavioral_summary.json"

  require_dir "$rho_rotation"
  require_dir "$m_rotation"
  require_file "$rho_rotation/rotation_weight.pt"
  require_file "$rho_rotation/rotation_weight_metadata.json"
  require_file "$rho_rotation/summary_metrics.json"
  require_file "$m_rotation/rotation_weight.pt"
  require_file "$m_rotation/rotation_weight_metadata.json"
  require_file "$m_rotation/summary_metrics.json"

  echo
  echo "######## Section 6.2: Qwen r64 seed $seed, $cell ########"

  # R2: full-square necessity/zero controls and same-rho portability.
  if [[ -s "$r2_dir/run_manifest.json" &&
        -s "$r2_dir/ablation_summary.json" &&
        -s "$r2_dir/base_scored.csv" ]]; then
    echo "=== skip completed Section 6.2 R2: $r2_dir ==="
  else
    python3 code/run_section62_ablation.py \
      --behavioral-summary "$behavioral_summary" \
      --sites "$site" \
      --rho-rotation-dir "$rho_rotation" \
      --m-rotation-dir "$m_rotation" \
      --rotation-rank 64 \
      --rotation-seed "$seed" \
      --analysis-population full-square \
      --batch-size "$SECTION62_BATCH_SIZE" \
      --device-map none \
      --device cuda \
      --torch-dtype "$DTYPE" \
      --local-files-only \
      --output-dir "$r2_dir"
  fi

  # R2 portability target from the paper text: write the opposite rho value
  # from the paired MNLI cell, so the expected T/F decision reverses.
  if [[ -s "$opposite_dir/run_manifest.json" &&
        -s "$opposite_dir/opposite_summary.json" &&
        -s "$opposite_dir/rho_opposite_cross_cell.csv" ]]; then
    echo "=== skip completed Section 6.2 opposite-rho: $opposite_dir ==="
  else
    python3 code/run_section62_ablation.py \
      --opposite-rho-only \
      --behavioral-summary "$behavioral_summary" \
      --existing-run-dir "$r2_dir" \
      --sites "$site" \
      --rho-rotation-dir "$rho_rotation" \
      --m-rotation-dir "$m_rotation" \
      --rotation-rank 64 \
      --rotation-seed "$seed" \
      --analysis-population full-square \
      --batch-size "$SECTION62_BATCH_SIZE" \
      --device-map none \
      --device cuda \
      --torch-dtype "$DTYPE" \
      --local-files-only \
      --output-dir "$opposite_dir"
  fi

  # R3: m coordinate transfer between polarized and all-U strata.  rho and
  # rank-matched random coordinate transfers are negative controls.
  if [[ -s "$r3_dir/run_manifest.json" &&
        -s "$r3_dir/summary.json" &&
        -s "$r3_dir/intervention_scored.csv" ]]; then
    echo "=== skip completed Section 6.2 m-gate transfer: $r3_dir ==="
  else
    python3 code/run_section62_m_gate_transfer.py \
      --behavioral-summary "$behavioral_summary" \
      --existing-run-dir "$r2_dir" \
      --sites "$site" \
      --rho-rotation-dir "$rho_rotation" \
      --m-rotation-dir "$m_rotation" \
      --rotation-rank 64 \
      --rotation-seed "$seed" \
      --batch-size "$SECTION62_BATCH_SIZE" \
      --device-map none \
      --device cuda \
      --torch-dtype "$DTYPE" \
      --local-files-only \
      --output-dir "$r3_dir"
  fi
}

for required in \
  "$SECTION61_GENERATED/e1_behavioral.csv" \
  "$SECTION61_GENERATED/e2_rho_within.csv" \
  "$SECTION61_GENERATED/e3_rho_cross.csv" \
  "$SECTION61_GENERATED/e4_pi_within.csv" \
  "$SECTION61_GENERATED/e4_pc_within.csv" \
  "$SECTION61_GENERATED/e5_m_did_not.csv" \
  "$SECTION61_GENERATED/e5_m_did_not_ever.csv" \
  "$SECTION61_GENERATED/e5_m_didnt.csv" \
  "$SECTION61_GENERATED/e5_m_failed_to.csv" \
  "$SECTION61_GENERATED/e5_m_never.csv" \
  "$SECTION61_GENERATED/e5_m_not_the_case.csv" \
  "$SECTION61_GENERATED/e6_rho_mixed.csv" \
  "$SECTION61_GENERATED/e7_behavioral.csv" \
  "$SECTION62_GENERATED/manifest.json" \
  "$SECTION62_GENERATED/mnli_square_valid.csv" \
  "$SECTION62_GENERATED/mnli_all_u.csv"
do
  require_file "$required"
done

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


ensure_section62_behavioral
ensure_qwen_m_seed0_cell 18 claim_final
ensure_qwen_m_seed0_cell 24 answer_token

for rotation_seed in 0 1 2; do
  run_section62_profile "$rotation_seed" 18 claim_final
  run_section62_profile "$rotation_seed" 24 answer_token
done

run_section61_seed 1
run_section61_seed 2

echo
echo "DONE."
echo "Section 6.1: $SECTION61_OUTPUT_ROOT/seed{1,2}/"
echo "Section 6.2: $SECTION62_OUTPUT_ROOT/seed{0,1,2}/"
