#!/usr/bin/env bash
# One-GPU queue:
#   1. Qwen Section 6.1 (negation-form robustness), rotation seeds 1 and 2.
#   2. Qwen Section 6.2 (MNLI naturalistic transfer), rotation seeds 0, 1, 2.
#
# Section 6.2 is NOT the Section 5 joint rho+m patch.  It evaluates:
#   - zero/necessity effects for frozen rho and m, with rank-matched controls;
#   - opposite-rho coordinate transfer within each MNLI polarity square;
#   - m transfer between polarized and all-U strata, with rho/random controls.
# Section 6.2 freezes each variable at its own preregistered peak at both sites:
# rho at L18_claim_final/L24_answer_token and m at
# L16_claim_final/L22_answer_token. MNLI is never used to choose a layer.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#     bash code/run_qwen_section61_seed1_seed2_then_section62_3seed.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DTYPE="${DTYPE:-bfloat16}"
SECTION61_BATCH_SIZE="${SECTION61_BATCH_SIZE:-64}"
SECTION62_BATCH_SIZE="${SECTION62_BATCH_SIZE:-16}"
RUN_SECTION61="${RUN_SECTION61:-1}"
SECTION62_START_SEED="${SECTION62_START_SEED:-0}"
SECTION62_NUM_SEEDS="${SECTION62_NUM_SEEDS:-3}"

SECTION61_GENERATED="${SECTION61_GENERATED:-data/section61/generated}"
SECTION61_OUTPUT_ROOT="${SECTION61_OUTPUT_ROOT:-data/section61/qwen_seeds}"

SECTION62_GENERATED="${SECTION62_GENERATED:-data/section62/generated}"
SECTION62_BEHAVIORAL="${SECTION62_BEHAVIORAL:-data/section62/qwen/behavioral}"
SECTION62_OUTPUT_ROOT="${SECTION62_OUTPUT_ROOT:-data/section62/qwen_r64_seeds}"

if [[ "$RUN_SECTION61" != "0" && "$RUN_SECTION61" != "1" ]]; then
  echo "ERROR: RUN_SECTION61 must be 0 or 1; got: $RUN_SECTION61" >&2
  exit 2
fi
if ! [[ "$SECTION62_START_SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SECTION62_START_SEED must be a non-negative integer; got: $SECTION62_START_SEED" >&2
  exit 2
fi
if ! [[ "$SECTION62_NUM_SEEDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SECTION62_NUM_SEEDS must be a positive integer; got: $SECTION62_NUM_SEEDS" >&2
  exit 2
fi

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

run_section62_site() {
  local seed="$1"
  local site="$2"
  local rho_layer="$3"
  local m_layer="$4"
  local site_label="$5"

  local rho_peak="data/das/seed_sweep_qwen_rho_r64/seed${seed}/L${rho_layer}_${site}"
  local m_peak="data/das/seed_sweep_qwen_m_r64/seed${seed}/L${m_layer}_${site}"
  local rho_at_m_peak="data/das/seed_sweep_qwen_rho_r64/seed${seed}/L${m_layer}_${site}"
  local seed_root="$SECTION62_OUTPUT_ROOT/seed${seed}"
  local r2_dir="$seed_root/r2_m_peak_${site_label}"
  local opposite_dir="$seed_root/r2_opposite_rho_peak_${site_label}"
  local r3_dir="$seed_root/r3_m_gate_peak_${site_label}"
  local behavioral_summary="$SECTION62_BEHAVIORAL/behavioral_summary.json"

  for rotation_dir in "$rho_peak" "$m_peak" "$rho_at_m_peak"; do
    require_dir "$rotation_dir"
    require_file "$rotation_dir/rotation_weight.pt"
    require_file "$rotation_dir/rotation_weight_metadata.json"
  done

  echo
  echo "--- Section 6.2 seed $seed, $site: rho=L$rho_layer, m=L$m_layer ---"

  # Establish verified base scores and m-peak zero/random controls. rho at the
  # m layer is only a same-cell negative control for the m experiment.
  if [[ -s "$r2_dir/run_manifest.json" &&
        -s "$r2_dir/ablation_summary.json" &&
        -s "$r2_dir/base_scored.csv" ]]; then
    echo "=== skip completed Section 6.2 m-peak R2: $r2_dir ==="
  else
    python3 code/run_section62_ablation.py \
      --behavioral-summary "$behavioral_summary" \
      --sites "$site" \
      --rho-rotation-dir "$rho_at_m_peak" \
      --m-rotation-dir "$m_peak" \
      --rotation-rank 64 \
      --rotation-layer "$m_layer" \
      --rotation-seed "$seed" \
      --analysis-population full-square \
      --batch-size "$SECTION62_BATCH_SIZE" \
      --device-map none \
      --device cuda \
      --torch-dtype "$DTYPE" \
      --local-files-only \
      --output-dir "$r2_dir"
  fi

  # Primary rho portability at rho's independently selected site-specific peak.
  # Verified base scores are independent of the intervention layer.
  if [[ -s "$opposite_dir/run_manifest.json" &&
        -s "$opposite_dir/opposite_summary.json" &&
        -s "$opposite_dir/rho_opposite_cross_cell.csv" ]]; then
    echo "=== skip completed Section 6.2 opposite-rho peak: $opposite_dir ==="
  else
    python3 code/run_section62_ablation.py \
      --opposite-rho-only \
      --behavioral-summary "$behavioral_summary" \
      --existing-run-dir "$r2_dir" \
      --sites "$site" \
      --rho-rotation-dir "$rho_peak" \
      --rotation-rank 64 \
      --rotation-layer "$rho_layer" \
      --rotation-seed "$seed" \
      --analysis-population full-square \
      --batch-size "$SECTION62_BATCH_SIZE" \
      --device-map none \
      --device cuda \
      --torch-dtype "$DTYPE" \
      --local-files-only \
      --output-dir "$opposite_dir"
  fi

  # Primary m transfer at m's independently selected site-specific peak.
  if [[ -s "$r3_dir/run_manifest.json" &&
        -s "$r3_dir/summary.json" &&
        -s "$r3_dir/intervention_scored.csv" ]]; then
    echo "=== skip completed Section 6.2 m-gate peak: $r3_dir ==="
  else
    python3 code/run_section62_m_gate_transfer.py \
      --behavioral-summary "$behavioral_summary" \
      --existing-run-dir "$r2_dir" \
      --sites "$site" \
      --rho-rotation-dir "$rho_at_m_peak" \
      --m-rotation-dir "$m_peak" \
      --rotation-rank 64 \
      --rotation-layer "$m_layer" \
      --rotation-seed "$seed" \
      --batch-size "$SECTION62_BATCH_SIZE" \
      --device-map none \
      --device cuda \
      --torch-dtype "$DTYPE" \
      --local-files-only \
      --output-dir "$r3_dir"
  fi
}

run_section62_seed() {
  local seed="$1"
  echo
  echo "######## Section 6.2: Qwen r64 rotation seed $seed ########"
  run_section62_site "$seed" claim_final 18 16 claim
  run_section62_site "$seed" answer_token 24 22 answer
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

if [[ "$RUN_SECTION61" == "1" ]]; then
  run_section61_seed 1
  run_section61_seed 2
else
  echo "=== skip Section 6.1 (RUN_SECTION61=0) ==="
fi

ensure_section62_behavioral

section62_last_seed=$((SECTION62_START_SEED + SECTION62_NUM_SEEDS - 1))
for ((seed = SECTION62_START_SEED; seed <= section62_last_seed; seed++)); do
  run_section62_seed "$seed"
done

echo
echo "DONE."
if [[ "$RUN_SECTION61" == "1" ]]; then
  echo "Section 6.1: $SECTION61_OUTPUT_ROOT/seed{1,2}/"
fi
echo "Section 6.2: $SECTION62_OUTPUT_ROOT/seed${SECTION62_START_SEED}..seed${section62_last_seed}/"
