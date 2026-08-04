#!/usr/bin/env bash
# Queue the remaining Phi-4 Section 6.1 evaluations (rotation seeds 1 and 2),
# then train the Phi-4 p_i matched-assumption-site sweep for seed 2.
#
# Expected wall time on one A40/A6000-class GPU:
#   Section 6.1 seed 1: about 2h20m
#   Section 6.1 seed 2: about 2h20m
#   assumption-site p_i seed 2: depends on DAS training speed
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#     bash code/run_phi4_section61_seed1_seed2_then_assumption_seed2.sh
#
# Completed Section 6.1 stages are skipped at stage granularity. The
# assumption-site runner uses --resume internally and skips completed cells.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-microsoft/Phi-4-mini-instruct}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
GENERATED_DIR="${GENERATED_DIR:-data/section61/generated}"
SECTION61_OUTPUT_ROOT="${SECTION61_OUTPUT_ROOT:-data/section61/phi4_seeds}"
ASSUMPTION_OUTPUT_ROOT="${ASSUMPTION_OUTPUT_ROOT:-data/das/seed_sweep_phi4_pi_r64_matched_assumption_final}"

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
  local eval_seed="$3"
  shift 3
  local samples=("$@")

  if [[ -s "$output_dir/das_summary.json" && -s "$output_dir/scored.csv" ]]; then
    echo "=== skip completed DAS stage: $output_dir ==="
    return
  fi

  require_dir "$rotation_dir"
  echo "=== DAS: $output_dir ==="
  python3 code/run_section61.py das \
    --samples "${samples[@]}" \
    --rotation-dir "$rotation_dir" \
    --seed "$eval_seed" \
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
echo "Section 6.1 Phi-4 rotation seeds: 1 2"
echo "Section 6.1 output root: $SECTION61_OUTPUT_ROOT"

for seed in 1 2; do
  output_root="$SECTION61_OUTPUT_ROOT/seed$seed"
  rho_root="data/das/seed_sweep_phi4_rho/seed$seed"
  pi_root="data/das/seed_sweep_phi4_pi/seed$seed"
  pc_root="data/das/seed_sweep_phi4_pc/seed$seed"
  m_root="data/das/seed_sweep_phi4_m/seed$seed"

  echo
  echo "######## Section 6.1 Phi-4 seed $seed ########"

  # E1 and E7 are behavioral gates. Phi-4's E7 double-negation gate currently
  # fails, so—as in seed0—there is no E7 DAS invocation.
  run_behavioral "$GENERATED_DIR/e1_behavioral.csv" "$output_root/e1"
  run_behavioral "$GENERATED_DIR/e7_behavioral.csv" "$output_root/e7_behavior"

  # E2: within-form rho transfer.
  run_das "$rho_root/L12_claim_final" "$output_root/e2_claim" "$seed" \
    "$GENERATED_DIR/e2_rho_within.csv"
  run_das "$rho_root/L16_answer_token" "$output_root/e2_answer" "$seed" \
    "$GENERATED_DIR/e2_rho_within.csv"

  # E3: cross-form rho transfer.
  run_das "$rho_root/L12_claim_final" "$output_root/e3_cross_claim" "$seed" \
    "$GENERATED_DIR/e3_rho_cross.csv"
  run_das "$rho_root/L16_answer_token" "$output_root/e3_cross_answer" "$seed" \
    "$GENERATED_DIR/e3_rho_cross.csv"

  # E4: raw p_i and p_c identification.
  run_das "$pi_root/L10_claim_final" "$output_root/e4_pi_claim" "$seed" \
    "$GENERATED_DIR/e4_pi_within.csv"
  run_das "$pi_root/L16_answer_token" "$output_root/e4_pi_answer" "$seed" \
    "$GENERATED_DIR/e4_pi_within.csv"
  run_das "$pc_root/L10_claim_final" "$output_root/e4_pc_claim" "$seed" \
    "$GENERATED_DIR/e4_pc_within.csv"
  run_das "$pc_root/L16_answer_token" "$output_root/e4_pc_answer" "$seed" \
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
  run_das "$m_root/L12_claim_final" "$output_root/e5_m_claim" "$seed" \
    "${e5_samples[@]}"
  run_das "$m_root/L16_answer_token" "$output_root/e5_m_answer" "$seed" \
    "${e5_samples[@]}"

  # E6: mixed-form rho transfer.
  run_das "$rho_root/L12_claim_final" "$output_root/e6_mixed_claim" "$seed" \
    "$GENERATED_DIR/e6_rho_mixed.csv"
  run_das "$rho_root/L16_answer_token" "$output_root/e6_mixed_answer" "$seed" \
    "$GENERATED_DIR/e6_rho_mixed.csv"

  echo "=== summarize Section 6.1 Phi-4 seed $seed ==="
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
done

echo
echo "######## Phi-4 p_i matched-assumption-site seed 2 ########"

# Do not let two writers train the same seed2 output concurrently. A separate
# seed0/seed1 process is allowed; only an active relay-map writer targeting the
# seed2 assumption output is a conflict. An already completed seed2 is harmless
# because the called runner resumes/skips cells.
if pgrep -af '[r]un_das_relay_map.py.*matched_assumption_final.*seed2' >/dev/null; then
  echo "ERROR: matched-assumption seed2 is already being written by another process." >&2
  echo "Section 6.1 seeds 1 and 2 are complete, but assumption seed2 was not started." >&2
  echo "Wait for the active seed2 process, then rerun (completed Section 6.1 stages will skip):" >&2
  echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES START_SEED=2 NUM_SEEDS=1 \\" >&2
  echo "    OUTPUT_ROOT=$ASSUMPTION_OUTPUT_ROOT \\" >&2
  echo "    bash code/run_phi4_pi_matched_assumption_seed_sweep.sh" >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
START_SEED=2 NUM_SEEDS=1 \
DTYPE="$DTYPE" EVAL_INTERVAL=0 \
OUTPUT_ROOT="$ASSUMPTION_OUTPUT_ROOT" \
bash code/run_phi4_pi_matched_assumption_seed_sweep.sh

echo
echo "DONE."
echo "Section 6.1: $SECTION61_OUTPUT_ROOT/seed{1,2}/"
echo "Assumption-site p_i seed2: $ASSUMPTION_OUTPUT_ROOT/seed2/"
