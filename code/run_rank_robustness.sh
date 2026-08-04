#!/usr/bin/env bash
# Rank-robustness DAS sweeps on focused causal-flow layers.
#
# Defaults:
#   - ranks 16, 128, 256, 512 (rank 64 already has the main full sweeps)
#   - seed 0 only
#   - both claim_final and answer_token sites
#   - focused30 layer profile: about 30% of model layers, including the causal
#     core plus one early and one late negative-control anchor
#   - one full epoch, exact headline control mix and batch sizes
#   - one visible CUDA device, with device_map disabled to prevent auto-sharding
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 bash code/run_rank_robustness.sh qwen_pi qwen_pc
#   CUDA_VISIBLE_DEVICES=1 bash code/run_rank_robustness.sh qwen_rho qwen_m
#   CUDA_VISIBLE_DEVICES=2 bash code/run_rank_robustness.sh phi4_pi phi4_pc
#   CUDA_VISIBLE_DEVICES=3 bash code/run_rank_robustness.sh phi4_rho phi4_m
#
# Useful overrides:
#   RANKS="16 128" START_SEED=1 NUM_SEEDS=2 bash code/run_rank_robustness.sh qwen_rho
#   PROFILE=peaks DRY_RUN=1 bash code/run_rank_robustness.sh phi4_pi
#   PROFILE=full RANKS="128" bash code/run_rank_robustness.sh qwen_m
#   LAYERS="12 14 16 18 20 22" bash code/run_rank_robustness.sh qwen_rho
set -euo pipefail
cd "$(dirname "$0")/.."

RANKS="${RANKS:-16 128 256 512}"
START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-1}"
PROFILE="${PROFILE:-focused30}"  # focused30 | peaks | full
LAYERS="${LAYERS:-}"             # optional global override
SITES="${SITES:-claim_final answer_token}"
DTYPE="${DTYPE:-bfloat16}"
STEPS="${STEPS:-}"
EPOCHS="${EPOCHS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-none}"
OUTPUT_BASE="${OUTPUT_BASE:-data/das/rank_robustness}"
DRY_RUN="${DRY_RUN:-0}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
EVAL_BATCH_SIZE_OVERRIDE="${EVAL_BATCH_SIZE_OVERRIDE:-}"

read -r -a RANK_LIST <<< "$RANKS"
read -r -a RUN_SITES <<< "$SITES"

if [ "${#RANK_LIST[@]}" -eq 0 ]; then
  echo "RANKS must contain at least one rank" >&2
  exit 2
fi
for rank in "${RANK_LIST[@]}"; do
  if ! [[ "$rank" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid rank in RANKS: $rank" >&2
    exit 2
  fi
done
if ! [[ "$START_SEED" =~ ^[0-9]+$ ]] || ! [[ "$NUM_SEEDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "START_SEED must be non-negative and NUM_SEEDS must be positive" >&2
  exit 2
fi
if [ "$PROFILE" != "focused30" ] && [ "$PROFILE" != "peaks" ] && [ "$PROFILE" != "full" ]; then
  echo "PROFILE must be focused30, peaks, or full" >&2
  exit 2
fi
if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [ -n "$STEPS" ]; then
  DURATION=(--steps "$STEPS")
else
  DURATION=(--epochs "$EPOCHS")
fi

# name      target dataset model                           batch evalb original cell for exact control mix
TRN=(
  "qwen_rho rho rho_v1 Qwen/Qwen3-8B                 32 64 qwen_rho_v1_r16_stride2_1ep_b32/L18_claim_final"
  "qwen_pc  pc  pc_v4  Qwen/Qwen3-8B                 32 64 qwen_pc_v4_r16_stride2_1ep/L14_claim_final"
  "qwen_m   m   m_v4   Qwen/Qwen3-8B                 32 64 qwen_m_v4_r16_stride2_1ep_b32/L28_answer_token"
  "qwen_pi  pi  pi_v5  Qwen/Qwen3-8B                 20 60 qwen_pi_v5_rawid_r16_stride2_1ep_b20/L16_claim_final"
  "phi4_rho rho rho_v1 microsoft/Phi-4-mini-instruct 32 64 phi4_rho_r64_stride2_1ep/L12_row"
  "phi4_pc  pc  pc_v4  microsoft/Phi-4-mini-instruct 32 64 phi4_pc_v4_r64_stride2_1ep/L10_claim_final"
  "phi4_m   m   m_v4   microsoft/Phi-4-mini-instruct 20 64 phi4_m_v4_r64_stride2/L12_claim_final"
  "phi4_pi  pi  pi_v5  microsoft/Phi-4-mini-instruct 20 60 phi4_pi_v5_rawid_r64_stride2_1ep_b20/L12_claim_final"
)

KNOWN=(qwen_rho qwen_pc qwen_m qwen_pi phi4_rho phi4_pc phi4_m phi4_pi)
WANT=("$@")

is_known() {
  local candidate=$1
  local known
  for known in "${KNOWN[@]}"; do
    [ "$candidate" = "$known" ] && return 0
  done
  return 1
}

for requested in "${WANT[@]}"; do
  if ! is_known "$requested"; then
    echo "Unknown combo: $requested" >&2
    echo "Valid combos: ${KNOWN[*]}" >&2
    exit 2
  fi
done

want() {
  local candidate=$1
  local requested
  [ "${#WANT[@]}" -eq 0 ] && return 0
  for requested in "${WANT[@]}"; do
    [ "$candidate" = "$requested" ] && return 0
  done
  return 1
}

control_flags() {
  python - "$1" <<'PY'
import json
import sys

metadata = json.load(open(f"data/das/{sys.argv[1]}/rotation_weight_metadata.json"))
proportions = metadata.get("train_control_proportions")
if proportions:
    print(
        "--train-control-types all --train-control-proportions "
        + " ".join(f"{name}={weight:g}" for name, weight in proportions.items())
    )
PY
}

layers_for() {
  local name=$1
  if [ -n "$LAYERS" ]; then
    echo "$LAYERS"
    return
  fi

  case "$PROFILE:$name" in
    # About 30% of layers. These sets preserve the core transition and include
    # early/late negative-control anchors to detect high-rank overfitting.
    focused30:qwen_pc)  echo "0 8 10 12 14 16 18 20 22 24 34" ;;
    focused30:qwen_pi)  echo "0 10 12 14 16 18 20 22 24 26 34" ;;
    focused30:qwen_rho) echo "0 12 14 16 18 20 22 24 26 32 34" ;;
    focused30:qwen_m)   echo "0 10 12 14 16 18 20 22 24 28 34" ;;
    focused30:phi4_pc)  echo "0 6 8 10 12 14 16 18 22 30" ;;
    focused30:phi4_pi)  echo "0 6 8 10 12 14 16 18 22 30" ;;
    focused30:phi4_rho) echo "0 6 8 10 12 14 16 18 24 30" ;;
    focused30:phi4_m)   echo "0 6 8 10 12 14 16 18 22 30" ;;

    # Cheap pilot: peak neighborhoods and handoff cells only.
    peaks:qwen_pc)  echo "12 14 16 20 22 24" ;;
    peaks:qwen_pi)  echo "14 16 18 20 22 24" ;;
    peaks:qwen_rho) echo "14 16 18 22 24 32" ;;
    peaks:qwen_m)   echo "16 18 20 22 24 28" ;;
    peaks:phi4_pc)  echo "8 10 12 14 16 18" ;;
    peaks:phi4_pi)  echo "8 10 12 14 16 18" ;;
    peaks:phi4_rho) echo "10 12 14 16 18 20" ;;
    peaks:phi4_m)   echo "10 12 14 16 18 20" ;;

    full:qwen_*) echo "0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34" ;;
    full:phi4_*) echo "0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30" ;;
    *)
      echo "No layer profile for $PROFILE:$name" >&2
      return 2
      ;;
  esac
}

run_command() {
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [ "$DRY_RUN" = "0" ]; then
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "WARNING: CUDA_VISIBLE_DEVICES is unset; DEVICE=cuda will use logical cuda:0." >&2
  fi
  python - <<'PY'
import os
import torch

print(
    "CUDA audit:",
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
    f"device_count={torch.cuda.device_count()}",
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY
fi

for row in "${TRN[@]}"; do
  set -- $row
  NAME=$1
  VAR=$2
  DS=$3
  MODEL=$4
  BATCH=$5
  EVAL_BATCH=$6
  ORIG=$7
  want "$NAME" || continue

  [ -n "$BATCH_SIZE_OVERRIDE" ] && BATCH=$BATCH_SIZE_OVERRIDE
  [ -n "$EVAL_BATCH_SIZE_OVERRIDE" ] && EVAL_BATCH=$EVAL_BATCH_SIZE_OVERRIDE
  read -r -a RUN_LAYERS <<< "$(layers_for "$NAME")"
  read -r -a CTRL_FLAGS <<< "$(control_flags "$ORIG")"

  echo
  echo "### $NAME layers=${RUN_LAYERS[*]} sites=${RUN_SITES[*]}"
  echo "### control mix: ${CTRL_FLAGS[*]:-<unstratified>}"

  for rank in "${RANK_LIST[@]}"; do
    for seed in $(seq "$START_SEED" $((START_SEED + NUM_SEEDS - 1))); do
      OUTPUT_DIR="$OUTPUT_BASE/$NAME/r$rank/seed$seed"
      CMD=(
        python code/run_das_relay_map.py
        --samples "data/das/$DS/pairs.csv"
        --model-name "$MODEL"
        --target-var "$VAR"
        --layers "${RUN_LAYERS[@]}"
        --sites "${RUN_SITES[@]}"
        --rank "$rank"
        "${DURATION[@]}"
        --batch-size "$BATCH"
        --eval-batch-size "$EVAL_BATCH"
        --learning-rate 0.002
        --eval-interval "$EVAL_INTERVAL"
        --seed "$seed"
        "${CTRL_FLAGS[@]}"
        --torch-dtype "$DTYPE"
        --local-files-only
        --device-map "$DEVICE_MAP"
        --device "$DEVICE"
        --output-dir "$OUTPUT_DIR"
        --resume
      )
      echo "=== rank robustness $NAME rank=$rank seed=$seed output=$OUTPUT_DIR ==="
      run_command "${CMD[@]}"
    done
  done
done

echo
echo "DONE. Results under $OUTPUT_BASE/<combo>/r<rank>/seed<seed>/"
