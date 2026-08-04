#!/usr/bin/env bash
# DAS multi-seed retrain at each peak cell (needs the torch.manual_seed fix in
# das_pyvene.py so --seed varies the rotation init, not just batch order).
# Trains one peak cell by default, or the original full stride-2 layer/site grid with FULL_SWEEP=1.
#
# Usage:
#   bash code/run_seed_train.sh                  # all 8 combos, seeds 0..4
#   NUM_SEEDS=3 bash code/run_seed_train.sh       # fewer seeds
#   bash code/run_seed_train.sh qwen_rho phi4_pi  # only listed combos
#   START_SEED=1 NUM_SEEDS=2 FULL_SWEEP=1 bash code/run_seed_train.sh  # Qwen defaults to r64; resume partial sweeps
#   QWEN_RANK=16 START_SEED=1 NUM_SEEDS=1 FULL_SWEEP=1 bash code/run_seed_train.sh qwen_m  # resume legacy r16
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_SEEDS="${NUM_SEEDS:-5}"
START_SEED="${START_SEED:-0}"   # seed 0 = original headline run; set 1 to add fresh seeds only
DTYPE="${DTYPE:-bfloat16}"
QWEN_RANK="${QWEN_RANK:-64}" # default new Qwen sweeps to r64; set 16 to resume legacy r16 outputs
FULL_SWEEP="${FULL_SWEEP:-0}" # 1 = all original stride-2 layers/sites; 0 = peak cell only
QWEN_SWEEP_LAYERS=(0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34)
PHI4_SWEEP_LAYERS=(0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30)
if [ "$FULL_SWEEP" != "0" ] && [ "$FULL_SWEEP" != "1" ]; then
  echo "FULL_SWEEP must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$QWEN_RANK" =~ ^[1-9][0-9]*$ ]]; then
  echo "QWEN_RANK must be a positive integer" >&2
  exit 2
fi
# STEPS: override the default (--epochs 1). Empty = full epoch (matches headline).
#        Set e.g. STEPS=600 only AFTER verifying IIA has plateaued by then.
# EVAL_INTERVAL: 0 (default) = no mid-training eval. Set e.g. 100 for a convergence
#        check run so summary_metrics.json / stdout show IIA vs step.
STEPS="${STEPS:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
if [ -n "$STEPS" ]; then DURATION=(--steps "$STEPS"); else DURATION=(--epochs 1); fi

# The control-training mix MUST match the original headline run or the retrained
# subspace aces `main` but fails the audit controls (pc_macro / test_IIA collapse
# to chance). We read train_control_proportions straight from each original cell's
# metadata so the seeds reproduce the exact recipe; --epochs 1 then also lands on
# the same step count.
#
# name          target dataset model                          rank layer site         batch evalb  orig-cell-dir (for control mix)
TRN=(
  "qwen_rho rho rho_v1 Qwen/Qwen3-8B                 $QWEN_RANK 18 claim_final  32 64 qwen_rho_v1_r16_stride2_1ep_b32/L18_claim_final"
  "qwen_pc  pc  pc_v4  Qwen/Qwen3-8B                 $QWEN_RANK 14 claim_final  32 64 qwen_pc_v4_r16_stride2_1ep/L14_claim_final"
  "qwen_m   m   m_v4   Qwen/Qwen3-8B                 $QWEN_RANK 28 answer_token 32 64 qwen_m_v4_r16_stride2_1ep_b32/L28_answer_token"
  "qwen_pi  pi  pi_v5  Qwen/Qwen3-8B                 $QWEN_RANK 16 claim_final  20 60 qwen_pi_v5_rawid_r16_stride2_1ep_b20/L16_claim_final"
  "phi4_rho rho rho_v1 microsoft/Phi-4-mini-instruct 64 12 row         32 64 phi4_rho_r64_stride2_1ep/L12_row"
  "phi4_pc  pc  pc_v4  microsoft/Phi-4-mini-instruct 64 10 claim_final 32 64 phi4_pc_v4_r64_stride2_1ep/L10_claim_final"
  "phi4_m   m   m_v4   microsoft/Phi-4-mini-instruct 64 12 claim_final 20 64 phi4_m_v4_r64_stride2/L12_claim_final"
  "phi4_pi  pi  pi_v5  microsoft/Phi-4-mini-instruct 64 12 claim_final 20 60 phi4_pi_v5_rawid_r64_stride2_1ep_b20/L12_claim_final"
)

# Emit `--train-control-types all --train-control-proportions k=v ...` from an
# original cell's metadata; nothing if that run was unstratified (props null).
control_flags() {
  python - "$1" <<'PY'
import json, sys
d = json.load(open(f"data/das/{sys.argv[1]}/rotation_weight_metadata.json"))
p = d.get("train_control_proportions")
if p:
    print("--train-control-types all --train-control-proportions " +
          " ".join(f"{k}={v:g}" for k, v in p.items()))
PY
}

WANT=("$@")
want() { [ ${#WANT[@]} -eq 0 ] && return 0; for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

for row in "${TRN[@]}"; do
  set -- $row; NAME=$1; VAR=$2; DS=$3; MODEL=$4; RANK=$5; L=$6; SITE=$7; B=$8; EB=$9; ORIG=${10}
  want "$NAME" || continue
  CTRL=$(control_flags "$ORIG")
  echo "### $NAME control mix: ${CTRL:-<unstratified>}"
  RANK_SUFFIX=""
  if [[ "$NAME" == qwen_* ]] && [ "$RANK" != "16" ]; then
    RANK_SUFFIX="_r${RANK}"
  fi
  if [ "$FULL_SWEEP" = "1" ]; then
    if [[ "$NAME" == qwen_* ]]; then
      RUN_LAYERS=("${QWEN_SWEEP_LAYERS[@]}")
    else
      RUN_LAYERS=("${PHI4_SWEEP_LAYERS[@]}")
    fi
    RUN_SITES=(claim_final answer_token)
    OUTPUT_ROOT="data/das/seed_sweep_${NAME}${RANK_SUFFIX}"
    RESUME_FLAGS=(--resume)
  else
    RUN_LAYERS=("$L")
    RUN_SITES=("$SITE")
    OUTPUT_ROOT="data/das/seed_${NAME}${RANK_SUFFIX}"
    RESUME_FLAGS=()
  fi
  for S in $(seq "$START_SEED" $((START_SEED + NUM_SEEDS - 1))); do
    echo "=== train $NAME seed $S  layers=${RUN_LAYERS[*]} sites=${RUN_SITES[*]} rank$RANK ==="
    python code/run_das_relay_map.py \
      --samples data/das/$DS/pairs.csv \
      --model-name $MODEL \
      --target-var $VAR \
      --layers "${RUN_LAYERS[@]}" --sites "${RUN_SITES[@]}" \
      --rank $RANK "${DURATION[@]}" \
      --batch-size $B --eval-batch-size $EB \
      --learning-rate 0.002 --eval-interval "$EVAL_INTERVAL" \
      --seed $S \
      $CTRL \
      --torch-dtype "$DTYPE" --local-files-only \
      --output-dir "$OUTPUT_ROOT/seed$S" "${RESUME_FLAGS[@]}"
  done
done
if [ "$FULL_SWEEP" = "1" ]; then
  echo "DONE. Full sweeps under data/das/seed_sweep_<combo>[_r<rank>]/seed*/L{layer}_{site}/"
else
  echo "DONE. Peak cells under data/das/seed_<combo>[_r<rank>]/seed*/L{layer}_{site}/"
fi
