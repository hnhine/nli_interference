#!/usr/bin/env bash
set -euo pipefail

# Score one model on the exact shared 42k Section 4 protocol.
# Run one invocation per GPU/tmux pane.

cd /workspace/nhi/nli_interference

if (( $# != 1 )); then
  echo "Usage: bash code/run_section4_screening.sh MODEL_KEY" >&2
  echo "Keys: qwen phi4 gemma3 gemma2 olmo3 granite llama31 gemma4" >&2
  exit 2
fi

MODEL_KEY="$1"
case "$MODEL_KEY" in
  qwen)    MODEL="Qwen/Qwen3-8B" ;;
  phi4)    MODEL="microsoft/Phi-4-mini-instruct" ;;
  gemma3)  MODEL="google/gemma-3-12b-it" ;;
  gemma2)  MODEL="google/gemma-2-9b" ;;
  olmo3)   MODEL="allenai/Olmo-3-7B-Instruct" ;;
  granite) MODEL="ibm-granite/granite-4.1-8b" ;;
  llama31) MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
  gemma4)  MODEL="google/gemma-4-12B" ;;
  *)
    echo "Unknown MODEL_KEY '$MODEL_KEY'" >&2
    exit 2
    ;;
esac

PROTOCOL_DIR="${PROTOCOL_DIR:-data/section4_screening/protocol_seed0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/section4_screening/results}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_KEY}"

if [[ ! -s "${PROTOCOL_DIR}/samples.csv" || ! -s "${PROTOCOL_DIR}/manifest.json" ]]; then
  python3 code/prepare_section4_screening.py \
    --n-base-events 1000 \
    --seed 0 \
    --output-dir "$PROTOCOL_DIR"
fi

if [[ -s "${OUTPUT_DIR}/summary/summary_metrics.json" && "${FORCE:-0}" != 1 ]]; then
  echo "Completed output exists; skipping ${MODEL_KEY}: ${OUTPUT_DIR}"
  exit 0
fi

echo "=== Section 4 screening: ${MODEL_KEY} (${MODEL}) ==="
python3 code/run_interference_suite.py exp6-run \
  --samples "${PROTOCOL_DIR}/samples.csv" \
  --model-name "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --device-map none \
  --device cuda \
  --torch-dtype bfloat16 \
  --local-files-only \
  --output-dir "$OUTPUT_DIR" \
  --csv-name samples.csv

echo "DONE: ${OUTPUT_DIR}/summary/summary_metrics.json"
