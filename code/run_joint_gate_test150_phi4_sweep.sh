#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SAMPLES="data/das/joint_gate_test150/triples.csv"
OUTPUT_ROOT="data/das/joint_gate_test150_sweep_phi4"

if [[ ! -f "${SAMPLES}" ]]; then
  echo "Missing dataset: ${SAMPLES}" >&2
  echo "Generate joint_gate_test150 before starting the sweep." >&2
  exit 1
fi

for SITE in claim_final answer_token; do
  if [[ "${SITE}" == "claim_final" ]]; then
    RHO_SITE="row"
  else
    RHO_SITE="answer_token"
  fi

  for LAYER in $(seq 0 2 30); do
    printf -v L2 "%02d" "${LAYER}"
    OUT="${OUTPUT_ROOT}/${SITE}/L${L2}"

    if [[ -f "${OUT}/joint_gate_scored.csv" ]]; then
      echo "SKIP completed: Phi-4 ${SITE}/L${L2}"
      continue
    fi

    echo "START: Phi-4 ${SITE}/L${L2}"

    python3 code/run_das_joint_gate.py \
      --samples "${SAMPLES}" \
      --model-name microsoft/Phi-4-mini-instruct \
      --m-rotation "data/das/phi4_m_v4_r64_stride2/L${L2}_${SITE}" \
      --rho-rotation "data/das/phi4_rho_r64_stride2_1ep/L${L2}_${RHO_SITE}" \
      --composition-mode common \
      --site "${SITE}" \
      --eval-batch-size 32 \
      --random-seeds 0 1 2 \
      --bootstrap-samples 0 \
      --checkpoint-every 5 \
      --torch-dtype bfloat16 \
      --local-files-only \
      --output-dir "${OUT}"
  done
done

echo "PHI-4 FULL SWEEP COMPLETE"
