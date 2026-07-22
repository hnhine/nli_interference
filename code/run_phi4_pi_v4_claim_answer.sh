#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 code/run_das_relay_map.py \
  --samples data/das/pi_v4/pairs.csv \
  --model-name microsoft/Phi-4-mini-instruct \
  --target-var pi \
  --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 \
  --sites claim_final answer_token \
  --rank 64 \
  --epochs 1 \
  --batch-size 20 \
  --eval-batch-size 60 \
  --learning-rate 0.002 \
  --eval-interval 0 \
  --seed 0 \
  --train-control-types all \
  --train-control-proportions \
    main=0.40 \
    active_source_m0=0.40 \
    gate_m0=0.05 \
    label_copy_trap=0.05 \
    distractor=0.10 \
  --torch-dtype bfloat16 \
  --local-files-only \
  --resume \
  --output-dir data/das/phi4_pi_v4_r64_stride2_1ep_b20
