#!/usr/bin/env bash
# Export a training checkpoint's actor weights as a HuggingFace model dir for the eval runner.
# This verl (Megatron-Bridge) already writes the weights in HF format to actor/model/huggingface, so no conversion is
# needed: the files are hard-linked (instant, no extra disk), and survive verl rotating the checkpoint away later.
# Usage: ./merge_ckpt.sh RUN_NAME STEP   ->  harness_rl/outputs/models/RUN_NAME/step_STEP   (override with OUT=...)
set -euo pipefail
RUN=${1:?run}; STEP=${2:?step}
SRC=/home/ubuntu/playground/outputs/checkpoints/$RUN/global_step_$STEP/actor/model/huggingface
OUT=${OUT:-/home/ubuntu/rsi/harness_rl/outputs/models/$RUN/step_$STEP}
[ -f "$SRC/config.json" ] || { echo "no HF weights at $SRC" >&2; exit 1; }
[ -f "$OUT/config.json" ] && { echo "already exported: $OUT"; exit 0; }
mkdir -p "$(dirname "$OUT")"
cp -al "$SRC" "$OUT"
echo "exported -> $OUT"
