#!/usr/bin/env bash
# End-to-end: symmetric W4A16 QAT -> dense checkpoint -> compressed-tensors.
#
# Paths are relative to the repository so this runs anywhere; override any of
# the variables below from the environment.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_MODEL="${SOURCE_MODEL:?set SOURCE_MODEL to a dense base model directory or hub id}"
GROUP_SIZE="${GROUP_SIZE:-128}"
STEPS="${STEPS:-40}"
QAT_DIR="${QAT_DIR:-$REPO_DIR/checkpoints/qat_dense}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/checkpoints/compressed_tensors_model}"
IGNORE="${IGNORE:-lm_head}"

cd "$REPO_DIR"

echo ">>> [1/3] QAT finetune (symmetric INT4, group_size=$GROUP_SIZE)"
"$PYTHON_BIN" examples/qat_finetune.py \
    --source "$SOURCE_MODEL" \
    --data-file examples/data/behavior_learning_data.json \
    --steps "$STEPS" \
    --group-size "$GROUP_SIZE" \
    --ignore $IGNORE \
    --output-dir "$QAT_DIR"

echo ">>> [2/3] Convert to compressed-tensors"
"$PYTHON_BIN" -m torchao_to_compressed_tensors.adapter \
    --source "$QAT_DIR" \
    --output-dir "$OUT_DIR" \
    --group-size "$GROUP_SIZE" \
    --ignore $IGNORE

echo ">>> [3/3] Verify"
"$PYTHON_BIN" examples/verify_inference.py --model-dir "$OUT_DIR"

echo
echo "Done. Serve with:  vllm serve $OUT_DIR"
echo "Confirm the log reports: Using MarlinLinearKernel for CompressedTensorsWNA16"
