#!/usr/bin/env bash
set -e

PROJECT_DIR="/home/dungvpt/workspace/dungvpt/sprint27/torchao_to_compressed_tensors"
PYTHON_BIN="/home/uney/miniconda3/envs/axolotl/bin/python"
GPU_ID=2

cd "$PROJECT_DIR"

echo "================================================================================"
echo "🎯 RUNNING END-TO-END PIPELINE: QAT SMOKE FINETUNE -> ADAPTER -> INFERENCE"
echo "================================================================================"

echo ""
echo ">>> STEP 1: Running Smoke QAT Finetuning (TorchAO)..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN smoke_qat_finetune.py \
    --data-file data/behavior_learning_data.json \
    --steps 40 \
    --group-size 128 \
    --lr 3.5e-5 \
    --output-dir checkpoints/torchao_model

echo ""
echo ">>> STEP 2: Running TorchAO to compressed-tensors Adapter..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN torchao_to_compressed_tensors_adapter.py \
    --source checkpoints/torchao_model \
    --output-dir checkpoints/compressed_tensors_model

echo ""
echo ">>> STEP 3: Running Inference Verification on compressed-tensors Checkpoint..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN verify_inference.py \
    --model-dir checkpoints/compressed_tensors_model

echo ""
echo "================================================================================"
echo "🎉 ALL STEPS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
