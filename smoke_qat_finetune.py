#!/usr/bin/env python3
"""
Smoke QAT Finetuning for Summarizer-600M using TorchAO.
Designed to run as fast as possible (< 1-2 minutes) for smoke validation.
Outputs a TorchAO INT4 quantized checkpoint.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
import torchao
from torchao.quantization import quantize_, Int4WeightOnlyConfig
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
from torchao.quantization.qat import Int4WeightOnlyQATQuantizer


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke QAT finetune summarizer-600M with TorchAO")
    parser.add_argument(
        "--source",
        type=str,
        default="/home/dungvpt/workspace/dungvpt/sprint27/summarization-testing/models/summarizer-600m",
        help="Path to source model",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="/home/dungvpt/workspace/dungvpt/sprint27/torchao_to_compressed_tensors/data/smoke_summarization_data.json",
        help="Path to smoke summarization dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/dungvpt/workspace/dungvpt/sprint27/torchao_to_compressed_tensors/checkpoints/torchao_model",
        help="Output directory for TorchAO checkpoint",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of smoke training steps")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    return parser.parse_args()


def load_dense_model_from_safetensors(model_path: str, device: str) -> tuple:
    print(f"[1/5] Initializing base model structure from: {model_path} ...", flush=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    
    # Initialize dense model in bfloat16 directly on device to avoid slow CPU init
    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    
    model = meta_model.to_empty(device=device)
    
    # Dequantize INT8 weights from compressed-tensors into bfloat16
    print("      Dequantizing source weights into bfloat16...", flush=True)
    safetensors_path = os.path.join(model_path, "model.safetensors")
    clean_state = {}
    with safe_open(safetensors_path, framework="pt", device=device) as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith("_scale"):
                continue
            tensor = f.get_tensor(k)
            scale_key = k + "_scale"
            if scale_key in keys:
                scale = f.get_tensor(scale_key)
                param = (tensor.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
            else:
                param = tensor.to(torch.bfloat16)
            clean_state[k] = param

    model.load_state_dict(clean_state, strict=False)
    # Crucial: tie lm_head to embed_tokens so projection is not random uninitialized memory
    model.tie_weights()
    print(f"      Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model, config


def prepare_dataset(tokenizer, data_path: str, max_length: int = 512):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        messages = [
            {"role": "user", "content": f"{item['instruction']}\n\n{item['text']}"},
            {"role": "assistant", "content": item["summary"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        enc = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
        input_ids = enc.input_ids[0]
        # Train on assistant response: mask user prompt tokens with -100
        labels = input_ids.clone()
        user_prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        user_len = len(tokenizer(user_prompt)["input_ids"])
        labels[:user_len] = -100
        samples.append({"input_ids": input_ids, "labels": labels})
    return samples


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 SMOKE QAT FINETUNING PIPELINE (TorchAO Backend)")
    print(f"   Source model : {args.source}")
    print(f"   Output dir   : {args.output_dir}")
    print(f"   Steps        : {args.steps}")
    print(f"   Group size   : {args.group_size}")
    print(f"   Device       : {args.device}")
    print("=" * 80, flush=True)

    # 1. Load tokenizer & dense base model
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    model, config = load_dense_model_from_safetensors(args.source, args.device)

    # 2. Prepare TorchAO QAT
    print(f"[2/5] Injecting TorchAO QAT fake-quantizers into transformer layers (group_size={args.group_size})...", flush=True)
    quantizer = Int4WeightOnlyQATQuantizer(groupsize=args.group_size)
    model.model = quantizer.prepare(model.model)
    model.train()

    # 3. Prepare data & optimizer
    dataset = prepare_dataset(tokenizer, args.data_file)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 4. Smoke Training Loop
    print(f"[3/5] Starting smoke QAT training loop ({args.steps} steps)...", flush=True)
    start_time = time.time()
    for step in range(1, args.steps + 1):
        sample = dataset[(step - 1) % len(dataset)]
        input_ids = sample["input_ids"].unsqueeze(0).to(args.device)
        labels = sample["labels"].unsqueeze(0).to(args.device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"   Step [{step:02d}/{args.steps:02d}] - Loss: {loss.item():.4f}", flush=True)

    elapsed = time.time() - start_time
    print(f"      QAT training complete in {elapsed:.2f}s!", flush=True)

    # 5. Convert QAT fake-quant to real TorchAO INT4 weights
    print("[4/5] Converting QAT model to real TorchAO INT4 weights...", flush=True)
    model.model = quantizer.convert(model.model)

    # 6. Save TorchAO Checkpoint
    print(f"[5/5] Saving TorchAO checkpoint to {output_dir} ...", flush=True)
    qconfig = Int4WeightOnlyConfig(
        group_size=args.group_size,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
    )
    model.config.quantization_config = TorchAoConfig(quant_type=qconfig)
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy optional template files
    for fname in ["chat_template.jinja", "generation_config.json"]:
        src_f = Path(args.source) / fname
        if src_f.exists():
            shutil.copy2(src_f, output_dir / fname)

    state_dict = model.state_dict()
    if "lm_head.weight" not in state_dict and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        state_dict["lm_head.weight"] = model.lm_head.weight
    torch.save(state_dict, output_dir / "pytorch_model.bin")

    print("=" * 80)
    print(f"✅ TorchAO checkpoint successfully saved to: {output_dir}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
