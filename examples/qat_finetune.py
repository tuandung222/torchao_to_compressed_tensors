#!/usr/bin/env python3
"""
Symmetric W4A16 QAT finetune, producing a dense checkpoint for the adapter.

Deliberately ends at a *dense* bf16 checkpoint rather than a TorchAO-quantised
one. ``QATConfig(step="convert")`` with no base config swaps the fake-quantised
layers back to ``nn.Linear``, leaving ordinary weights that carry the QAT
training. Quantisation then happens exactly once, in the adapter, onto the grid
that is actually served. See docs/SCOPE.md.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao_to_compressed_tensors import (
    DEFAULT_IGNORE,
    MARLIN_GROUP_SIZES,
    convert_w4a16_qat,
    prepare_w4a16_qat,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="dense base model directory or hub id")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=REPO_ROOT / "examples" / "data" / "smoke_summarization_data.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "qat_dense",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=128, choices=MARLIN_GROUP_SIZES)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=list(DEFAULT_IGNORE),
        metavar="NAME",
        help="modules to leave dense. Must match the adapter's --ignore",
    )
    parser.add_argument("--quantize-embeddings", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_samples(tokenizer, data_file: Path, max_length: int) -> list[dict]:
    """Tokenise the dataset, masking prompt tokens so loss covers the response only."""
    records = json.loads(data_file.read_text(encoding="utf-8"))

    samples = []
    for record in records:
        messages = [
            {"role": "user", "content": f"{record['instruction']}\n\n{record['text']}"},
            {"role": "assistant", "content": record["summary"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        ).input_ids[0]

        prompt = tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        prompt_length = len(tokenizer(prompt)["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        samples.append({"input_ids": input_ids, "labels": labels})
    return samples


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.source} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    model = AutoModelForCausalLM.from_pretrained(args.source, dtype=torch.bfloat16)
    model.to(args.device)

    print(
        f"Preparing symmetric W4A16 QAT (group_size={args.group_size}, "
        f"ignore={args.ignore}, embeddings={args.quantize_embeddings}) ...",
        flush=True,
    )
    prepare_w4a16_qat(
        model,
        group_size=args.group_size,
        ignore=tuple(args.ignore),
        quantize_embeddings=args.quantize_embeddings,
    )
    model.train()

    samples = build_samples(tokenizer, args.data_file, args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"Training for {args.steps} steps ...", flush=True)
    started = time.time()
    for step in range(1, args.steps + 1):
        sample = samples[(step - 1) % len(samples)]
        optimizer.zero_grad()
        outputs = model(
            input_ids=sample["input_ids"].unsqueeze(0).to(args.device),
            labels=sample["labels"].unsqueeze(0).to(args.device),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"  step {step:3d}/{args.steps}  loss {outputs.loss.item():.4f}", flush=True)
    print(f"Done in {time.time() - started:.1f}s", flush=True)

    print("Removing fake quantisers, keeping QAT-trained bf16 weights ...", flush=True)
    convert_w4a16_qat(model)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Dense QAT checkpoint written to {args.output_dir}", flush=True)
    print(
        "Next: torchao-to-ct --source "
        f"{args.output_dir} --output-dir <out> --group-size {args.group_size} "
        f"--ignore {' '.join(args.ignore)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
