#!/usr/bin/env python3
"""
Structural and inference checks on a converted compressed-tensors checkpoint.

Checks what can be checked without a GPU: that the config is a valid
compressed-tensors block, that every quantised layer carries a consistent set of
tensors, that the packed weights unpack to their declared shape, and that the
model loads and generates. Kernel selection itself is only observable when vLLM
loads the checkpoint on the target GPU -- see the note printed at the end.
"""

import argparse
import json
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationConfig
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from compressed_tensors.compressors.pack_quantized import unpack_from_int32
except ImportError:  # pragma: no cover
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        unpack_from_int32,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "compressed_tensors_model",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def load_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            if f.metadata() is None or f.metadata().get("format") != "pt":
                raise AssertionError(f"{shard.name} is missing metadata format=pt")
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def verify_structure(model_dir: Path) -> None:
    print("[1/2] Verifying checkpoint structure ...", flush=True)

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    parsed = QuantizationConfig.model_validate(config["quantization_config"])
    weights = parsed.config_groups["group_0"].weights
    print(
        f"      format={parsed.format} num_bits={weights.num_bits} "
        f"strategy={weights.strategy} group_size={weights.group_size} "
        f"symmetric={weights.symmetric} ignore={parsed.ignore}",
        flush=True,
    )
    assert parsed.format == "pack-quantized", parsed.format

    if weights.group_size not in (32, 64, 128):
        raise AssertionError(
            f"group_size={weights.group_size} is not Marlin-servable; vLLM will "
            "fall back to a slower kernel"
        )

    tensors = load_tensors(model_dir)
    prefixes = sorted(
        k[: -len(".weight_packed")] for k in tensors if k.endswith(".weight_packed")
    )
    assert prefixes, "no packed weights found"
    print(f"      {len(prefixes)} quantised layers, {len(tensors)} tensors", flush=True)

    for prefix in prefixes:
        scale = tensors[f"{prefix}.weight_scale"]
        shape = torch.Size(tensors[f"{prefix}.weight_shape"].tolist())
        unpacked = unpack_from_int32(tensors[f"{prefix}.weight_packed"], 4, shape)

        assert unpacked.shape == shape, f"{prefix}: {unpacked.shape} != {shape}"
        assert unpacked.dtype == torch.int8, f"{prefix}: {unpacked.dtype}"
        assert unpacked.min() >= -8 and unpacked.max() <= 7, f"{prefix}: out of range"
        assert scale.shape == (shape[0], shape[1] // weights.group_size), (
            f"{prefix}: scale shape {tuple(scale.shape)} inconsistent with "
            f"weight shape {tuple(shape)} at group_size={weights.group_size}"
        )

        zero_point = tensors.get(f"{prefix}.weight_zero_point")
        if weights.symmetric:
            assert zero_point is None, f"{prefix}: symmetric export carries a zero-point"
        else:
            assert zero_point is not None, f"{prefix}: asymmetric export lacks a zero-point"
            assert zero_point.dtype == torch.int32, f"{prefix}: {zero_point.dtype}"
            assert tuple(zero_point.shape) == (-(-shape[0] // 8), scale.shape[-1])

    print("      all layers consistent", flush=True)


def verify_inference(model_dir: Path, device: str, max_new_tokens: int) -> None:
    print("[2/2] Loading and generating ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.to(device).eval()

    prompt = "Summarize compactly: Machine learning models need optimization to serve efficiently."
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()

    print(f"      generated: {generated!r}", flush=True)
    assert generated, "generation returned an empty string"


def main() -> None:
    args = parse_args()
    verify_structure(args.model_dir)
    if not args.skip_generation:
        verify_inference(args.model_dir, args.device, args.max_new_tokens)

    print(
        "\nStructure and inference are sound. Kernel selection is only decided "
        "when vLLM loads this on the target GPU -- run `vllm serve` and confirm "
        "the log line:\n"
        "  Using MarlinLinearKernel for CompressedTensorsWNA16",
        flush=True,
    )


if __name__ == "__main__":
    main()
