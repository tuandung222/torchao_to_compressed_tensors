"""
Core conversion pipeline for converting TorchAO checkpoints into Compressed-Tensors checkpoints.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import save_file

from .schemas import (
    SchemaType,
    detect_tensor_schema,
    extract_group_size_from_config,
)
from .handlers import (
    convert_int4_tinygemm,
    convert_int4_plain,
    convert_int4_preshuffled,
    convert_int8_weight_only,
    convert_int8_dynamic_act,
    convert_int8_static_act,
    convert_fp8_weight_only,
    convert_fp8_dynamic_act,
)
from .config import generate_compressed_tensors_config


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Schema TorchAO to Compressed-Tensors Adapter")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source TorchAO checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory for compressed-tensors checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for tensor transformations",
    )
    parser.add_argument(
        "--int4-asymmetric",
        action="store_true",
        help="Export INT4 weights with integer zero-point (AWQ/Marlin-ZP style)",
    )
    parser.add_argument(
        "--act-asymmetric",
        action="store_true",
        help="Export dynamic activations as asymmetric (vLLM CUTLASS AZP style)",
    )
    return parser.parse_args()


def convert_checkpoint(
    source_dir: Path,
    output_dir: Path,
    device_str: str = "cuda:0",
    int4_asymmetric: bool = False,
    act_asymmetric: bool = False,
):
    """Main checkpoint conversion entry point."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    print("=" * 80)
    print("🔄 MULTI-SCHEMA TORCHAO -> COMPRESSED-TENSORS ADAPTER")
    print(f"   Source checkpoint : {source_dir}")
    print(f"   Output directory  : {output_dir}")
    print(f"   Compute Device    : {device}")
    print(f"   INT4 Asymmetric   : {int4_asymmetric}")
    print(f"   Act Asymmetric    : {act_asymmetric}")
    print("=" * 80, flush=True)

    config_path = source_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {source_dir}")

    group_size = extract_group_size_from_config(config_path)

    # 1. Load weights
    bin_path = source_dir / "pytorch_model.bin"
    st_path = source_dir / "model.safetensors"
    if bin_path.exists():
        print(f"[1/4] Loading TorchAO state_dict from {bin_path} ...", flush=True)
        start_time = time.time()
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    elif st_path.exists():
        print(f"[1/4] Loading TorchAO state_dict from {st_path} ...", flush=True)
        start_time = time.time()
        from safetensors import safe_open
        state_dict = {}
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    else:
        raise FileNotFoundError(f"No checkpoint weights found in {source_dir}")

    print(f"      Loaded {len(state_dict)} tensor keys in {time.time() - start_time:.2f}s", flush=True)

    # 2. Schema detection and conversion
    print("[2/4] Detecting quantization schemas and converting layers...", flush=True)
    target_state_dict = {}
    schema_counts = {s: 0 for s in SchemaType}
    processed_prefixes = set()

    for name, tensor in state_dict.items():
        prefix = name[:-len(".weight")] if name.endswith(".weight") else name
        if prefix in processed_prefixes:
            continue

        schema = detect_tensor_schema(name, tensor, state_dict)

        if schema == SchemaType.INT4_TINYGEMM:
            prefix, converted = convert_int4_tinygemm(
                name, tensor, state_dict, group_size, device, export_asymmetric=int4_asymmetric
            )
            target_state_dict.update(converted)
            processed_prefixes.add(prefix)
            schema_counts[SchemaType.INT4_TINYGEMM] += 1

        elif schema == SchemaType.INT4_PLAIN:
            prefix, converted = convert_int4_plain(name, tensor, state_dict, group_size, device)
            target_state_dict.update(converted)
            processed_prefixes.add(prefix)
            schema_counts[SchemaType.INT4_PLAIN] += 1

        elif schema == SchemaType.INT4_PRESHUFFLED:
            prefix, converted = convert_int4_preshuffled(name, tensor, state_dict, group_size, device)
            target_state_dict.update(converted)
            processed_prefixes.add(prefix)
            schema_counts[SchemaType.INT4_PRESHUFFLED] += 1

        elif schema == SchemaType.INT8_WEIGHT_ONLY:
            prefix, converted = convert_int8_weight_only(name, tensor, state_dict, device)
            target_state_dict.update(converted)
            processed_prefixes.add(prefix)
            schema_counts[SchemaType.INT8_WEIGHT_ONLY] += 1

        elif schema == SchemaType.INT8_DYNAMIC_ACT:
            prefix, converted = convert_int8_dynamic_act(name, tensor, state_dict, device)
            target_state_dict.update(converted)
            processed_prefixes.add(prefix)
            schema_counts[SchemaType.INT8_DYNAMIC_ACT] += 1

        elif schema == SchemaType.INT8_STATIC_ACT:
            prefix, converted = convert_int8_static_act(name, tensor, state_dict, device)
            target_state_dict.update(converted)
            processed_prefixes.add(prefix)
            schema_counts[SchemaType.INT8_STATIC_ACT] += 1

        elif schema == SchemaType.FLOAT8_WEIGHT_ONLY:
            prefix, converted = convert_fp8_weight_only(name, tensor, state_dict, device)
            target_state_dict.update(converted)
            processed_prefixes.add(prefix)
            schema_counts[SchemaType.FLOAT8_WEIGHT_ONLY] += 1

        elif schema == SchemaType.FLOAT8_DYNAMIC_ACT:
            prefix, converted = convert_fp8_dynamic_act(name, tensor, state_dict, device)
            target_state_dict.update(converted)
            processed_prefixes.add(prefix)
            schema_counts[SchemaType.FLOAT8_DYNAMIC_ACT] += 1

        elif schema == SchemaType.PASSTHROUGH:
            if not name.endswith(".scales_and_zeros"):
                # Clone tensor storage buffer to avoid safetensors shared-memory conflicts
                target_state_dict[name] = tensor.cpu().clone().contiguous()
                schema_counts[SchemaType.PASSTHROUGH] += 1

    print("      Conversion statistics across detected schemas:")
    for s, cnt in schema_counts.items():
        if cnt > 0:
            print(f"       - {s.value}: {cnt} layers")

    # Determine dominant schema
    quant_schemas = [s for s in SchemaType if s != SchemaType.PASSTHROUGH and schema_counts[s] > 0]
    dominant_schema = quant_schemas[0] if quant_schemas else SchemaType.PASSTHROUGH

    # 3. Generate spec-compliant configuration
    print("[3/4] Generating config.json and writing .safetensors ...", flush=True)
    base_config = json.loads((source_dir / "config.json").read_text())
    target_config = generate_compressed_tensors_config(
        dominant_schema,
        group_size,
        base_config,
        symmetric=(not int4_asymmetric),
        act_symmetric=(not act_asymmetric),
    )
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(target_config, f, indent=2)

    # 4. Save model.safetensors
    safetensors_output = output_dir / "model.safetensors"
    save_file(target_state_dict, safetensors_output)
    print(f"      Saved: {safetensors_output} ({safetensors_output.stat().st_size / (1024**2):.2f} MB)")

    # 5. Copy tokenizer and auxiliary files
    print("[4/4] Synchronizing tokenizer & metadata assets...", flush=True)
    for fname in os.listdir(source_dir):
        if fname not in ["pytorch_model.bin", "model.safetensors", "config.json"]:
            src_f = source_dir / fname
            if src_f.is_file():
                shutil.copy2(src_f, output_dir / fname)

    print("=" * 80)
    print(f"🎉 Conversion Complete! Spec-compliant checkpoint saved to: {output_dir}")
    print("=" * 80, flush=True)


def main():
    args = parse_args()
    convert_checkpoint(
        Path(args.source),
        Path(args.output_dir),
        args.device,
        int4_asymmetric=args.int4_asymmetric,
        act_asymmetric=args.act_asymmetric,
    )


if __name__ == "__main__":
    main()
