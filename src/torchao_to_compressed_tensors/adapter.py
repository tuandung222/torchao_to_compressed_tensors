"""
Checkpoint conversion: TorchAO / dense QAT checkpoints -> compressed-tensors
weight-only INT4 or INT8 (W4A16 / W8A16).
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .config import (
    build_model_config,
    build_w8a8_quantization_config,
    close_ignore_over_unquantised,
)
from .handlers import dequantize_tinygemm
from .qat import MARLIN_GROUP_SIZES, validate_group_size
from .quantize import (
    DEFAULT_NUM_BITS,
    SUPPORTED_NUM_BITS,
    quantize_weight,
    quantize_weight_int8,
)
from .schemas import (
    DEFAULT_IGNORE,
    ModuleKind,
    SourceFormat,
    classify_modules,
    detect_source_format,
    extract_group_size,
    iter_quantizable_weights,
)

__all__ = ["convert_checkpoint", "main"]


#: Roughly HF's default. Kept below the 2^31 byte safetensors-friendly range and
#: small enough that conversion never holds the whole model twice in memory.
DEFAULT_MAX_SHARD_BYTES = 5 * 1000**3

_AUXILIARY_SKIP = {"pytorch_model.bin", "model.safetensors", "config.json"}


def _load_state_dict(source_dir: Path) -> dict[str, Any]:
    """Load the source weights from either a .bin or a .safetensors checkpoint."""
    bin_path = source_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=False)

    shards = sorted(source_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No checkpoint weights found in {source_dir}")

    from safetensors import safe_open

    state_dict: dict[str, Any] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def _densify_tinygemm(
    state_dict: dict[str, Any], group_size: int, device: torch.device
) -> dict[str, Any]:
    """Replace every tinygemm INT4 tensor with the dense weight it encodes."""
    dense: dict[str, Any] = {}
    for name, tensor in state_dict.items():
        if name.endswith(".scales_and_zeros"):
            continue
        sidecar = state_dict.get(f"{name[: -len('.weight')]}.scales_and_zeros")
        if name.endswith(".weight") and sidecar is not None:
            dense[name] = dequantize_tinygemm(tensor, sidecar, group_size, device).cpu()
        else:
            dense[name] = tensor
    return dense


def _write_sharded(
    tensors: dict[str, torch.Tensor], output_dir: Path, max_shard_bytes: int
) -> None:
    """Write tensors as one or more safetensors shards plus an index.

    Sharding matters beyond convention: a single-file export holds the entire
    model in memory and fails outright past the safetensors size limit.
    """
    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    for name in sorted(tensors):
        tensor = tensors[name]
        nbytes = tensor.numel() * tensor.element_size()
        if sizes[-1] and sizes[-1] + nbytes > max_shard_bytes:
            shards.append({})
            sizes.append(0)
        shards[-1][name] = tensor
        sizes[-1] += nbytes

    # transformers requires this metadata; without it the checkpoint is rejected.
    metadata = {"format": "pt"}

    if len(shards) == 1:
        save_file(shards[0], output_dir / "model.safetensors", metadata=metadata)
        return

    total = len(shards)
    weight_map = {}
    for index, shard in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{total:05d}.safetensors"
        save_file(shard, output_dir / filename, metadata=metadata)
        for name in shard:
            weight_map[name] = filename

    index_path = output_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map},
            indent=2,
        ),
        encoding="utf-8",
    )


def convert_checkpoint(
    source_dir: Path,
    output_dir: Path,
    group_size: int | None = None,
    symmetric: bool = True,
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
    quantize_embeddings: bool = False,
    device_str: str = "cpu",
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    num_bits: int = DEFAULT_NUM_BITS,
    scheme: str = "weight-only",
) -> None:
    """Convert a source checkpoint into a compressed-tensors checkpoint.

    Two output formats, selected by ``scheme``:

    ``"weight-only"``
        ``pack-quantized``, per-group, ``num_bits`` wide -- W4A16 or W8A16.
    ``"w8a8"``
        ``int-quantized``, per-channel int8 weights plus a dynamic per-token
        int8 activation scheme. ``group_size``, ``num_bits`` and
        ``quantize_embeddings`` do not apply and are rejected rather than
        silently ignored: a grouped W8A8 config is served as plain weight-only
        with the activations left in bf16, which is the failure worth catching.

    :param source_dir: directory holding the source checkpoint
    :param output_dir: directory to write the converted checkpoint to
    :param group_size: elements per group; read from the source config if omitted
    :param symmetric: export without a zero-point. Matches a symmetric QAT run
        exactly and lets Marlin use its faster bias-encoded path (``uint4b8`` at
        4 bits, ``uint8b128`` at 8)
    :param num_bits: weight width, 4 or 8. Both produce the ``pack-quantized``
        format and the same vLLM scheme; 8 must be symmetric
    :param ignore: module-name fragments to leave dense. Must match what QAT used
    :param quantize_embeddings: also quantise ``nn.Embedding`` weights
    :param device_str: device to run the tensor transforms on
    :param max_shard_bytes: shard size threshold for the output
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    config_path = source_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {source_dir}")
    base_config = json.loads(config_path.read_text(encoding="utf-8"))

    w8a8 = scheme == "w8a8"
    if w8a8:
        if not symmetric:
            raise ValueError("W8A8 weights must be symmetric; vLLM accepts no other")
        if quantize_embeddings:
            raise ValueError(
                "W8A8 says nothing about embeddings -- a lookup has no matmul to feed "
                "int8 activations into. Quantise the table with --scheme weight-only."
            )
        if group_size is not None:
            raise ValueError(
                "W8A8 is per-channel, not per-group. vLLM's W8A8 predicates accept "
                "only TENSOR or CHANNEL, so a grouped checkpoint falls through to "
                "weight-only WNA16 and is served with bf16 activations."
            )
    elif scheme != "weight-only":
        raise ValueError(f"unknown scheme {scheme!r}")
    else:
        group_size = validate_group_size(extract_group_size(config_path, group_size))

    print(f"[1/4] Loading state dict from {source_dir} ...", flush=True)
    started = time.time()
    state_dict = _load_state_dict(source_dir)
    source_format = detect_source_format(state_dict)
    print(
        f"      {len(state_dict)} tensors in {time.time() - started:.1f}s "
        f"| source format: {source_format.value}",
        flush=True,
    )

    if source_format is SourceFormat.INT4_TINYGEMM:
        print("[2/4] Dequantising tinygemm INT4 to dense ...", flush=True)
        print(
            "      note: tinygemm uses a float zero-point, compressed-tensors an "
            "integer one, so this re-quantisation is not lossless. A symmetric "
            "QAT run exported from dense weights is.",
            flush=True,
        )
        state_dict = _densify_tinygemm(state_dict, group_size, device)
    else:
        print("[2/4] Source is already dense, no decoding needed.", flush=True)

    if w8a8:
        print(
            f"[3/4] Quantising to W8A8 int8 (per-channel weights, dynamic "
            f"per-token activations, ignore={list(ignore)}) ...",
            flush=True,
        )
    else:
        print(
            f"[3/4] Quantising to W{num_bits}A16 (group_size={group_size}, "
            f"symmetric={symmetric}, ignore={list(ignore)}, "
            f"embeddings={quantize_embeddings}) ...",
            flush=True,
        )
    module_kinds = classify_modules(config_path)
    if not module_kinds:
        print(
            "      warning: could not introspect the architecture; treating every "
            "2D float weight as Linear",
            flush=True,
        )

    output: dict[str, torch.Tensor] = {}
    converted: set[str] = set()
    counts = {ModuleKind.LINEAR: 0, ModuleKind.EMBEDDING: 0}

    for prefix, weight, kind in iter_quantizable_weights(
        state_dict, module_kinds, ignore, quantize_embeddings
    ):
        if w8a8:
            int8 = quantize_weight_int8(weight, device)
            output[f"{prefix}.weight"] = int8.weight
            output[f"{prefix}.weight_scale"] = int8.weight_scale
        else:
            quantized = quantize_weight(
                weight, group_size, symmetric, device, num_bits=num_bits
            )
            output[f"{prefix}.weight_packed"] = quantized.weight_packed
            output[f"{prefix}.weight_scale"] = quantized.weight_scale
            output[f"{prefix}.weight_shape"] = quantized.weight_shape
            if quantized.weight_zero_point is not None:
                output[f"{prefix}.weight_zero_point"] = quantized.weight_zero_point
        converted.add(f"{prefix}.weight")
        counts[kind] += 1

    if not converted:
        raise ValueError(
            "No weights were quantised. Check --ignore and that the source "
            "checkpoint actually holds Linear weights."
        )
    print(
        f"      {counts[ModuleKind.LINEAR]} Linear, "
        f"{counts[ModuleKind.EMBEDDING]} Embedding",
        flush=True,
    )

    lm_head_is_tied = bool(base_config.get("tie_word_embeddings", False))
    if lm_head_is_tied and quantize_embeddings and "lm_head" not in ignore:
        print(
            "      warning: tie_word_embeddings is set, so lm_head shares the "
            "quantised embedding table. vLLM routes lm_head through the Linear "
            "scheme and the embedding through the Embedding scheme; with one "
            "shared tensor that pairing is ambiguous. Add lm_head to --ignore, "
            "or untie the model, unless you have verified this loads.",
            flush=True,
        )

    for name, tensor in state_dict.items():
        if name in converted:
            continue
        if lm_head_is_tied and name == "lm_head.weight":
            # Tied to the embedding; safetensors cannot store shared storage and
            # transformers re-ties on load.
            continue
        output[name] = tensor.detach().cpu().clone().contiguous()

    print("[4/4] Writing checkpoint ...", flush=True)
    quantised_prefixes = {name[: -len(".weight")] for name in converted}
    linear_modules = {
        name for name, kind in module_kinds.items() if kind is ModuleKind.LINEAR
    }
    effective_ignore = close_ignore_over_unquantised(
        ignore, quantised_prefixes, linear_modules
    )
    if len(effective_ignore) > len(ignore):
        added = list(effective_ignore[len(ignore) :])
        print(
            f"      adding {len(added)} unquantised Linear module(s) to ignore "
            f"(e.g. {added[:3]}) -- the config must not promise tensors the "
            "checkpoint does not contain",
            flush=True,
        )

    # vLLM matches a class target by substring against its own module class
    # names, and lm_head is a ParallelLMHead -- no "Linear" in it, no "Embedding"
    # either. Unless the config names lm_head literally, vLLM builds it dense and
    # the load fails on the missing lm_head.weight. See build_quantization_config.
    lm_head_targets = tuple(
        name
        for name in sorted(quantised_prefixes)
        if name == "lm_head" or name.endswith(".lm_head")
    )

    if w8a8:
        target_config = dict(base_config)
        target_config.pop("quantization_config", None)
        target_config["quantization_config"] = build_w8a8_quantization_config(
            effective_ignore, lm_head_targets
        )
        target_config["tie_word_embeddings"] = lm_head_is_tied
    else:
        target_config = build_model_config(
            base_config,
            group_size,
            symmetric,
            lm_head_is_tied,
            effective_ignore,
            quantize_embeddings,
            lm_head_targets,
            num_bits,
        )
    (output_dir / "config.json").write_text(
        json.dumps(target_config, indent=2), encoding="utf-8"
    )
    _write_sharded(output, output_dir, max_shard_bytes)

    for filename in os.listdir(source_dir):
        if filename in _AUXILIARY_SKIP or filename.endswith(".safetensors"):
            continue
        candidate = source_dir / filename
        if candidate.is_file():
            shutil.copy2(candidate, output_dir / filename)

    print(f"Done. compressed-tensors checkpoint written to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a TorchAO or dense QAT checkpoint to compressed-tensors "
        "weight-only INT4/INT8 (W4A16 / W8A16)"
    )
    parser.add_argument("--source", required=True, help="source checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        choices=MARLIN_GROUP_SIZES,
        help="elements per quantisation group; read from the source config if "
        "omitted. Not valid with --scheme w8a8, which is per-channel",
    )
    parser.add_argument(
        "--scheme",
        default="weight-only",
        choices=("weight-only", "w8a8"),
        help="weight-only produces pack-quantized W4A16/W8A16 for Marlin; w8a8 "
        "produces int-quantized per-channel int8 weights with dynamic per-token "
        "int8 activations, for the CUTLASS int8 GEMM",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=DEFAULT_NUM_BITS,
        choices=SUPPORTED_NUM_BITS,
        help="weight width. 8 halves the compression against 4 but keeps the same "
        "format, scheme and Marlin kernel, and must stay symmetric",
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="export with a zero-point. Only correct if QAT was also asymmetric; "
        "symmetric is the supported default, and the only option at 8 bits",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=list(DEFAULT_IGNORE),
        metavar="NAME",
        help="module-name fragments to leave dense (default: lm_head). Pass with "
        "no values to quantise everything, including lm_head. Must match QAT",
    )
    parser.add_argument(
        "--quantize-embeddings",
        action="store_true",
        help="also quantise nn.Embedding weights (vLLM serves these via "
        "CompressedTensorsEmbeddingWNA16Int)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="device for the tensor transforms",
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=DEFAULT_MAX_SHARD_BYTES,
        help="output shard size threshold in bytes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(
        Path(args.source),
        Path(args.output_dir),
        group_size=args.group_size,
        symmetric=not args.asymmetric,
        ignore=tuple(args.ignore),
        quantize_embeddings=args.quantize_embeddings,
        num_bits=args.num_bits,
        scheme=args.scheme,
        device_str=args.device,
        max_shard_bytes=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
