#!/usr/bin/env python3
"""
Numerical acceptance check: does the exported checkpoint hold the weights the
source implies?

Structural checks cannot answer this. A conversion bug that reorders elements
produces a checkpoint with correct shapes, correct dtypes, a correct value
distribution and a valid config -- it loads cleanly in both transformers and
vLLM and generates fluent-looking tokens. The only thing that catches it is
comparing, layer by layer, against the source.

Cosine similarity is the right detector precisely because it is sensitive to
permutation: a scrambled layer scores ~0.70 where a correct one scores >0.99.

Exit code is 0 on pass, 1 on fail, so this can gate a pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import calculate_qparams
from safetensors import safe_open

try:
    from compressed_tensors.compressors.pack_quantized import unpack_from_int32
except ImportError:  # pragma: no cover
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        unpack_from_int32,
    )

from torchao_to_compressed_tensors import (
    SourceFormat,
    dequantize_tinygemm,
    detect_source_format,
    make_quantization_args,
)

#: A dense source is re-quantised exactly, so anything below this is a defect.
#: A tinygemm source is re-quantised through an integer zero-point it was not
#: built for, which costs roughly 1e-2 of cosine -- hence the looser bar.
THRESHOLDS = {SourceFormat.DENSE: 1.0 - 1e-6, SourceFormat.INT4_TINYGEMM: 0.97}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="minimum per-layer cosine; defaults to a per-source-format value",
    )
    parser.add_argument("--report-worst", type=int, default=5)
    return parser.parse_args()


def load_source(source_dir: Path) -> dict[str, torch.Tensor]:
    bin_path = source_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=False)
    tensors = {}
    for shard in sorted(source_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def load_converted(model_dir: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def expected_weight(
    dense: torch.Tensor,
    group_size: int,
    symmetric: bool,
    scale_dtype: torch.dtype | None = None,
    num_bits: int = 4,
) -> torch.Tensor:
    """What the exported layer should decompress to, per compressed-tensors.

    Uses ``quantize`` then ``dequantize`` rather than ``fake_quantize``. The two
    are not interchangeable: on CUDA the fused QDQ path breaks rounding ties
    differently from quantise-then-cast-to-int8, disagreeing by one level on
    roughly 0.1% of elements (they agree exactly on CPU). ``quantize`` is what
    ``PackedQuantizationCompressor.compress`` actually calls, so it is the
    authoritative reference; ``fake_quantize`` is the calibration-time
    simulation.
    """
    args = make_quantization_args(group_size, symmetric, num_bits)
    grouped = dense.view(dense.shape[0], -1, group_size)
    scale, zero_point = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
    if scale_dtype is not None:
        scale = scale.to(scale_dtype)
    quantized = quantize(
        x=dense, scale=scale, zero_point=zero_point, args=args, dtype=torch.int8
    )
    return dequantize(
        x_q=quantized,
        scale=scale,
        zero_point=None if symmetric else zero_point,
        args=args,
    )


def expected_int8_weight(dense: torch.Tensor) -> torch.Tensor:
    """What a W8A8 layer should decompress to: per-channel symmetric int8."""
    from torchao_to_compressed_tensors.quantize import make_int8_quantization_args

    args = make_int8_quantization_args()
    grouped = dense.view(dense.shape[0], 1, -1)
    scale, zero_point = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
    quantized = quantize(
        x=dense, scale=scale, zero_point=zero_point, args=args, dtype=torch.int8
    )
    return dequantize(x_q=quantized, scale=scale, zero_point=None, args=args)


def actual_int8_weight(
    tensors: dict[str, torch.Tensor], prefix: str, device: torch.device
) -> torch.Tensor:
    """Decompress a W8A8 layer: a multiply, with nothing to unpack."""
    from torchao_to_compressed_tensors.quantize import make_int8_quantization_args

    args = make_int8_quantization_args()
    return dequantize(
        x_q=tensors[f"{prefix}.weight"].to(device),
        scale=tensors[f"{prefix}.weight_scale"].to(device),
        zero_point=None,
        args=args,
    )


def group_size_of(tensors: dict[str, torch.Tensor], prefix: str) -> int:
    """Recover the group size from the stored shape and scale."""
    in_features = int(tensors[f"{prefix}.weight_shape"][1])
    return in_features // tensors[f"{prefix}.weight_scale"].shape[-1]


def actual_weight(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    symmetric: bool,
    device: torch.device,
    num_bits: int = 4,
) -> torch.Tensor:
    """Decompress a layer along the same path vLLM's loader takes."""
    args = make_quantization_args(group_size_of(tensors, prefix), symmetric, num_bits)
    shape = torch.Size(tensors[f"{prefix}.weight_shape"].tolist())
    scale = tensors[f"{prefix}.weight_scale"].to(device)

    zero_point = None
    if not symmetric:
        zero_point = unpack_from_int32(
            tensors[f"{prefix}.weight_zero_point"].to(device),
            num_bits,
            (shape[0], scale.shape[-1]),
            packed_dim=0,
        )
    unpacked = unpack_from_int32(
        tensors[f"{prefix}.weight_packed"].to(device), num_bits, shape
    )
    return dequantize(x_q=unpacked, scale=scale, zero_point=zero_point, args=args)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    config = json.loads((args.converted / "config.json").read_text(encoding="utf-8"))
    parsed = QuantizationConfig.model_validate(config["quantization_config"])
    weights = parsed.config_groups["group_0"].weights
    group_size, symmetric = weights.group_size, weights.symmetric
    # Taken from the checkpoint, not a flag. The config already records the
    # width, and a second source of truth is a second thing to get wrong.
    num_bits = weights.num_bits

    source = load_source(args.source)
    source_format = detect_source_format(source)
    threshold = args.threshold
    if threshold is None:
        threshold = THRESHOLDS[source_format]

    print(
        f"source={source_format.value} group_size={group_size} "
        f"symmetric={symmetric} num_bits={num_bits} threshold={threshold:.6f}",
        flush=True,
    )

    converted = load_converted(args.converted)
    # int-quantized (W8A8) stores the weight unpacked, so there is no
    # .weight_packed to key off. Checking the format rather than guessing from the
    # tensor names keeps a W8A8 checkpoint from reporting "no quantised layers".
    int_quantized = config["quantization_config"].get("format") == "int-quantized"
    suffix = ".weight_scale" if int_quantized else ".weight_packed"
    prefixes = sorted(
        k[: -len(suffix)] for k in converted if k.endswith(suffix)
    )
    if not prefixes:
        print("FAIL: no quantised layers in the converted checkpoint")
        return 1

    results: list[tuple[float, str]] = []
    for prefix in prefixes:
        raw = source.get(f"{prefix}.weight")
        if raw is None:
            print(f"FAIL: {prefix} is quantised but absent from the source")
            return 1

        if source_format is SourceFormat.INT4_TINYGEMM:
            sidecar = source.get(f"{prefix}.scales_and_zeros")
            if sidecar is None:
                print(f"FAIL: {prefix} has no tinygemm sidecar in the source")
                return 1
            dense = dequantize_tinygemm(raw, sidecar, group_size, device)
        else:
            dense = raw.to(device=device, dtype=torch.float32).contiguous()

        stored_scale_dtype = converted[f"{prefix}.weight_scale"].dtype
        if int_quantized:
            expected = expected_int8_weight(dense)
            actual = actual_int8_weight(converted, prefix, device)
        else:
            expected = expected_weight(
                dense, group_size, symmetric, stored_scale_dtype, num_bits
            )
            actual = actual_weight(converted, prefix, symmetric, device, num_bits)
        cosine = torch.nn.functional.cosine_similarity(
            expected.flatten().float(), actual.flatten().float(), dim=0
        ).item()
        results.append((cosine, prefix))

    cosines = torch.tensor([c for c, _ in results])
    failures = [(c, name) for c, name in results if c < threshold]

    print(
        f"{len(results)} layers | cosine min={cosines.min():.6f} "
        f"p05={cosines.quantile(0.05):.6f} median={cosines.median():.6f}",
        flush=True,
    )
    for cosine, name in sorted(results)[: args.report_worst]:
        print(f"   {cosine:.6f}  {name}")

    if failures:
        print(f"\nFAIL: {len(failures)} layer(s) below {threshold:.6f}")
        print(
            "A layer whose shapes and dtypes are right but whose cosine is far "
            "below 1 usually means the packing reordered elements, not that "
            "quantisation lost precision."
        )
        return 1

    print(f"\nPASS: every layer at or above {threshold:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
