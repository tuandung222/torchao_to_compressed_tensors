"""
Weight-only INT4/INT8 quantisation, delegated to compressed-tensors itself.

The scale formula, the zero-point convention, the clamp range and the epsilon
handling are all owned by compressed-tensors. Reimplementing any of them here
would create a second source of truth that drifts silently -- the original
adapter did exactly that and ended up exporting a grid 1.45x worse than the one
TorchAO had trained. So this module computes per-group statistics and hands
everything else to ``calculate_qparams`` and ``quantize``.

Two export shapes live here, because the runtime wants two different things:

``quantize_weight``
    Per-group, 4 or 8 bits, packed into int32 -- the ``pack-quantized`` format
    behind W4A16 and W8A16, served by Marlin.
``quantize_weight_int8``
    Per-channel int8, stored unpacked -- the ``int-quantized`` format behind the
    weight half of W8A8, served by a CUTLASS int8 GEMM.

They are kept apart rather than unified behind a flag: nothing is shared past the
call to ``calculate_qparams``, and vLLM reads them through different schemes with
different parameter names. See docs/SCOPE.md.
"""

from typing import NamedTuple

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.utils import calculate_qparams

try:  # compressed-tensors >= 0.17 moved the packing helpers
    from compressed_tensors.compressors.pack_quantized import pack_to_int32
except ImportError:  # pragma: no cover - older layout
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        pack_to_int32,
    )


__all__ = [
    "DEFAULT_NUM_BITS",
    "NUM_BITS",
    "SUPPORTED_NUM_BITS",
    "Int8QuantizedWeight",
    "QuantizedWeight",
    "make_int8_quantization_args",
    "make_quantization_args",
    "quantize_weight",
    "quantize_weight_int8",
    "validate_num_bits",
]


#: Weight widths Marlin serves: ``uint4b8`` and ``uint8b128``.
SUPPORTED_NUM_BITS = (4, 8)

DEFAULT_NUM_BITS = 4

#: Retained so existing importers keep working. New code takes ``num_bits`` as an
#: argument instead -- a module-level constant cannot express a per-call width.
NUM_BITS = DEFAULT_NUM_BITS


def validate_num_bits(num_bits: int, symmetric: bool) -> int:
    """Reject widths and sign conventions Marlin cannot serve, at config time.

    Eight-bit must be symmetric. vLLM's
    ``query_marlin_supported_quant_types(has_zp=True)`` returns ``[uint4]`` alone,
    so an asymmetric 8-bit checkpoint loads happily and then runs on a fallback
    kernel -- a silent performance loss rather than an error, which is exactly the
    kind of thing worth failing loudly at export time.
    """
    if num_bits not in SUPPORTED_NUM_BITS:
        raise ValueError(
            f"num_bits={num_bits} is not servable by Marlin. "
            f"Supported: {list(SUPPORTED_NUM_BITS)}."
        )
    if num_bits == 8 and not symmetric:
        raise ValueError(
            "asymmetric 8-bit is not servable by Marlin: it offers a zero-point "
            "path for 4 bits only (uint4). Use symmetric, or 4 bits."
        )
    return num_bits


class QuantizedWeight(NamedTuple):
    """The tensors a pack-quantized checkpoint stores for one Linear layer."""

    weight_packed: torch.Tensor  # int32, packed along in_features
    weight_scale: torch.Tensor
    weight_shape: torch.Tensor  # int32 [out_features, in_features]
    weight_zero_point: torch.Tensor | None  # int32, packed along out_features


def make_quantization_args(
    group_size: int, symmetric: bool, num_bits: int = DEFAULT_NUM_BITS
) -> QuantizationArgs:
    """Build the ``QuantizationArgs`` the served config will declare."""
    return QuantizationArgs(
        num_bits=num_bits,
        type=QuantizationType.INT,
        symmetric=symmetric,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
    )


def quantize_weight(
    weight: torch.Tensor,
    group_size: int,
    symmetric: bool = True,
    device: torch.device | None = None,
    scale_dtype: torch.dtype | None = None,
    num_bits: int = DEFAULT_NUM_BITS,
) -> QuantizedWeight:
    """Quantise one dense weight into compressed-tensors pack-quantized tensors.

    Both 4 and 8 bits use the same ``pack-quantized`` format and the same vLLM
    scheme (``CompressedTensorsWNA16``); only the pack factor differs, 8 values
    per int32 word against 4.

    :param weight: dense ``[out_features, in_features]`` weight, any float dtype
    :param group_size: elements per group; must divide ``in_features``
    :param symmetric: symmetric export (no zero-point) or asymmetric
    :param num_bits: 4 or 8; 8 must be symmetric (see :func:`validate_num_bits`)
    :param device: device to run the transform on; defaults to the weight's own
    :param scale_dtype: dtype to store ``weight_scale`` in; defaults to the
        weight's own. This must be the model's dtype: decompression returns the
        weight in the scale's dtype, so a float32 scale on a bfloat16 model
        yields a float32 weight and the first matmul dies with
        ``expected mat1 and mat2 to have the same dtype``. vLLM casts params on
        load and so hides this; transformers does not.
    :return: the packed tensors, all on CPU and contiguous, ready for safetensors
    :raises ValueError: if ``in_features`` is not divisible by ``group_size``
    """
    validate_num_bits(num_bits, symmetric)
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")

    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )

    device = device or weight.device
    # float32 throughout: bf16 has too little mantissa for a faithful w/scale.
    # .contiguous() is load-bearing, not hygiene: pack_to_int32 reshapes the
    # quantised tensor, and on a transposed view (as produced by dequantising a
    # tinygemm checkpoint) that silently packs elements in the wrong order. The
    # result decompresses to plausible-looking but scrambled weights.
    w = weight.to(device=device, dtype=torch.float32).contiguous()
    args = make_quantization_args(group_size, symmetric, num_bits)

    grouped = w.view(out_features, -1, group_size)
    scale, zero_point = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)

    # Round the scale to its storage dtype *before* quantising, so the integers
    # written are the ones that reproduce the weight against the scale that is
    # also written. Quantising against a float32 scale and then storing a
    # bfloat16 one would leave the two inconsistent.
    scale = scale.to(scale_dtype if scale_dtype is not None else weight.dtype)

    q = quantize(x=w, scale=scale, zero_point=zero_point, args=args, dtype=torch.int8)

    packed = pack_to_int32(q, num_bits, packed_dim=1)
    shape = torch.tensor([out_features, in_features], dtype=torch.int32)

    packed_zp = None
    if not symmetric:
        # compressed-tensors packs the zero-point along out_features, not
        # in_features (PackedQuantizationCompressor.compress), and vLLM declares
        # it as an int32 parameter of shape [ceil(out/8), n_groups].
        packed_zp = pack_to_int32(
            zero_point.to(torch.int8), num_bits, packed_dim=0
        ).cpu().contiguous()

    return QuantizedWeight(
        weight_packed=packed.cpu().contiguous(),
        weight_scale=scale.cpu().contiguous(),
        weight_shape=shape.contiguous(),
        weight_zero_point=packed_zp,
    )


# ---------------------------------------------------------------------------
# W8A8: int-quantized, per-channel, unpacked
# ---------------------------------------------------------------------------


class Int8QuantizedWeight(NamedTuple):
    """The tensors an ``int-quantized`` checkpoint stores for one Linear layer.

    Note what is *not* here. There is no ``weight_shape``: the weight is stored at
    its true shape rather than packed, so nothing needs recording to recover it.
    And there is no ``weight_zero_point``, because vLLM's ``W8A8Int8`` accepts
    symmetric weights only -- ``_is_dynamic_token_w8a8`` and
    ``_is_static_tensor_w8a8`` both test ``weight_quant.symmetric``.
    """

    weight: torch.Tensor  # int8, [out_features, in_features]
    weight_scale: torch.Tensor  # float32, [out_features, 1]


def make_int8_quantization_args() -> QuantizationArgs:
    """Weight args for W8A8: 8-bit, symmetric, per-channel.

    Per-channel rather than per-group is not a tuning choice. vLLM rejects a
    grouped weight strategy for W8A8 outright -- both predicates accept only
    ``TENSOR`` or ``CHANNEL`` -- so a grouped checkpoint would fall through to the
    weight-only WNA16 path and be served without activation quantisation at all.
    """
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        symmetric=True,
        strategy=QuantizationStrategy.CHANNEL,
    )


def quantize_weight_int8(
    weight: torch.Tensor,
    device: torch.device | None = None,
) -> Int8QuantizedWeight:
    """Quantise one dense weight into compressed-tensors ``int-quantized`` tensors.

    The scale stays float32 regardless of the model dtype, unlike the packed path.
    vLLM declares it that way (``ChannelQuantScaleParameter(..., dtype=torch.float32)``
    in ``CompressedTensorsW8A8Int8.create_weights``), and the int8 GEMM consumes it
    as a float32 dequantisation factor rather than folding it into a bf16 matmul,
    so there is no dtype to match here.

    :param weight: dense ``[out_features, in_features]`` weight, any float dtype
    :param device: device to run the transform on; defaults to the weight's own
    :return: the int8 weight and its per-channel scale, on CPU and contiguous
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")

    device = device or weight.device
    # .contiguous() for the same reason as the packed path: a transposed view
    # reshapes to something plausible and wrong.
    w = weight.to(device=device, dtype=torch.float32).contiguous()
    args = make_int8_quantization_args()

    # Per-channel is per-group with a single group spanning the row, which is how
    # calculate_qparams reduces it.
    grouped = w.view(w.shape[0], 1, -1)
    scale, zero_point = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)

    q = quantize(x=w, scale=scale, zero_point=zero_point, args=args, dtype=torch.int8)

    return Int8QuantizedWeight(
        weight=q.cpu().contiguous(),
        weight_scale=scale.to(torch.float32).cpu().contiguous(),
    )
