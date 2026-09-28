"""
Source-specific decoding.

Every supported source is reduced to a dense weight here; :mod:`quantize` then
picks the export grid exactly once. Keeping decode and quantise apart is what
stops the two from disagreeing -- the original adapter fused them and ended up
re-using TorchAO's asymmetric scale on a symmetric grid.
"""

import torch

__all__ = ["dequantize_tinygemm"]


def _int4pack_mm(
    activations: torch.Tensor,
    packed_weight: torch.Tensor,
    group_size: int,
    scales_and_zeros: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to the tinygemm kernel matching the weight's packing layout.

    The two kernels take incompatible layouts, so the choice is driven by the
    tensor rather than by the device: the CUDA kernel wants the tile-packed 4D
    int32 layout, the CPU one a 2D uint8 layout. Dispatching on device instead
    silently hands a 4D int32 tensor to the CPU kernel, which rejects it.
    """
    is_cpu_layout = packed_weight.dtype == torch.uint8 and packed_weight.ndim == 2
    if is_cpu_layout:
        cpu_op = getattr(torch.ops.aten, "_weight_int4pack_mm_for_cpu", None)
        if cpu_op is None:
            raise RuntimeError(
                "This checkpoint uses the CPU tinygemm layout but this build of "
                "torch has no _weight_int4pack_mm_for_cpu."
            )
        return cpu_op(activations, packed_weight, group_size, scales_and_zeros)

    if packed_weight.device.type != "cuda":
        raise RuntimeError(
            "This checkpoint uses TorchAO's tile-packed 4D INT4 layout "
            f"(shape {tuple(packed_weight.shape)}, {packed_weight.dtype}), which "
            "only the CUDA tinygemm kernel can decode. Re-run the conversion with "
            "--device cuda:0.\n"
            "Dense checkpoints -- the output of QATConfig(step='convert') -- "
            "convert on CPU and are the preferred input; see docs/SCOPE.md."
        )

    return torch.ops.aten._weight_int4pack_mm(
        activations, packed_weight, group_size, scales_and_zeros
    )


def dequantize_tinygemm(
    packed_weight: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Recover the dense weight from a TorchAO tinygemm INT4 tensor.

    The tile-packed 4D layout is awkward to unpack directly, so the weight is
    recovered by multiplying it with an identity matrix through the same kernel
    that serves it. That is layout-agnostic and therefore survives TorchAO
    changing its packing, which string-matching on tensor class names did not.

    Note that tinygemm is asymmetric with a *float* zero-point, while
    compressed-tensors requires an integer one. The re-quantisation in
    :func:`quantize_weight` is therefore not lossless for this source; a
    symmetric QAT run exported straight from dense weights is. See docs/SCOPE.md.

    :param packed_weight: the tile-packed INT4 tensor
    :param scales_and_zeros: the tinygemm sidecar, ``[num_groups, out_features, 2]``
    :param group_size: elements per quantisation group
    :param device: device to run the dequantisation on
    :return: dense ``[out_features, in_features]`` float32 weight on ``device``
    """
    num_groups, out_features = scales_and_zeros.shape[0], scales_and_zeros.shape[1]
    in_features = num_groups * group_size

    eye = torch.eye(in_features, dtype=torch.bfloat16, device=device)
    dense = _int4pack_mm(
        eye,
        packed_weight.to(device),
        group_size,
        scales_and_zeros.to(device),
    ).t()

    if tuple(dense.shape) != (out_features, in_features):
        raise ValueError(
            f"tinygemm dequantisation produced {tuple(dense.shape)}, expected "
            f"{(out_features, in_features)}; group_size={group_size} is likely wrong"
        )
    # .t() above leaves a transposed view; downstream packing reshapes the
    # tensor, so hand back real row-major memory rather than a stride trick.
    return dense.to(torch.float32).contiguous()
