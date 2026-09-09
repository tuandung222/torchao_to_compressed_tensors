"""
Modular tensor transformation handlers for TorchAO quantization formats.
Converts tensors to compressed-tensors compatible structures (Marlin, int-quantized, float-quantized).
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

import torchao
from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros

try:
    from compressed_tensors.compressors.pack_quantized import pack_to_int32, unpack_from_int32
except ImportError:
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import pack_to_int32, unpack_from_int32

from .schemas import extract_prefix


def convert_int4_tinygemm(
    name: str,
    tensor: Any,
    state_dict: Dict[str, Any],
    group_size: int,
    device: torch.device,
    export_asymmetric: bool = False,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Convert TorchAO Tinygemm / TilePackedTo4D tensor into compressed-tensors pack-quantized format."""
    prefix = extract_prefix(name)
    sz_key = f"{prefix}.scales_and_zeros" if prefix else "scales_and_zeros"

    if sz_key in state_dict:
        w = tensor
        sz = state_dict[sz_key]
        out_features = sz.shape[1]
        num_groups = sz.shape[0]
        in_features = num_groups * group_size

        scales, zeros = unpack_tinygemm_scales_and_zeros(sz)
        valid_scales = scales.squeeze(-1)[:, :num_groups].to(torch.float16)

        # Dequantize using hardware aten._weight_int4pack_mm
        eye = torch.eye(in_features, dtype=torch.bfloat16, device=device)
        dequant_w = torch.ops.aten._weight_int4pack_mm(eye, w.to(device), group_size, sz.to(device)).t()
    else:
        # Instance of Int4TilePackedTo4dTensor
        out_features, in_features = tensor.shape
        num_groups = in_features // group_size
        scales, zeros = unpack_tinygemm_scales_and_zeros(tensor.scale_and_zero)
        valid_scales = scales.squeeze(-1)[:, :num_groups].to(torch.float16)

        tensor_gpu = tensor.to(device)
        eye = torch.eye(in_features, dtype=torch.bfloat16, device=device)
        dequant_w = F.linear(eye, tensor_gpu).t()

    scales_exp = valid_scales.to(device).repeat_interleave(group_size, dim=1)

    if export_asymmetric:
        valid_zeros = zeros.squeeze(-1)[:, :num_groups].to(device)
        z_uint = torch.round(8.0 - valid_zeros / valid_scales.to(device)).clamp(0, 15).to(torch.int8)
        z_uint_exp = z_uint.repeat_interleave(group_size, dim=1)
        q_unsigned = torch.round((dequant_w / scales_exp) + z_uint_exp.float()).clamp(0, 15).to(torch.int8)
        q_signed = (q_unsigned - 8).to(torch.int8)
        weight_packed = pack_to_int32(q_signed.to(device), num_bits=4, packed_dim=1).cpu()
        weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)
        k = lambda s: f"{prefix}.{s}" if prefix else s
        return prefix, {
            k("weight_packed"): weight_packed.contiguous(),
            k("weight_scale"): valid_scales.cpu().contiguous(),
            k("weight_shape"): weight_shape.contiguous(),
            k("weight_zero_point"): z_uint.cpu().contiguous(),
        }

    # Exact symmetric integer grid [-8, 7]
    q_signed = torch.round(dequant_w / scales_exp).clamp(-8, 7).to(torch.int8)

    weight_packed = pack_to_int32(q_signed.to(device), num_bits=4, packed_dim=1).cpu()
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)

    k = lambda s: f"{prefix}.{s}" if prefix else s
    return prefix, {
        k("weight_packed"): weight_packed.contiguous(),
        k("weight_scale"): valid_scales.cpu().contiguous(),
        k("weight_shape"): weight_shape.contiguous(),
    }


def convert_int4_plain(
    name: str,
    tensor: Any,
    state_dict: Dict[str, Any],
    group_size: int,
    device: torch.device,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Convert Int4PlainInt32Tensor into compressed-tensors pack-quantized format."""
    prefix = extract_prefix(name)
    if hasattr(tensor, "qdata") and hasattr(tensor, "scale"):
        qdata = tensor.qdata.cpu()
        scale = tensor.scale.cpu().to(torch.float16)
        orig_shape = tensor.shape
    else:
        qdata = tensor.cpu()
        scale_key = f"{prefix}.scale" if prefix else "scale"
        scale = state_dict[scale_key].cpu().to(torch.float16)
        orig_shape = (scale.shape[0], scale.shape[1] * group_size)

    # Convert plain uint4 to signed int8 [-8, 7]
    q_signed = (qdata.to(torch.int8) - 8).to(device)
    weight_packed = pack_to_int32(q_signed, num_bits=4, packed_dim=1).cpu()
    weight_shape = torch.tensor(list(orig_shape), dtype=torch.int32)

    k = lambda s: f"{prefix}.{s}" if prefix else s
    return prefix, {
        k("weight_packed"): weight_packed.contiguous(),
        k("weight_scale"): scale.contiguous(),
        k("weight_shape"): weight_shape.contiguous(),
    }


def convert_int4_preshuffled(
    name: str,
    tensor: Any,
    state_dict: Dict[str, Any],
    group_size: int,
    device: torch.device,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Convert Int4PreshuffledTensor into compressed-tensors pack-quantized format."""
    prefix = extract_prefix(name)
    out_features, in_features = tensor.shape
    num_groups = in_features // group_size

    # Dequantize Marlin layout tensor to obtain exact signed integers
    tensor_gpu = tensor.to(device)
    eye = torch.eye(in_features, dtype=torch.bfloat16, device=device)
    dequant_w = F.linear(eye, tensor_gpu).t()

    scale = getattr(tensor, "group_scale", None)
    if scale is None:
        scale_key = f"{prefix}.group_scale" if prefix else "group_scale"
        scale = state_dict.get(scale_key)
    valid_scales = scale.squeeze(-1).to(torch.float16)

    scales_exp = valid_scales.to(device).repeat_interleave(group_size, dim=1)
    q_signed = torch.round(dequant_w / scales_exp).clamp(-8, 7).to(torch.int8)

    weight_packed = pack_to_int32(q_signed.to(device), num_bits=4, packed_dim=1).cpu()
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)

    k = lambda s: f"{prefix}.{s}" if prefix else s
    return prefix, {
        k("weight_packed"): weight_packed.contiguous(),
        k("weight_scale"): valid_scales.cpu().contiguous(),
        k("weight_shape"): weight_shape.contiguous(),
    }


def convert_int8_weight_only(
    name: str,
    tensor: Any,
    state_dict: Dict[str, Any],
    device: torch.device,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Convert Int8WeightOnlyConfig tensor into compressed-tensors int-quantized format."""
    prefix = extract_prefix(name)

    if hasattr(tensor, "tensor_impl"):
        ti = tensor.tensor_impl
        raw_tensor = getattr(ti, "int_data", getattr(ti, "data", None))
        raw_int8 = raw_tensor.cpu().to(torch.int8)
        scale = ti.scale.cpu().to(torch.float16)
        zero_point = getattr(ti, "zero_point", None)
    else:
        raw_int8 = tensor.cpu().to(torch.int8)
        scale_key = f"{prefix}.scale" if prefix else "scale"
        zp_key = f"{prefix}.zero_point" if prefix else "zero_point"
        scale = state_dict[scale_key].cpu().to(torch.float16)
        zero_point = state_dict.get(zp_key)

    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)

    k = lambda s: f"{prefix}.{s}" if prefix else s
    result = {
        k("weight"): raw_int8.contiguous(),
        k("weight_scale"): scale.contiguous(),
    }
    if zero_point is not None:
        zp = zero_point.cpu().to(torch.int8)
        if zp.ndim == 1:
            zp = zp.unsqueeze(-1)
        result[k("weight_zero_point")] = zp.contiguous()

    return prefix, result


def convert_int8_dynamic_act(
    name: str,
    tensor: Any,
    state_dict: Dict[str, Any],
    device: torch.device,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Convert Int8DynamicActivationInt8WeightConfig into compressed-tensors int-quantized format."""
    prefix = extract_prefix(name)
    if hasattr(tensor, "original_weight_tensor"):
        underlying = tensor.original_weight_tensor
        ti = getattr(underlying, "tensor_impl", None)
        if ti is not None:
            raw_int8 = getattr(ti, "int_data", getattr(ti, "data", None)).cpu().to(torch.int8)
            scale = ti.scale.cpu().to(torch.float16)
            zero_point = getattr(ti, "zero_point", None)
        else:
            raw_int8 = underlying.cpu().to(torch.int8)
            scale = getattr(underlying, "scale", torch.tensor([1.0])).cpu().to(torch.float16)
            zero_point = getattr(underlying, "zero_point", None)
    elif hasattr(tensor, "tensor_impl"):
        ti = tensor.tensor_impl
        raw_int8 = getattr(ti, "int_data", getattr(ti, "data", None)).cpu().to(torch.int8)
        scale = ti.scale.cpu().to(torch.float16)
        zero_point = getattr(ti, "zero_point", None)
    elif hasattr(tensor, "qdata"):
        raw_int8 = tensor.qdata.cpu().to(torch.int8)
        scale = tensor.scale.cpu().to(torch.float16)
        zero_point = getattr(tensor, "zero_point", None)
    else:
        raw_int8 = tensor.cpu().to(torch.int8)
        scale_key = f"{prefix}.scale" if prefix else "scale"
        zp_key = f"{prefix}.zero_point" if prefix else "zero_point"
        scale = state_dict.get(scale_key)
        if scale is not None:
            scale = scale.cpu().to(torch.float16)
        else:
            scale = torch.ones((raw_int8.shape[0], 1), dtype=torch.float16)
        zero_point = state_dict.get(zp_key)

    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)

    k = lambda s: f"{prefix}.{s}" if prefix else s
    result = {
        k("weight"): raw_int8.contiguous(),
        k("weight_scale"): scale.contiguous(),
    }
    if zero_point is not None:
        zp = zero_point.cpu().to(torch.int8)
        if zp.ndim == 1:
            zp = zp.unsqueeze(-1)
        result[k("weight_zero_point")] = zp.contiguous()

    return prefix, result


def convert_int8_static_act(
    name: str,
    tensor: Any,
    state_dict: Dict[str, Any],
    device: torch.device,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Convert Int8StaticActivationInt8WeightConfig into compressed-tensors int-quantized format."""
    prefix, result = convert_int8_weight_only(name, tensor, state_dict, device)
    
    act_scale_key = f"{prefix}.input_scale" if prefix else "input_scale"
    act_zp_key = f"{prefix}.input_zero_point" if prefix else "input_zero_point"
    
    if act_scale_key in state_dict:
        result[act_scale_key] = state_dict[act_scale_key].cpu().to(torch.float16).contiguous()
    elif hasattr(tensor, "act_quant_scale"):
        result[act_scale_key] = tensor.act_quant_scale.cpu().to(torch.float16).contiguous()

    if act_zp_key in state_dict:
        result[act_zp_key] = state_dict[act_zp_key].cpu().to(torch.int8).contiguous()

    return prefix, result


def convert_fp8_weight_only(
    name: str,
    tensor: Any,
    state_dict: Dict[str, Any],
    device: torch.device,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Convert Float8WeightOnlyConfig into compressed-tensors float-quantized format."""
    prefix = extract_prefix(name)
    if hasattr(tensor, "qdata") and hasattr(tensor, "scale"):
        qdata = tensor.qdata.cpu()
        scale = tensor.scale.cpu().to(torch.float32)
    else:
        qdata = tensor.cpu()
        scale_key = f"{prefix}.scale" if prefix else "scale"
        scale = state_dict[scale_key].cpu().to(torch.float32)

    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)

    k = lambda s: f"{prefix}.{s}" if prefix else s
    return prefix, {
        k("weight"): qdata.contiguous(),
        k("weight_scale"): scale.contiguous(),
    }


def convert_fp8_dynamic_act(
    name: str,
    tensor: Any,
    state_dict: Dict[str, Any],
    device: torch.device,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Convert Float8DynamicActivationFloat8WeightConfig into compressed-tensors float-quantized format."""
    return convert_fp8_weight_only(name, tensor, state_dict, device)
