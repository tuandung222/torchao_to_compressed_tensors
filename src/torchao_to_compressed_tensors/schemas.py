"""
Quantization schema definitions and detection logic for TorchAO tensors.
"""

import enum
import json
from pathlib import Path
from typing import Any, Dict

import torch


class SchemaType(enum.Enum):
    INT4_TINYGEMM = "int4_tinygemm"
    INT4_PLAIN = "int4_plain"
    INT4_PRESHUFFLED = "int4_preshuffled"
    INT8_WEIGHT_ONLY = "int8_weight_only"
    INT8_DYNAMIC_ACT = "int8_dynamic_act"
    INT8_STATIC_ACT = "int8_static_act"
    FLOAT8_WEIGHT_ONLY = "float8_weight_only"
    FLOAT8_DYNAMIC_ACT = "float8_dynamic_act"
    PASSTHROUGH = "passthrough"


def extract_prefix(name: str) -> str:
    """Extract parent module prefix from parameter name."""
    if name == "weight":
        return ""
    elif name.endswith(".weight"):
        return name[:-len(".weight")]
    return name


def extract_group_size_from_config(config_path: Path) -> int:
    """Extract quantization group size from model config.json."""
    if not config_path.exists():
        return 128
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    qcfg = cfg.get("quantization_config", {})
    if "quant_type" in qcfg:
        qt = qcfg["quant_type"]
        if isinstance(qt, dict):
            return qt.get("group_size") or qt.get("_data", {}).get("group_size", 128)
    return 128


def detect_tensor_schema(name: str, tensor: Any, state_dict: Dict[str, Any]) -> SchemaType:
    """Accurately identify the quantization schema of a tensor in the state dict."""
    if name.endswith(".scales_and_zeros"):
        return SchemaType.PASSTHROUGH  # Handled with corresponding .weight

    sz_key = f"{name[:-len('.weight')]}.scales_and_zeros" if name.endswith(".weight") else f"{name}.scales_and_zeros"
    if sz_key in state_dict:
        return SchemaType.INT4_TINYGEMM

    tensor_cls = tensor.__class__.__name__

    # Float8 checks
    if "Float8Layout" in tensor_cls:
        if hasattr(tensor, "act_quant_kwargs") or (getattr(tensor, "scale", None) is not None and getattr(tensor, "scale").shape == (1, 1)):
            return SchemaType.FLOAT8_DYNAMIC_ACT
        return SchemaType.FLOAT8_WEIGHT_ONLY

    if getattr(tensor, "dtype", None) in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return SchemaType.FLOAT8_WEIGHT_ONLY

    if "Int4TilePackedTo4dTensor" in tensor_cls:
        return SchemaType.INT4_TINYGEMM
    elif "Int4PlainInt32Tensor" in tensor_cls:
        return SchemaType.INT4_PLAIN
    elif "Int4PreshuffledTensor" in tensor_cls:
        return SchemaType.INT4_PRESHUFFLED
    elif "LinearActivationQuantizedTensor" in tensor_cls:
        return SchemaType.INT8_DYNAMIC_ACT
    elif "AffineQuantizedTensor" in tensor_cls:
        return SchemaType.INT8_WEIGHT_ONLY

    # Key naming heuristics for exported state dicts
    if name.endswith(".qdata") and f"{name[:-len('.qdata')]}.scale" in state_dict:
        scale_t = state_dict[f"{name[:-len('.qdata')]}.scale"]
        if tensor.dtype == torch.int32:
            return SchemaType.INT4_PLAIN
        elif tensor.dtype == torch.int8:
            return SchemaType.INT8_WEIGHT_ONLY
        elif tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            return SchemaType.FLOAT8_WEIGHT_ONLY

    return SchemaType.PASSTHROUGH
