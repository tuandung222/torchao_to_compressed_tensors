"""
TorchAO to Compressed-Tensors Adapter
A high-performance adapter for converting TorchAO checkpoints into Compressed-Tensors format.
"""

from .schemas import (
    SchemaType,
    detect_tensor_schema,
    extract_prefix,
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
from .adapter import convert_checkpoint, main

__version__ = "0.2.0"

__all__ = [
    "SchemaType",
    "detect_tensor_schema",
    "extract_prefix",
    "extract_group_size_from_config",
    "convert_int4_tinygemm",
    "convert_int4_plain",
    "convert_int4_preshuffled",
    "convert_int8_weight_only",
    "convert_int8_dynamic_act",
    "convert_int8_static_act",
    "convert_fp8_weight_only",
    "convert_fp8_dynamic_act",
    "generate_compressed_tensors_config",
    "convert_checkpoint",
    "main",
]
