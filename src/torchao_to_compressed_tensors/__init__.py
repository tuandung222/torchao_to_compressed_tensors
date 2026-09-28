"""
TorchAO QAT -> compressed-tensors W4A16, for serving with vLLM's Marlin kernel.

Scope is deliberately one scheme wide; see docs/SCOPE.md for what was removed
and why.
"""

from .adapter import convert_checkpoint, main
from .config import build_model_config, build_quantization_config
from .handlers import dequantize_tinygemm
from .qat import (
    DEFAULT_IGNORE,
    MARLIN_GROUP_SIZES,
    convert_w4a16_qat,
    patch_axolotl_qat,
    prepare_w4a16_qat,
    validate_group_size,
    w4a16_convert_config,
    w4a16_qat_config,
)
from .quantize import QuantizedWeight, make_quantization_args, quantize_weight
from .schemas import (
    ModuleKind,
    SourceFormat,
    classify_modules,
    detect_source_format,
    extract_group_size,
    iter_quantizable_weights,
)

__version__ = "0.3.0"

# Grouped by role rather than sorted; the grouping is the documentation.
__all__ = [  # noqa: RUF022
    # QAT
    "MARLIN_GROUP_SIZES",
    "DEFAULT_IGNORE",
    "w4a16_qat_config",
    "w4a16_convert_config",
    "prepare_w4a16_qat",
    "convert_w4a16_qat",
    "validate_group_size",
    "patch_axolotl_qat",
    # Quantisation
    "QuantizedWeight",
    "make_quantization_args",
    "quantize_weight",
    # Source inspection
    "SourceFormat",
    "ModuleKind",
    "detect_source_format",
    "extract_group_size",
    "classify_modules",
    "iter_quantizable_weights",
    "dequantize_tinygemm",
    # Config + entrypoint
    "build_quantization_config",
    "build_model_config",
    "convert_checkpoint",
    "main",
]
