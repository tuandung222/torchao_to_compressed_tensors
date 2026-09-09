"""
Specification-compliant configuration generator for compressed-tensors format.
"""

from typing import Any, Dict
from .schemas import SchemaType


def generate_compressed_tensors_config(
    dominant_schema: SchemaType,
    group_size: int,
    base_config: Dict[str, Any],
    symmetric: bool = True,
    act_symmetric: bool = True,
) -> Dict[str, Any]:
    """Generate spec-compliant compressed-tensors quantization_config."""
    config_copy = dict(base_config)

    if dominant_schema in [SchemaType.INT4_TINYGEMM, SchemaType.INT4_PLAIN, SchemaType.INT4_PRESHUFFLED]:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": symmetric,
                        "strategy": "group",
                        "group_size": group_size,
                        "actorder": None,
                    },
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": ["lm_head"],
            "quantization_status": "compressed",
            "version": "0.18.0",
        }
    elif dominant_schema == SchemaType.INT8_DYNAMIC_ACT:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "actorder": None,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": act_symmetric,
                        "strategy": "token",
                        "dynamic": True,
                    },
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": ["lm_head"],
            "quantization_status": "compressed",
            "version": "0.18.0",
        }
    elif dominant_schema == SchemaType.INT8_STATIC_ACT:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "actorder": None,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": act_symmetric,
                        "strategy": "tensor",
                        "dynamic": False,
                    },
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": ["lm_head"],
            "quantization_status": "compressed",
            "version": "0.18.0",
        }
    elif dominant_schema == SchemaType.FLOAT8_WEIGHT_ONLY:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "channel",
                    },
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": ["lm_head"],
            "quantization_status": "compressed",
            "version": "0.18.0",
        }
    elif dominant_schema == SchemaType.FLOAT8_DYNAMIC_ACT:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "channel",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "token",
                        "dynamic": True,
                    },
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": ["lm_head"],
            "quantization_status": "compressed",
            "version": "0.18.0",
        }
    else:  # INT8_WEIGHT_ONLY
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": symmetric,
                        "strategy": "channel" if group_size == 0 or group_size is None else "group",
                        "actorder": None,
                    },
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": ["lm_head"],
            "quantization_status": "compressed",
            "version": "0.18.0",
        }

    config_copy["quantization_config"] = quant_config
    config_copy["tie_word_embeddings"] = False
    return config_copy
