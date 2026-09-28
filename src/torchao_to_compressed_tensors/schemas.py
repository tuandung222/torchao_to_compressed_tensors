"""
Source checkpoint inspection: what kind of checkpoint is this, and what group
size was it trained with.
"""

import enum
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "DEFAULT_IGNORE",
    "ModuleKind",
    "SourceFormat",
    "classify_modules",
    "detect_source_format",
    "extract_group_size",
    "iter_quantizable_weights",
]


class SourceFormat(enum.Enum):
    """The checkpoint layouts this adapter accepts."""

    #: Dense fp16/bf16 weights, typically a QAT run after ``QATConfig(step="convert")``.
    #: The preferred input: nothing has been packed yet, so the export grid is
    #: chosen once, here, and matches what vLLM serves.
    DENSE = "dense"

    #: TorchAO tinygemm INT4, recognised by the ``.scales_and_zeros`` sidecar.
    #: Produced by the legacy ``Int4WeightOnlyQATQuantizer``. Supported for
    #: existing checkpoints; asymmetric by construction.
    INT4_TINYGEMM = "int4_tinygemm"


#: Default modules to leave dense -- a convention, not a limitation. vLLM serves
#: a quantised ``lm_head`` through the same Marlin path as any other Linear, so
#: passing ``ignore=()`` to quantise the whole model is supported. Must match
#: whatever QAT used; see :data:`qat.DEFAULT_IGNORE`.
DEFAULT_IGNORE = ("lm_head",)

_TINYGEMM_SUFFIX = ".scales_and_zeros"


class ModuleKind(enum.Enum):
    """How compressed-tensors will target a module."""

    LINEAR = "Linear"
    EMBEDDING = "Embedding"
    OTHER = "other"


def detect_source_format(state_dict: dict[str, Any]) -> SourceFormat:
    """Classify a loaded state dict.

    :param state_dict: the source checkpoint's tensors
    :return: the detected format
    :raises ValueError: if the checkpoint holds a quantised layout this adapter
        does not support, rather than silently passing those tensors through
    """
    if any(k.endswith(_TINYGEMM_SUFFIX) for k in state_dict):
        return SourceFormat.INT4_TINYGEMM

    exotic = sorted(
        {
            type(v).__name__
            for v in state_dict.values()
            if type(v) is not torch.Tensor and type(v).__name__ != "Parameter"
        }
    )
    if exotic:
        raise ValueError(
            f"Unsupported quantised tensor types in source checkpoint: {exotic}. "
            "Only dense checkpoints and TorchAO tinygemm INT4 are supported; see "
            "docs/SCOPE.md. Re-export with QATConfig(step='convert') to get a "
            "dense checkpoint."
        )

    non_float = sorted(
        {str(v.dtype) for v in state_dict.values() if not v.dtype.is_floating_point}
    )
    if non_float:
        raise ValueError(
            f"Source checkpoint holds non-float tensors ({non_float}) but no "
            "recognised quantisation sidecar. Refusing to guess its layout."
        )

    return SourceFormat.DENSE


def _find_group_size(node: Any) -> int | None:
    """Recursively search a decoded config fragment for a ``group_size``.

    TorchAO nests it differently across versions -- ``quant_type.group_size`` in
    some, ``quant_type.default._data.group_size`` in others. Searching instead of
    hardcoding one shape is what stops a group_size=64 checkpoint from being
    silently exported as 128.
    """
    if isinstance(node, dict):
        value = node.get("group_size")
        if isinstance(value, int):
            return value
        for child in node.values():
            found = _find_group_size(child)
            if found is not None:
                return found
    elif isinstance(node, list):
        for child in node:
            found = _find_group_size(child)
            if found is not None:
                return found
    return None


def extract_group_size(config_path: Path, override: int | None = None) -> int:
    """Determine the quantisation group size for a source checkpoint.

    :param config_path: path to the source ``config.json``
    :param override: explicit group size from the CLI, which always wins
    :return: the group size to quantise with
    :raises ValueError: if no group size can be established. Never falls back to
        a default -- exporting a group_size=64 checkpoint as 128 produces a
        checkpoint that loads cleanly and returns nonsense.
    """
    if override is not None:
        return override

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        found = _find_group_size(config.get("quantization_config", {}))
        if found is not None:
            return found

    raise ValueError(
        f"Could not determine group_size from {config_path}. Pass --group-size "
        "explicitly; it must match the group size the model was QAT-trained with."
    )


def classify_modules(config_path: Path) -> dict[str, ModuleKind]:
    """Map every module prefix to the kind compressed-tensors will target it as.

    A state dict cannot distinguish an ``nn.Linear`` weight from an
    ``nn.Embedding`` one -- both are 2D float -- but compressed-tensors targets
    by module *class*. So the architecture is instantiated on the ``meta`` device
    (no allocation, no weights read) and its real module tree is walked.

    :param config_path: path to the source ``config.json``
    :return: prefix -> kind, empty if the architecture could not be instantiated
    """
    try:
        import torch.nn as nn
        import transformers
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    except ImportError:  # pragma: no cover - transformers is a hard dep in practice
        return {}

    try:
        config = AutoConfig.from_pretrained(config_path.parent)
        config = type(config).from_dict(
            {k: v for k, v in config.to_dict().items() if k != "quantization_config"}
        )
    except Exception:
        return {}

    # The class named in `architectures` first, because the auto classes can
    # build a *different* tree. For a multimodal checkpoint like Qwen3.5,
    # AutoModelForCausalLM yields a text-only model rooted at `model.layers.*`
    # while the checkpoint is `model.language_model.layers.*` -- zero prefixes
    # match, and every weight would be classified OTHER and skipped.
    candidates = []
    for name in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, name, None)
        if cls is not None:
            candidates.append(cls)
    candidates.extend([AutoModelForCausalLM, AutoModel])

    for cls in candidates:
        try:
            with torch.device("meta"):
                model = (
                    cls.from_config(config)
                    if hasattr(cls, "from_config")
                    else cls(config)
                )
        except Exception:
            # Custom architectures, trust_remote_code models, unavailable classes.
            continue

        kinds: dict[str, ModuleKind] = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                kinds[name] = ModuleKind.LINEAR
            elif isinstance(module, nn.Embedding):
                kinds[name] = ModuleKind.EMBEDDING
        if kinds:
            return kinds

    # Nothing instantiated; the caller falls back to treating every 2D float
    # weight as Linear.
    return {}


def iter_quantizable_weights(
    state_dict: dict[str, torch.Tensor],
    module_kinds: dict[str, ModuleKind],
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
    quantize_embeddings: bool = False,
) -> Iterator[tuple[str, torch.Tensor, ModuleKind]]:
    """Yield ``(prefix, weight, kind)`` for every weight to be quantised.

    A weight qualifies when it is 2D, float, named ``<prefix>.weight``, its
    prefix contains none of the ``ignore`` fragments, and its module kind is
    targeted. Norms and biases fall out via the 2D check.

    When ``module_kinds`` is empty (architecture could not be introspected)
    every eligible weight is treated as Linear.

    :param state_dict: the source checkpoint's tensors
    :param module_kinds: output of :func:`classify_modules`
    :param ignore: module-name fragments to keep dense
    :param quantize_embeddings: whether ``nn.Embedding`` weights are targeted
    """
    # A classification that names none of the checkpoint's modules is worse than
    # none at all: every weight would resolve to OTHER and be silently skipped.
    # Treat that as "unavailable" and fall back to the 2D-float heuristic.
    if module_kinds:
        prefixes = {k[: -len(".weight")] for k in state_dict if k.endswith(".weight")}
        if not (prefixes & module_kinds.keys()):
            module_kinds = {}

    for name, tensor in state_dict.items():
        if not name.endswith(".weight"):
            continue
        prefix = name[: -len(".weight")]
        if any(fragment in prefix for fragment in ignore):
            continue
        if tensor.ndim != 2 or not tensor.dtype.is_floating_point:
            continue

        kind = module_kinds.get(prefix, ModuleKind.LINEAR if not module_kinds else ModuleKind.OTHER)
        if kind is ModuleKind.EMBEDDING and not quantize_embeddings:
            continue
        if kind is ModuleKind.OTHER:
            continue
        yield prefix, tensor, kind
