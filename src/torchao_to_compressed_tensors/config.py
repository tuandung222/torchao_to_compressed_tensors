"""
Generation of the ``quantization_config`` block for the exported checkpoint.

The block is built from compressed-tensors' own pydantic models rather than
assembled as a dict literal, so a schema change upstream surfaces as a
validation error here instead of as a checkpoint vLLM silently mis-reads.
"""

import re
from typing import Any

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)

from .quantize import (
    DEFAULT_NUM_BITS,
    make_int8_quantization_args,
    make_quantization_args,
)

__all__ = [
    "build_model_config",
    "build_quantization_config",
    "build_w8a8_quantization_config",
    "to_ignore_patterns",
]


def to_ignore_patterns(ignore: tuple[str, ...]) -> list[str]:
    """Turn module-name fragments into targets compressed-tensors will actually match.

    ``compressed_tensors.utils.match.match_name`` matches a target *exactly*
    unless it starts with ``re:``. A bare fragment like ``visual`` therefore
    never matches ``model.visual.blocks.0.attn.qkv``, so the serving runtime
    builds a quantised layer for a module this adapter deliberately left dense.
    The checkpoint then lacks the tensors that layer expects and the load dies
    with an unrelated-looking ``AttributeError``.

    Widening fragments into regexes makes them mean what they look like. It also
    makes fused modules work: matching runs through vLLM's
    ``packed_modules_mapping``, so a pattern covering ``in_proj_a`` also excludes
    the fused ``in_proj_ba`` that vLLM actually builds.

    Entries already written as ``re:...`` pass through untouched.
    """
    return [
        fragment
        if fragment.startswith("re:")
        else f"re:.*{re.escape(fragment)}.*"
        for fragment in ignore
    ]


def close_ignore_over_unquantised(
    ignore: tuple[str, ...],
    quantised: set[str],
    linear_modules: set[str],
) -> tuple[str, ...]:
    """Extend ``ignore`` to cover Linear modules that were not actually quantised.

    The config's ``ignore`` is a promise about the served model, not a record of
    the caller's intent. Any Linear the runtime would otherwise quantise must be
    listed, or it builds a quantised layer and then finds no ``weight_packed``
    to fill it.

    The gap this closes is not hypothetical: with ``tie_word_embeddings`` set,
    ``lm_head`` shares the embedding and has no tensor of its own, so there is
    nothing to quantise and nothing written -- yet unless it is named here,
    transformers fails the load with ``Linear has no attribute 'weight'``. (vLLM
    happens to tolerate it, which is worse: the checkpoint looks fine until
    someone loads it the other way.)

    :param ignore: patterns requested by the caller
    :param quantised: module prefixes that were actually quantised
    :param linear_modules: every Linear prefix in the architecture
    :return: ``ignore`` plus the exact names of any uncovered, unquantised Linear
    """
    from compressed_tensors.utils.match import match_name

    patterns = to_ignore_patterns(ignore)
    missing = sorted(
        name
        for name in linear_modules - quantised
        if not any(match_name(name, pattern) for pattern in patterns)
    )
    return tuple(ignore) + tuple(missing)


def build_quantization_config(
    group_size: int,
    symmetric: bool,
    ignore: tuple[str, ...],
    quantize_embeddings: bool = False,
    extra_linear_targets: tuple[str, ...] = (),
    num_bits: int = DEFAULT_NUM_BITS,
) -> dict[str, Any]:
    """Build the weight-only ``quantization_config`` block.

    Embeddings get their own group because compressed-tensors targets by module
    class: ``["Linear"]`` does not match an ``nn.Embedding``, and vLLM routes the
    two to different schemes (``CompressedTensorsWNA16`` vs
    ``CompressedTensorsEmbeddingWNA16Int``).

    ``extra_linear_targets`` exists for ``lm_head``. vLLM resolves a class target
    by asking whether the target string is *contained* in its own module class
    name: ``MergedColumnParallelLinear`` and ``RowParallelLinear`` both contain
    "Linear", and ``VocabParallelEmbedding`` contains "Embedding". ``lm_head`` is
    a ``ParallelLMHead``, which contains neither, so a config that only says
    ``["Linear"]`` leaves it dense -- and the load then dies with
    ``no module or parameter named 'lm_head.weight_packed'``. Naming it as a
    literal target makes the first match arm, on the layer name, succeed.

    :param group_size: elements per quantisation group
    :param symmetric: whether weights were exported without a zero-point
    :param ignore: module names to record as left dense
    :param quantize_embeddings: whether an ``Embedding`` group is emitted
    :param extra_linear_targets: literal layer names to add to the Linear group
    :param num_bits: 4 or 8; both use the pack-quantized format and the same
        vLLM scheme, so this is the only thing that distinguishes W4A16 from W8A16
    :return: the JSON-serialisable config block
    """
    weights: QuantizationArgs = make_quantization_args(group_size, symmetric, num_bits)
    groups = {
        "group_0": QuantizationScheme(
            targets=["Linear", *extra_linear_targets], weights=weights
        )
    }
    if quantize_embeddings:
        groups["embedding"] = QuantizationScheme(
            targets=["Embedding"], weights=weights
        )

    config = QuantizationConfig(
        config_groups=groups,
        format=CompressionFormat.pack_quantized.value,
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=to_ignore_patterns(ignore),
    )
    return config.model_dump(mode="json")


def build_model_config(
    base_config: dict[str, Any],
    group_size: int,
    symmetric: bool,
    lm_head_is_tied: bool,
    ignore: tuple[str, ...],
    quantize_embeddings: bool = False,
    extra_linear_targets: tuple[str, ...] = (),
    num_bits: int = DEFAULT_NUM_BITS,
) -> dict[str, Any]:
    """Derive the exported ``config.json`` from the source one.

    Everything the source declares is preserved except the quantisation block.
    In particular ``tie_word_embeddings`` is carried over rather than forced to
    ``False``: forcing it obliges the checkpoint to carry a duplicate copy of the
    embedding matrix as ``lm_head.weight``, which for a large vocabulary is
    hundreds of megabytes of redundancy.

    :param base_config: the decoded source ``config.json``
    :param group_size: elements per quantisation group
    :param symmetric: whether weights were exported without a zero-point
    :param lm_head_is_tied: whether ``lm_head`` shares the embedding weight
    :param ignore: module names left dense
    :param quantize_embeddings: whether an ``Embedding`` group is emitted
    :param extra_linear_targets: literal layer names to add to the Linear group
    :param num_bits: weight width, 4 or 8
    :return: the exported config
    """
    config = dict(base_config)
    config.pop("quantization_config", None)
    config["quantization_config"] = build_quantization_config(
        group_size,
        symmetric,
        ignore,
        quantize_embeddings,
        extra_linear_targets,
        num_bits,
    )
    config["tie_word_embeddings"] = lm_head_is_tied
    return config


def build_w8a8_quantization_config(
    ignore: tuple[str, ...],
    extra_linear_targets: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the ``quantization_config`` block for W8A8 int8.

    Three things differ from the weight-only block and all three are forced by
    what vLLM will accept, not chosen:

    *Per-channel weights.* ``_is_dynamic_token_w8a8`` tests the weight strategy
    against ``TENSOR`` and ``CHANNEL`` only. A grouped W8A8 config does not fail --
    it falls through to the weight-only ``WNA16`` scheme and is served with no
    activation quantisation at all, which is a silent loss of the entire point.

    *Dynamic per-token activations.* The alternative, static per-tensor, needs a
    calibrated ``input_scale`` in the checkpoint. Calibration means running data
    through the model, which is llm-compressor's job rather than this adapter's.

    *Asymmetric activations.* vLLM accepts either, so this is the one place the
    choice looks free -- and it is not. TorchAO cannot fake-quantise activations
    symmetrically at all (``IntxFakeQuantizer._per_token_forward`` raises), so a
    symmetric declaration here could never be matched by QAT. The asymmetric grids
    do agree bit for bit; ``tests/test_grid_parity.py`` pins that down.

    There is no ``Embedding`` group. An embedding lookup has no matmul to feed
    int8 activations into, so W8A8 says nothing about it; quantising the table is
    a separate, weight-only decision and vLLM reads it through a different scheme.

    :param ignore: module names to record as left dense
    :param extra_linear_targets: literal layer names to add to the Linear group
    :return: the JSON-serialisable config block
    """
    weights = make_int8_quantization_args()
    input_activations = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        symmetric=False,
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    )
    groups = {
        "group_0": QuantizationScheme(
            targets=["Linear", *extra_linear_targets],
            weights=weights,
            input_activations=input_activations,
        )
    }
    config = QuantizationConfig(
        config_groups=groups,
        format=CompressionFormat.int_quantized.value,
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=to_ignore_patterns(ignore),
    )
    return config.model_dump(mode="json")
