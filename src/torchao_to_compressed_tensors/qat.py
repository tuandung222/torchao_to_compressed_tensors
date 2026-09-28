"""
QAT configuration that trains against the grid vLLM's Marlin kernel actually serves.

TorchAO exposes several INT4 fake-quantisation numerics and they are not
interchangeable. Picking the wrong one trains the model for a grid it will never
be served on, which shows up as "QAT barely helped" rather than as an error.

For weight-only INT4 and INT8 on A100 the only correct choice is a symmetric
per-group ``IntxFakeQuantizeConfig``; ``tests/test_grid_parity.py`` proves it
matches compressed-tensors bit for bit at both widths. See docs/SCOPE.md for the
full reasoning.

Eight bits must be symmetric. vLLM's Marlin offers a zero-point path for 4 bits
only, so asymmetric INT8 loads and then quietly runs on a slower kernel.
"""

import torch
import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.qat import (
    FakeQuantizedEmbedding,
    IntxFakeQuantizeConfig,
    QATConfig,
)

from .quantize import DEFAULT_NUM_BITS, validate_num_bits

__all__ = [
    "DEFAULT_IGNORE",
    "MARLIN_GROUP_SIZES",
    "convert_w4a16_qat",
    "patch_axolotl_qat",
    "prepare_w4a16_qat",
    "prepare_w8a8_qat",
    "prepare_weight_only_qat",
    "validate_group_size",
    "w4a16_convert_config",
    "w4a16_qat_config",
    "w8a8_qat_config",
    "weight_only_qat_config",
]


# vLLM: MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]. -1 (channelwise) is
# excluded here because QAT per-channel INT4 loses far too much accuracy to be
# worth offering.
MARLIN_GROUP_SIZES = (32, 64, 128)

DEFAULT_GROUP_SIZE = 128

#: Default modules to leave dense. This is a *convention*, not a limitation:
#: vLLM serves a quantised ``lm_head`` through the same Marlin path as any other
#: Linear (``ParallelLMHead`` is handled as a linear in
#: ``compressed_tensors.py``), and quantised embeddings through
#: ``CompressedTensorsEmbeddingWNA16Int``. Pass ``ignore=()`` to quantise
#: everything.
#:
#: Whatever is chosen here must match the ``ignore`` passed to the adapter at
#: export time. A module fake-quantised during QAT but served dense -- or the
#: reverse -- is trained against a grid it never sees.
DEFAULT_IGNORE = ("lm_head",)


def validate_group_size(group_size: int) -> int:
    """Reject group sizes Marlin cannot serve, at config time rather than at load time."""
    if group_size not in MARLIN_GROUP_SIZES:
        raise ValueError(
            f"group_size={group_size} is not servable by Marlin. "
            f"Supported: {list(MARLIN_GROUP_SIZES)}. A checkpoint quantised with "
            "any other group size loads in vLLM but silently falls back to a "
            "slower kernel."
        )
    return group_size


#: TorchAO's dtype per weight width. ``torch.int4`` is a torchao sentinel rather
#: than a storage dtype; ``torch.int8`` is a real one.
_AO_WEIGHT_DTYPE = {4: torch.int4, 8: torch.int8}


def weight_only_qat_config(
    num_bits: int = DEFAULT_NUM_BITS, group_size: int = DEFAULT_GROUP_SIZE
) -> QATConfig:
    """QAT config for symmetric INT4/INT8 weight-only, per-group -- the Marlin target.

    Deliberately does *not* use ``Int4WeightFakeQuantizeConfig``: despite the
    name, that config simulates MSLK preshuffled-kernel numerics and defaults to
    fp8 input activations (torchao ``fake_quantize_config.py``), which is a
    Hopper/fbgemm path with a different grid.

    Both widths use the same ``IntxFakeQuantizeConfig`` path and both match
    compressed-tensors bit for bit on the scale; ``tests/test_grid_parity.py``
    covers 4 and 8. Eight bits must stay symmetric -- see
    :func:`quantize.validate_num_bits`.

    :param num_bits: 4 or 8
    :param group_size: elements per quantisation group; must be Marlin-servable
    :return: a ``QATConfig`` ready for ``quantize_(model, config)``
    """
    validate_num_bits(num_bits, symmetric=True)
    validate_group_size(group_size)
    weight_config = IntxFakeQuantizeConfig(
        _AO_WEIGHT_DTYPE[num_bits],
        group_size=group_size,
        is_symmetric=True,
    )
    return QATConfig(weight_config=weight_config, step="prepare")


def w4a16_qat_config(group_size: int = DEFAULT_GROUP_SIZE) -> QATConfig:
    """W4A16 specialisation of :func:`weight_only_qat_config`."""
    return weight_only_qat_config(4, group_size)


def w4a16_convert_config() -> QATConfig:
    """Config that strips fake quantisers and leaves plain bf16 weights.

    Passing no ``base_config`` makes the convert step a pure module swap
    (``FakeQuantizedLinear -> nn.Linear``), so the result is an ordinary dense
    checkpoint carrying QAT-trained weights. That checkpoint is the input to
    either the adapter or an llm-compressor oneshot run.
    """
    return QATConfig(step="convert")


def _make_filter(kind: type, ignore: tuple[str, ...]):
    def _filter(module: nn.Module, fqn: str) -> bool:
        if type(module) is not kind and not isinstance(module, kind):
            return False
        return not any(fragment in fqn for fragment in ignore)

    return _filter


def prepare_weight_only_qat(
    model: nn.Module,
    num_bits: int = DEFAULT_NUM_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
    quantize_embeddings: bool = False,
) -> nn.Module:
    """Swap the targeted modules for fake-quantised ones.

    :param model: model to prepare in place
    :param num_bits: 4 or 8
    :param group_size: elements per quantisation group; must be Marlin-servable
    :param ignore: module-name fragments to leave dense; pass ``()`` to quantise
        every Linear, ``lm_head`` included
    :param quantize_embeddings: also fake-quantise ``nn.Embedding`` weights.
        Served by vLLM's ``CompressedTensorsEmbeddingWNA16Int``; worth it mainly
        for large-vocabulary models where the table is a real share of memory
    :return: the same model, prepared
    """
    config = weight_only_qat_config(num_bits, group_size)
    quantize_(model, config, filter_fn=_make_filter(nn.Linear, ignore))
    if quantize_embeddings:
        quantize_(model, config, filter_fn=_make_filter(nn.Embedding, ignore))
    return model


def prepare_w4a16_qat(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
    quantize_embeddings: bool = False,
) -> nn.Module:
    """W4A16 specialisation of :func:`prepare_weight_only_qat`."""
    return prepare_weight_only_qat(
        model, 4, group_size, ignore, quantize_embeddings
    )


def convert_w4a16_qat(model: nn.Module) -> nn.Module:
    """Strip fake quantisers, leaving plain dense weights carrying QAT training.

    Embeddings need their own pass: ``quantize_`` defaults to a Linear-only
    filter, so a ``FakeQuantizedEmbedding`` is never visited and survives the
    convert step. The saved weights are unaffected -- the fake quantiser wraps a
    normal Parameter -- but the in-memory model keeps quantising its lookups,
    which silently skews anything evaluated on it before export.

    :param model: prepared model to convert in place
    :return: the same model, with ``nn.Linear`` and ``nn.Embedding`` restored
    """
    config = w4a16_convert_config()
    quantize_(model, config)
    quantize_(
        model,
        config,
        filter_fn=lambda module, _: isinstance(module, FakeQuantizedEmbedding),
    )
    return model


def patch_axolotl_qat(ignore: tuple[str, ...] = DEFAULT_IGNORE) -> None:
    """Redirect Axolotl's ``qat`` block to the numerics vLLM actually serves.

    Axolotl's ``_make_qat_config`` maps INT4 weights onto
    ``Int4WeightFakeQuantizeConfig``. With ``activation_dtype: null`` that config
    falls back to its default ``activation_dtype=e4m3`` and fake-quantises the
    weight through fp8 per-row *before* INT4 -- numerics meant for
    ``torch.ops.mslk.*``, not for Marlin W4A16.

    It also applies ``ignore`` so that QAT covers exactly the modules the export
    will quantise. Axolotl itself fake-quantises every Linear including
    ``lm_head``; pass ``ignore=()`` to keep that behaviour, and give the adapter
    the same ``--ignore`` at export time.

    Patches ``prepare_model_for_qat`` rather than the private
    ``_make_qat_config``: the former has the same signature across Axolotl
    versions and is the single call site (``loaders/model.py``), the latter does
    not exist before 0.17. Three cases are redirected -- INT4 and INT8
    weight-only, and INT8 weight with INT8 activations (W8A8). Every other
    combination falls through to Axolotl's own implementation untouched.

    Call this before the model is loaded -- an Axolotl plugin's import hook is a
    convenient place.

    :raises ImportError: if Axolotl is not installed
    """
    from axolotl.utils import quantization as axolotl_quantization
    from axolotl.utils.schemas.enums import TorchAOQuantDType

    original = axolotl_quantization.prepare_model_for_qat
    if getattr(original, "_w4a16_patched", False):
        return

    def _patched_prepare_model_for_qat(
        model,
        weight_dtype,
        group_size=None,
        activation_dtype=None,
        quantize_embedding=False,
    ):
        # INT8 weight-only is redirected for a second reason on top of the grid:
        # Axolotl refuses it outright. ``get_quantization_config`` raises
        # "Int8WeightOnlyConfig is not supported by torchao QAT" -- but that call
        # sits inside the function being replaced here, so returning early never
        # reaches it and W8A16 QAT becomes available without touching Axolotl.
        widths = {TorchAOQuantDType.int4: 4, TorchAOQuantDType.int8: 8}
        num_bits = widths.get(weight_dtype)

        # W8A8. Axolotl refuses this one too ("Int8DynamicActivationInt8WeightConfig
        # is not supported by torchao QAT"), from the same call site, so the same
        # early return makes it available. Note that group_size is deliberately
        # dropped: W8A8 weights are per-channel, and honouring a group size here
        # would train against a grid vLLM will not serve.
        if (
            weight_dtype == TorchAOQuantDType.int8
            and activation_dtype == TorchAOQuantDType.int8
        ):
            if quantize_embedding:
                raise ValueError(
                    "quantize_embedding has no meaning under W8A8: an embedding "
                    "lookup has no matmul to feed int8 activations into. Quantise "
                    "the table with a weight-only scheme instead."
                )
            prepare_w8a8_qat(model, ignore=ignore)
            return None

        is_weight_only = num_bits is not None and activation_dtype is None
        if not is_weight_only:
            return original(
                model, weight_dtype, group_size, activation_dtype, quantize_embedding
            )

        gs = group_size if group_size is not None else DEFAULT_GROUP_SIZE
        prepare_weight_only_qat(
            model,
            num_bits,
            gs,
            ignore=ignore,
            quantize_embeddings=bool(quantize_embedding),
        )

    _patched_prepare_model_for_qat._w4a16_patched = True
    axolotl_quantization.prepare_model_for_qat = _patched_prepare_model_for_qat


# ---------------------------------------------------------------------------
# W8A8: per-channel weights, per-token activations
# ---------------------------------------------------------------------------


def w8a8_qat_config() -> QATConfig:
    """QAT config for W8A8 int8: per-channel weights, dynamic per-token activations.

    Every part of this is pinned by something outside our control:

    ``PerAxis(0)`` rather than a group size, because vLLM's W8A8 predicates accept
    only ``TENSOR`` or ``CHANNEL`` weight strategies. A grouped config is not an
    error -- it is served as plain weight-only WNA16, with the activations left in
    bf16 and the entire point of W8A8 quietly dropped.

    ``is_symmetric=True`` on the weights, matching ``make_int8_quantization_args``
    and required by both W8A8 predicates.

    ``is_symmetric=False`` on the activations, which looks like the free choice and
    is not: ``IntxFakeQuantizer._per_token_forward`` raises ``NotImplementedError``
    for symmetric, so there is no symmetric path to train against. Fortunately the
    asymmetric grids agree exactly -- TorchAO and
    ``vllm._custom_ops.scaled_int8_quant(symmetric=False)`` derive the same
    ``(max - min) / 255`` scale and the same zero-point, verified in
    ``tests/test_grid_parity.py``. vLLM's *symmetric* per-token path uses
    ``max_abs / 127``, a third convention, so this must not be flipped without
    re-measuring.

    :return: a ``QATConfig`` ready for ``quantize_(model, config)``
    """
    from torchao.quantization.granularity import PerAxis, PerToken

    weight_config = IntxFakeQuantizeConfig(
        dtype=torch.int8,
        granularity=PerAxis(0),
        is_symmetric=True,
    )
    activation_config = IntxFakeQuantizeConfig(
        dtype=torch.int8,
        granularity=PerToken(),
        is_symmetric=False,
    )
    return QATConfig(
        weight_config=weight_config,
        activation_config=activation_config,
        step="prepare",
    )


def prepare_w8a8_qat(
    model: nn.Module,
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
) -> nn.Module:
    """Swap Linears for W8A8 fake-quantised ones.

    Embeddings are deliberately not touched, and take no ``quantize_embeddings``
    flag. W8A8 is about feeding int8 activations into an int8 matmul; a lookup has
    no matmul, so the scheme says nothing about the table. Quantising the embedding
    remains a separate weight-only decision, served through a different vLLM
    scheme, and mixing it in here would imply otherwise.

    :param model: model to prepare in place
    :param ignore: module-name fragments to leave dense; must match the export
    :return: the same model, prepared
    """
    quantize_(model, w8a8_qat_config(), filter_fn=_make_filter(nn.Linear, ignore))
    return model
