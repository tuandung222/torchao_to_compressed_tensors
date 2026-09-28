"""
Grid parity: the lattice TorchAO QAT trains against must equal the lattice
compressed-tensors (and therefore vLLM's Marlin kernel) serves.

This is the foundational test of the project. If it fails, QAT is optimising the
model for a quantisation grid that differs from the one used at inference, and
every downstream accuracy number is meaningless regardless of how correct the
checkpoint conversion is.

Covered here:

* INT4 and INT8 weight-only, per-group, on the group sizes Marlin accepts
  (``MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]``) -- W4A16 and W8A16.
* INT8 weight-only, per-channel -- the weight half of W8A8, which cannot use
  per-group at all (vLLM's ``_is_dynamic_token_w8a8`` accepts only TENSOR or
  CHANNEL).
* INT8 activations, per-token -- the other half of W8A8.

Asymmetric weight quantisation stays unsupported; the tripwires below say why.

Runs on CPU. The activation tests compare against a reference implementation of
vLLM's formula rather than calling vLLM, which lives in a separate environment;
the reference was checked against ``vllm._custom_ops.scaled_int8_quant`` directly
and agreed bit for bit. See ``test_per_token_activation_matches_vllm_formula``.
"""

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams
from torchao.quantization.qat import IntxFakeQuantizeConfig
from torchao.quantization.qat.fake_quantizer import IntxFakeQuantizer
from torchao.quantization.utils import (
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
)

MARLIN_GROUP_SIZES = [32, 64, 128]

#: Weight bit-widths Marlin serves symmetrically (``uint4b8`` and ``uint8b128``).
#: Note the asymmetry in the asymmetry: ``query_marlin_supported_quant_types``
#: returns ``[uint4]`` when ``has_zp``, so 8-bit has no zero-point path and must
#: stay symmetric to keep the fast kernel.
WEIGHT_BITS = [4, 8]

#: The torchao dtype for each width. ``torch.int4`` exists only as a torchao
#: sentinel; INT8 is a real torch dtype.
_AO_DTYPE = {4: torch.int4, 8: torch.int8}

# Shapes drawn from real Qwen3/Llama projections: in_features must be divisible
# by the group size, out_features by 64 for Marlin tile alignment.
SHAPES = [
    (256, 512),
    (2048, 1024),
    (3072, 1024),
    (1024, 3072),
]


def _ct_args(
    group_size: int | None, symmetric: bool, num_bits: int = 4
) -> QuantizationArgs:
    """``group_size=None`` selects per-channel, the strategy W8A8 requires."""
    if group_size is None:
        return QuantizationArgs(
            num_bits=num_bits,
            type=QuantizationType.INT,
            symmetric=symmetric,
            strategy=QuantizationStrategy.CHANNEL,
        )
    return QuantizationArgs(
        num_bits=num_bits,
        type=QuantizationType.INT,
        symmetric=symmetric,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
    )


def _ct_qparams(
    w: torch.Tensor, group_size: int | None, symmetric: bool, num_bits: int = 4
):
    """Quantisation parameters exactly as compressed-tensors would derive them."""
    args = _ct_args(group_size, symmetric, num_bits)
    # Per-channel is per-group with one group spanning the whole row, which is
    # how compressed-tensors reduces it internally.
    grouped = (
        w.view(w.shape[0], 1, -1)
        if group_size is None
        else w.view(w.shape[0], -1, group_size)
    )
    return calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)


def _weight(out_features: int, in_features: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(out_features, in_features) * 0.02).float()


# ----------------------------------------------------------------------------
# Symmetric: the supported configuration
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("num_bits", WEIGHT_BITS)
@pytest.mark.parametrize("group_size", MARLIN_GROUP_SIZES)
@pytest.mark.parametrize("out_features,in_features", SHAPES)
def test_symmetric_qparams_are_bit_identical(
    out_features, in_features, group_size, num_bits
):
    """TorchAO and compressed-tensors must derive the same scale and zero-point.

    Both use scale = max_abs / (2**(num_bits-1) - 0.5) -- the full asymmetric
    integer range, so 7.5 at 4 bits and 127.5 at 8, rather than 7 and 127 -- with
    zero-point 0. Any drift here means the two libraries disagree on the grid
    itself, not merely on rounding.
    """
    w = _weight(out_features, in_features)

    s_ct, z_ct = _ct_qparams(w, group_size, symmetric=True, num_bits=num_bits)
    s_ao, z_ao = get_group_qparams_symmetric(w, num_bits, group_size, torch.float32)
    s_ao = s_ao.view(s_ct.shape)
    z_ao = z_ao.view(z_ct.shape)

    assert torch.equal(s_ct, s_ao), (
        f"scale mismatch: max rel diff "
        f"{((s_ct - s_ao).abs() / s_ct).max().item():.3e}"
    )
    assert z_ct.count_nonzero() == 0 and z_ao.count_nonzero() == 0, (
        "symmetric zero-point must be 0 on both sides"
    )


#: Elements allowed to disagree away from the rounding tie. Zero at 4 bits. At 8
#: bits the levels are 16x finer, so a value can sit close enough to a boundary
#: that the two float operation orders round it differently without being at the
#: exact half-way point: measured 2 such elements in 3,145,728 (a 1024x3072
#: weight at group 128), against 2,512 genuine tie disagreements. Kept tight so a
#: real grid divergence still fails rather than hiding under the allowance.
_OFF_TIE_ALLOWANCE = {4: 0, 8: 8}


@pytest.mark.parametrize("num_bits", WEIGHT_BITS)
@pytest.mark.parametrize("group_size", MARLIN_GROUP_SIZES)
@pytest.mark.parametrize("out_features,in_features", SHAPES)
def test_symmetric_lattice_matches(out_features, in_features, group_size, num_bits):
    """Fake-quantised weights must agree, except at the exact rounding tie.

    With scale = max_abs / (2**(num_bits-1) - 0.5), the largest-magnitude element
    of every group maps to exactly +-that bound -- a rounding tie. The two
    libraries reach it through different float operation orders, so they can land
    on different sides. That affects at most one element per group and always by
    exactly one level, which is ordinary rounding noise rather than a grid
    disagreement.

    Anything beyond the tie, past ``_OFF_TIE_ALLOWANCE``, is a real mismatch.
    """
    w = _weight(out_features, in_features)
    args = _ct_args(group_size, symmetric=True, num_bits=num_bits)

    s, z = _ct_qparams(w, group_size, symmetric=True, num_bits=num_bits)
    ct = fake_quantize(w, s, z, args)

    config = IntxFakeQuantizeConfig(
        _AO_DTYPE[num_bits], group_size=group_size, is_symmetric=True
    )
    ao = IntxFakeQuantizer(config)(w)

    scale_expanded = s.repeat_interleave(group_size, dim=1)
    mismatch = (ao - ct).abs() > 1e-7

    if not mismatch.any():
        return

    # Every mismatching element must sit on the tie ...
    tie = 2 ** (num_bits - 1) - 0.5
    at_tie = ((w / scale_expanded).abs() - tie).abs() < 1e-4
    off_tie = mismatch & ~at_tie
    assert off_tie.sum().item() <= _OFF_TIE_ALLOWANCE[num_bits], (
        f"{off_tie.sum().item()} element(s) disagree away from the rounding tie "
        f"(allowance {_OFF_TIE_ALLOWANCE[num_bits]}) -- this looks like a genuine "
        "grid mismatch, not rounding noise"
    )

    # ... and differ by at most one quantisation level.
    max_levels = ((ao - ct).abs() / scale_expanded).max().item()
    assert max_levels < 1.01, f"disagreement of {max_levels:.3f} levels exceeds one"

    # Ties are bounded by one element per group: the group's own maximum is the
    # only value that can land exactly on the bound.
    num_groups = w.shape[0] * (in_features // group_size)
    assert mismatch.sum().item() <= num_groups, (
        f"{mismatch.sum().item()} mismatches for only {num_groups} groups"
    )


# ----------------------------------------------------------------------------
# Per-channel weights: the weight half of W8A8
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("num_bits", WEIGHT_BITS)
@pytest.mark.parametrize("out_features,in_features", SHAPES)
def test_channel_symmetric_qparams_are_bit_identical(
    out_features, in_features, num_bits
):
    """Per-channel weights must agree too, because W8A8 cannot use per-group.

    vLLM's ``_is_dynamic_token_w8a8`` and ``_is_static_tensor_w8a8`` accept only
    ``TENSOR`` or ``CHANNEL`` weight strategies, so the grid checked above under
    per-group says nothing about the scheme W8A8 actually serves. TorchAO reaches
    per-channel through ``PerAxis(0)`` rather than a group size.
    """
    from torchao.quantization.granularity import PerAxis

    w = _weight(out_features, in_features)

    s_ct, z_ct = _ct_qparams(w, None, symmetric=True, num_bits=num_bits)
    quantizer = IntxFakeQuantizer(
        IntxFakeQuantizeConfig(
            dtype=_AO_DTYPE[num_bits], granularity=PerAxis(0), is_symmetric=True
        )
    )
    quantizer(w)

    s_ao = quantizer.scale.view(s_ct.shape).to(s_ct.dtype)
    assert torch.equal(s_ct, s_ao), (
        f"per-channel scale mismatch: max rel diff "
        f"{((s_ct - s_ao).abs() / s_ct).max().item():.3e}"
    )
    assert z_ct.count_nonzero() == 0, "symmetric zero-point must be 0"


# ----------------------------------------------------------------------------
# Per-token activations: the other half of W8A8
# ----------------------------------------------------------------------------


def _activations(rows: int, cols: int, seed: int = 11) -> torch.Tensor:
    """Activations shaped like what a Linear actually sees.

    Deliberately post-SiLU rather than centred noise: real activations are skewed
    hard to one side, which is the case where a symmetric grid wastes half its
    range and where the asymmetric zero-point has to be non-zero to be correct.
    """
    torch.manual_seed(seed)
    return torch.nn.functional.silu(torch.randn(rows, cols) * 2).float()


def _vllm_per_token_asymmetric(x: torch.Tensor):
    """Reference for ``vllm._custom_ops.scaled_int8_quant(..., symmetric=False)``.

    Reimplemented rather than imported because vLLM lives in a separate
    environment from the training stack. Verified against the real op on an
    8x1024 post-SiLU tensor: scale matched bit for bit (max relative difference
    0.0) and every zero-point was identical.

    Note the range: 255 levels across [min, max], unlike the weight path which
    divides max_abs by ``2**(bits-1) - 0.5``. Two different conventions inside one
    W8A8 checkpoint, which is exactly why both halves need their own test.
    """
    minimum = x.amin(-1, keepdim=True)
    maximum = x.amax(-1, keepdim=True)
    scale = (maximum - minimum) / 255.0
    zero_point = -128 - torch.round(minimum / scale)
    return scale, zero_point


def test_per_token_activation_matches_vllm_formula():
    """TorchAO's per-token activation grid must equal the one vLLM computes.

    This is the half of W8A8 that has no checkpoint tensor to inspect: the
    activation scale is derived at runtime, so a mismatch here cannot be caught by
    verifying the exported weights. It would show up only as a model that trained
    against one activation grid and is served on another.
    """
    from torchao.quantization.granularity import PerToken

    x = _activations(8, 1024)
    quantizer = IntxFakeQuantizer(
        IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity=PerToken(), is_symmetric=False
        )
    )
    quantizer(x)

    scale_ref, zp_ref = _vllm_per_token_asymmetric(x)
    assert torch.equal(
        quantizer.scale.flatten().float(), scale_ref.flatten().float()
    ), (
        "per-token activation scale differs from vLLM's: max rel diff "
        f"{((quantizer.scale.flatten() - scale_ref.flatten()).abs() / scale_ref.flatten()).max().item():.3e}"
    )
    assert torch.equal(
        quantizer.zero_point.flatten().to(torch.int64),
        zp_ref.flatten().to(torch.int64),
    ), "per-token activation zero-point differs from vLLM's"


def test_per_token_activation_zero_point_is_real():
    """Tripwire: the per-token zero-point must not collapse the way per-group's does.

    ``test_asymmetric_qat_zero_point_collapses_upstream`` documents TorchAO
    truncating asymmetric per-group zero-points to 0. The per-token path does not
    share that bug -- it derives integer zero-points directly -- and W8A8 depends
    on that, because TorchAO has no symmetric per-token implementation to fall
    back on. If this ever starts failing, W8A8 QAT has silently lost its
    activation grid and must be disabled.
    """
    from torchao.quantization.granularity import PerToken

    x = _activations(8, 1024)
    quantizer = IntxFakeQuantizer(
        IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity=PerToken(), is_symmetric=False
        )
    )
    quantizer(x)
    assert quantizer.zero_point.count_nonzero() == quantizer.zero_point.numel(), (
        "per-token zero-points collapsed toward 0 -- asymmetric activations are no "
        "longer trustworthy, so W8A8 QAT has no correct activation grid"
    )


def test_symmetric_per_token_is_unsupported_upstream():
    """Tripwire: TorchAO cannot fake-quantise activations symmetrically.

    ``IntxFakeQuantizer._per_token_forward`` raises for ``is_symmetric=True``.
    That is why W8A8 must declare asymmetric activations even though vLLM accepts
    both -- and it is not merely a missing feature, because vLLM's symmetric
    per-token path divides max_abs by 127 while the weight path uses 127.5. If
    TorchAO implements it, check which convention it picked before switching.
    """
    from torchao.quantization.granularity import PerToken

    quantizer = IntxFakeQuantizer(
        IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity=PerToken(), is_symmetric=True
        )
    )
    with pytest.raises(NotImplementedError, match="[Ss]ymmetric per token"):
        quantizer(_activations(4, 256))


# ----------------------------------------------------------------------------
# Asymmetric weights: unsupported, and a tripwire for the bug that makes it so
# ----------------------------------------------------------------------------


def test_asymmetric_qat_zero_point_collapses_upstream():
    """Tripwire: TorchAO's asymmetric INT4 QAT silently degenerates to symmetric.

    ``IntxFakeQuantizer._per_channel_or_group_forward`` derives asymmetric
    qparams via ``get_groupwise_affine_qparams``, which returns a *float-domain*
    zero-point (tinygemm convention, e.g. 0.0119). It then casts that to
    ``zero_point_precision`` (torch.int32 by default) while
    ``zero_point_domain`` is ``ZeroPointDomain.INT`` -- so every zero-point
    truncates to 0 and the asymmetric grid collapses into a symmetric one that
    still carries an asymmetric scale.

    Verified against torchao 0.16.0. This test asserts the broken behaviour on
    purpose: when it starts failing, TorchAO has fixed the bug and asymmetric
    W4A16 becomes worth re-evaluating (Marlin does support it, via uint4).
    Until then, only symmetric QAT is supported -- see docs/SCOPE.md.
    """
    w = _weight(256, 512)
    group_size = 128

    config = IntxFakeQuantizeConfig(torch.int4, group_size=group_size, is_symmetric=False)
    assert config.zero_point_precision == torch.int32
    assert str(config.zero_point_domain) == "ZeroPointDomain.INT"

    _, zeros_float = get_groupwise_affine_qparams(w, 4, group_size, torch.float32)
    assert zeros_float.dtype.is_floating_point
    assert zeros_float.abs().max() < 1.0, (
        "float-domain zero-points are expected to be small; the truncation "
        "argument below depends on it"
    )

    quantizer = IntxFakeQuantizer(config)
    quantizer(w)
    assert quantizer.zero_point.count_nonzero() == 0, (
        "TorchAO asymmetric QAT now produces non-zero integer zero-points -- the "
        "upstream bug is fixed, re-evaluate asymmetric W4A16 support"
    )


def test_asymmetric_qat_does_not_match_compressed_tensors():
    """Consequence of the above: asymmetric QAT must not be used for serving.

    compressed-tensors derives genuine non-zero integer zero-points for the same
    weights, so the two grids differ. Guards against anyone wiring
    ``is_symmetric=False`` into the QAT config and assuming it round-trips.
    """
    w = _weight(256, 512)
    group_size = 128
    args = _ct_args(group_size, symmetric=False)

    s, z = _ct_qparams(w, group_size, symmetric=False)
    assert z.count_nonzero() > 0, "compressed-tensors should find real zero-points here"

    ct = fake_quantize(w, s, z, args)
    config = IntxFakeQuantizeConfig(torch.int4, group_size=group_size, is_symmetric=False)
    ao = IntxFakeQuantizer(config)(w)

    scale_expanded = s.repeat_interleave(group_size, dim=1)
    max_levels = ((ao - ct).abs() / scale_expanded).max().item()
    assert max_levels > 1.5, (
        "asymmetric QAT unexpectedly agrees with compressed-tensors -- if TorchAO "
        "fixed the zero-point domain bug, lift the symmetric-only restriction"
    )


# ----------------------------------------------------------------------------
# Tied embeddings
# ----------------------------------------------------------------------------


def _tied_model():
    """A tiny causal LM whose lm_head shares the embedding tensor."""
    transformers = pytest.importorskip("transformers")
    import json
    import tempfile
    from pathlib import Path

    directory = Path(tempfile.mkdtemp())
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "vocab_size": 2048,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": True,
                "dtype": "bfloat16",
                "layer_types": ["full_attention"] * 2,
            }
        )
    )
    config = transformers.AutoConfig.from_pretrained(directory)
    return transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)


def test_tied_weight_is_fake_quantised_consistently():
    """A tied weight must see one grid, whichever module reaches it.

    With ``tie_word_embeddings``, ``embed_tokens`` and ``lm_head`` are the same
    tensor used two ways: a row lookup and a matmul. QAT wraps them separately,
    so the question is whether the two fake quantisers agree. They must, or the
    shared weight is being pulled toward two different lattices at once.
    """
    from torchao_to_compressed_tensors import prepare_w4a16_qat

    model = _tied_model()
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    assert embed.weight.data_ptr() == head.weight.data_ptr(), "fixture must be tied"

    prepare_w4a16_qat(model, group_size=128, ignore=(), quantize_embeddings=True)
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()

    assert embed.weight.data_ptr() == head.weight.data_ptr(), "prepare broke the tie"
    assert torch.equal(
        embed.weight_fake_quantizer(embed.weight.float()),
        head.weight_fake_quantizer(head.weight.float()),
    ), "the two fake quantisers disagree on the shared weight"


def test_convert_restores_embeddings():
    """Regression: the convert step must unwrap embeddings, not just Linears.

    ``quantize_`` defaults to a Linear-only filter, so a
    ``FakeQuantizedEmbedding`` survives ``QATConfig(step="convert")`` unless a
    second pass targets it. The saved weights look fine either way -- the fake
    quantiser wraps a normal Parameter -- so this only shows up as a model that
    keeps quantising its lookups after conversion.
    """
    import torch.nn as nn

    from torchao_to_compressed_tensors import convert_w4a16_qat, prepare_w4a16_qat

    model = _tied_model()
    prepare_w4a16_qat(model, group_size=128, ignore=(), quantize_embeddings=True)
    convert_w4a16_qat(model)

    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    assert type(embed) is nn.Embedding, f"embedding left as {type(embed).__name__}"
    assert type(head) is nn.Linear, f"lm_head left as {type(head).__name__}"
    assert embed.weight.data_ptr() == head.weight.data_ptr(), "convert broke the tie"
