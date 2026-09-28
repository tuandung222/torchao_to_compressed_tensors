"""
Round-trip parity: what the adapter writes must decompress to what it promised.

The old suite compared the export against TorchAO with a cosine threshold of
0.990, which is loose enough to pass a conversion that had silently dropped the
zero-point. These tests instead decompress the exported tensors along exactly
the path ``PackedQuantizationCompressor`` and vLLM take, and require an exact
match against the reference quantisation of the same weights.

Runs on CPU.
"""

import json
from pathlib import Path

import pytest
import torch
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import calculate_qparams

try:
    from compressed_tensors.compressors.pack_quantized import unpack_from_int32
except ImportError:  # pragma: no cover
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        unpack_from_int32,
    )

from torchao_to_compressed_tensors import (
    convert_checkpoint,
    detect_source_format,
    extract_group_size,
    make_quantization_args,
    quantize_weight,
)

GROUP_SIZE = 128


def _tiny_config(**overrides) -> dict:
    config = {
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
    config.update(overrides)
    return config


@pytest.fixture(scope="module")
def dense_checkpoint(tmp_path_factory) -> Path:
    """A small dense checkpoint, standing in for a QAT run after convert."""
    transformers = pytest.importorskip("transformers")
    source = tmp_path_factory.mktemp("dense_src")
    (source / "config.json").write_text(json.dumps(_tiny_config()), encoding="utf-8")

    config = transformers.AutoConfig.from_pretrained(source)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.save_pretrained(source)
    return source


def _load_exported(output_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    tensors = {}
    for shard in sorted(output_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def _decompress(tensors: dict[str, torch.Tensor], prefix: str, symmetric: bool):
    """Decompress one layer exactly as PackedQuantizationCompressor.decompress does.

    ``args`` is passed explicitly: compressed-tensors >= 0.19 refuses to infer
    the strategy from tensor shapes.
    """
    args = make_quantization_args(GROUP_SIZE, symmetric)
    packed = tensors[f"{prefix}.weight_packed"]
    scale = tensors[f"{prefix}.weight_scale"]
    shape = torch.Size(tensors[f"{prefix}.weight_shape"].tolist())

    zero_point = None
    if not symmetric:
        zp_shape = (shape[0], scale.shape[-1])
        zero_point = unpack_from_int32(
            tensors[f"{prefix}.weight_zero_point"], 4, zp_shape, packed_dim=0
        )

    unpacked = unpack_from_int32(packed, 4, shape)
    return dequantize(x_q=unpacked, scale=scale, zero_point=zero_point, args=args)


# ----------------------------------------------------------------------------
# Export parity
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("symmetric", [True, False])
def test_exported_weights_decompress_exactly(dense_checkpoint, tmp_path, symmetric):
    """Decompressed weights must equal the reference quantisation, bit for bit."""
    output = tmp_path / f"out_{symmetric}"
    convert_checkpoint(
        dense_checkpoint,
        output,
        group_size=GROUP_SIZE,
        symmetric=symmetric,
        device_str="cpu",
    )

    source = _load_exported(dense_checkpoint)
    exported = _load_exported(output)

    prefixes = sorted(
        k[: -len(".weight_packed")] for k in exported if k.endswith(".weight_packed")
    )
    assert prefixes, "nothing was quantised"

    args = make_quantization_args(GROUP_SIZE, symmetric)
    for prefix in prefixes:
        original = source[f"{prefix}.weight"].float()
        grouped = original.view(original.shape[0], -1, GROUP_SIZE)
        scale, zero_point = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
        # The scale is stored in the model's dtype, and quantisation must use
        # the stored value -- see quantize_weight.
        scale = scale.to(exported[f"{prefix}.weight_scale"].dtype)

        # quantize + dequantize, not fake_quantize: the fused QDQ path breaks
        # rounding ties differently on CUDA (~0.1% of elements, one level), and
        # `quantize` is what PackedQuantizationCompressor.compress actually
        # calls. The two agree on CPU, so using fake_quantize here would pass
        # while silently disagreeing with the real compressor.
        quantized = quantize(
            x=original, scale=scale, zero_point=zero_point, args=args, dtype=torch.int8
        )
        reference = dequantize(
            x_q=quantized,
            scale=scale,
            zero_point=None if symmetric else zero_point,
            args=args,
        )

        recovered = _decompress(exported, prefix, symmetric)
        assert torch.equal(recovered, reference), f"{prefix} does not round-trip"


def test_scale_dtype_matches_model_dtype(dense_checkpoint, tmp_path):
    """Regression: ``weight_scale`` must be stored in the model's dtype.

    Decompression returns the weight in the scale's dtype, so a float32 scale on
    a bfloat16 model produces a float32 weight and the first matmul fails with
    ``expected mat1 and mat2 to have the same dtype``. vLLM casts parameters on
    load and so never sees it; transformers does.
    """
    output = tmp_path / "out_scale_dtype"
    convert_checkpoint(
        dense_checkpoint, output, group_size=GROUP_SIZE, device_str="cpu"
    )

    source = _load_exported(dense_checkpoint)
    exported = _load_exported(output)
    model_dtype = next(
        v.dtype for k, v in source.items() if k.endswith(".weight") and v.ndim == 2
    )
    assert model_dtype == torch.bfloat16, "fixture is meant to be bfloat16"

    scales = [v for k, v in exported.items() if k.endswith(".weight_scale")]
    assert scales
    assert all(s.dtype == model_dtype for s in scales), (
        f"scales are {  {s.dtype for s in scales} }, expected {model_dtype}"
    )


def test_ignore_is_written_as_matchable_patterns(dense_checkpoint, tmp_path):
    """Regression: bare fragments in ``ignore`` never match, so they must be regexes.

    ``compressed_tensors.utils.match.match_name`` compares a target *exactly*
    unless it starts with ``re:``. Writing ``ignore: ["visual"]`` therefore
    excludes nothing: the serving runtime builds a quantised layer for a module
    this adapter left dense, then fails at load looking for ``weight_packed``
    where the checkpoint only has ``weight``. Observed on Qwen3.5, where vLLM
    fuses ``in_proj_a``/``in_proj_b`` into ``in_proj_ba``.
    """
    from compressed_tensors.utils.match import match_name

    output = tmp_path / "out_ignore_patterns"
    convert_checkpoint(
        dense_checkpoint,
        output,
        group_size=GROUP_SIZE,
        ignore=("down_proj", "re:.*already_a_regex.*"),
        device_str="cpu",
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    patterns = config["quantization_config"]["ignore"]
    assert all(p.startswith("re:") for p in patterns), patterns
    assert "re:.*already_a_regex.*" in patterns, "existing regexes must pass through"

    # The pattern must match a real, fully-qualified module name.
    assert any(
        match_name("model.layers.0.mlp.down_proj", p) for p in patterns
    ), patterns

    # And it must reach the fused module vLLM actually builds.
    assert any(
        match_name(
            "model.layers.0.mlp.gate_up_proj",
            p,
            fused={"gate_up_proj": ["gate_proj", "down_proj"]},
        )
        for p in patterns
    ), patterns


def test_tied_lm_head_is_added_to_ignore(dense_checkpoint, tmp_path):
    """Regression: the config must not promise tensors the checkpoint lacks.

    With ``tie_word_embeddings`` set, ``lm_head`` shares the embedding and has no
    tensor of its own, so there is nothing to quantise and nothing written. If
    the config still targets it, transformers builds a quantised ``lm_head``,
    finds no ``weight_packed``, and dies with ``Linear has no attribute
    'weight'``. vLLM happens to tolerate it, which is worse -- the checkpoint
    looks fine until someone loads it the other way.

    Uses an empty ``ignore`` so nothing covers lm_head by request; the adapter
    has to work out that it was left dense.
    """
    output = tmp_path / "out_tied_lm_head"
    convert_checkpoint(
        dense_checkpoint, output, group_size=GROUP_SIZE, ignore=(), device_str="cpu"
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["tie_word_embeddings"] is True

    exported = _load_exported(output)
    assert not any("lm_head" in k for k in exported), "nothing to quantise for lm_head"

    from compressed_tensors.utils.match import match_name

    patterns = config["quantization_config"]["ignore"]
    assert any(match_name("lm_head", p) for p in patterns), (
        f"lm_head is unquantised but not ignored: {patterns}"
    )


def test_exported_config_is_valid(dense_checkpoint, tmp_path):
    """The generated block must parse back into compressed-tensors' own model."""
    output = tmp_path / "out_config"
    convert_checkpoint(
        dense_checkpoint, output, group_size=GROUP_SIZE, device_str="cpu"
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    parsed = QuantizationConfig.model_validate(config["quantization_config"])

    assert parsed.format == "pack-quantized"
    weights = parsed.config_groups["group_0"].weights
    assert (weights.num_bits, weights.group_size) == (4, GROUP_SIZE)
    assert weights.strategy == "group"
    assert parsed.config_groups["group_0"].targets == ["Linear"]


def test_symmetric_export_omits_zero_point(dense_checkpoint, tmp_path):
    """A symmetric checkpoint must not carry zero-points; Marlin picks uint4b8."""
    output = tmp_path / "out_sym"
    convert_checkpoint(
        dense_checkpoint, output, group_size=GROUP_SIZE, symmetric=True, device_str="cpu"
    )
    exported = _load_exported(output)
    assert not any(k.endswith(".weight_zero_point") for k in exported)


def test_asymmetric_zero_point_is_packed_int32(dense_checkpoint, tmp_path):
    """Regression: zero-points must be packed along out_features as int32.

    Writing a raw int8 zero-point makes both ``unpack_from_int32`` and vLLM's
    ``PackedvLLMParameter`` reject the checkpoint at load.
    """
    output = tmp_path / "out_asym"
    convert_checkpoint(
        dense_checkpoint,
        output,
        group_size=GROUP_SIZE,
        symmetric=False,
        device_str="cpu",
    )
    exported = _load_exported(output)

    zp_keys = [k for k in exported if k.endswith(".weight_zero_point")]
    assert zp_keys
    for key in zp_keys:
        prefix = key[: -len(".weight_zero_point")]
        zero_point = exported[key]
        out_features = int(exported[f"{prefix}.weight_shape"][0])
        num_groups = exported[f"{prefix}.weight_scale"].shape[-1]

        assert zero_point.dtype == torch.int32
        assert tuple(zero_point.shape) == (-(-out_features // 8), num_groups)


def test_ignored_modules_stay_dense(dense_checkpoint, tmp_path):
    """``--ignore`` must leave the named modules untouched, and ``()`` must not skip."""
    with_ignore = tmp_path / "out_ignore"
    convert_checkpoint(
        dense_checkpoint,
        with_ignore,
        group_size=GROUP_SIZE,
        ignore=("down_proj",),
        device_str="cpu",
    )
    exported = _load_exported(with_ignore)

    assert not any("down_proj.weight_packed" in k for k in exported)
    assert any(k.endswith("down_proj.weight") for k in exported)
    assert any("gate_proj.weight_packed" in k for k in exported)


def test_embeddings_quantised_on_request(dense_checkpoint, tmp_path):
    """Embeddings are opt-in and get their own config group targeting Embedding."""
    output = tmp_path / "out_embed"
    convert_checkpoint(
        dense_checkpoint,
        output,
        group_size=GROUP_SIZE,
        ignore=("lm_head",),
        quantize_embeddings=True,
        device_str="cpu",
    )

    exported = _load_exported(output)
    assert any("embed_tokens.weight_packed" in k for k in exported)

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    groups = config["quantization_config"]["config_groups"]
    assert groups["embedding"]["targets"] == ["Embedding"]


def test_tied_embeddings_are_not_duplicated(dense_checkpoint, tmp_path):
    """A tied model must not gain a redundant lm_head copy of the vocab matrix."""
    output = tmp_path / "out_tied"
    convert_checkpoint(
        dense_checkpoint, output, group_size=GROUP_SIZE, device_str="cpu"
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["tie_word_embeddings"] is True
    assert not any("lm_head" in k for k in _load_exported(output))


def test_safetensors_metadata_present(dense_checkpoint, tmp_path):
    """transformers rejects a checkpoint whose metadata lacks a known format."""
    from safetensors import safe_open

    output = tmp_path / "out_meta"
    convert_checkpoint(
        dense_checkpoint, output, group_size=GROUP_SIZE, device_str="cpu"
    )
    with safe_open(output / "model.safetensors", framework="pt") as f:
        assert f.metadata() == {"format": "pt"}


def test_sharding_emits_index(dense_checkpoint, tmp_path):
    """Above the shard threshold the export must be sharded with a valid index."""
    output = tmp_path / "out_shard"
    convert_checkpoint(
        dense_checkpoint,
        output,
        group_size=GROUP_SIZE,
        device_str="cpu",
        max_shard_bytes=64 * 1024,
    )

    index_path = output / "model.safetensors.index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))

    shards = {p.name for p in output.glob("*.safetensors")}
    assert len(shards) > 1
    assert set(index["weight_map"].values()) == shards
    assert set(index["weight_map"]) == set(_load_exported(output))


# ----------------------------------------------------------------------------
# Source inspection regressions
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quantization_config,expected",
    [
        # The layout TorchAO actually writes: the group size is three levels deep.
        # The old extractor looked one level down, missed it, and silently
        # defaulted to 128 -- so a group_size=64 checkpoint exported as garbage.
        ({"quant_type": {"default": {"_data": {"group_size": 64}}}}, 64),
        ({"quant_type": {"group_size": 32}}, 32),
        ({"quant_type": {"default": {"_data": {"group_size": 128}}}}, 128),
    ],
)
def test_group_size_found_at_any_nesting(tmp_path, quantization_config, expected):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"quantization_config": quantization_config}), encoding="utf-8"
    )
    assert extract_group_size(config_path) == expected


def test_group_size_refuses_to_guess(tmp_path):
    """No silent default: an unreadable group size must stop the conversion."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"quantization_config": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="group_size"):
        extract_group_size(config_path)

    assert extract_group_size(config_path, override=64) == 64


def test_unknown_quantised_source_is_rejected():
    """Unrecognised layouts must raise, not pass through as opaque tensors.

    The previous detector matched TorchAO tensor subclasses by class name. When
    TorchAO renamed them, every INT8 and FP8 layer fell through to a passthrough
    branch and was written into the checkpoint unconverted.
    """

    class FakeQuantizedTensor(torch.Tensor):
        pass

    state_dict = {"model.layers.0.mlp.down_proj.weight": FakeQuantizedTensor()}
    with pytest.raises(ValueError, match="Unsupported quantised tensor types"):
        detect_source_format(state_dict)


def test_dense_source_detected():
    state_dict = {"model.layers.0.mlp.down_proj.weight": torch.randn(64, 128)}
    assert detect_source_format(state_dict).value == "dense"


def test_tinygemm_source_detected():
    state_dict = {
        "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 2, 32, 4, dtype=torch.int32),
        "model.layers.0.mlp.down_proj.scales_and_zeros": torch.zeros(2, 64, 2),
    }
    assert detect_source_format(state_dict).value == "int4_tinygemm"


def test_group_size_must_be_marlin_servable(dense_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="not servable by Marlin"):
        convert_checkpoint(
            dense_checkpoint, tmp_path / "bad", group_size=256, device_str="cpu"
        )


def test_quantize_weight_rejects_indivisible_group(tmp_path):
    with pytest.raises(ValueError, match="not divisible"):
        quantize_weight(torch.randn(64, 100), 128)


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs a CUDA device"
            ),
        ),
    ],
)
def test_quantize_weight_handles_non_contiguous_input(device, symmetric):
    """Regression: a transposed view must quantise identically to its clone.

    Dequantising a tinygemm checkpoint ends in ``.t()``, which yields a
    transposed view. ``pack_to_int32`` is stride-sensitive **on CUDA only**: given
    a non-contiguous input it packs elements in the wrong order, and every int32
    word comes out different while the scales stay correct. The export then
    decompresses to weights with the right shape, dtype and value distribution,
    but scrambled.

    The CPU path is unaffected, so this is invisible to a CPU-only suite while
    the conversion itself normally runs on GPU. The ``cuda`` parameter is the one
    that matters; the ``cpu`` one only pins the invariant.
    """
    contiguous = torch.randn(256, 512, device=device)
    transposed = contiguous.t().contiguous().t()
    assert not transposed.is_contiguous()
    assert torch.equal(transposed, contiguous)

    from_view = quantize_weight(transposed, GROUP_SIZE, symmetric=symmetric)
    from_clone = quantize_weight(contiguous, GROUP_SIZE, symmetric=symmetric)

    assert torch.equal(from_view.weight_packed, from_clone.weight_packed)
    assert torch.equal(from_view.weight_scale, from_clone.weight_scale)
    if symmetric:
        assert from_view.weight_zero_point is None
    else:
        assert torch.equal(from_view.weight_zero_point, from_clone.weight_zero_point)
