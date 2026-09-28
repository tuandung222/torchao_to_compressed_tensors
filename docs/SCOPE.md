# Scope

This adapter supports **INT4 and INT8 weight-only, per-group, symmetric**
(`W4A16` and `W8A16`), at group sizes 32, 64 and 128.

Everything else was removed. This document records why, so the decisions can be
revisited when the hardware target or the upstream libraries change.

`W8A16` was added after `W4A16`, and it cost almost nothing to add because the
two are the same scheme at a different width — same `pack-quantized` format, same
`CompressedTensorsWNA16`, same Marlin kernel. See
[Adding INT8 weights](#adding-int8-weights). `W8A8` is *not* supported; it is a
different format with an activation scheme attached, and
[the section on it](#w8a8-is-a-separate-path-not-a-wider-w8a16) says what it
would take.

## The target

Serving on **NVIDIA A100 (sm80)** through vLLM's **Marlin** kernel.

Marlin on A100 is confirmed working by the maintainer on real hardware. The
table below was produced separately, by running vLLM's own kernel selector
(`choose_mp_linear_kernel`) with `compute_capability=80` — that is what pins
down *which* schemes reach Marlin, not merely that Marlin runs:

| Scheme | A100 kernel | H100 kernel |
| --- | --- | --- |
| W4A16 symmetric, group 128 | `MarlinLinearKernel` | `MacheteLinearKernel` |
| W4A16 asymmetric, group 128 | `MarlinLinearKernel` | `MacheteLinearKernel` |
| W4A16 symmetric, channelwise | `MarlinLinearKernel` | `MacheteLinearKernel` |
| W8A16 symmetric | `MarlinLinearKernel` | `MacheteLinearKernel` |
| W4A8 int | `HummingLinearKernel` | `HummingLinearKernel` |
| W2A16 / W3A16 | `HummingLinearKernel` | `HummingLinearKernel` |

The weight types Marlin accepts on sm80:

```
has_zp=False: uint4b8, uint8b128, float8_e4m3fn, float4_e2m1f
has_zp=True : uint4
```

And the minimum compute capability of each compressed-tensors scheme:

| Scheme | Min capability | A100 |
| --- | --- | --- |
| `wNa16` (W4A16, W8A16) | 75 | yes |
| `w8a8_int8` | 75 | yes, via CUTLASS rather than Marlin |
| `w8a16_fp8` | 75 | yes, but no faster than INT4 |
| `w8a8_fp8` | **89** | no |
| `w4a8_fp8` | **90** | no |
| `w4a4_nvfp4` / `w4a4_mxfp4` | 75 / 80 | emulated only; Blackwell formats |

## Adding INT8 weights

`W8A16` reuses every part of the `W4A16` path. What had to be checked before
relying on that, and what the checks found:

| Question | Answer |
| --- | --- |
| Same vLLM scheme? | Yes. `CompressedTensorsWNA16` takes `num_bits`; `WNA16_SUPPORTED_TYPES_MAP` maps 8 to `uint8b128` |
| Same compression format? | Yes, `pack-quantized`. `pack_to_int32(q, 8)` round-trips exactly; the pack factor goes from 8 values per int32 word to 4 |
| Marlin on sm80? | Yes, `uint8b128` is in the `has_zp=False` list |
| Scale formula identical to compressed-tensors? | Yes, bit for bit, per-group *and* per-channel |
| Lattice identical? | Within one level, as at 4 bits — but see below |

The one genuine difference is in the rounding ties. At 4 bits every disagreement
between TorchAO and compressed-tensors sits exactly on the ±7.5 half-way point,
so `assert not off_tie.any()` holds. At 8 bits the levels are 16× finer and two
elements out of 3,145,728 (a 1024×3072 weight at group 128) disagree *without*
being at the tie — float32 rounding near a boundary rather than a grid
divergence. `tests/test_grid_parity.py` carries a small per-width allowance
(`_OFF_TIE_ALLOWANCE`) with that measurement recorded, kept tight enough that a
real divergence still fails.

### Eight bits must be symmetric

`query_marlin_supported_quant_types(has_zp=True)` returns `[uint4]` and nothing
else. An asymmetric INT8 checkpoint therefore loads without complaint and then
runs on a slower kernel — a silent performance loss, which is why
`quantize.validate_num_bits` rejects it at export time instead.

## W8A8 is a separate path, not a wider W8A16

Supported through `--scheme w8a8`, as a second export path rather than another
`num_bits` value, because it differs at every layer of the stack:

| | W4A16 / W8A16 | W8A8 int8 |
| --- | --- | --- |
| Format | `pack-quantized` | `int-quantized` |
| Weight tensor | `weight_packed` (int32) | `weight` (plain int8) |
| Scale | per-group | per-channel or per-tensor, fp32 `(out, 1)` |
| Strategy | `GROUP` | `CHANNEL` or `TENSOR`; `GROUP` is rejected |
| Config | `weights` only | plus `input_activations` |
| Kernel | Marlin | CUTLASS int8 |
| QAT | weight fake-quant | weight *and* per-token activation fake-quant |

Two constraints found by measurement, both of which narrow the design to a single
viable combination:

* TorchAO cannot fake-quantise activations symmetrically —
  `IntxFakeQuantizer._per_token_forward` raises `NotImplementedError`. So the
  activations must be declared asymmetric, which vLLM does accept.
* That is fortunate, because the asymmetric per-token grids *do* match:
  `scale = (max - min) / 255` and the zero-point agree bit for bit between
  TorchAO and `vllm._custom_ops.scaled_int8_quant(symmetric=False)`, checked on a
  post-SiLU tensor. vLLM's *symmetric* per-token path divides `max_abs` by 127,
  a different convention from the weight path's 127.5, so if TorchAO ever
  implements symmetric per-token it must not be adopted without rechecking.

So the only correct W8A8 recipe is **symmetric per-channel weights with
asymmetric dynamic per-token activations**. The static per-tensor variant needs
activation calibration data, which belongs to llm-compressor rather than to a
QAT-export adapter.

vLLM confirms the routing: a W8A8 export of Qwen3.5-4B loads and generates, and
the log reads `Selected CutlassInt8ScaledMMLinearKernel for
CompressedTensorsW8A8Int8` — the intended scheme and kernel, not a fallback.

### W8A8 costs far more accuracy than W4A16

Measured on the v6-finetuned Qwen3.5-4B, perplexity over assistant tokens against
the bf16 model:

| Scheme | v6 train | val |
| --- | --- | --- |
| W8A16 (RTN or GPTQ) | −0.2% | −0.0% |
| W4A16 GPTQ | +5.1% | +3.0% |
| **W8A8** | **+34.8%** | see results/ |

Twice the weight bits of W4A16 and six times the damage. The weights are not the
problem — W8A16 is free — so all of it comes from quantising the activations. Any
throughput case for W8A8 has to be made against that, not against the weight
compression, which is worse than W4A16's anyway (int8 unpacked is the same size as
int8 packed, so W8A8 saves nothing over W8A16).

This also relocates where QAT might be worth the GPU hours. At 4 bits QAT lost to
GPTQ, and at 8 bits weight-only there is nothing left to recover. Activation
quantisation is the one place where a post-training method has no lever at all:
GPTQ compensates weight rounding error and cannot touch activations. Untested.

## What was removed

| Removed | Reason |
| --- | --- |
| FP8 weight-only and FP8 dynamic-activation | FP8 activations need sm89+. FP8 weight-only runs on A100 but loses to INT4 on both memory and speed. |
| INT8 static-activation | No Marlin path, and TorchAO QAT does not support it. |
| INT8 weight-only, INT8 dynamic-activation | W8A16 does reach Marlin, but INT4 dominates it on A100. Decisive factor: Axolotl's `get_quantization_config` raises `"Int8WeightOnlyConfig is not supported by torchao QAT"` and the same for `Int8DynamicActivationInt8WeightConfig`, so neither is reachable from the training stack at all. |
| INT4 preshuffled | An fbgemm/Hopper layout. Marlin repacks in `process_weights_after_loading`, so pre-shuffling is wasted work. |
| INT4 plain-int32 | An Intel XPU layout. |
| Sub-4-bit (2, 3, 5, 6, 7) | Not in Marlin's supported types; falls through to `HummingLinearKernel`. |

## Grid parity: the rule everything else follows

QAT is only worth running if the lattice it trains against is the lattice that
is served. `tests/test_grid_parity.py` asserts it directly, and two findings came
out of writing it.

**Symmetric INT4 matches exactly.** TorchAO and compressed-tensors both derive
`scale = max_abs / 7.5` with a zero-point of 0 over the range `[-8, 7]`. Scales
are bit-identical. Fake-quantised outputs agree everywhere except at the exact
`±7.5` rounding tie, which affects at most one element per group (~0.07% of
weights) by exactly one level -- float operation ordering, not a grid
disagreement.

**Asymmetric INT4 does not, because TorchAO's is broken.**
`IntxFakeQuantizer._per_channel_or_group_forward` derives asymmetric qparams via
`get_groupwise_affine_qparams`, which returns a *float-domain* zero-point
(tinygemm convention, e.g. `0.0119`). It then casts that to
`zero_point_precision` -- `torch.int32` by default -- while `zero_point_domain`
is `ZeroPointDomain.INT`. Every zero-point truncates to 0, so asymmetric QAT
silently degenerates into a symmetric grid carrying an asymmetric scale.
Verified against torchao 0.16.0; `test_asymmetric_qat_zero_point_collapses_upstream`
is a tripwire that fails when this is fixed upstream.

So: **QAT symmetric.** The adapter can still export asymmetrically
(`--asymmetric`, correctly packed) for checkpoints quantised elsewhere, but it is
not the supported path.

## Why dense input is preferred

The adapter accepts two source formats:

- **Dense** (`SourceFormat.DENSE`) -- the output of `QATConfig(step="convert")`
  with no base config, which swaps `FakeQuantizedLinear` back to `nn.Linear` and
  leaves plain bf16 weights carrying the QAT training. **Preferred.** The export
  grid is chosen once, here, and provably matches what is served.
- **TorchAO tinygemm INT4** (`SourceFormat.INT4_TINYGEMM`) -- for existing
  checkpoints. Supported, but *not* lossless: tinygemm uses a float zero-point
  and compressed-tensors requires an integer one, so re-quantisation rounds it.
  Its tile-packed 4D layout also only decodes on CUDA.

Detection is by checkpoint structure, not by TorchAO tensor class name. The
previous implementation matched on names like `AffineQuantizedTensor` and
`LinearActivationQuantizedTensor`; TorchAO replaced those with `Int8Tensor` and
`Float8Tensor`, after which every INT8 and FP8 layer silently fell through to a
passthrough branch. Unrecognised layouts now raise.

## What is *not* restricted

`lm_head` and embeddings are **not** hardcoded exclusions.

- `--ignore lm_head` is the default because it is the common convention, not a
  limitation. vLLM handles `ParallelLMHead` as a linear with the full scheme
  (`compressed_tensors.py`), so a quantised `lm_head` is served through the same
  Marlin path. Pass `--ignore` with no values to quantise the whole model.
- `--quantize-embeddings` emits a second config group targeting `["Embedding"]`,
  which vLLM serves via `CompressedTensorsEmbeddingWNA16Int`. Worth it mainly for
  large-vocabulary models.

  **Check the serving model implementation first.** Whether an embedding can be
  quantised is decided per model in vLLM, not by the config. Llama builds it as
  `VocabParallelEmbedding(..., quant_config=quant_config)`; Qwen3.5 (vLLM 0.28)
  omits that argument, so the embedding is always dense and loading a checkpoint
  that quantises it fails with *"no module or parameter named
  `embed_tokens.weight_packed`"*. The export itself is correct — verified at
  cosine 1.000000, and it takes a 2B checkpoint from 2.3 GB to 1.6 GB — so this
  is a one-line upstream gap rather than a format problem. On Qwen3.5 the
  embedding is 34% of the exported checkpoint, so it is worth checking whether
  the gap has been closed before giving it up.

Whatever is chosen must match what QAT used -- `prepare_w4a16_qat(..., ignore=...)`
and `convert_checkpoint(..., ignore=...)` take the same argument for that reason.
A module fake-quantised during training but served dense, or the reverse, is
trained against a grid it never sees.

One caveat: with `tie_word_embeddings` set, `lm_head` shares the embedding
tensor, so quantising embeddings while `lm_head` is not ignored leaves the
Linear/Embedding scheme pairing ambiguous. The adapter warns.

### `ignore` must survive module fusion

`ignore` is written into the config as `re:` patterns, not bare fragments.
`compressed_tensors.utils.match.match_name` compares targets *exactly* unless
they start with `re:`, so `ignore: ["visual"]` excludes nothing.

That matters because the serving runtime decides what to quantise from *its own*
module tree, which is not the checkpoint's. The adapter skipping a tensor does
not stop vLLM building a quantised layer for it; the load then fails looking for
`weight_packed` where the checkpoint only has `weight`, with an
`AttributeError` that names neither the module nor the cause.

Fusion makes the mismatch worse. On Qwen3.5 vLLM fuses `in_proj_a` and
`in_proj_b` into one `in_proj_ba`, and `in_proj_qkv` + `in_proj_z` into
`in_proj_qkvz`. compressed-tensors resolves this through vLLM's
`packed_modules_mapping`, but only for targets that actually match — so a
regex covering `in_proj_a` also excludes the fused `in_proj_ba`, while the bare
fragment excludes neither.

## Verified on a real model

Qwen3.5-2B (`Qwen3_5ForConditionalGeneration`: hybrid 18 linear-attention + 6
full-attention layers, vision tower, MTP head, `tie_word_embeddings`), with the
v6 LoRA merged in:

```
--ignore visual in_proj_a in_proj_b mtp   ->  150 Linear quantised
verify_conversion: 150 layers, cosine min = 1.000000 (bit-exact)
vLLM: loads, CompressedTensorsWNA16, generates coherent text
```

`in_proj_a`/`in_proj_b` are `(16, 2048)`. They are excluded on purpose: Marlin
requires `output_size_per_partition >= MIN_THREAD_N = 64`, which the Qwen3.5
non-interleaved layout violates. vLLM carries a workaround
(`maybe_disable_tp`) but it is gated on `AutoAWQConfig, AutoGPTQConfig,
INCConfig` — **not** `CompressedTensorsConfig`. They are also the SSM decay
gates, where 4 bits is reckless regardless.
