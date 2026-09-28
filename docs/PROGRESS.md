# Progress and status

Last updated 2026-09-28.

Companion to [SCOPE.md](SCOPE.md), which records *what* is supported and why.
This file records what was measured, what it cost, and what is still unknown.

## Status

| Scheme | Export | Numerical verify | vLLM load + generate | Kernel vLLM selected |
| --- | --- | --- | --- | --- |
| W4A16 | yes | 202 layers, cosine 1.000000 | yes | `MacheteLinearKernel` |
| W8A16 | yes | 201 layers, cosine 1.000000 | yes | `MacheteLinearKernel` |
| W8A8 int8 | yes | 201 layers, cosine 1.000000 | yes | `CutlassInt8ScaledMMLinearKernel` |

89 tests pass, 63 of them in `tests/test_grid_parity.py`.

Machete rather than Marlin because the development box is an H200 (sm90). On the
A100 target (sm80) the selector reaches Marlin for both weight-only widths --
`uint4b8` and `uint8b128` are both in its supported list. That is read from
vLLM's `marlin_utils.py`, not observed on hardware; see [Limitations](#limitations).

## The question this project existed to answer

Does QAT beat post-training quantisation? It had never been measured. It has now,
on the v6-finetuned Qwen3.5-4B (4.84B parameters, untied lm_head, vocab 248320),
scored as perplexity over assistant tokens only -- the objective training actually
optimised, since `train_on_inputs` defaults to false.

```
                              v6 train                    val (held out)
                          ppl   vs bf16  top-1        ppl   vs bf16  top-1
  bf16                 1.5416     --     --        2.6114     --     --
  W8A16 RTN            1.5386   -0.2%  99.3%       2.6110   -0.0%  99.1%
  W8A16 GPTQ           1.5408   -0.1%  99.4%       2.6087   -0.1%  99.2%
  W4A16 GPTQ           1.6207   +5.1%  92.1%       2.6893   +3.0%  91.7%
  W4A16 QAT            1.6181   +5.0%  91.1%       3.0987  +18.7%  86.7%
  W4A16 RTN            1.6642   +8.0%  92.1%       2.7534   +5.4%  91.4%
  W8A8 (lm_head bf16)  2.0484  +32.9%  84.3%       3.3473  +28.2%  82.7%
  W8A8                 2.0788  +34.8%  84.0%       3.4553  +32.3%  81.8%
```

`top-1` is the fraction of positions where the quantised model would greedily emit
the same token as bf16. It catches damage that averages out of perplexity.

### QAT did not win, in two separate ways

Comparing a quantised model against bf16 conflates two costs: what quantisation
broke, and what the extra training epoch broke. Scoring the QAT checkpoint *before*
quantisation separates them.

| | v6 train | val |
| --- | --- | --- |
| Cost of quantisation alone -- QAT | +5.8% | **+6.5%** |
| Cost of quantisation alone -- GPTQ | +5.1% | **+3.0%** |
| Cost of the extra training epoch alone | -0.8% | **+11.4%** |

Robustness to 4-bit quantisation is the entire reason QAT exists, and it lost to
GPTQ on both sets. Separately, the extra epoch cost 11.4% of held-out ability to
buy 0.8% on the target distribution -- ordinary forgetting, unrelated to
quantisation. **4.5 GPU-hours produced a worse result than 20 minutes of GPTQ.**

One caveat worth keeping: this is one recipe (1 epoch, lr 1e-5, no replay data).
Mixing in original data or lowering the learning rate could fix the forgetting. It
would not fix the first finding, which is measured against QAT's own dense model.

### Quality does not follow bit count

**W8A16 ≈ bf16 ≫ W4A16 ≫ W8A8.**

W8A8 spends twice W4A16's weight bits and does six times the damage, while saving
nothing on disk -- unpacked int8 is exactly the size of int8 packed into int32.
The cause is isolated: W8A16 with the same 8-bit weights is free, so all of the
+33% comes from quantising activations. Excluding `lm_head` recovers only 2 points,
so the damage is spread across layers rather than concentrated in the logit
projection.

At 8 bits GPTQ is no better than round-to-nearest (1.5408 vs 1.5386). Calibration
buys nothing once the step size is small enough, so the "use GPTQ" conclusion
applies at 4 bits only.

| Checkpoint | Size |
| --- | --- |
| bf16 source | 9.0 GB |
| W4A16, embedding 4-bit | 2.4 GB |
| W4A16, embedding bf16 | 3.3 GB |
| W8A16, embedding 8-bit | 4.7 GB |
| W8A8 | 5.2 GB |

## Recommendations

**W8A16 round-to-nearest** when quality matters: indistinguishable from bf16, and
no calibration data is needed at all.

**W4A16 GPTQ** when size matters: +3-5% perplexity for 2.4 GB.

**Not W8A8**, unless a measured throughput win justifies +33% perplexity and the
worst top-1 agreement of any scheme tested. Its only argument is int8 tensor-core
speed; it is not a compression win.

**Not QAT** for weight-only at either width. Measured, twice, and it loses.

The one place QAT still has a principled advantage is **activation** quantisation,
because GPTQ compensates weight rounding error via a Hessian and has no lever on
activations at all. `qat: {weight_dtype: int8, activation_dtype: int8}` routes
through `patch_axolotl_qat` to `prepare_w8a8_qat` and is ready to run. Untested:
each attempt is ~4.5 GPU-hours against a +33% deficit.

## Defects found by running real models

Every one of these passed structural inspection. None produced an error where the
mistake was made.

| Defect | Consequence if unnoticed |
| --- | --- |
| Adapter omitted `lm_head` from `targets` | **vLLM cannot load any untied checkpoint.** `ParallelLMHead` contains neither "Linear" nor "Embedding", so vLLM's substring class match leaves it dense and the load dies on a missing `lm_head.weight` |
| llm-compressor writes an all-zero embedding `weight_scale` | Checkpoint saves, config is correct, size matches, 202 modules report compressed -- and every token dequantises to the zero vector. Detected as perplexity 248319.6 against a vocabulary of 248320 |
| Perplexity scored over whole sequences | bf16 measured 844 on its own training data, because it was graded on thousands of tool-schema tokens it was never trained to predict |
| `max_len` of 4096 | The first assistant turn starts at token 7898 of 9001, so nothing scoreable survived truncation |
| `tool_calls.arguments` stored as a JSON string | **47% of eval samples silently dropped** -- specifically the conversations with tool calls, which is what the model is for. Assistant tokens rose from 34,897 to 121,299 once fixed |
| Eval harness hard-coded 4 bits | An 8-bit checkpoint unpacked as 4-bit scores badly rather than failing |
| `pack_to_int32` is stride-sensitive on CUDA | A non-contiguous input packs in the wrong order: scales correct, 100% of int32 words wrong, weights plausible-looking and scrambled |

## Limitations

**Marlin has never been observed.** Both weight-only schemes are verified through
Machete on sm90. Marlin selection on sm80 is inferred from vLLM's source.

**W8A8 numbers are simulated.** Measured by fake-quantising in transformers using
the grids `test_grid_parity.py` proves match vLLM bit for bit, not by running the
CUTLASS int8 kernel. Accumulation order differs.

**A quantised embedding cannot be served on Qwen3.5.** vLLM's `qwen3_5.py` builds
`VocabParallelEmbedding` without passing `quant_config`, so no quantisation method
is ever attached. `llama.py` does pass it, so this is a per-model gap, fixable
upstream in one line. Until then every servable export leaves the embedding in
bf16, costing 1.27 GB.

**Tensor parallelism untested.** **MoE unsupported** -- vLLM routes `RoutedExperts`
to `CompressedTensorsMoEMethod`, a path this adapter does not produce.

**Static per-tensor W8A8 not implemented.** It needs calibrated activation scales,
which belongs to llm-compressor rather than to a QAT-export adapter.

## Reproducing

Experiment scripts live outside this repository, in `sprint27/qat_exp/scripts`
(not version-controlled):

| Script | Purpose |
| --- | --- |
| `run_untied_1gpu.sh` | QAT, single GPU, with restart-and-resume |
| `ptq_baseline.sh` | RTN export plus numerical verification |
| `gptq_baseline.sh` | GPTQ via llm-compressor, `NUM_BITS` selects 4 or 8 |
| `post_qat_untied.sh` | Export a QAT checkpoint in both comparison and servable forms |
| `eval_ppl.py` | Assistant-masked perplexity, top-1 agreement, W8A8 simulation |
| `run_comparison.sh` | Every model on both eval sets, against one pinned tokenizer |

Two things there are load-bearing and easy to get wrong. The tokenizer must be
pinned: the shared base checkpoint's chat template was replaced partway through the
first comparison, and runs either side of that edit rendered 155,108 versus 147,258
assistant tokens from the same 150 samples, making their perplexities
incomparable. And W8A8 must be scored by simulation rather than by decompressing
its export, because the export carries only the weight half of the scheme --
dequantising it measures W8A16 while appearing to measure W8A8.
