# torchao-to-compressed-tensors

Convert TorchAO QAT checkpoints into [compressed-tensors](https://github.com/vllm-project/compressed-tensors)
`W4A16`, for serving on **NVIDIA A100** through vLLM's **Marlin** kernel.

One scheme: INT4 weight-only, per-group, symmetric, at group sizes 32 / 64 / 128.
[docs/SCOPE.md](docs/SCOPE.md) records what was removed and why.

## Why this exists

There is no framework that trains QAT and emits compressed-tensors directly.
llm-compressor ships only `oneshot` and `model_free` entrypoints, and every
quantisation forward in compressed-tensors is decorated `@torch.no_grad()` — the
stack is post-training only. So QAT happens in TorchAO, and the checkpoint has to
cross over.

## The rule everything follows

**The grid QAT trains against must be the grid that is served.** A checkpoint can
be structurally perfect and still be worthless if the model was optimised for a
different lattice.

`tests/test_grid_parity.py` asserts this directly and is the most important test
here. Two results came out of writing it:

- **Symmetric INT4 matches exactly.** TorchAO and compressed-tensors both use
  `scale = max_abs / 7.5`, zero-point 0, range `[-8, 7]`. Scales are
  bit-identical; fake-quantised outputs agree everywhere except the `±7.5`
  rounding tie (~0.07% of weights, one level, float operation ordering).
- **Asymmetric INT4 does not, because TorchAO's is broken.** Its asymmetric QAT
  derives a float-domain zero-point then casts it to `torch.int32`, truncating
  every value to 0 — so asymmetric QAT silently degenerates into a symmetric grid
  carrying an asymmetric scale. Verified on torchao 0.16.0; there is a tripwire
  test that fails once upstream fixes it.

Hence: **QAT symmetric.**

## Install

```bash
pip install -e .
```

## Use

### 1. QAT

Either through this package directly:

```python
from torchao_to_compressed_tensors import prepare_w4a16_qat, convert_w4a16_qat

prepare_w4a16_qat(model, group_size=128)   # swap in fake quantisers
...                                         # train as usual
convert_w4a16_qat(model)                    # back to plain bf16 weights
model.save_pretrained("qat_dense")
```

Or through Axolotl, via the bundled plugin:

```yaml
plugins:
  - torchao_to_compressed_tensors.axolotl_plugin.W4A16QATPlugin

w4a16_qat_ignore: [lm_head]   # must match the adapter's --ignore

qat:
  weight_dtype: int4
  activation_dtype: null
  group_size: 128
```

Without the patch, Axolotl maps INT4 weights onto `Int4WeightFakeQuantizeConfig`,
which despite the name simulates MSLK preshuffled-kernel numerics and defaults to
fp8 input activations — a Hopper/fbgemm grid, not Marlin's. It also
fake-quantises `lm_head`, which the exported config leaves dense by default. Both
mismatches are silent; they show up as "QAT barely helped".

Note that `activation_dtype: int8` gives correct weight numerics but produces
W4A8, which on A100 falls through to `HummingLinearKernel` rather than Marlin.

### 2. Convert

```bash
torchao-to-ct --source qat_dense --output-dir model-w4a16 --group-size 128
```

### 3. Verify, then serve

```bash
# numerical: does the export hold the weights the source implies? (exit 1 on fail)
python examples/verify_conversion.py --source qat_dense --converted model-w4a16

# structural + generation
python examples/verify_inference.py --model-dir model-w4a16

vllm serve model-w4a16
```

**Run `verify_conversion.py`, not just `verify_inference.py`.** A conversion bug
that reorders elements yields a checkpoint with correct shapes, correct dtypes, a
correct value distribution and a valid config. It loads cleanly in transformers
*and* in vLLM and generates fluent-looking tokens; every structural check passes.
Only the per-layer comparison against the source catches it — cosine is the right
detector because it is sensitive to permutation, scoring ~0 on a scrambled layer
where a correct one scores >0.999.

This is not hypothetical: `pack_to_int32` is stride-sensitive on CUDA and silently
mispacks a non-contiguous input, which is exactly what dequantising a tinygemm
checkpoint produces. See `test_quantize_weight_handles_non_contiguous_input`.

Kernel selection is decided when vLLM loads the checkpoint, so confirm the log
line on the target GPU:

```
Using MarlinLinearKernel for CompressedTensorsWNA16
```

On A100 this is the expected line and Marlin there is confirmed working. On
newer hardware the same checkpoint is served by a different kernel — an H200
reports `MacheteLinearKernel`, which is correct, not a misconfiguration.

`examples/run_pipeline.sh` runs all three steps.

## Quantising the whole model

`lm_head` and embeddings are **not** hardcoded exclusions. `--ignore lm_head` is
the default because it is the common convention.

```bash
# quantise everything, lm_head included
torchao-to-ct --source qat_dense --output-dir out --ignore

# also quantise the embedding table (worth it for large vocabularies)
torchao-to-ct --source qat_dense --output-dir out --quantize-embeddings
```

vLLM serves a quantised `lm_head` through the same Marlin path as any other
Linear, and quantised embeddings through `CompressedTensorsEmbeddingWNA16Int`.

Whatever you choose must match what QAT used — `prepare_w4a16_qat(..., ignore=...)`
takes the same argument. A module fake-quantised during training but served dense,
or the reverse, is trained against a grid it never sees.

## Source formats

| Format | Detected by | Notes |
| --- | --- | --- |
| Dense fp16/bf16 | no quantisation sidecar | **Preferred.** Output of `QATConfig(step="convert")`. Converts on CPU. |
| TorchAO tinygemm INT4 | `.scales_and_zeros` | For existing checkpoints. Not lossless — tinygemm's zero-point is float, compressed-tensors' is integer. Its tile-packed 4D layout decodes on CUDA only. |

Anything else raises rather than passing through silently.

## Tests

```bash
pytest tests/          # CPU only
```

- `test_grid_parity.py` — QAT grid vs serving grid
- `test_roundtrip.py` — exported tensors decompress to exactly the reference
  quantisation, along the same path `PackedQuantizationCompressor` and vLLM take

## Limitations

- INT4 per-group only, group size 32 / 64 / 128. Other sizes are rejected at
  config time rather than silently falling back to a slower kernel.
- Asymmetric export exists (`--asymmetric`, correctly packed) but is not the
  supported path while TorchAO's asymmetric QAT is broken.
- tinygemm sources need a CUDA device.
- With `tie_word_embeddings` set, quantising embeddings while `lm_head` is not
  ignored leaves the Linear/Embedding scheme pairing ambiguous. The adapter warns.
