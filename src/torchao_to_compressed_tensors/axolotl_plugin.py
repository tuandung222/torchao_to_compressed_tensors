"""
Axolotl plugin: make ``qat: weight_dtype: int4`` train on the grid vLLM serves.

Add to an Axolotl config::

    plugins:
      - torchao_to_compressed_tensors.axolotl_plugin.W4A16QATPlugin

    w4a16_qat_ignore: [visual, in_proj_a, in_proj_b, mtp]

    qat:
      weight_dtype: int4
      activation_dtype: null
      group_size: 128

Without this plugin Axolotl maps INT4 weights onto
``Int4WeightFakeQuantizeConfig``, which simulates MSLK preshuffled-kernel
numerics with fp8 activations -- a different grid from the one a
compressed-tensors W4A16 checkpoint is served on. See :mod:`.qat`.

``w4a16_qat_ignore`` must match the ``--ignore`` given to the adapter at export
time. A module fake-quantised during training but served dense, or the reverse,
is trained against a grid it never sees.
"""

from axolotl.integrations.base import BasePlugin
from pydantic import BaseModel, Field

from .qat import DEFAULT_IGNORE, patch_axolotl_qat

__all__ = ["W4A16QATArgs", "W4A16QATPlugin"]


class W4A16QATArgs(BaseModel):
    """Config keys this plugin adds."""

    w4a16_qat_ignore: list[str] | None = Field(
        default=None,
        description=(
            "Module-name fragments to leave dense during QAT. Must match the "
            "adapter's --ignore at export time. Defaults to ['lm_head']."
        ),
    )


class W4A16QATPlugin(BasePlugin):
    """Redirects Axolotl's INT4 QAT to symmetric per-group numerics."""

    def get_input_args(self) -> str:
        return "torchao_to_compressed_tensors.axolotl_plugin.W4A16QATArgs"

    def pre_model_load(self, cfg) -> None:
        ignore = tuple(cfg.get("w4a16_qat_ignore") or DEFAULT_IGNORE)
        patch_axolotl_qat(ignore=ignore)
