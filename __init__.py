"""
ComfyUI ARC Save — sign-on-arrival save nodes for ArcVelvetOS.

Registers two ComfyUI custom nodes:

  - ARCSave        — image-side, v1.0.0+ stable, terminal/trusted.
                     Replaces SaveImage; POSTs PNG to arcIngest,
                     writes signed PNG + sidecar receipt.

  - ARCSaveAudio   — audio-side, v1.1.0 alpha. Encodes ComfyUI AUDIO
                     to 16-bit PCM WAV, POSTs to arcIngest, writes
                     signed WAV + sidecar receipt. Text-prompt
                     moderation only — audio-content moderation is
                     not yet covered. See nodes/arc_save_audio.py
                     module docstring for the full alpha framing.

Loading is structured so the audio node is OPTIONAL: if the audio
module fails to import for any reason (a future torch/numpy/runtime
incompatibility, an upstream bug, anything), the image node remains
registered and functional. The image node is the trusted production
surface and must not be regressed by any audio-side change.

See README.md for setup (API key issuance, arc_config.json) and the
node's wire protocol against arcIngest.
"""

import sys

from .nodes.arc_save import (
    NODE_CLASS_MAPPINGS as _IMAGE_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_NODE_DISPLAY_NAMES,
)

NODE_CLASS_MAPPINGS: dict = dict(_IMAGE_NODE_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS: dict = dict(_IMAGE_NODE_DISPLAY_NAMES)

# AUDIO-1 — register the audio sibling node. Isolated in a try/except
# so a failure to import the audio module (e.g. a runtime env without
# the expected numpy/torch features the audio encoder uses) cannot
# break the image node's registration. The image node above is the
# trusted production surface and its availability must not depend on
# the audio node's import success.
try:
    from .nodes.arc_save_audio import (
        NODE_CLASS_MAPPINGS as _AUDIO_NODE_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _AUDIO_NODE_DISPLAY_NAMES,
    )
    NODE_CLASS_MAPPINGS.update(_AUDIO_NODE_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_AUDIO_NODE_DISPLAY_NAMES)
except Exception as e:  # noqa: BLE001 — broad on purpose for isolation
    print(
        f"[comfyui-arc-save] ARCSaveAudio node not registered "
        f"(image node remains available): {type(e).__name__}: {e}",
        file=sys.stderr,
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
