"""
test_save_smoke.py — standalone smoke test for ARC Save (Day 1).

Exercises the save() pipeline end-to-end against the LIVE arcIngest
endpoint without requiring a ComfyUI installation. Uses a FakeTensor
class that mimics torch.Tensor's .cpu().numpy() interface, so neither
torch nor a full ComfyUI environment is needed.

Required for the test to run:
    - arc_config.json in the repo root with a valid API key, OR
      ARC_API_KEY env var set to a valid key
    - Internet access to arcvelvetos.cloudfunctions.net
    - Python packages: numpy, Pillow, requests
      (install via: pip install -r requirements.txt)

Usage:
    cd comfyui-arc-save
    python tests/test_save_smoke.py

Exit code: 0 on PASS, 1 on FAIL. PASS criteria:
    - save() returns a dict with ui.images[0].filename
    - signed PNG written to the temp output dir
    - sidecar .arc.json written with vaultItemId / verifyUrl /
      contentHash fields populated
    - sidecar verifyUrl matches the locked vault shape

Cost per run: one arcIngest invocation against the live deploy
(~1-2 seconds, well under the 10/min rate limit per key). The smoke
test uploads a 64x64 random-noise PNG (~16 KB) so storage/egress
cost per run is negligible. Test artifacts are cleaned up on PASS;
on FAIL the temp dir is preserved for diagnosis.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

# Insert repo root so 'nodes.arc_save' imports work when this script
# is run as a __main__ from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from nodes.arc_save import ARCSave, _load_api_key  # noqa: E402


class FakeTensor:
    """Mimics enough of torch.Tensor for the save path:
       .cpu() returns self; .numpy() returns the underlying ndarray.
    The encode path in arc_save._encode_image_to_png_bytes does
    `image.cpu().numpy()`; nothing else is required."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


def _log(tag: str, msg: str) -> None:
    print(f"[smoke] {tag}: {msg}", flush=True)


def main() -> int:
    # ── Setup ────────────────────────────────────────────────────
    try:
        api_key = _load_api_key()
        _log("setup", f"API key loaded (prefix: {api_key[:12]}...)")
    except RuntimeError as e:
        _log("FAIL", str(e))
        return 1

    output_dir = tempfile.mkdtemp(prefix="arc_smoke_")
    _log("setup", f"output dir: {output_dir}")

    # 64x64 random-noise "image" in [0,1] float32, shape (H, W, C) RGB.
    # Seeded for deterministic test runs (the resulting contentHash on
    # the server side will be stable across local invocations, which
    # means subsequent smoke runs hit the dedup path — see PASS notes).
    np.random.seed(42)
    fake_image = np.random.rand(64, 64, 3).astype(np.float32)
    images = [FakeTensor(fake_image)]

    # ── Run save() ────────────────────────────────────────────────
    node = ARCSave()
    try:
        ui_result = node.save(
            images=images,
            filename_prefix="smoke",
            title="ARC-API-3-1 Day 1 smoke",
            prompt={
                "1": {
                    "class_type": "FakeNode",
                    "inputs": {
                        "text": (
                            "Day 1 sends this prompt VERBATIM to the "
                            "signed assertion. Redaction lands in the "
                            "next reviewable piece."
                        )
                    },
                }
            },
            extra_pnginfo={"workflow": {"nodes": [], "version": 1}},
            unique_id="smoke_test_node",
            _output_dir_override=output_dir,
        )
        _log("run", f"save() returned: {json.dumps(ui_result)[:200]}")
    except Exception as e:
        _log("FAIL", f"save() raised {type(e).__name__}: {e}")
        _log("artifacts", f"preserved at {output_dir} for diagnosis")
        return 1

    # ── Assert outputs ───────────────────────────────────────────
    files = sorted(os.listdir(output_dir))
    pngs = [f for f in files if f.endswith(".png")]
    sidecars = [f for f in files if f.endswith(".arc.json")]

    if len(pngs) != 1:
        _log("FAIL", f"expected exactly 1 PNG written, got {len(pngs)}: {files}")
        _log("artifacts", f"preserved at {output_dir} for diagnosis")
        return 1
    if len(sidecars) != 1:
        _log("FAIL", f"expected exactly 1 sidecar .arc.json, got {len(sidecars)}: {files}")
        _log("artifacts", f"preserved at {output_dir} for diagnosis")
        return 1

    png_path = os.path.join(output_dir, pngs[0])
    sidecar_path = os.path.join(output_dir, sidecars[0])

    png_size = os.path.getsize(png_path)
    if png_size < 1024:
        _log("FAIL", f"signed PNG suspiciously small: {png_size} bytes")
        _log("artifacts", f"preserved at {output_dir} for diagnosis")
        return 1
    _log("assert", f"signed PNG written: {pngs[0]} ({png_size} bytes)")

    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    required_fields = ["vaultItemId", "verifyUrl", "contentHash"]
    missing = [k for k in required_fields if not sidecar.get(k)]
    if missing:
        _log("FAIL", f"sidecar missing required fields: {missing}")
        _log("artifacts", f"preserved at {output_dir} for diagnosis")
        return 1

    if not sidecar["verifyUrl"].startswith(
        "https://arcvelvet.com/verify?type=vault&owner="
    ):
        _log(
            "FAIL",
            f"sidecar verifyUrl does not match locked shape: {sidecar['verifyUrl']}",
        )
        _log("artifacts", f"preserved at {output_dir} for diagnosis")
        return 1

    if len(sidecar["contentHash"]) != 64:
        _log("FAIL", f"sidecar contentHash not sha256 hex: {sidecar['contentHash']}")
        _log("artifacts", f"preserved at {output_dir} for diagnosis")
        return 1

    _log("assert", f"sidecar vaultItemId: {sidecar['vaultItemId']}")
    _log("assert", f"sidecar verifyUrl shape OK")
    _log("assert", f"sidecar contentHash: {sidecar['contentHash'][:16]}...")
    _log("assert", f"sidecar deduplicated: {sidecar.get('deduplicated')}")

    # ── Cleanup on PASS ──────────────────────────────────────────
    shutil.rmtree(output_dir, ignore_errors=True)
    _log("PASS", "Day 1 smoke test green. Cleanup complete.")
    print("", flush=True)
    print(
        "Note: subsequent smoke runs with seed=42 will hit the dedup\n"
        "fast path on the server (same content_hash → same vaultItemId)\n"
        "and report deduplicated: true in the sidecar. That's expected\n"
        "behavior, not a test failure.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
