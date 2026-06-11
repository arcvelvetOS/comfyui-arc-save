"""
arc_save.py — ARC-API-3-1 Day 1 scope

The ARC Save ComfyUI custom node. Encodes the input IMAGE tensor to
PNG bytes (matching SaveImage's encoding path exactly), POSTs to the
live ArcVelvet arcIngest endpoint, downloads the signed copy, writes
the signed PNG plus a sidecar .arc.json receipt to the output
directory.

Current scope (Day 1 + batch/retry + redaction):
    - Per-image loop across the IMAGE batch tensor.
    - Single retry with 2-second backoff on TRANSIENT errors only
      (HTTP 503 + requests.Timeout + requests.ConnectionError).
      Every other non-200 (401/403/413/415/429/500/etc.) raises
      immediately — no retry for terminal failures.
    - Prompt redaction (REDACTION PASS, this commit):
        - include_prompt_text widget in INPUT_TYPES, default False
          (redact). True bypasses redaction; prompt text passes
          verbatim into the cryptographic assertion.
        - When False: walk the PROMPT graph, find any node whose
          class_type matches the configured pattern list, and
          replace string values in known text-input field names
          with SHA-256 hashes envelope-wrapped as
              "[REDACTED:sha256:<64hex>]"
          The workflow structure is preserved; only the text
          payloads are hashed. A creator who later wants to prove
          a specific prompt produced this image can reveal the
          plaintext and any relying party hashes it to verify.
        - Pattern list = LOCKED_CATCHALL_PATTERN ("textencode",
          case-insensitive substring) PLUS the optional
          text_encoder_patterns field in arc_config.json. The
          catch-all is fail-closed — new/unknown CLIPTextEncode-
          like variants redact rather than leak. Creators can
          ADD patterns; they cannot REMOVE the catch-all.
        - generation assertion gains a meaningful
          redacted_prompt: bool field so relying parties can
          tell whether text fields in workflow_prompt are
          plaintext or hashes.
    - arc_config.json + ARC_API_KEY env var key loading
    - Sidecar .arc.json with vaultItemId / verifyUrl / contentHash

Out of current scope (later pieces):
    - Manager / Registry packaging (pyproject.toml for the
      Comfy Registry + a PR against the ComfyUI Manager
      custom-node-list.json)

Additive-fingerprint slot (ARC-API-3-0 addendum):
    The generation assertion is built as an open dict. A future
    perceptual-fingerprint block can be added as a top-level
    namespaced key (e.g. "fingerprint": {...}) without restructuring
    any of the call sites here. Same compatibility posture as
    com.arcvelvet.generation's loose-typed union member on the server.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from typing import Any
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image


# ─── Constants ─────────────────────────────────────────────────────

# Live arcIngest endpoint deployed under ARC-API-2 Commit 4.
INGEST_URL = "https://us-central1-arcvelvetos.cloudfunctions.net/arcIngest"

# Request timeouts (seconds). The benchmark from ARC-API-1 put sign
# latency at <1s warm even for 20 MB inputs; the 60s ceiling here
# absorbs cold start + upload time + headroom.
INGEST_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 60

# Retry posture for transient ingest failures. Operator-locked:
# single retry on 503 / Timeout / ConnectionError, 2-second backoff
# between attempts. Any other status code (auth, rate-limit, payload-
# too-large, sign-failed) is terminal — no retry.
RETRY_BACKOFF_SECONDS = 2.0

# Compression level for PIL Image.save (matches SaveImage's
# compress_level=4 default for output files; PreviewImage uses 1).
PNG_COMPRESS_LEVEL = 4

# ── Redaction config ──────────────────────────────────────────────
# LOCKED catch-all substring (case-insensitive). Cannot be removed
# via arc_config.json — creators can only ADD patterns. Fail-closed
# default: any class_type containing "textencode" gets redacted.
LOCKED_CATCHALL_PATTERN = "textencode"

# Field names whose string values get hashed when the parent node
# matches the encoder pattern list. Covers vanilla CLIPTextEncode
# ("text"), CLIPTextEncodeSDXL ("text_g", "text_l"), and the common
# positive/negative split used by many workflows. Lowercase for
# case-insensitive matching against arbitrary case (e.g. "Text",
# "PROMPT") in custom encoder shapes.
DEFAULT_TEXT_INPUT_FIELDS_LOWER = (
    "text",
    "prompt",
    "text_g",
    "text_l",
    "positive",
    "negative",
    "string",
)

# Envelope around each hash so a relying party reading the assertion
# can identify the redaction unambiguously. The envelope IS structured
# so anyone who later receives the plaintext can hash it and check
# match without parsing tooling.
_REDACTED_PREFIX = "[REDACTED:sha256:"
_REDACTED_SUFFIX = "]"


# ─── Config + API key loading ──────────────────────────────────────


def _load_arc_config() -> dict:
    """Read arc_config.json from the repo root if present. Returns
    {} if absent. Raises RuntimeError on parse failure so a typo'd
    config surfaces immediately rather than silently falling back
    to defaults.

    Read on every workflow execution — no caching. Config-edit
    iterations land without ComfyUI restart at the cost of two
    extra disk reads per execution (negligible).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, "arc_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(
            f"ArcVelvet: arc_config.json exists but failed to parse: {e}"
        )


def _load_api_key() -> str:
    """Load the API key from arc_config.json in the repo root, or
    fall back to the ARC_API_KEY env var. Raises RuntimeError with a
    creator-facing setup message if neither is present.

    The key is read at execute time. It NEVER appears in INPUT_TYPES,
    so it cannot leak into the workflow JSON on share/export.
    """
    cfg = _load_arc_config()
    key = cfg.get("api_key")
    if isinstance(key, str) and key.startswith("arc_live_"):
        return key
    env_key = os.environ.get("ARC_API_KEY")
    if env_key and env_key.startswith("arc_live_"):
        return env_key
    raise RuntimeError(
        "ArcVelvet API key not configured. Create arc_config.json with "
        '{"api_key": "arc_live_..."} in the comfyui-arc-save directory '
        "root, OR set the ARC_API_KEY env var. "
        "See README.md for the issuance flow."
    )


# ─── Redaction ─────────────────────────────────────────────────────


def _load_text_encoder_patterns() -> tuple[str, ...]:
    """Return the active text-encoder pattern list.

    Composition:
      LOCKED_CATCHALL_PATTERN (always included; cannot be removed)
      + optional creator-supplied patterns from arc_config.json's
        text_encoder_patterns array

    Each pattern is a case-insensitive SUBSTRING checked against
    each node's class_type. A node's class_type matches the
    pattern list if ANY pattern is a substring of the lower-cased
    class_type.

    Invalid arc_config.json entries (non-string, empty string) are
    silently skipped; a typo'd patterns list does NOT prevent
    redaction — the catch-all still fires.
    """
    custom: list[str] = []
    try:
        cfg = _load_arc_config()
    except RuntimeError:
        # _load_arc_config raises on malformed JSON. If we got here
        # via the save() path, _load_api_key has already raised the
        # same error first — the workflow halted. If we got here via
        # the unit-test path, defaults-only is the right fallback.
        return (LOCKED_CATCHALL_PATTERN,)
    raw = cfg.get("text_encoder_patterns")
    if isinstance(raw, list):
        custom = [p for p in raw if isinstance(p, str) and p]
    return (LOCKED_CATCHALL_PATTERN, *custom)


def _class_type_matches(class_type: Any, patterns: tuple[str, ...]) -> bool:
    """Case-insensitive substring match. Returns False for any non-
    string class_type so malformed nodes don't crash the walk."""
    if not isinstance(class_type, str):
        return False
    lower = class_type.lower()
    for p in patterns:
        if p.lower() in lower:
            return True
    return False


def _redact_string_value(s: str) -> str:
    """Hash a text payload and wrap in the redaction envelope."""
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return f"{_REDACTED_PREFIX}{digest}{_REDACTED_SUFFIX}"


def _redact_prompt_text(prompt: Any, patterns: tuple[str, ...]) -> Any:
    """Walk the PROMPT graph and replace text-input strings in
    matching encoder nodes with SHA-256-hashed envelopes.

    Pure function. Returns a NEW dict; does not mutate the input.
    Non-dict prompts (None, malformed, etc.) pass through unchanged
    so a degenerate input cannot fail the whole signing.

    A node matches when:
      isinstance(node_spec, dict)
      and _class_type_matches(node_spec["class_type"], patterns)
      and isinstance(node_spec["inputs"], dict)

    Within a matching node's inputs dict, a field is redacted when:
      field name (lower-cased) is in DEFAULT_TEXT_INPUT_FIELDS_LOWER
      and isinstance(value, str)

    Non-string values (numbers, lists, tensor refs) are left alone
    even in matching fields — ComfyUI's "text" input on
    CLIPTextEncode is normally a string, but some workflows route
    other-node outputs through, and those values are already
    structural (not plaintext).
    """
    if not isinstance(prompt, dict):
        return prompt

    out: dict = {}
    for node_id, node_spec in prompt.items():
        if not isinstance(node_spec, dict):
            out[node_id] = node_spec
            continue
        if not _class_type_matches(node_spec.get("class_type"), patterns):
            out[node_id] = node_spec
            continue
        inputs = node_spec.get("inputs")
        if not isinstance(inputs, dict):
            out[node_id] = node_spec
            continue
        new_inputs: dict = {}
        for field_name, value in inputs.items():
            if (
                isinstance(field_name, str)
                and field_name.lower() in DEFAULT_TEXT_INPUT_FIELDS_LOWER
                and isinstance(value, str)
            ):
                new_inputs[field_name] = _redact_string_value(value)
            else:
                new_inputs[field_name] = value
        new_node = dict(node_spec)
        new_node["inputs"] = new_inputs
        out[node_id] = new_node
    return out


# ─── Encoding ──────────────────────────────────────────────────────


def _encode_image_to_png_bytes(image_tensor: Any) -> bytes:
    """Encode a ComfyUI IMAGE-shaped tensor to PNG bytes.

    Mirrors SaveImage in ComfyUI/nodes.py verbatim — same numpy ops,
    same clamp, same uint8 cast, same PIL.Image.fromarray pattern.
    Result is byte-equivalent to what a SaveImage node would write
    for the same input (modulo PngInfo metadata, which we add
    server-side via the assertion, not here).

    Returns the encoded PNG bytes ready to POST as the request body.
    """
    arr = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buf.getvalue()


# ─── Generation assertion build ────────────────────────────────────


def _detect_comfyui_version() -> str:
    """Best-effort version detection. ComfyUI does not expose a
    stable __version__ on all releases; falls back to 'unknown' or
    'not-comfyui' for standalone test environments."""
    try:
        import comfy  # type: ignore[import-not-found]

        return getattr(comfy, "__version__", "unknown")
    except ImportError:
        return "not-comfyui"


def _build_generation_assertion(
    prompt: Any,
    extra_pnginfo: Any,
    unique_id: Any,
    title: str,
    comfyui_version: str,
    batch_index: int,
    batch_size: int,
    redacted_prompt: bool,
) -> dict:
    """Build the data block sent as X-Arc-Generation-Metadata.

    Caller passes the workflow_prompt already-redacted-or-not.
    The redacted_prompt flag tells the relying party which case it
    is: True means text inputs in matching encoder nodes are
    SHA-256-hashed under the [REDACTED:sha256:...] envelope; False
    means workflow_prompt carries plaintext.

    batch_index and batch_size are baked into every per-image
    assertion so a relying party can see "this is image 2 of 4
    from the same workflow execution." Useful for video frame
    chains, multi-sample ablations, and any downstream tooling
    that wants to reconstruct the batch grouping after the fact.

    The returned dict is open-shaped — additional top-level keys
    (e.g. a future perceptual fingerprint block) can be added by
    later code without restructuring this builder.
    """
    return {
        "schema": "com.arcvelvet.generation.v0",
        "platform_client": "comfyui-arc-save",
        "comfyui_version": comfyui_version,
        "node_id": str(unique_id) if unique_id is not None else "unknown",
        "title": title,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "redacted_prompt": redacted_prompt,
        "workflow_prompt": prompt or {},
        # extra_pnginfo typically has a "workflow" key holding the
        # editor UI graph (positions, widget values, group annotations).
        # Separate from the executable PROMPT graph; both are useful.
        # extra_pnginfo is NOT redacted — the editor-UI graph is
        # what creators see in their own ComfyUI; passing it through
        # verbatim matches SaveImage behavior and the redacted
        # workflow_prompt above is the authoritative provenance copy.
        "extra_pnginfo": extra_pnginfo or {},
        # Reserved slot for future per-image perceptual fingerprint.
        # ARC-API-3-0 addendum: confirmed the assertion path can carry
        # one more optional namespaced block without core-flow change.
        # Current build leaves it absent; future code adds it here as
        # "fingerprint": { "algorithm": "...", "value": "..." }.
    }


# ─── HTTP ──────────────────────────────────────────────────────────


class _TransientIngestError(Exception):
    """Internal sentinel — POST failed in a way that's worth one retry
    (HTTP 503 from the server, or a Timeout / ConnectionError on the
    transport). NEVER surfaced to the caller as-is; the retry wrapper
    either succeeds on the second attempt or wraps this in a final
    RuntimeError that includes the prior-attempt cause."""


def _post_to_arc_ingest(
    png_bytes: bytes,
    title: str,
    generation_assertion: dict,
    api_key: str,
) -> dict:
    """Single POST attempt to arcIngest.

    Three exits:
      - 200 → returns the parsed receipt dict.
      - 503 / requests.Timeout / requests.ConnectionError →
        raises _TransientIngestError (the retry wrapper handles).
      - Anything else (auth, rate-limit, payload-too-large,
        sign-failed, other transport classes, 200 with non-JSON
        body, etc.) → raises RuntimeError immediately. Terminal,
        no retry — these reflect creator action items (revoke +
        reissue a key, back off rate, shrink payload) or server
        bugs that won't fix themselves in 2 seconds.

    The X-Arc-Title header is URL-encoded so non-ASCII titles ride
    through without HTTP header byte-set violations. The server
    URL-decodes on receipt. X-Arc-Generation-Metadata is base64-
    encoded JSON; the server validates ≤ 64 KB decoded.
    """
    encoded_title = quote(title or "", safe="")
    encoded_metadata = base64.b64encode(
        json.dumps(generation_assertion, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "image/png",
        "X-Arc-Title": encoded_title,
        "X-Arc-Generation-Metadata": encoded_metadata,
    }

    try:
        resp = requests.post(
            INGEST_URL,
            headers=headers,
            data=png_bytes,
            timeout=INGEST_TIMEOUT_SECONDS,
        )
    except (requests.Timeout, requests.ConnectionError) as e:
        raise _TransientIngestError(
            f"transport {type(e).__name__}: {e}"
        )
    except requests.RequestException as e:
        # Non-retryable transport class (e.g. InvalidURL, MissingSchema).
        raise RuntimeError(
            f"ArcVelvet ingest transport error (terminal): "
            f"{type(e).__name__}: {e}"
        )

    if resp.status_code == 200:
        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"ArcVelvet ingest returned 200 but body was not JSON: {e}"
            )

    if resp.status_code == 503:
        raise _TransientIngestError(
            f"server unavailable: HTTP 503 {resp.text[:200]}"
        )

    # Any other non-200 — terminal. Surface the server's stable error
    # code (ERR_AUTH_FAILED, ERR_RATE_LIMITED, ERR_PAYLOAD_TOO_LARGE,
    # ERR_SIGN_FAILED, etc.) so the creator can act without digging.
    body_preview = resp.text[:400]
    raise RuntimeError(
        f"ArcVelvet ingest failed: HTTP {resp.status_code} — {body_preview}"
    )


def _post_to_arc_ingest_with_retry(
    png_bytes: bytes,
    title: str,
    generation_assertion: dict,
    api_key: str,
) -> dict:
    """Wraps _post_to_arc_ingest with a single 2-second retry on
    transient failure (HTTP 503, Timeout, ConnectionError).

    Terminal failures raise immediately on the first attempt — no
    retry. A second-attempt transient failure surfaces as a
    RuntimeError that includes both attempts' causes so the creator
    sees the full picture in ComfyUI's error tile.
    """
    try:
        return _post_to_arc_ingest(png_bytes, title, generation_assertion, api_key)
    except _TransientIngestError as first_err:
        time.sleep(RETRY_BACKOFF_SECONDS)
        try:
            return _post_to_arc_ingest(
                png_bytes, title, generation_assertion, api_key
            )
        except _TransientIngestError as second_err:
            raise RuntimeError(
                f"ArcVelvet ingest failed after 1 retry — "
                f"first: {first_err}; second: {second_err}"
            )
        # A terminal RuntimeError on the second attempt propagates
        # as-is; no need to catch + rewrap.


def _download_credentialed_bytes(ingest_result: dict) -> bytes:
    """Download the credentialed file with verify-route fallback.

    Two-tier download:
      Tier 1 — signedFileUrl: pre-signed Storage URL, 15-min TTL,
        no rate limit. Fastest path. The server's contract
        explicitly allows it to be '' when getSignedUrl mints
        fail (e.g. the runtime SA lacks
        iam.serviceAccounts.signBlob — see arcIngest.ts
        ingest.signed_url_failed WARN path).
      Tier 2 — {verifyUrl}&format=file: ARC-API-2 Commit 5 vault
        verify route, public, rate-limited at 60/min per item
        via assertFlatRateLimit. Durable — uses Admin SDK to
        serve the binary; does NOT depend on URL-signing IAM,
        so it works even if Tier 1 is permanently broken.

    The signed URL is now an OPTIMIZATION. The verify route is
    the durable source of truth. This function fails loudly only
    if BOTH paths fail; the resulting error message includes
    each tier's failure cause.
    """
    signed_file_url = ingest_result.get("signedFileUrl") or ""
    verify_url = ingest_result.get("verifyUrl") or ""

    primary_err: str = ""
    if signed_file_url:
        try:
            resp = requests.get(signed_file_url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return resp.content
            primary_err = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            primary_err = f"{type(e).__name__}: {e}"
    else:
        primary_err = (
            "empty (server URL-minting failed; see ingest.signed_url_failed)"
        )

    if not verify_url:
        raise RuntimeError(
            "ArcVelvet credentialed-bytes download failed — "
            f"signedFileUrl: {primary_err}; verifyUrl was also empty."
        )

    fallback_url = (
        verify_url + ("&" if "?" in verify_url else "?") + "format=file"
    )
    try:
        resp = requests.get(fallback_url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise RuntimeError(
            "ArcVelvet credentialed-bytes download failed — "
            f"signedFileUrl: {primary_err}; "
            f"verify-route fallback transport: {type(e).__name__}: {e}"
        )
    if resp.status_code != 200:
        raise RuntimeError(
            "ArcVelvet credentialed-bytes download failed — "
            f"signedFileUrl: {primary_err}; "
            f"verify-route fallback: HTTP {resp.status_code}"
        )
    return resp.content


# ─── Output writing ────────────────────────────────────────────────


def _write_outputs(
    signed_bytes: bytes,
    ingest_result: dict,
    output_dir: str,
    filename_prefix: str,
) -> str:
    """Write the signed PNG + sidecar .arc.json to output_dir.

    Filename shape: <prefix>_<vaultItemId>.png plus
    <prefix>_<vaultItemId>.arc.json. The vaultItemId is content-hash-
    derived deterministic (arc_ingest_<16hex>), so the same content
    re-saved produces the same filename — dedup hits land on the
    same file on disk.

    Returns the filename written (without path) for ComfyUI's
    preview-tile UI hook.
    """
    os.makedirs(output_dir, exist_ok=True)
    vault_item_id = ingest_result.get("vaultItemId", "unknown")
    safe_prefix = filename_prefix.replace("/", "_").replace("\\", "_")

    png_name = f"{safe_prefix}_{vault_item_id}.png"
    sidecar_name = f"{safe_prefix}_{vault_item_id}.arc.json"

    png_path = os.path.join(output_dir, png_name)
    sidecar_path = os.path.join(output_dir, sidecar_name)

    with open(png_path, "wb") as f:
        f.write(signed_bytes)

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vaultItemId": ingest_result.get("vaultItemId"),
                "verifyUrl": ingest_result.get("verifyUrl"),
                "contentHash": ingest_result.get("contentHash"),
                "traceId": ingest_result.get("traceId"),
                "deduplicated": bool(ingest_result.get("deduplicated", False)),
                "timings": ingest_result.get("timings"),
            },
            f,
            indent=2,
        )

    return png_name


def _resolve_output_dir(output_dir_override: str | None) -> str:
    """Resolve where the signed PNG should land.

    Inside ComfyUI: defers to folder_paths.get_output_directory().
    Standalone tests: caller passes output_dir_override explicitly
    via the save() kwarg; ComfyUI does not pass this kwarg because
    it is not declared in INPUT_TYPES.

    Bare fallback (no ComfyUI, no override): current working
    directory. Useful only if someone runs this outside its
    intended environment.
    """
    if output_dir_override:
        return output_dir_override
    try:
        import folder_paths  # type: ignore[import-not-found]

        return folder_paths.get_output_directory()
    except ImportError:
        return os.getcwd()


# ─── The node class ────────────────────────────────────────────────


class ARCSave:
    """ARC Save — sign-on-arrival ComfyUI save node.

    OUTPUT_NODE = True marks this as a terminal/sink node that
    executes for its side effects (the HTTP POST + signed-file
    write), not for downstream consumption.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # include_prompt_text is the redaction toggle. Default OFF
        # (False) means the prompt graph is redacted before signing:
        # CLIPTextEncode-style nodes have their text inputs replaced
        # with SHA-256-hashed envelopes. Turning it ON ships the
        # prompt VERBATIM into the signed assertion — useful when
        # the creator wants their wording cryptographically bound
        # to the file (attribution, dataset provenance, etc.).
        # Either way the structural workflow graph is preserved;
        # only the text payloads differ.
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image tensor from upstream node "
                            "(typically VAE Decode)."
                        )
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "ArcVelvet"},
                ),
                "title": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional creator-supplied title baked into "
                            "the signed assertion."
                        ),
                    },
                ),
                "include_prompt_text": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "OFF (default): prompt text in CLIPTextEncode-"
                            "style nodes is SHA-256-hashed before signing "
                            "(\"[REDACTED:sha256:...]\" envelope). Workflow "
                            "structure is preserved; you can reveal the "
                            "plaintext later and anyone can verify the "
                            "hash. ON: prompt rides verbatim — useful when "
                            "you want your wording cryptographically bound "
                            "to the file."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES: tuple = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image/save"

    def save(
        self,
        images,
        filename_prefix,
        title,
        include_prompt_text,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        # Test-only override; ComfyUI does not pass this. See
        # _resolve_output_dir docstring.
        _output_dir_override: str | None = None,
    ):
        if not hasattr(images, "__len__") or len(images) == 0:
            raise RuntimeError("ARC Save: no images received from upstream node.")

        api_key = _load_api_key()
        output_dir = _resolve_output_dir(_output_dir_override)
        comfyui_version = _detect_comfyui_version()
        batch_size = len(images)

        # Redaction decision is made ONCE per workflow execution and
        # applied to all images in the batch. Same toggle setting,
        # same prompt graph, same redaction outcome — relying
        # parties comparing two signed files from the same batch
        # should see identical workflow_prompt fields.
        if include_prompt_text:
            workflow_prompt = prompt
        else:
            patterns = _load_text_encoder_patterns()
            workflow_prompt = _redact_prompt_text(prompt, patterns)
        redacted_prompt_flag = not include_prompt_text

        # Per-image loop. Each image's sign is INDEPENDENT — a failure
        # on image 3 raises immediately and the workflow halts; images
        # 1 and 2 are already saved (the API's content-hash idempotency
        # means re-running the workflow won't double-charge those two).
        # Fail-loud posture applies per-image: any non-retryable error
        # on any image stops the batch.
        #
        # Rate-limit note: arcIngest is 10/min per API key. A batch of
        # 11+ images will trigger ERR_RATE_LIMITED partway through and
        # raise loudly with the exact failure point. The creator
        # decides whether to wait + re-run (idempotent dedup on the
        # already-saved images) or to split the workflow.
        saved_filenames: list[str] = []
        for batch_index, image_tensor in enumerate(images):
            png_bytes = _encode_image_to_png_bytes(image_tensor)

            generation_assertion = _build_generation_assertion(
                prompt=workflow_prompt,
                extra_pnginfo=extra_pnginfo,
                unique_id=unique_id,
                title=title,
                comfyui_version=comfyui_version,
                batch_index=batch_index,
                batch_size=batch_size,
                redacted_prompt=redacted_prompt_flag,
            )

            ingest_result = _post_to_arc_ingest_with_retry(
                png_bytes=png_bytes,
                title=title,
                generation_assertion=generation_assertion,
                api_key=api_key,
            )

            signed_bytes = _download_credentialed_bytes(ingest_result)

            filename = _write_outputs(
                signed_bytes=signed_bytes,
                ingest_result=ingest_result,
                output_dir=output_dir,
                filename_prefix=filename_prefix,
            )
            saved_filenames.append(filename)

        # OUTPUT_NODE ui-result shape — ComfyUI's editor uses this to
        # render saved images as the node's preview tile. One entry
        # per image so the full batch shows. Matches SaveImage's
        # batch return shape.
        return {
            "ui": {
                "images": [
                    {"filename": f, "subfolder": "", "type": "output"}
                    for f in saved_filenames
                ]
            }
        }


# ─── ComfyUI registration ──────────────────────────────────────────

NODE_CLASS_MAPPINGS = {"ARCSave": ARCSave}
NODE_DISPLAY_NAME_MAPPINGS = {"ARCSave": "ARC Save (sign on arrival)"}
