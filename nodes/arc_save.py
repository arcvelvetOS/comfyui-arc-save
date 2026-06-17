"""
arc_save.py — v1.0.0 (MINOR-SAFETY-1 Sprint 2C cutover)

The ARC Save ComfyUI custom node. Encodes the input IMAGE tensor to
PNG bytes (matching SaveImage's encoding path exactly), POSTs to the
live ArcVelvet arcIngest endpoint, downloads the signed copy, writes
the signed PNG plus a sidecar .arc.json receipt to the output
directory.

Wire contract (v1.0.0):
    - Per-image loop across the IMAGE batch tensor.
    - Single retry with 2-second backoff on TRANSIENT errors only
      (HTTP 503 + requests.Timeout + requests.ConnectionError).
      Every other non-200 (401/403/413/415/429/500/etc.) raises
      immediately.
    - Three prompt-related fields in the generation assertion:
        workflow_prompt:           PLAINTEXT ComfyUI PROMPT graph.
                                   Server is the manifest-redaction
                                   authority and walks this with the
                                   same pattern set the node uses to
                                   extract for moderation.
        promptTextForModeration:   Concatenated plaintext from text-
                                   encoder nodes. Server scans via
                                   OpenAI moderation and DISCARDS on
                                   pass; preserves in a sealed
                                   minor_safety_escalations doc on a
                                   sexual/minors flag (NCMEC handoff
                                   pipeline). NEVER in the manifest.
        include_prompt_text:       Creator opt-in BOOLEAN. When false
                                   (default), the server replaces text-
                                   encoder text values with
                                   [REDACTED:sha256:<hex>] envelopes
                                   before signing the manifest. When
                                   true, the server embeds plaintext.
    - The node-side `redacted_prompt` field is no longer set by the
      node. The server sets it based on include_prompt_text and writes
      it into the signed manifest assertion.
    - Local redaction code (LOCKED_CATCHALL_PATTERN, _redact_prompt_text,
      _redact_string_value, text_encoder_patterns config) is REMOVED.
      The node retains the same walk pattern for EXTRACTION only —
      see _extract_prompt_text_for_moderation().
    - arc_config.json + ARC_API_KEY env var key loading.
    - Sidecar .arc.json with vaultItemId / verifyUrl / contentHash.

Why server-side redaction (over client-side):
    A single canonical redaction authority avoids the trust gap where
    each node-version's local walk can drift from the server's
    expectation. The server's promptModeration.ts walks workflow_prompt
    with the IDENTICAL substring catch-all ("textencode") and field-
    name match (/text|prompt/i) the node uses here for extraction. The
    walks are pinned in lockstep so the corpus the node sends matches
    what the server would have extracted on its own.

Additive-fingerprint slot (ARC-API-3-0 addendum, unchanged):
    The generation assertion is built as an open dict. A future
    perceptual-fingerprint block can be added as a top-level
    namespaced key (e.g. "fingerprint": {...}) without restructuring
    any of the call sites here.
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse

import numpy as np
import requests
from PIL import Image


# ─── Constants ─────────────────────────────────────────────────────

# Live arcIngest endpoint deployed under ARC-API-2 Commit 4.
INGEST_URL = "https://us-central1-arcvelvetos.cloudfunctions.net/arcIngest"

# Live c2paVerify MACHINE-API endpoint (Cloud Run direct URL). This
# is the function URL from the ARC-API-2 Commit 5 deploy receipt —
# NOT arcvelvet.com/verify (which is a Firebase Hosting rewrite to
# public/verify.html, the human-facing SPA). The two URLs serve
# distinct audiences:
#   - arcvelvet.com/verify        → human page (SPA)
#   - c2paverify-*.run.app        → machine API (this constant)
# The node's fallback path MUST use the machine API; pointing at the
# human page returns HTML, not the credentialed binary. See SMOKE-2.
VERIFY_API_URL = "https://c2paverify-crvmppbxka-uc.a.run.app"

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

# ── Prompt extraction config (1.0.0 cutover) ──────────────────────
#
# MINOR-SAFETY-1 Sprint 2C cutover: as of v1.0.0, the SERVER is the
# manifest-redaction authority. The node no longer hashes text fields
# before sending. Instead, the node:
#
#   1. Always sends `workflow_prompt` PLAINTEXT to arcIngest.
#   2. Always sends a NEW field `promptTextForModeration` carrying the
#      concatenated plaintext from text-encoder nodes — this is the
#      corpus the server feeds to OpenAI moderation. The server discards
#      it on pass; on a flagged hit the server preserves it in the
#      sealed minor_safety_escalations collection for NCMEC handoff
#      (see ArcVelvet server-side policy docs).
#   3. Sends `include_prompt_text` as a creator opt-in BOOLEAN. When
#      false (default), the server walks workflow_prompt and replaces
#      text-encoder text values with [REDACTED:sha256:<hex>] envelopes
#      before signing the manifest. When true, the server embeds
#      plaintext in the manifest.
#
# The node-side LOCKED_CATCHALL_PATTERN and the field-name walk live on
# in TEXTENCODE_CATCHALL / TEXT_INPUT_FIELDS_LOWER below, but they are
# now used ONLY to compute promptTextForModeration (extraction), NOT
# to redact (transformation). The server uses the IDENTICAL walk to
# ensure node corpus matches server extraction.
TEXTENCODE_CATCHALL = "textencode"

# Field names whose string values get extracted for the moderation
# corpus. Covers vanilla CLIPTextEncode ("text"), CLIPTextEncodeSDXL
# ("text_g", "text_l"), and the common positive/negative split used
# by many workflows. Lowercase for case-insensitive matching against
# arbitrary case (e.g. "Text", "PROMPT") in custom encoder shapes.
TEXT_INPUT_FIELDS_LOWER = (
    "text",
    "prompt",
    "text_g",
    "text_l",
    "positive",
    "negative",
    "string",
)


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


# ─── Prompt extraction (for moderation) ────────────────────────────


def _is_textencode_node(class_type: Any) -> bool:
    """Case-insensitive substring match against TEXTENCODE_CATCHALL.
    Returns False for any non-string class_type so malformed nodes
    don't crash the walk.

    Mirrors the server's isTextEncodeNode() in promptModeration.ts to
    guarantee node corpus extraction matches server-side extraction —
    this is the contract that makes the gate's moderation decision
    reproducible across the wire.
    """
    if not isinstance(class_type, str):
        return False
    return TEXTENCODE_CATCHALL in class_type.lower()


def _is_text_field(field_name: Any) -> bool:
    """Field-name match against TEXT_INPUT_FIELDS_LOWER. Mirrors the
    server's isTextField() in promptModeration.ts."""
    if not isinstance(field_name, str):
        return False
    return field_name.lower() in TEXT_INPUT_FIELDS_LOWER


def _extract_prompt_text_for_moderation(prompt: Any) -> str:
    """Walk the PROMPT graph and return the concatenated plaintext
    from every text-encoder node's text-input field, joined by '\\n'.

    This is the corpus the server feeds to OpenAI moderation. The
    walk MUST match the server-side extractPromptCorpus() walk in
    functions/src/moderation/promptModeration.ts so the corpus the
    node generates is byte-identical to what the server would extract
    from workflow_prompt itself.

    Pure function. Does not mutate the input. Non-dict prompts pass
    through to '' so a degenerate input doesn't crash the signing.
    """
    if not isinstance(prompt, dict):
        return ""
    texts: list[str] = []
    for node_id, node_spec in prompt.items():
        if not isinstance(node_spec, dict):
            continue
        if not _is_textencode_node(node_spec.get("class_type")):
            continue
        inputs = node_spec.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for field_name, value in inputs.items():
            if not _is_text_field(field_name):
                continue
            if isinstance(value, str) and value:
                texts.append(value)
    return "\n".join(texts)


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
    include_prompt_text: bool,
    prompt_text_for_moderation: str,
) -> dict:
    """Build the metadata block POSTed as the multipart `metadata` part
    of the arcIngest request.

    1.0.0 cutover (MINOR-SAFETY-1 Sprint 2C):
      - `workflow_prompt` is always sent PLAINTEXT. The server is the
        manifest-redaction authority and walks workflow_prompt with the
        SAME pattern set used here for extraction; when
        include_prompt_text=false (default), the server replaces text-
        encoder text values with [REDACTED:sha256:<hex>] envelopes
        BEFORE signing the manifest. The node-side `redacted_prompt`
        assertion field is no longer set by the node — the server sets
        it based on include_prompt_text.
      - `promptTextForModeration` carries the concatenated plaintext
        from text-encoder nodes for OpenAI moderation. Server discards
        on pass; preserves in a sealed collection on a sexual/minors
        flag (NCMEC handoff). NEVER in the manifest.
      - `include_prompt_text` is the creator's opt-in boolean,
        forwarded so the server knows whether to redact for manifest.

    batch_index and batch_size are baked into every per-image
    assertion so a relying party can see "this is image 2 of 4 from the
    same workflow execution."

    The returned dict is open-shaped — additional top-level keys can be
    added by later code without restructuring this builder.
    """
    return {
        "schema": "com.arcvelvet.generation.v0",
        "platform_client": "comfyui-arc-save",
        "comfyui_version": comfyui_version,
        "node_id": str(unique_id) if unique_id is not None else "unknown",
        "title": title,
        "batch_index": batch_index,
        "batch_size": batch_size,
        # 1.0.0 cutover: workflow_prompt is plaintext; server redacts
        # for manifest based on include_prompt_text. redacted_prompt is
        # set by the server, not here.
        "workflow_prompt": prompt or {},
        "include_prompt_text": bool(include_prompt_text),
        "promptTextForModeration": prompt_text_for_moderation,
        # extra_pnginfo typically has a "workflow" key holding the
        # editor UI graph (positions, widget values, group annotations).
        # Separate from the executable PROMPT graph; both are useful.
        # extra_pnginfo passes through verbatim — matches SaveImage
        # behavior and is independent of the workflow_prompt
        # redaction-for-manifest decision the server makes.
        "extra_pnginfo": extra_pnginfo or {},
        # Reserved slot for future per-image perceptual fingerprint.
        # ARC-API-3-0 addendum: confirmed the assertion path can carry
        # one more optional namespaced block without core-flow change.
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

    Transport (B1-FIX-1): multipart/form-data with 'image' (raw
    PNG bytes) and 'metadata' (JSON object) parts. The legacy
    header transport (image bytes as body + X-Arc-Generation-
    Metadata base64-encoded header) had a quiet GFE 431 cliff
    that fell over on the first real ComfyUI workflow. The body
    transport has no header-size ceiling.

    X-Arc-Title remains a header (small, bounded; encoded via
    URL-quoting so non-ASCII titles ride through without HTTP
    header byte-set violations).

    The 431 path is kept as a safety net even though multipart
    eliminates the failure mode: if any future change to the
    transport (header bloat from a new SDK middleware, proxy
    config etc.) re-introduces it, the message stays actionable
    rather than trailing into a dash.
    """
    encoded_title = quote(title or "", safe="")
    metadata_json = json.dumps(
        generation_assertion, separators=(",", ":")
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Arc-Title": encoded_title,
    }

    # requests builds the multipart body when 'files' is passed. The
    # filename slot is intentionally None on the metadata part so the
    # server's busboy parser sees it as a regular form field (no
    # filename in Content-Disposition); the image part carries a
    # filename so it surfaces as a file event.
    files = {
        "image": ("image.png", png_bytes, "image/png"),
        "metadata": (None, metadata_json, "application/json"),
    }

    try:
        resp = requests.post(
            INGEST_URL,
            headers=headers,
            files=files,
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

    if resp.status_code == 431:
        # Safety net for a transport-layer regression. The current
        # multipart-only path puts metadata in the body and shouldn't
        # trip this; if it does, that's a node bug.
        raise RuntimeError(
            "ArcVelvet ingest failed: HTTP 431 (Request Header Fields "
            "Too Large). Headers exceeded the transport limit even "
            "though metadata is sent in the body. This is a node bug. "
            "Please open an issue at "
            "https://github.com/arcvelvetOS/comfyui-arc-save/issues."
        )

    # Any other non-200 — terminal. Surface the server's stable error
    # code (ERR_AUTH_FAILED, ERR_RATE_LIMITED, ERR_PAYLOAD_TOO_LARGE,
    # ERR_METADATA_TOO_LARGE, ERR_SIGN_FAILED, etc.) so the creator can
    # act without digging.
    body_preview = resp.text[:400]
    raise RuntimeError(
        f"ArcVelvet ingest failed: HTTP {resp.status_code} - {body_preview}"
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
      Tier 2 — machine-API verify route at VERIFY_API_URL with
        ?type=vault&owner={uid}&item={itemId}&format=file. ARC-
        API-2 Commit 5 vault route, public, rate-limited at
        60/min per item via assertFlatRateLimit. Durable — uses
        Admin SDK to serve the binary; does NOT depend on URL-
        signing IAM, so it works even if Tier 1 is permanently
        broken.

    IMPORTANT — human page vs machine API:
        The receipt's verifyUrl points at arcvelvet.com/verify,
        which Firebase Hosting rewrites to public/verify.html
        (the human-facing SPA). Fetching that URL returns HTML,
        not the credentialed binary. The fallback MUST target
        VERIFY_API_URL (the Cloud Run function URL) — the
        machine API. See SMOKE-2 for the live failure that
        forced this distinction.

        We extract owner+item from the receipt's verifyUrl via
        urllib.parse rather than reusing the URL itself, so the
        node never fetches the human page expecting bytes.

    The signed URL is an OPTIMIZATION. The verify-route machine
    API is the durable source of truth. This function fails
    loudly only if BOTH paths fail; the resulting error message
    includes each tier's failure cause.
    """
    signed_file_url = ingest_result.get("signedFileUrl") or ""
    receipt_verify_url = ingest_result.get("verifyUrl") or ""

    # Tier 1 — try the pre-signed Storage URL when present.
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

    # Tier 2 — extract owner + item from the receipt verifyUrl and
    # rebuild against the machine-API endpoint. urllib.parse handles
    # encoding edge cases properly; never string-split a URL.
    if not receipt_verify_url:
        raise RuntimeError(
            "ArcVelvet credentialed-bytes download failed — "
            f"signedFileUrl: {primary_err}; verifyUrl was also empty."
        )

    parsed = urlparse(receipt_verify_url)
    params = parse_qs(parsed.query)
    owner_list = params.get("owner") or []
    item_list = params.get("item") or []
    if not owner_list or not item_list or not owner_list[0] or not item_list[0]:
        raise RuntimeError(
            "ArcVelvet credentialed-bytes download failed — "
            f"signedFileUrl: {primary_err}; "
            f"could not extract owner+item from verifyUrl "
            f"(parsed query: {parsed.query!r})."
        )

    fallback_query = urlencode(
        {
            "type": "vault",
            "owner": owner_list[0],
            "item": item_list[0],
            "format": "file",
        }
    )
    fallback_url = f"{VERIFY_API_URL}?{fallback_query}"

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
            f"verify-route fallback: HTTP {resp.status_code} "
            f"(body preview: {resp.text[:200]!r})"
        )

    # Content-Type guard: refuse non-image responses. If the verify-
    # route ever serves HTML (route mismatch, error page, hostname
    # collision with the human page), this catches it BEFORE the
    # bytes get written to disk as a .png the creator can't open.
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if not content_type.startswith("image/"):
        raise RuntimeError(
            "ArcVelvet credentialed-bytes download failed — "
            f"signedFileUrl: {primary_err}; "
            f"verify-route fallback returned non-image Content-Type "
            f"{content_type!r} (likely fetched a human page instead "
            f"of the machine API). Body preview: {resp.text[:200]!r}"
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

        # 1.0.0 cutover (MINOR-SAFETY-1 Sprint 2C): the node no longer
        # redacts workflow_prompt locally. The server is the manifest-
        # redaction authority and walks workflow_prompt with the same
        # pattern set we use here for extraction.
        #
        # The node's responsibilities are now:
        #   - send workflow_prompt PLAINTEXT (the server hashes it for
        #     the manifest when include_prompt_text=false)
        #   - extract the moderation corpus and send it as
        #     promptTextForModeration (the server scans + discards on
        #     pass, preserves in a sealed escalation collection on a
        #     sexual/minors hit)
        #   - forward include_prompt_text so the server knows the
        #     creator's manifest-redaction opt-in choice
        #
        # The extraction is deterministic per workflow execution and
        # identical across the batch — every image's assertion carries
        # the same workflow_prompt + promptTextForModeration so relying
        # parties comparing two signed files from the same batch see
        # identical provenance fields.
        prompt_text_for_moderation = _extract_prompt_text_for_moderation(prompt)

        # Per-image loop. Each image's sign is INDEPENDENT — a failure
        # on image 3 raises immediately and the workflow halts; images
        # 1 and 2 are already saved (the API's content-hash idempotency
        # means re-running the workflow won't double-charge those two).
        # Fail-loud posture applies per-image: any non-retryable error
        # on any image stops the batch.
        #
        # Rate-limit note: arcIngest is 5/min per API key (free-door cap;
        # Sprint A0 tightening) PLUS 20/min per uid as an aggregate cap
        # across all keys on the same account. A batch of 6+ images on
        # one key, or 21+ across keys on one uid, will trigger
        # ERR_RATE_LIMITED partway through and raise loudly with the
        # exact failure point. The creator decides whether to wait +
        # re-run (idempotent dedup on the already-saved images) or to
        # split the workflow.
        saved_filenames: list[str] = []
        for batch_index, image_tensor in enumerate(images):
            png_bytes = _encode_image_to_png_bytes(image_tensor)

            generation_assertion = _build_generation_assertion(
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
                unique_id=unique_id,
                title=title,
                comfyui_version=comfyui_version,
                batch_index=batch_index,
                batch_size=batch_size,
                include_prompt_text=bool(include_prompt_text),
                prompt_text_for_moderation=prompt_text_for_moderation,
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
