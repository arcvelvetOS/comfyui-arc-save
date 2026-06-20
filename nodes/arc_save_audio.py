"""
arc_save_audio.py — AUDIO-1 v1.1.0 (audio sibling of arc_save.py)

The ARC Save Audio ComfyUI custom node. Encodes a ComfyUI AUDIO tensor
to a WAV (16-bit PCM, lossless) and POSTs it to ArcVelvet's arcIngest
endpoint for sign-on-arrival. Same wire contract as the image node —
the multipart part is still named "image" (legacy wire label; the
server routes by Content-Type, not by part name).

────────────────────────────────────────────────────────────────────
ALPHA STATUS — what this node attests, what it does NOT attest
────────────────────────────────────────────────────────────────────

Signing a WAV file with ArcVelvet's C2PA key attests the CREATION
EVENT: that on a specific date, against a specific platform-bound
identity, this exact bytes-on-disk WAV was emitted by this user's
ComfyUI workflow. The signed manifest carries the workflow graph
(model, sampler, seed, connectivity), the platform-attested
identity, and a vault-bound verify URL.

The signature does NOT attest:
  - That the voice in the audio is the speaker's real voice
  - That the audio is not a deepfake or voice clone of a real person
  - That the audio contents are factually true, lawful, or consented to
  - That generated audio (TTS, music with implicit lyrics, etc.)
    matches anything the moderation gate inspected

Moderation coverage in v1.1.0:
  - The text prompt that drives the audio generation IS moderated
    server-side via OpenAI's omni-moderation endpoint, same path as
    the image node. A sexual/minors hit refuses signing with HTTP
    451 and the corpus is preserved in a sealed escalation doc.
  - Audio-content moderation (transcript scanning, voice-clone
    detection, audio-borne policy violations) is NOT yet covered
    by an audio-specific scan. This is a known gap, held for a
    separate later decision.

Implication: this node is appropriate for personal / alpha use and
internal experimentation. It is NOT cleared for a public audio
launch until audio-content moderation lands.
────────────────────────────────────────────────────────────────────

Per-clip loop across the AUDIO batch tensor (waveform shape (B, C, T)).
Each clip is encoded to a standalone 16-bit PCM WAV, POSTed, signed
server-side, downloaded, and written to the output directory as
<prefix>_<vaultItemId>.wav with a sidecar <prefix>_<vaultItemId>.arc.json.
A failure on any clip raises immediately and the workflow halts;
clips already signed are saved (server-side content-hash idempotency
means re-running the workflow won't double-charge them).

Code reuse: the HTTP retry wrapper, API key loader, prompt extraction,
ComfyUI version detection, output-dir resolution, and generation
assertion builder are imported from arc_save.py verbatim — text-
prompt moderation is workflow-graph-based and identical whether the
output is an image or audio clip. Only three functions diverge:

  - _encode_audio_to_wav_bytes(): replaces _encode_image_to_png_bytes
  - _post_to_arc_ingest_audio():  replaces _post_to_arc_ingest (the
    multipart now sends audio/wav bytes; part name stays "image" per
    the wire-label decision in server-side Phase A)
  - _download_credentialed_bytes_audio(): the Content-Type guard
    accepts audio/* instead of image/*
  - _write_outputs_audio(): writes .wav extension and returns the
    OUTPUT_NODE ui shape under the "audio" key (instead of "images")

The image node (arc_save.py) is untouched.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np
import requests

# Reused verbatim from the image node. These are pure functions /
# constants — the image node's behavior is not affected by their use
# from this module.
from .arc_save import (
    DOWNLOAD_TIMEOUT_SECONDS,
    INGEST_TIMEOUT_SECONDS,
    INGEST_URL,
    VERIFY_API_URL,
    _TransientIngestError,
    _build_generation_assertion,
    _detect_comfyui_version,
    _extract_prompt_text_for_moderation,
    _load_api_key,
    _resolve_output_dir,
    RETRY_BACKOFF_SECONDS,
)


# ─── Audio encoding ────────────────────────────────────────────────


def _encode_audio_to_wav_bytes(waveform: Any, sample_rate: int) -> bytes:
    """Encode a single ComfyUI audio clip to 16-bit PCM WAV bytes.

    Input:
      waveform — torch.Tensor of shape (C, T) with float values
                 nominally in [-1, 1]. C = channel count
                 (1 = mono, 2 = stereo, etc.); T = sample frames.
                 This is a single clip from the batch — callers
                 slice the (B, C, T) batch tensor before passing.
      sample_rate — int, samples per second (e.g. 44100, 48000).

    Output: bytes of a standalone valid WAV file (RIFF/WAVE/fmt/data),
    16-bit PCM, suitable to POST directly as the multipart `image`
    part with Content-Type audio/wav.

    Hand-rolled WAV header avoids a torchaudio dependency. The format
    is the lowest-common-denominator WAV that any C2PA-compliant
    reader accepts; the smoke spike at functions/scripts/c2pa-audio-
    spike.mjs proved a hand-rolled 1-second silence WAV of this exact
    shape round-trips through the c2pa-node signer.

    Pure function. Does not mutate the input.
    """
    # Convert to numpy. The duck-typed check supports both torch
    # tensors (the ComfyUI AUDIO contract) and numpy arrays (some
    # custom audio nodes pass numpy directly). Anything else is a
    # contract violation — let the AttributeError surface.
    if hasattr(waveform, "cpu") and hasattr(waveform, "numpy"):
        arr = waveform.cpu().numpy()  # torch.Tensor → np.ndarray
    elif isinstance(waveform, np.ndarray):
        arr = waveform
    else:
        raise TypeError(
            "ARC Save Audio: waveform must be a torch.Tensor or "
            f"numpy.ndarray; got {type(waveform).__name__}. "
            "Check that the upstream audio node is emitting a "
            "standard ComfyUI AUDIO {'waveform': tensor, "
            "'sample_rate': int} shape."
        )

    if arr.ndim != 2:
        raise ValueError(
            "ARC Save Audio: expected per-clip waveform of shape "
            f"(channels, samples); got ndim={arr.ndim} shape={arr.shape}. "
            "If you are passing the full batch tensor (B, C, T), "
            "slice it per-clip before encoding."
        )

    channels, num_samples = arr.shape
    if channels < 1 or num_samples < 1:
        raise ValueError(
            f"ARC Save Audio: degenerate waveform shape {arr.shape}."
        )

    # Clamp to [-1, 1] and scale to int16 range.
    arr = np.clip(arr, -1.0, 1.0)
    samples_int16 = (arr * 32767.0).astype(np.int16)  # (C, T)

    # Interleave channels: WAV expects sample-major layout
    # (sample0_ch0, sample0_ch1, sample1_ch0, sample1_ch1, ...).
    # (C, T) → (T, C) → flatten C-order = interleaved.
    interleaved = samples_int16.T.flatten(order="C")  # (T*C,)

    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = num_samples * block_align
    file_size = 36 + data_size  # 36 = RIFF(4) + size(4) + WAVE(4) + fmt header(24) - 8

    header = bytearray()
    header += b"RIFF"
    header += file_size.to_bytes(4, "little")
    header += b"WAVE"
    header += b"fmt "
    header += (16).to_bytes(4, "little")            # fmt chunk size
    header += (1).to_bytes(2, "little")             # audio format = 1 (PCM)
    header += channels.to_bytes(2, "little")
    header += sample_rate.to_bytes(4, "little")
    header += byte_rate.to_bytes(4, "little")
    header += block_align.to_bytes(2, "little")
    header += bits_per_sample.to_bytes(2, "little")
    header += b"data"
    header += data_size.to_bytes(4, "little")

    return bytes(header) + interleaved.tobytes()


# ─── HTTP (audio-mime variants) ────────────────────────────────────


def _post_to_arc_ingest_audio(
    wav_bytes: bytes,
    title: str,
    generation_assertion: dict,
    api_key: str,
) -> dict:
    """Single POST attempt to arcIngest with audio/wav bytes.

    Sibling of arc_save._post_to_arc_ingest. The wire contract is
    identical except for the multipart `image` part's content-type
    (audio/wav) and filename (clip.wav). The part name remains
    "image" because the server-side parser routes by Content-Type,
    not by part name — see functions/src/shared/multipartIngest.ts
    wire-contract comment.

    Same three exits as the image variant:
      - 200 → returns the parsed receipt dict.
      - 503 / requests.Timeout / requests.ConnectionError →
        raises _TransientIngestError (the retry wrapper handles).
      - Anything else (auth, rate-limit, payload-too-large,
        sign-failed, etc.) → raises RuntimeError immediately.
    """
    from urllib.parse import quote

    encoded_title = quote(title or "", safe="")
    metadata_json = json.dumps(
        generation_assertion, separators=(",", ":")
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Arc-Title": encoded_title,
    }

    # Part name stays "image" — wire label, not type-coupled. The
    # server's parseMultipartIngest() routes by mime allowlist; an
    # audio/wav payload in a part named "image" is the intended shape.
    files = {
        "image": ("clip.wav", wav_bytes, "audio/wav"),
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

    if resp.status_code == 415:
        # Audio variant-specific guidance: 415 here almost always means
        # the deployed server hasn't picked up the AUDIO-1 Phase A
        # changes yet (the allowlist still rejects audio/wav). Help
        # the operator diagnose without log-diving.
        raise RuntimeError(
            "ArcVelvet ingest rejected audio/wav (HTTP 415). The "
            "server-side audio support (AUDIO-1 Phase A) may not be "
            "deployed yet. If you just installed this node and the "
            "server has not been updated, audio signing will not "
            "work. Contact support@arcvelvet.com if this persists. "
            f"Server body: {resp.text[:200]}"
        )

    body_preview = resp.text[:400]
    raise RuntimeError(
        f"ArcVelvet ingest failed: HTTP {resp.status_code} - {body_preview}"
    )


def _post_to_arc_ingest_audio_with_retry(
    wav_bytes: bytes,
    title: str,
    generation_assertion: dict,
    api_key: str,
) -> dict:
    """Wraps _post_to_arc_ingest_audio with a single 2-second retry
    on transient failure. Same retry posture as the image variant."""
    import time

    try:
        return _post_to_arc_ingest_audio(
            wav_bytes, title, generation_assertion, api_key
        )
    except _TransientIngestError as first_err:
        time.sleep(RETRY_BACKOFF_SECONDS)
        try:
            return _post_to_arc_ingest_audio(
                wav_bytes, title, generation_assertion, api_key
            )
        except _TransientIngestError as second_err:
            raise RuntimeError(
                f"ArcVelvet ingest failed after 1 retry — "
                f"first: {first_err}; second: {second_err}"
            )


def _download_credentialed_bytes_audio(ingest_result: dict) -> bytes:
    """Download the credentialed WAV with verify-route fallback.

    Sibling of arc_save._download_credentialed_bytes. Two-tier
    download is identical (Tier 1 signedFileUrl → Tier 2 verify-
    route machine API). The only difference is the Content-Type
    guard: this variant accepts audio/* responses and rejects
    anything else (an image/* response here would indicate a wrong-
    type fetch; an HTML response would indicate the human page
    leaked into the machine path).
    """
    signed_file_url = ingest_result.get("signedFileUrl") or ""
    receipt_verify_url = ingest_result.get("verifyUrl") or ""

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

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if not content_type.startswith("audio/"):
        raise RuntimeError(
            "ArcVelvet credentialed-bytes download failed — "
            f"signedFileUrl: {primary_err}; "
            f"verify-route fallback returned non-audio Content-Type "
            f"{content_type!r} (likely fetched a human page or wrong-"
            f"type binary). Body preview: {resp.text[:200]!r}"
        )

    return resp.content


# ─── Output writing ────────────────────────────────────────────────


def _write_outputs_audio(
    signed_bytes: bytes,
    ingest_result: dict,
    output_dir: str,
    filename_prefix: str,
) -> str:
    """Write the signed WAV + sidecar .arc.json to output_dir.

    Filename shape: <prefix>_<vaultItemId>.wav plus
    <prefix>_<vaultItemId>.arc.json. The vaultItemId is content-hash-
    derived deterministic (arc_ingest_<16hex>), so the same audio
    re-saved produces the same filename — server-side dedup hits land
    on the same file on disk.

    Returns the filename written (without path) for ComfyUI's
    preview-tile UI hook.
    """
    os.makedirs(output_dir, exist_ok=True)
    vault_item_id = ingest_result.get("vaultItemId", "unknown")
    safe_prefix = filename_prefix.replace("/", "_").replace("\\", "_")

    wav_name = f"{safe_prefix}_{vault_item_id}.wav"
    sidecar_name = f"{safe_prefix}_{vault_item_id}.arc.json"

    wav_path = os.path.join(output_dir, wav_name)
    sidecar_path = os.path.join(output_dir, sidecar_name)

    with open(wav_path, "wb") as f:
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

    return wav_name


# ─── The audio node class ──────────────────────────────────────────


class ARCSaveAudio:
    """ARC Save Audio — sign-on-arrival ComfyUI save node for AUDIO.

    Alpha. Signs the CREATION EVENT, not the authenticity of the
    voice. Text-prompt moderation only; audio-content moderation
    (transcript scanning, voice-clone detection) is not yet covered.
    See module docstring for the full alpha framing.

    OUTPUT_NODE = True marks this as a terminal/sink node that
    executes for its side effects (the HTTP POST + signed-WAV write).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Audio dict from upstream node "
                            "(typically a TTS or music-gen node). "
                            "Expected shape: "
                            "{'waveform': torch.Tensor of (B, C, T), "
                            "'sample_rate': int}."
                        )
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "ArcVelvetAudio"},
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
                            "OFF (default): text-prompt fields in the "
                            "generation graph are SHA-256-hashed before "
                            "signing. Workflow structure is preserved. "
                            "ON: prompt rides verbatim in the signed "
                            "manifest. Same semantics as the image node. "
                            "NOTE: this controls TEXT-PROMPT redaction "
                            "only — it does not gate the audio bytes, "
                            "which are always signed verbatim."
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
    CATEGORY = "audio/save"

    def save(
        self,
        audio,
        filename_prefix,
        title,
        include_prompt_text,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        # Test-only override; ComfyUI does not pass this. Mirrors the
        # image node's pattern.
        _output_dir_override: str | None = None,
    ):
        if not isinstance(audio, dict):
            raise RuntimeError(
                "ARC Save Audio: expected AUDIO input as a dict "
                "{'waveform': tensor, 'sample_rate': int}; "
                f"got {type(audio).__name__}."
            )
        waveform_batch = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform_batch is None or sample_rate is None:
            raise RuntimeError(
                "ARC Save Audio: AUDIO dict missing 'waveform' or "
                "'sample_rate'. Check the upstream node's output shape."
            )
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise RuntimeError(
                "ARC Save Audio: sample_rate must be a positive int; "
                f"got {sample_rate!r}."
            )
        # Convert batch to numpy once for shape inspection. The per-
        # clip encoder converts again — cheap; lets us validate the
        # shape with a clear error before the loop.
        if hasattr(waveform_batch, "cpu") and hasattr(waveform_batch, "numpy"):
            batch_arr = waveform_batch.cpu().numpy()
        elif isinstance(waveform_batch, np.ndarray):
            batch_arr = waveform_batch
        else:
            raise RuntimeError(
                "ARC Save Audio: waveform must be a torch.Tensor or "
                f"numpy.ndarray; got {type(waveform_batch).__name__}."
            )
        if batch_arr.ndim != 3:
            raise RuntimeError(
                "ARC Save Audio: expected batched waveform of shape "
                f"(B, C, T); got ndim={batch_arr.ndim} shape={batch_arr.shape}."
            )
        batch_size = batch_arr.shape[0]
        if batch_size < 1:
            raise RuntimeError("ARC Save Audio: empty audio batch.")

        api_key = _load_api_key()
        output_dir = _resolve_output_dir(_output_dir_override)
        comfyui_version = _detect_comfyui_version()
        prompt_text_for_moderation = _extract_prompt_text_for_moderation(prompt)

        # Per-clip loop. Same fail-fast posture as the image node:
        # any error on any clip raises immediately. Clips already
        # signed are kept (server-side content-hash dedup avoids
        # double-charging on a re-run).
        saved_filenames: list[str] = []
        for batch_index in range(batch_size):
            clip = batch_arr[batch_index]  # (C, T)
            wav_bytes = _encode_audio_to_wav_bytes(clip, sample_rate)

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
            # Tag the assertion as audio-variant emission so server-
            # side log analytics + downstream consumers can distinguish
            # image-node and audio-node provenance. The image node's
            # builder leaves platform_client = "comfyui-arc-save"; we
            # override here.
            generation_assertion["platform_client"] = "comfyui-arc-save-audio"
            # Audio-specific provenance: include the sample rate +
            # channel count + duration as factual metadata about the
            # encoded WAV. Cheap, useful for downstream verifiers, and
            # cryptographically bound by the signed manifest.
            num_channels, num_samples = clip.shape
            generation_assertion["audio"] = {
                "sample_rate_hz": int(sample_rate),
                "channels": int(num_channels),
                "samples": int(num_samples),
                "duration_seconds": float(num_samples) / float(sample_rate),
                "format": "audio/wav",
                "encoding": "pcm_s16le",
            }

            ingest_result = _post_to_arc_ingest_audio_with_retry(
                wav_bytes=wav_bytes,
                title=title,
                generation_assertion=generation_assertion,
                api_key=api_key,
            )

            signed_bytes = _download_credentialed_bytes_audio(ingest_result)

            filename = _write_outputs_audio(
                signed_bytes=signed_bytes,
                ingest_result=ingest_result,
                output_dir=output_dir,
                filename_prefix=filename_prefix,
            )
            saved_filenames.append(filename)

        # OUTPUT_NODE ui-result. ComfyUI renders audio results under
        # the "audio" key, same record shape as images but a different
        # bucket — matches SaveAudio's ui-result convention.
        return {
            "ui": {
                "audio": [
                    {"filename": f, "subfolder": "", "type": "output"}
                    for f in saved_filenames
                ]
            }
        }


# ─── ComfyUI registration ──────────────────────────────────────────

NODE_CLASS_MAPPINGS = {"ARCSaveAudio": ARCSaveAudio}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ARCSaveAudio": "ARC Save Audio (sign on arrival)"
}
