# ComfyUI ARC Save

Sign-on-arrival save node for ArcVelvetOS. Replaces ComfyUI's `SaveImage` — encodes the image, POSTs the bytes to the ArcVelvet ingest API, writes the C2PA-signed copy returned by the server to your output directory.

**Status: V0 / Day 1 build. Not yet published. Single-image, no retry, no prompt redaction.**

## Why this node

ComfyUI's default `SaveImage` writes an unsigned PNG to disk. ARC Save signs the file before it touches disk: the unsigned bytes never exist on the file system. The signed PNG carries a C2PA manifest with platform-attested identity, your generation workflow, and a vault-bound verify URL. Anyone who later receives the file can verify it cryptographically.

## Setup (development)

### 1. Issue an API key (operator-only — server side)

```bash
cd <arcvelvetos repo>
ARC_API_KEY_PEPPER="$(firebase functions:secrets:access ARC_API_KEY_PEPPER)" \
  node functions/scripts/arc-issue-key.mjs \
    --uid <YOUR_UID> \
    --label "comfyui-arc-save dev"
```

The script prints the plaintext key **exactly once**. Save it.

### 2. Configure the node with your key

Copy the plaintext into `arc_config.json` in this repo's root:

```json
{ "api_key": "arc_live_..." }
```

Or set the `ARC_API_KEY` environment variable.

### ⚠️ Credential safety

**`arc_config.json` holds a live API credential.** A leaked key gives the holder the ability to sign content as you, against your rate-limited budget, until you revoke it.

- The `.gitignore` in this repo prevents accidental commits.
- **Humans need to manually avoid**: screenshots of the file, ZIP-then-share of the custom_nodes directory, screen-shares with the file open, "send me your config" support requests.
- If a key is ever exposed: revoke it by setting `revokedAt` on its api_keys doc (see arcvelvet's `api_keys` collection), then re-issue.

### 3. Install into ComfyUI

```bash
# Symlink or copy this directory into your ComfyUI's custom_nodes:
ln -s "$(pwd)" "<path-to-ComfyUI>/custom_nodes/comfyui-arc-save"

# Or just copy:
cp -r . "<path-to-ComfyUI>/custom_nodes/comfyui-arc-save"
```

Restart ComfyUI. The node appears as `ARC Save (sign on arrival)` under the `image/save` category.

### 4. Use in a workflow

Replace any `SaveImage` node with `ARC Save (sign on arrival)`. The node has the same input shape (`images`, `filename_prefix`) plus a `title` field. (The prompt-redaction toggle ships with the redaction pass — Day 1 deliberately omits it rather than show a visible widget that does nothing.)

On a successful workflow execution, two files land in your output directory:

- `<prefix>_<vaultItemId>.png` — the signed PNG, ready to share
- `<prefix>_<vaultItemId>.arc.json` — sidecar receipt with `vaultItemId`, `verifyUrl`, `contentHash`, `traceId`

Anyone with the verify URL can hit `https://arcvelvet.com/verify?type=vault&owner={uid}&item={vaultItemId}` to see the provenance JSON, and append `&format=file` to download the signed bytes.

## Verification (development / pre-publication)

### Standalone smoke test

```bash
pip install -r requirements.txt
python tests/test_save_smoke.py
```

Sends a 64×64 random-noise PNG against the live `arcIngest`, asserts the receipt structure, cleans up. No ComfyUI install required.

### Manual smoke test in a real ComfyUI

Before any creator depends on this build, run a real workflow (SD1.5 KSampler → VAE Decode → ARC Save) end-to-end through a local ComfyUI install. Verify the signed PNG opens correctly in [Adobe Content Credentials inspector](https://contentauthenticity.adobe.com/inspect) and shows the ArcVelvet platform claim + `com.arcvelvet.generation` assertion. (The inspector will note an unknown CA until the SSL.com cert cutover; see "Cert status" below.)

## What this V0 includes

- Per-image encode + POST + signed-bytes write across the full batch
- Single 2-second retry on transient errors (HTTP 503, Timeout, ConnectionError); all other non-200s raise immediately
- **Prompt redaction by default**: the `include_prompt_text` widget defaults OFF. With it off, text inputs in CLIPTextEncode-style nodes are SHA-256-hashed in the signed assertion (workflow structure preserved; you can reveal the plaintext later and anyone can verify by re-hashing). Flip the widget ON to send the prompt verbatim.
- **Fail-closed catch-all**: any node whose `class_type` contains the substring `textencode` (case-insensitive) gets redacted — new/unknown encoder variants stay safe by default rather than leaking plaintext.
- **Configurable pattern list**: extend the catch-all with `text_encoder_patterns` in `arc_config.json` to cover bespoke encoders whose names don't contain `TextEncode` (e.g. T5Encoder, PromptInput). You can only ADD; the catch-all is locked.
- `arc_config.json` + `ARC_API_KEY` env var key loading
- Fail-loudly on any terminal error (no silent unsigned fallback)
- Sidecar `.arc.json` per signed image with `vaultItemId` / `verifyUrl` / `contentHash`
- Open-shape generation assertion (`batch_index` / `batch_size` / `redacted_prompt` baked in; additive fingerprint slot reserved)

## What's NOT yet in V0

- ComfyUI Manager / Comfy Registry packaging (the install story is currently "symlink the directory into custom_nodes/"; one-line install via Manager lands with the packaging piece)

### Rate-limit note for batch workflows

`arcIngest` allows 5 signings per minute per API key (free-door cap; Sprint A0 tightening) plus 20 signings per minute per uid as an aggregate cap across all keys on the same account. A batch of 6+ images on one key, or 21+ across keys on one account, will trigger `ERR_RATE_LIMITED` partway through and the node will halt with the exact failure point named. Images already signed before the rate-limit hit are saved (and on rerun, the content-hash idempotency on the server will dedup them so you don't double-charge). For larger batches, either wait a minute and rerun the same workflow, or split the batch upstream.

## Cert status (read before sharing signed files publicly)

The signed PNGs verify cryptographically and the embedded provenance is real. Until the SSL.com production certificate is loaded into the platform's signing keys, third-party verifiers (Adobe's inspector, contentauthenticity.org) will note that the issuing CA is not in their trust list. This is expected for the pre-production phase.

For internal testing: works now. For public distribution / external creator demos: production cert cutover is the gate, not anything in this node.

## License

MIT.
