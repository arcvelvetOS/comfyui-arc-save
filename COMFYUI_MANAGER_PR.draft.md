# ComfyUI Manager submission — DRAFT (held)

This document captures the prepared submission to [ltdrdata/ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) for one-line install of ARC Save through Manager. **NOT opened.** Operator gates the upstream PR until the GitHub org + Registry registration land.

## Prerequisites (operator side, before this PR opens)

1. The repo `github.com/arcvelvetOS/comfyui-arc-save` must be public.
2. The repo's `pyproject.toml` (B2a) references the registered Comfy Registry PublisherId `arcvelvet` (done — registered 2026-06-12). Note: PublisherId and GitHub org name are independent; the publisher handle stays `arcvelvet` even though the GitHub org is `arcvelvetOS`.
3. A first release tag should exist (e.g. `v0.1.0`) so Manager's install path has a stable reference.

## PR target

- Repo: `ltdrdata/ComfyUI-Manager`
- Branch base: `main`
- File modified: `custom-node-list.json`

## Diff against `custom-node-list.json`

The file holds a top-level `custom_nodes` array. Insert this object in alphabetical order by `title` (between `Comfy*` and `D*` entries):

```json
{
  "author": "ArcVelvet Studios LLC",
  "title": "ARC Save (sign on arrival)",
  "id": "comfyui-arc-save",
  "reference": "https://github.com/arcvelvetOS/comfyui-arc-save",
  "files": [
    "https://github.com/arcvelvetOS/comfyui-arc-save"
  ],
  "install_type": "git-clone",
  "description": "Sign-on-arrival save node for ArcVelvetOS. Replaces ComfyUI's SaveImage: POSTs the encoded image to the ArcVelvet arcIngest API and writes the C2PA-signed copy returned by the server. The unsigned bytes never touch disk. Each saved file carries a platform-attested C2PA manifest with embedded generation parameters, a verify URL, and a vault-bound record id. Prompt redaction defaults ON (text-encoder inputs are SHA-256 hashed in the assertion; flip a config toggle to send verbatim). Requires an arc:sign-scoped API key issued through the ArcVelvet Credentials surface."
}
```

## PR title

```
[New] ARC Save (sign on arrival) — C2PA sign-on-arrival save node
```

## PR body

```markdown
Adds **ARC Save (sign on arrival)** to the Manager listing.

**What it does.** Replaces ComfyUI's `SaveImage`. Encodes the image to PNG, POSTs to the ArcVelvet `arcIngest` HTTP API, writes the C2PA-signed copy returned by the server to the output directory. Each saved file carries a platform-attested C2PA manifest containing:

- The exact generation parameters (model, sampler, prompt, seed, etc.) as a `com.arcvelvet.generation` assertion. Defaults to **redacted** — text-encoder inputs (CLIPTextEncode and any node whose `class_type` contains `textencode`, case-insensitive) are SHA-256 hashed in the assertion; flip `arc_config.json:include_prompt_text` to `true` to send verbatim.
- The source-bytes content hash (SHA-256), so the file is independently re-verifiable from bytes alone.
- A vault-bound verify URL (`arcvelvet.com/verify?type=vault&owner=...&item=...`) that resolves to a record page.

**Why it's distinct from other save nodes.** The unsigned bytes never touch disk. The signing happens server-side under a managed certificate (currently c2pa-rs pilot fixture, SSL.com production cert pending — see the repo README for cert posture). Standard C2PA verifiers (Adobe CAI, contentauthenticity.org) parse the manifest natively.

**Requirements.**
- ArcVelvet account with an `arc:sign` or `ingest:sign`-scoped API key issued via the Credentials surface in the user's profile.
- `arc_config.json` in the node directory carrying the plaintext key, OR the `ARC_API_KEY` env var.
- See repo README for setup + safety notes.

**Cost.** The `ingest:sign` scope (free door) supports the standard signing path with rate limits (5/min per key, 20/min per uid). The `arc:sign` scope (paid door) lands a commercial provenance record consuming 1 token from the user's ArcVelvet ARC balance per signing.

**Testing.** A standalone smoke script ships in `tests/test_save_smoke.py` that exercises the node end-to-end against the live arcIngest endpoint without requiring a ComfyUI install. CI in the repo runs it on every push.

**License.** MIT (matches the node directory's LICENSE).
```

## Post-merge checklist (operator)

- [ ] Verify the entry resolves correctly under `Manager → Install Custom Nodes → search "ARC"`.
- [ ] Click Install in a clean ComfyUI environment, confirm the node appears under `image/save` after restart.
- [ ] Run the workflow smoke (SD1.5 KSampler → VAE Decode → ARC Save) end-to-end, confirm output PNG validates in Adobe CAI inspector with the expected `com.arcvelvet.generation` assertion.

## Notes

- The Manager listing is independent of the Comfy Registry. Both exist; the Registry is the official packaging index that pyproject.toml feeds, while Manager is a community-maintained installer UI shipped inside ComfyUI. We submit to both — Registry first (B2b), then Manager (this draft).
- If Registry approval reveals a different canonical id (e.g. `arcvelvetOS/comfyui-arc-save` rather than the bare `comfyui-arc-save`), update the `id` field in the JSON entry above before opening.
