"""
test_redaction.py — unit tests for the prompt-redaction pure logic.

NO network. NO API key required. NO ComfyUI. Exercises
_class_type_matches, _redact_prompt_text, and the catch-all + pattern
list composition logic in isolation.

Run via:
    python tests/test_redaction.py

Exit 0 on PASS, 1 on FAIL. Each named test that fails prints the
expectation and what it got. Cost: zero network, runs in milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from nodes.arc_save import (  # noqa: E402
    DEFAULT_TEXT_INPUT_FIELDS_LOWER,
    LOCKED_CATCHALL_PATTERN,
    _class_type_matches,
    _load_text_encoder_patterns,
    _redact_prompt_text,
    _redact_string_value,
)


fail_count = 0


def _assert(name: str, ok: bool, detail: str = "") -> None:
    global fail_count
    if ok:
        print(f"  ok    {name}")
    else:
        fail_count += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


# ─── _class_type_matches ───────────────────────────────────────────


def test_class_type_matches():
    print("\ntest_class_type_matches")
    pats = (LOCKED_CATCHALL_PATTERN,)

    _assert("CLIPTextEncode matches catch-all", _class_type_matches("CLIPTextEncode", pats))
    _assert("CLIPTextEncodeSDXL matches catch-all", _class_type_matches("CLIPTextEncodeSDXL", pats))
    _assert(
        "case-insensitive: CLIPTEXTENCODE matches",
        _class_type_matches("CLIPTEXTENCODE", pats),
    )
    _assert(
        "case-insensitive: cliptextencode matches",
        _class_type_matches("cliptextencode", pats),
    )
    _assert(
        "mixed-case: BNK_CLIPTextEncodeAdvanced matches",
        _class_type_matches("BNK_CLIPTextEncodeAdvanced", pats),
    )
    _assert("KSampler does NOT match", not _class_type_matches("KSampler", pats))
    _assert("VAEDecode does NOT match", not _class_type_matches("VAEDecode", pats))
    _assert("LoraLoader does NOT match", not _class_type_matches("LoraLoader", pats))

    _assert("None safely returns False", not _class_type_matches(None, pats))
    _assert("int safely returns False", not _class_type_matches(42, pats))
    _assert("empty string returns False", not _class_type_matches("", pats))

    # Custom pattern extension
    pats_ext = (LOCKED_CATCHALL_PATTERN, "t5encoder", "myCustom")
    _assert(
        "extended pattern: T5EncoderNode matches via 't5encoder' substring",
        _class_type_matches("T5EncoderNode", pats_ext),
    )
    _assert(
        "extended pattern: myCustomThing matches via 'myCustom' substring",
        _class_type_matches("myCustomThing", pats_ext),
    )
    _assert(
        "extended pattern: KSampler still does NOT match",
        not _class_type_matches("KSampler", pats_ext),
    )


# ─── _redact_string_value ──────────────────────────────────────────


def test_redact_string_value():
    print("\ntest_redact_string_value")
    s = "a cosmic operator hovering above the abyss"
    expected_digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    out = _redact_string_value(s)
    _assert(
        "envelope shape [REDACTED:sha256:<64hex>]",
        out == f"[REDACTED:sha256:{expected_digest}]",
        f"got: {out}",
    )
    _assert("digest is 64 hex chars", len(expected_digest) == 64)

    # Idempotency: hashing a value already in the envelope is fine
    # (the envelope text becomes the new plaintext; the hash differs
    # from the original. We just confirm no exception.)
    rehash = _redact_string_value(out)
    _assert("rehashing returns another envelope", rehash.startswith("[REDACTED:sha256:"))

    # Empty string is a valid input — emits a deterministic hash.
    out_empty = _redact_string_value("")
    _assert("empty string emits standard SHA-256 empty hash", out_empty.endswith(
        hashlib.sha256(b"").hexdigest() + "]"
    ))


# ─── _redact_prompt_text ───────────────────────────────────────────


def test_redact_prompt_text_basic():
    print("\ntest_redact_prompt_text_basic")
    pats = (LOCKED_CATCHALL_PATTERN,)

    prompt = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a cosmic operator", "clip": ["model", 0]},
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {"seed": 12345, "steps": 20},
        },
        "3": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["1", 0], "vae": ["model", 1]},
        },
    }

    out = _redact_prompt_text(prompt, pats)
    encoded_hash = hashlib.sha256("a cosmic operator".encode("utf-8")).hexdigest()
    _assert(
        "CLIPTextEncode.inputs.text is redacted",
        out["1"]["inputs"]["text"] == f"[REDACTED:sha256:{encoded_hash}]",
        f"got: {out['1']['inputs']['text']}",
    )
    _assert(
        "CLIPTextEncode.inputs.clip is preserved (non-string)",
        out["1"]["inputs"]["clip"] == ["model", 0],
    )
    _assert(
        "KSampler.inputs.seed is preserved (non-text-field node)",
        out["2"]["inputs"]["seed"] == 12345,
    )
    _assert(
        "VAEDecode is preserved (non-encoder node)",
        out["3"]["inputs"]["samples"] == ["1", 0],
    )


def test_redact_prompt_text_sdxl_split():
    print("\ntest_redact_prompt_text_sdxl_split")
    pats = (LOCKED_CATCHALL_PATTERN,)
    prompt = {
        "1": {
            "class_type": "CLIPTextEncodeSDXL",
            "inputs": {
                "text_g": "cinematic",
                "text_l": "studio lighting",
                "width": 1024,
                "height": 1024,
            },
        },
    }
    out = _redact_prompt_text(prompt, pats)
    h_g = hashlib.sha256(b"cinematic").hexdigest()
    h_l = hashlib.sha256(b"studio lighting").hexdigest()
    _assert("text_g redacted", out["1"]["inputs"]["text_g"] == f"[REDACTED:sha256:{h_g}]")
    _assert("text_l redacted", out["1"]["inputs"]["text_l"] == f"[REDACTED:sha256:{h_l}]")
    _assert("width preserved", out["1"]["inputs"]["width"] == 1024)


def test_redact_prompt_text_immutability():
    print("\ntest_redact_prompt_text_immutability")
    pats = (LOCKED_CATCHALL_PATTERN,)
    prompt = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "original"},
        }
    }
    original_text = prompt["1"]["inputs"]["text"]
    _ = _redact_prompt_text(prompt, pats)
    _assert(
        "input dict NOT mutated",
        prompt["1"]["inputs"]["text"] == original_text,
        f"input mutated to: {prompt['1']['inputs']['text']}",
    )


def test_redact_prompt_text_malformed():
    print("\ntest_redact_prompt_text_malformed")
    pats = (LOCKED_CATCHALL_PATTERN,)

    _assert("None returns None", _redact_prompt_text(None, pats) is None)
    _assert("empty dict returns empty dict", _redact_prompt_text({}, pats) == {})
    _assert("list (non-dict) passes through", _redact_prompt_text([], pats) == [])
    _assert("string (non-dict) passes through", _redact_prompt_text("hi", pats) == "hi")

    # Malformed node spec (non-dict node_spec) should pass through
    weird = {"1": "not a dict", "2": None}
    _assert("non-dict node specs pass through", _redact_prompt_text(weird, pats) == weird)

    # Node with no class_type
    no_class = {"1": {"inputs": {"text": "x"}}}
    out = _redact_prompt_text(no_class, pats)
    _assert("node without class_type is not redacted", out["1"]["inputs"]["text"] == "x")

    # Node with non-dict inputs
    bad_inputs = {"1": {"class_type": "CLIPTextEncode", "inputs": "wrong shape"}}
    out = _redact_prompt_text(bad_inputs, pats)
    _assert("non-dict inputs pass through", out["1"]["inputs"] == "wrong shape")


def test_catchall_locked():
    print("\ntest_catchall_locked")
    # _load_text_encoder_patterns ALWAYS includes the catch-all,
    # regardless of what arc_config.json holds.
    pats = _load_text_encoder_patterns()
    _assert(
        "loaded patterns include the locked catch-all",
        LOCKED_CATCHALL_PATTERN in pats,
        f"loaded patterns: {pats}",
    )
    _assert(
        "loaded patterns is non-empty even without arc_config.json",
        len(pats) >= 1,
    )


def test_default_text_input_fields():
    print("\ntest_default_text_input_fields")
    # Sanity check on the field-name list — these are the canonical
    # CLIPTextEncode-family input names.
    expected_subset = {"text", "prompt", "text_g", "text_l", "positive", "negative"}
    _assert(
        "default text-input fields include the CLIPTextEncode canonical set",
        expected_subset.issubset(set(DEFAULT_TEXT_INPUT_FIELDS_LOWER)),
        f"got: {DEFAULT_TEXT_INPUT_FIELDS_LOWER}",
    )


def test_non_string_text_value_preserved():
    print("\ntest_non_string_text_value_preserved")
    # When the "text" input is a node-output reference (a list), the
    # redactor leaves it alone — it's not plaintext to begin with.
    pats = (LOCKED_CATCHALL_PATTERN,)
    prompt = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ["2", 0], "clip": ["model", 0]},
        }
    }
    out = _redact_prompt_text(prompt, pats)
    _assert(
        "node-output-reference (list) in text field is preserved",
        out["1"]["inputs"]["text"] == ["2", 0],
    )


# ─── Main ──────────────────────────────────────────────────────────


def main() -> int:
    test_class_type_matches()
    test_redact_string_value()
    test_redact_prompt_text_basic()
    test_redact_prompt_text_sdxl_split()
    test_redact_prompt_text_immutability()
    test_redact_prompt_text_malformed()
    test_catchall_locked()
    test_default_text_input_fields()
    test_non_string_text_value_preserved()

    print("")
    if fail_count == 0:
        print(f"PASS — all redaction unit tests green.")
        return 0
    print(f"FAIL — {fail_count} assertion(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
