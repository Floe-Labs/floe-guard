"""Tests for offline pricing — fail-closed resolution and token math."""

from __future__ import annotations

import pytest

from floe_guard.pricing import (
    ManualPrice,
    PricedModel,
    price_tokens,
    resolve_price,
)


def test_resolves_known_model_from_cost_map() -> None:
    priced = resolve_price("gpt-4o")
    assert priced is not None
    assert priced.source == "cost_map"
    assert priced.input_cost_per_token == pytest.approx(2.5e-6)
    assert priced.output_cost_per_token == pytest.approx(1e-5)


def test_strips_provider_prefix() -> None:
    bare = resolve_price("gpt-4o")
    prefixed = resolve_price("openai/gpt-4o")
    assert prefixed is not None and bare is not None
    assert prefixed.input_cost_per_token == bare.input_cost_per_token


def test_unknown_model_is_unpriceable() -> None:
    assert resolve_price("no-such-model-anywhere") is None


def test_groq_prefixed_multi_segment_ids_resolve() -> None:
    # LiteLLM/CrewAI pass "groq/<org>/<model>"; the map vendors the ChatGroq id
    # ("<org>/<model>"). Stripping the known "groq/" first segment must bridge
    # the two conventions (previously only the exact ChatGroq form resolved).
    for model in (
        "groq/qwen/qwen3-32b",
        "groq/meta-llama/llama-4-scout-17b-16e-instruct",
        "groq/openai/gpt-oss-120b",
    ):
        priced = resolve_price(model)
        assert priced is not None, model
        assert priced.source == "cost_map"
        chatgroq_form = resolve_price(model.removeprefix("groq/"))
        assert chatgroq_form is not None, model
        assert priced.input_cost_per_token == chatgroq_form.input_cost_per_token


def test_bare_multi_provider_names_stay_unpriceable() -> None:
    # Deliberate: a fully-bare name like "qwen3-32b" is served by many providers
    # at different prices; resolving it at Groq's cheap rate would under-meter.
    assert resolve_price("qwen3-32b") is None
    assert resolve_price("gpt-oss-120b") is None


def test_unknown_provider_prefix_is_not_bridged() -> None:
    # Only the groq/ prefix is stripped — another vendor serving the same
    # open-weights model must not inherit Groq's price.
    assert resolve_price("fireworks_ai/qwen/qwen3-32b") is None


def test_openai_and_anthropic_prefixes_do_not_bridge_to_groq_keys() -> None:
    # "openai/<model>" is LiteLLM's route for ANY OpenAI-compatible endpoint
    # (vLLM, OpenRouter, …), so a multi-segment remainder under openai/ or
    # anthropic/ is some other vendor's model and must fail closed — not price
    # at the Groq rate of the vendored "qwen/qwen3-32b"-style keys.
    assert resolve_price("openai/qwen/qwen3-32b") is None
    assert resolve_price("anthropic/qwen/qwen3-32b") is None
    assert resolve_price("openai/meta-llama/llama-4-scout-17b-16e-instruct") is None


def test_gemini_resolves_from_bare_and_prefixed_ids() -> None:
    # The map vendors Gemini under the BARE id the google-genai SDK and
    # @ai-sdk/google actually pass; LiteLLM's "gemini/<id>" and the older
    # "models/<id>" form reach the same entry via the bare-last-segment
    # fallback, so no prefix rule is needed for either.
    bare = resolve_price("gemini-2.5-flash")
    assert bare is not None
    assert bare.input_cost_per_token > 0
    for form in ("gemini/gemini-2.5-flash", "models/gemini-2.5-flash"):
        priced = resolve_price(form)
        assert priced is not None, form
        assert priced.input_cost_per_token == bare.input_cost_per_token, form
        assert priced.output_cost_per_token == bare.output_cost_per_token, form


def test_gemini_vertex_ids_price_at_ai_studio_rates() -> None:
    # DOCUMENTED LIMITATION, asserted so it cannot change silently.
    #
    # Only the AI Studio (Gemini Developer API) prices are vendored. Vertex AI
    # serves the same ids at its own — sometimes dearer — rates, and the id alone
    # cannot say which billing path a call took, so a "vertex_ai/<id>" caller
    # lands on the AI Studio price through the bare-last-segment fallback (the
    # same way "openrouter/openai/gpt-4o" already resolves to "gpt-4o").
    #
    # Vertex callers should pass price_overrides; the Gemini adapter detects them
    # via `client.vertexai`. If this ever needs to fail closed instead, that is a
    # change to the shared resolver in BOTH pricing.py and pricing.ts.
    vertex = resolve_price("vertex_ai/gemini-2.5-flash")
    ai_studio = resolve_price("gemini-2.5-flash")
    assert vertex is not None and ai_studio is not None
    assert vertex.input_cost_per_token == ai_studio.input_cost_per_token


def test_no_vendored_chat_model_bills_output_free() -> None:
    # Regression guard. Upstream mislabels some chat models as mode="embedding"
    # (gemini-1.5-flash was shipped that way, with output_cost_per_token=0), and
    # embedding mode zeroes the output rate — so a wrong mode silently bills a chat
    # model's output at $0, which fail-closed pricing cannot catch because 0 is a
    # finite, valid price. The refresh script now requires an embedding entry's id
    # to start with a known embedding family (EMBEDDING_ID_PREFIXES in
    # scripts/update-cost-map.mjs) — a prefix, not a substring, so a chat model
    # named "foo-embedding-chat" cannot claim the zeroed rate. Mirrored here as
    # the invariant that survives the script.
    from floe_guard.pricing import _COST_MAP

    embedding_prefixes = ("text-embedding-", "gemini-embedding-")
    for model, entry in _COST_MAP.items():
        if entry.get("mode") == "embedding":
            assert model.startswith(embedding_prefixes), (
                f"{model} claims embedding mode but is not named as a known embedding family"
            )
        else:
            assert entry.get("output_cost_per_token", 0) > 0, f"{model} bills output free"
        # Zero input bills every call free just as invisibly, embeddings included.
        assert entry.get("input_cost_per_token", 0) > 0, f"{model} bills input free"


def test_mislabelled_chat_model_is_not_vendored_as_an_embedding() -> None:
    # gemini-1.5-flash is a chat/multimodal model that upstream lists as
    # mode="embedding" with a 0 output rate. There is no correctly-priced variant
    # upstream to fall back to, so it stays unpriceable and fails closed rather
    # than metering chat completions with free output.
    assert resolve_price("gemini-1.5-flash") is None
    assert resolve_price("gemini/gemini-1.5-flash") is None


def test_gemini_free_tier_models_stay_unpriceable() -> None:
    # Upstream lists some experimental/free Gemini entries at 0/0. The refresh
    # script drops zero-priced CHAT models: fail-closed pricing cannot catch them
    # (0 is finite), so vendoring one would meter every call to it at $0 forever.
    # gemini-exp-1206 is listed twice upstream — 0/0 and a real price — and must
    # resolve to the real one.
    priced = resolve_price("gemini-exp-1206")
    assert priced is not None
    assert priced.input_cost_per_token > 0
    assert priced.output_cost_per_token > 0


def test_dated_snapshot_falls_back_to_alias_price() -> None:
    # Anthropic responses carry dated snapshot ids; a snapshot the map doesn't
    # list yet must price at its alias entry instead of failing closed.
    alias = resolve_price("claude-opus-4-8")
    dated = resolve_price("claude-opus-4-8-20991231")
    assert alias is not None and dated is not None
    assert dated.input_cost_per_token == alias.input_cost_per_token
    # OpenAI-style dashed dates, with and without a provider prefix.
    assert resolve_price("gpt-5.5-2099-01-01") is not None
    assert resolve_price("anthropic/claude-sonnet-5-20991231") is not None


def test_exact_dated_key_wins_over_alias_fallback() -> None:
    # A snapshot the map DOES list resolves via its own entry (raw id is the
    # most-specific candidate), not the alias fallback.
    exact = resolve_price("claude-sonnet-4-5-20250929")
    assert exact is not None
    assert exact.source == "cost_map"


def test_override_matches_provider_stripped_candidate() -> None:
    priced = resolve_price("groq/my-model", {"my-model": ManualPrice(1e-6, 2e-6)})
    assert priced is not None
    assert priced.source == "override"


def test_alias_override_does_not_shadow_exact_dated_map_entry() -> None:
    # gpt-4o-2024-05-13 has its OWN map entry at 2x the gpt-4o alias rate. An
    # alias-keyed override (a less-specific, date-stripped match) must not
    # absorb the snapshot — that would meter it at half its true cost.
    exact = resolve_price("gpt-4o-2024-05-13")
    assert exact is not None
    priced = resolve_price("gpt-4o-2024-05-13", {"gpt-4o": ManualPrice(2.5e-6, 1e-5)})
    assert priced is not None
    assert priced.source == "cost_map"
    assert priced.input_cost_per_token == exact.input_cost_per_token
    # The override still wins for the alias itself and for unlisted snapshots.
    assert resolve_price("gpt-4o", {"gpt-4o": ManualPrice(1e-9, 2e-9)}).source == "override"
    assert (
        resolve_price("gpt-4o-2099-01-01", {"gpt-4o": ManualPrice(1e-9, 2e-9)}).source
        == "override"
    )


def test_date_suffix_stripping_is_ascii_only() -> None:
    # TS "\d" is ASCII-only; the Python regex uses re.ASCII to match. A
    # Unicode-digit suffix must not strip to the alias in either package.
    assert resolve_price("gpt-4o-٢٠٢٥٠١٠١") is None


def test_override_wins_over_cost_map() -> None:
    priced = resolve_price("gpt-4o", {"gpt-4o": ManualPrice(1e-9, 2e-9)})
    assert priced is not None
    assert priced.source == "override"
    assert priced.input_cost_per_token == 1e-9


def test_override_with_non_finite_price_is_unpriceable() -> None:
    # Fail closed: a malformed override (NaN/inf) must not be used.
    assert resolve_price("x", {"x": ManualPrice(float("nan"), 1e-6)}) is None
    assert resolve_price("y", {"y": ManualPrice(1e-6, float("inf"))}) is None


def test_price_tokens_math() -> None:
    priced = PricedModel(input_cost_per_token=1e-6, output_cost_per_token=2e-6, source="cost_map")
    assert price_tokens(priced, 1_000, 500) == pytest.approx(1e-3 + 1e-3)


def test_price_tokens_clamps_negative_counts() -> None:
    priced = PricedModel(input_cost_per_token=1e-6, output_cost_per_token=2e-6, source="cost_map")
    assert price_tokens(priced, -50, -50) == 0.0


def test_price_tokens_with_prompt_caching_math() -> None:
    priced = PricedModel(input_cost_per_token=1e-6, output_cost_per_token=2e-6, source="cost_map")
    # Base input price: 1e-6
    # 5-minute write (1.25x): 100 * 1e-6 * 1.25 = 0.000125
    # 1-hour write (2.0x): 200 * 1e-6 * 2.0 = 0.0004
    # Read (0.1x): 1000 * 1e-6 * 0.1 = 0.0001
    # Regular prompt (0 tokens), Completion (0 tokens)
    expected_cost = 0.000125 + 0.0004 + 0.0001
    cost = price_tokens(
        priced,
        prompt_tokens=0,
        completion_tokens=0,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens_1h=200,
    )
    assert cost == pytest.approx(expected_cost)


def test_price_tokens_caching_constants() -> None:
    from floe_guard.pricing import (
        _CACHE_CREATION_1H_MULTIPLIER,
        _CACHE_CREATION_MULTIPLIER,
        _CACHE_READ_MULTIPLIER,
    )
    assert _CACHE_CREATION_MULTIPLIER == 1.25
    assert _CACHE_CREATION_1H_MULTIPLIER == 2.00
    assert _CACHE_READ_MULTIPLIER == 0.10

