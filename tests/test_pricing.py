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
