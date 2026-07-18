"""LangChain adapter (optional extra: ``pip install floe-guard[langchain]``).

:func:`budget_guard_callback_handler` builds a LangChain callback handler you pass
to any LLM or chat model. It checks the budget *before* the call
(``on_llm_start`` / ``on_chat_model_start``) — raising
:class:`~floe_guard.BudgetExceeded` aborts the call so it never runs — and records
spend *after* the response (``on_llm_end``).

    from langchain_openai import ChatOpenAI
    from floe_guard import BudgetGuard
    from floe_guard.integrations.langchain import budget_guard_callback_handler

    guard = BudgetGuard(limit_usd=1.00)
    llm = ChatOpenAI(model="gpt-4o", callbacks=[budget_guard_callback_handler(guard)])
    llm.invoke("hello")

Every priced response routes through the same :class:`~floe_guard.BudgetGuard` as
the other adapters, so token usage is priced via the bundled cost map.
"""

from __future__ import annotations

from typing import Any

from ..guard import BudgetGuard
from ..stream import approx_tokens


def _require_langchain() -> Any:
    try:
        import langchain_core  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The LangChain adapter requires langchain-core. "
            "Install with: pip install floe-guard[langchain]"
        ) from e
    return langchain_core


def _estimate_start(guard: BudgetGuard, serialized: Any, texts: list[str]) -> float | None:
    """Request-sized cost estimate for the pre-call check.

    ``check()`` defaults to predicting from the LAST call's cost — blind on the
    first call and wrong for a much larger one. Size the prediction to the
    request instead: model id and ``max_tokens`` from the serialized model
    config, prompt tokens via the ~4 chars/token heuristic (langchain-core has
    no tokenizer; a rough request-sized figure still beats a stale or zero
    baseline). Returns ``None`` when the model is unknown/unpriceable —
    ``check(None)`` keeps the old behaviour.
    """
    model_kwargs = serialized.get("kwargs") if isinstance(serialized, dict) else None
    if not isinstance(model_kwargs, dict):
        return None
    model = model_kwargs.get("model_name") or model_kwargs.get("model")
    if not model:
        return None
    prompt_tokens = sum(approx_tokens(t) for t in texts if isinstance(t, str))
    max_out = model_kwargs.get("max_tokens") or model_kwargs.get("max_completion_tokens") or 0
    try:
        max_out = max(0, int(max_out))
    except (TypeError, ValueError):
        max_out = 0
    return guard.estimate_call(str(model), prompt_tokens, max_out)


def _chat_texts(messages: Any) -> list[str]:
    """Flatten LangChain's batched chat messages to their text contents."""
    texts: list[str] = []
    for batch in messages or []:
        for message in batch or []:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                texts.append(content)
    return texts


def _model_from_result(response: Any) -> str:
    """Pull the model name from a LangChain ``LLMResult``."""
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        model = llm_output.get("model_name") or llm_output.get("model")
        if model:
            return str(model)
    return ""


def _usage_from_result(response: Any) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from a LangChain ``LLMResult``.

    Handles both shapes LangChain emits: the provider ``token_usage`` block in
    ``llm_output`` (e.g. OpenAI: ``prompt_tokens``/``completion_tokens``) and the
    standardized ``usage_metadata`` on a message (``input_tokens``/``output_tokens``).
    """
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)
            if prompt > 0 or completion > 0:
                return prompt, completion

    # Fall back to per-message usage_metadata (input_tokens/output_tokens), the
    # provider-agnostic shape newer chat models attach to each generation.
    prompt = completion = 0
    for batch in getattr(response, "generations", None) or []:
        for gen in batch:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if isinstance(meta, dict):
                prompt += int(meta.get("input_tokens", 0) or 0)
                completion += int(meta.get("output_tokens", 0) or 0)
    return prompt, completion


def _record_result(guard: BudgetGuard, response: Any) -> None:
    model = _model_from_result(response)
    prompt_tokens, completion_tokens = _usage_from_result(response)
    if prompt_tokens <= 0 and completion_tokens <= 0:
        # No tokens were spent — nothing to meter.
        return
    # There IS spend to account for. Route it through record() even when the model
    # id is missing, so the guard's configured policy applies (fail-closed → warn +
    # raise; fail-open → warn + skip). Silently skipping here would let a real,
    # completed call go unmetered and skew the next check().
    guard.record(model, prompt_tokens, completion_tokens)


def budget_guard_callback_handler(guard: BudgetGuard) -> Any:
    """Build a LangChain callback handler that enforces ``guard`` on every call.

    Pass it to any LLM or chat model::

        llm = ChatOpenAI(model="gpt-4o", callbacks=[budget_guard_callback_handler(guard)])

    ``on_llm_start``/``on_chat_model_start`` run ``guard.check()`` (raising
    :class:`~floe_guard.BudgetExceeded` to abort), and ``on_llm_end`` records the
    response's token cost.
    """
    _require_langchain()
    from langchain_core.callbacks import BaseCallbackHandler

    class BudgetGuardCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
        # LangChain swallows exceptions raised inside callbacks by default, which
        # would let a blocked call run anyway. raise_error=True propagates
        # BudgetExceeded so the call is actually aborted — the whole point.
        raise_error = True

        def __init__(self) -> None:
            super().__init__()
            self.guard = guard

        def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
            # Request-sized when derivable (see _estimate_start); check(None)
            # falls back to the last-cost prediction.
            texts = [p for p in (prompts or []) if isinstance(p, str)]
            self.guard.check(_estimate_start(self.guard, serialized, texts))

        def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
            self.guard.check(_estimate_start(self.guard, serialized, _chat_texts(messages)))

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            _record_result(self.guard, response)

    return BudgetGuardCallbackHandler()


__all__ = ["budget_guard_callback_handler"]
