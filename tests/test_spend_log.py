"""Tests for the per-call spend ledger (guard.spend_log / export_log / record_tool)."""

from __future__ import annotations

import json

import pytest

from floe_guard import BudgetExceeded, BudgetGuard, SpendEvent, UnpriceableModelWarning

MODEL = "gpt-4o"  # 1k in + 1k out = $0.0025 + $0.01 = $0.0125/call


def test_every_priced_call_appends_one_event_and_ledger_sums_to_spent() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    n = 7
    for _ in range(n):
        guard.record(MODEL, 1_000, 1_000)
    log = guard.spend_log
    assert len(log) == n
    assert all(isinstance(e, SpendEvent) for e in log)
    assert sum(e.cost_usd for e in log) == pytest.approx(guard.spent_usd)


def test_llm_event_schema() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    cost = guard.record(MODEL, 1_200, 350, label="researcher")
    (event,) = guard.spend_log
    assert event.kind == "llm"
    assert event.model_or_tool == MODEL
    assert event.prompt_tokens == 1_200
    assert event.completion_tokens == 350
    assert event.cost_usd == pytest.approx(cost)
    assert event.label == "researcher"
    assert event.reserved is None  # plain record(): no reservation to log
    assert event.timestamp > 0


def test_settle_logs_the_reservation_it_settled() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    handle = guard.reserve(0.05)
    guard.settle(MODEL, 1_000, 1_000, reserved=handle)
    (event,) = guard.spend_log
    assert event.reserved == pytest.approx(0.05)


def test_record_tool_accrues_and_logs_a_tool_event() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard.record(MODEL, 1_000, 1_000)
    returned = guard.record_tool("serpapi.search", 0.01, label="researcher")
    assert returned == pytest.approx(0.01)
    assert guard.spent_usd == pytest.approx(0.0125 + 0.01)
    tool_event = guard.spend_log[-1]
    assert tool_event.kind == "tool"
    assert tool_event.model_or_tool == "serpapi.search"
    assert tool_event.prompt_tokens is None
    assert tool_event.completion_tokens is None
    assert tool_event.cost_usd == pytest.approx(0.01)
    assert tool_event.label == "researcher"
    # The ledger invariant holds across mixed llm + tool events.
    assert sum(e.cost_usd for e in guard.spend_log) == pytest.approx(guard.spent_usd)


def test_record_tool_spend_counts_toward_the_ceiling() -> None:
    guard = BudgetGuard(limit_usd=0.05)
    guard.record_tool("scraper", 0.05)
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_record_tool_rejects_bad_costs() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    for bad in (float("nan"), float("inf"), -0.01):
        with pytest.raises(ValueError):
            guard.record_tool("tool", bad)
    assert guard.spend_log == []


def test_unpriceable_skip_path_logs_nothing() -> None:
    guard = BudgetGuard(limit_usd=1.00, fail_closed=False)
    with pytest.warns(UnpriceableModelWarning):
        guard.record("model-that-does-not-exist", 1_000, 1_000)
    assert guard.spend_log == []
    assert guard.spent_usd == 0.0


def test_max_log_events_is_a_ring_buffer_keeping_the_newest() -> None:
    guard = BudgetGuard(limit_usd=10.00, max_log_events=3)
    for i in range(5):
        guard.record(MODEL, 1_000, 1_000, label=f"call-{i}")
    log = guard.spend_log
    assert [e.label for e in log] == ["call-2", "call-3", "call-4"]
    # The cap bounds the ledger only — totals still cover all 5 calls.
    assert guard.spent_usd == pytest.approx(5 * 0.0125)


def test_max_log_events_zero_disables_the_ledger_but_not_the_totals() -> None:
    guard = BudgetGuard(limit_usd=1.00, max_log_events=0)
    guard.record(MODEL, 1_000, 1_000)
    assert guard.spend_log == []
    assert guard.export_log() == ""
    assert guard.spent_usd == pytest.approx(0.0125)


def test_max_log_events_validation() -> None:
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            BudgetGuard(limit_usd=1.00, max_log_events=bad)  # type: ignore[arg-type]


def test_spend_log_returns_a_snapshot_copy() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard.record(MODEL, 1_000, 1_000)
    snapshot = guard.spend_log
    snapshot.clear()
    assert len(guard.spend_log) == 1


def test_export_log_is_stable_jsonl() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    handle = guard.reserve(0.02)
    guard.settle(MODEL, 1_000, 1_000, reserved=handle, label="writer")
    guard.record(MODEL, 500, 100)
    guard.record_tool("browser", 0.001)

    out = guard.export_log()
    assert out.endswith("\n")
    lines = out.splitlines()
    assert len(lines) == 3

    settled, recorded, tool = (json.loads(line) for line in lines)
    # Fixed key order, optional fields present only when set — the schema the TS
    # package's exportLog() emits field-for-field.
    assert list(settled) == [
        "timestamp",
        "kind",
        "model_or_tool",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "label",
        "reserved",
    ]
    assert list(recorded) == [
        "timestamp",
        "kind",
        "model_or_tool",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
    ]
    assert tool["kind"] == "tool"
    assert tool["prompt_tokens"] is None and tool["completion_tokens"] is None
    assert sum(row["cost_usd"] for row in (settled, recorded, tool)) == pytest.approx(
        guard.spent_usd
    )


def test_export_log_keeps_unicode_raw_like_json_stringify() -> None:
    # JS JSON.stringify never \u-escapes non-ASCII; the Python export must not
    # either, or the same label would serialise differently per language.
    guard = BudgetGuard(limit_usd=1.00)
    guard.record(MODEL, 10, 10, label="café-agent")
    assert "café-agent" in guard.export_log()


def test_export_log_empty_ledger_is_empty_string() -> None:
    assert BudgetGuard(limit_usd=1.00).export_log() == ""
