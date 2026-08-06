"""UTC-day SQLite persistence and cross-process enforcement (issue #47)."""

from __future__ import annotations

import multiprocessing
import pickle
import queue
import time
from pathlib import Path

import pytest

import floe_guard.guard as guard_module
from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    ManualPrice,
    SqliteStore,
    StreamGuard,
    UnpriceableModelError,
    UnpriceableModelWarning,
)

_LIMIT = 0.05
_HOLD = 0.02


def _record_in_process(path: str) -> None:
    guard = BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))
    guard.record_tool("cron", 0.25)


def _overlapping_worker(
    path: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    guard = BudgetGuard(
        _LIMIT,
        window="utc-day",
        store=SqliteStore(path),
        on_block=lambda *_: None,
    )
    start.wait(timeout=10)
    try:
        reserved = guard.reserve_tool(_HOLD)
    except BudgetExceeded:
        results.put("blocked")
        return
    results.put("reserved")
    time.sleep(0.1)
    guard.settle_tool("cron", _HOLD, reserved=reserved)
    results.put("settled")


def test_sequential_process_loads_previous_spend(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_record_in_process, args=(str(path),))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    guard = BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))
    assert guard.spent_usd == pytest.approx(0.25)
    assert guard.remaining_usd == pytest.approx(0.75)


def test_reservation_survives_reconstruction_and_release(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    first = BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))
    reserved = first.reserve_tool(0.4)

    second = BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))
    assert second.spent_usd == 0.0
    assert second.remaining_usd == pytest.approx(0.6)

    second.release(reserved)
    assert second.remaining_usd == pytest.approx(1.0)
    assert SqliteStore(path).load(guard_module._utc_day_window_id(), 1.0) == (0.0, 0.0)


def test_utc_day_rollover_uses_a_fresh_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_day = ["2026-08-04"]
    monkeypatch.setattr(guard_module, "_utc_day_window_id", lambda: current_day[0])
    store = SqliteStore(tmp_path / "budget.sqlite3")
    guard = BudgetGuard(1.0, window="utc-day", store=store)
    guard.record_tool("day-one", 0.3)

    current_day[0] = "2026-08-05"
    assert guard.remaining_usd == pytest.approx(1.0)
    assert guard.advisory().expected_cost == 0.0
    assert guard.tool_costs == {"day-one": pytest.approx(0.3)}
    assert [event.model_or_tool for event in guard.spend_log] == ["day-one"]
    guard.record_tool("day-two", 0.1)

    assert guard.spent_usd == pytest.approx(0.1)
    assert guard.tool_costs == {
        "day-one": pytest.approx(0.3),
        "day-two": pytest.approx(0.1),
    }
    assert [event.model_or_tool for event in guard.spend_log] == ["day-one", "day-two"]
    assert store.load("2026-08-04", 1.0) == pytest.approx((0.3, 0.0))
    assert store.load("2026-08-05", 1.0) == pytest.approx((0.1, 0.0))


def test_reservations_settle_and_release_against_their_issuing_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_day = ["2026-08-04"]
    monkeypatch.setattr(guard_module, "_utc_day_window_id", lambda: current_day[0])
    store = SqliteStore(tmp_path / "budget.sqlite3")
    guard = BudgetGuard(1.0, window="utc-day", store=store)
    settled = guard.reserve_tool(0.2)
    released = pickle.loads(pickle.dumps(guard.reserve_tool(0.3)))

    current_day[0] = "2026-08-05"
    guard.record_tool("day-two", 0.1)
    guard.settle_tool("cross-midnight", 0.2, reserved=settled)
    assert guard.spent_usd == pytest.approx(0.1)
    guard.release(released)
    assert guard.spent_usd == pytest.approx(0.1)

    assert store.load("2026-08-04", 1.0) == pytest.approx((0.2, 0.0))
    assert store.load("2026-08-05", 1.0) == pytest.approx((0.1, 0.0))


def test_stream_settles_against_its_reservation_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_day = ["2026-08-04"]
    monkeypatch.setattr(guard_module, "_utc_day_window_id", lambda: current_day[0])
    store = SqliteStore(tmp_path / "budget.sqlite3")
    guard = BudgetGuard(1.0, window="utc-day", store=store)
    reserved = guard.reserve(0.2)
    stream = StreamGuard(
        guard,
        "manual",
        reserved=reserved,
        price=ManualPrice(input_cost_per_token=0.2, output_cost_per_token=0.0),
    )

    current_day[0] = "2026-08-05"
    stream.finish(prompt_tokens=1, completion_tokens=0)

    assert store.load("2026-08-04", 1.0) == pytest.approx((0.2, 0.0))
    assert store.load("2026-08-05", 1.0) == pytest.approx((0.0, 0.0))


def test_persistence_configuration_is_explicit(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "budget.sqlite3")
    with pytest.raises(ValueError, match="window and store must be configured together"):
        BudgetGuard(1.0, store=store)
    with pytest.raises(ValueError, match="window and store must be configured together"):
        BudgetGuard(1.0, window="utc-day")
    with pytest.raises(ValueError, match="window must be 'utc-day' or None"):
        BudgetGuard(1.0, window="hour", store=store)  # type: ignore[arg-type]


def test_conflicting_limit_for_same_window_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))

    with pytest.raises(ValueError, match="stored limit_usd"):
        BudgetGuard(2.0, window="utc-day", store=SqliteStore(path))


def test_record_paths_persist_atomically(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    first = BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))
    first.record("manual", 1_000, 0, price=ManualPrice(0.0001, 0.0))
    first.record_tool("search", 0.2)

    second = BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))
    assert second.spent_usd == pytest.approx(0.3)
    assert second.remaining_usd == pytest.approx(0.7)


def test_failed_settlement_releases_persisted_reservation(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    guard = BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))
    reserved = guard.reserve_tool(0.4)

    with pytest.warns(UnpriceableModelWarning):
        with pytest.raises(UnpriceableModelError):
            guard.settle("unknown-model", 1, 1, reserved=reserved)

    reloaded = BudgetGuard(1.0, window="utc-day", store=SqliteStore(path))
    assert reloaded.remaining_usd == pytest.approx(1.0)


def test_overlapping_processes_share_reservations_and_ceiling(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_overlapping_worker, args=(str(path), start, results))
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    messages: list[str] = []
    while True:
        try:
            messages.append(results.get_nowait())
        except queue.Empty:
            break

    assert messages.count("reserved") == 2
    assert messages.count("settled") == 2
    assert messages.count("blocked") == 6
    spent, reserved = SqliteStore(path).load(guard_module._utc_day_window_id(), _LIMIT)
    assert spent == pytest.approx(0.04)
    assert reserved == pytest.approx(0.0)
    assert spent <= _LIMIT


def test_no_store_keeps_the_in_memory_contract() -> None:
    guard = BudgetGuard(1.0)
    reserved = guard.reserve_tool(0.2)

    assert type(reserved) is float
    assert guard.remaining_usd == pytest.approx(0.8)
    guard.release(reserved)
    assert guard.remaining_usd == pytest.approx(1.0)
