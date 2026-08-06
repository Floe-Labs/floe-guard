"""Persistent state stores for cross-process budget enforcement.

The core guard remains in-memory unless a store is explicitly configured.
``SqliteStore`` uses only the Python standard library and serializes each
reserve/settle/release mutation in a SQLite write transaction, allowing
separate processes to enforce one shared UTC-day ceiling.
"""

from __future__ import annotations

import math
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

_EPS = 1e-12
_BUSY_TIMEOUT_MS = 5_000


class StateStore(Protocol):
    """Storage contract for one budget's windowed spend and reservations.

    Snapshots are returned as ``(spent_usd, reserved_usd)``. Implementations
    must make each mutation atomic across every caller sharing the store.
    """

    def load(self, window_id: str, limit_usd: float) -> tuple[float, float]:
        """Load or initialize a window and return its authoritative snapshot."""

    def reserve(
        self, window_id: str, limit_usd: float, amount_usd: float
    ) -> tuple[bool, float, float]:
        """Atomically hold ``amount_usd`` if it fits beneath the ceiling.

        Returns ``(accepted, spent_usd, reserved_usd)``. A rejected operation
        must leave the stored snapshot unchanged.
        """

    def settle(
        self,
        window_id: str,
        limit_usd: float,
        reserved_usd: float,
        actual_usd: float,
    ) -> tuple[float, float]:
        """Atomically consume a hold and add actual spend."""

    def release(self, window_id: str, limit_usd: float, reserved_usd: float) -> tuple[float, float]:
        """Atomically consume a hold without adding spend."""


class SqliteStore:
    """SQLite-backed :class:`StateStore` for one logical shared budget.

    Args:
        path: Filesystem path to the SQLite database. Every guard sharing this
            path and UTC-day window shares one ceiling.

    Connections are short-lived so a store can be used safely from multiple
    threads and processes. A process that exits without settling or releasing
    a reservation leaves that hold in place; this deliberately fails closed.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        connection = self._connect()
        try:
            # WAL lets readers proceed while one writer commits, so the read-heavy
            # check()/advisory() paths don't queue behind a settling process. The
            # mode is persisted on the file. Some network filesystems can't back
            # WAL's shared memory — fall back to the default journal rather than
            # failing to open the store.
            try:
                connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS budget_windows (
                    window_id TEXT PRIMARY KEY,
                    limit_usd REAL NOT NULL,
                    spent_usd REAL NOT NULL,
                    reserved_usd REAL NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def load(self, window_id: str, limit_usd: float) -> tuple[float, float]:
        """Load or initialize ``window_id`` and return its snapshot.

        Read-only fast path: an existing window is read in autocommit mode (a
        brief shared lock, no write lock), so the common case — every
        ``check()`` / ``advisory()`` / ``remaining_usd`` on a window that already
        exists — never serializes behind a writer. Only a missing window escalates
        to the ``BEGIN IMMEDIATE`` write transaction to create it, and that
        transaction re-reads first, so two processes racing to create the same
        window stay correct.
        """
        _validate_window(window_id, limit_usd)
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT limit_usd, spent_usd, reserved_usd
                FROM budget_windows
                WHERE window_id = ?
                """,
                (window_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is not None:
            return self._snapshot_from_row(row, window_id, limit_usd)
        with self._transaction() as connection:
            return self._load_or_create_locked(connection, window_id, limit_usd)

    def reserve(
        self, window_id: str, limit_usd: float, amount_usd: float
    ) -> tuple[bool, float, float]:
        """Atomically check the ceiling and add an in-flight hold."""
        _validate_amount("amount_usd", amount_usd)
        with self._transaction() as connection:
            spent, reserved = self._load_or_create_locked(connection, window_id, limit_usd)
            committed = spent + reserved
            if committed > limit_usd - _EPS or committed + amount_usd > limit_usd + _EPS:
                return False, spent, reserved
            reserved += amount_usd
            connection.execute(
                "UPDATE budget_windows SET reserved_usd = ? WHERE window_id = ?",
                (reserved, window_id),
            )
            return True, spent, reserved

    def settle(
        self,
        window_id: str,
        limit_usd: float,
        reserved_usd: float,
        actual_usd: float,
    ) -> tuple[float, float]:
        """Atomically consume a reservation and accrue actual spend."""
        _validate_amount("reserved_usd", reserved_usd)
        _validate_amount("actual_usd", actual_usd)
        with self._transaction() as connection:
            spent, total_reserved = self._load_or_create_locked(connection, window_id, limit_usd)
            total_reserved = _consume_reservation(total_reserved, reserved_usd)
            spent += actual_usd
            if 0.0 < spent - limit_usd < _EPS:
                spent = limit_usd
            connection.execute(
                """
                UPDATE budget_windows
                SET spent_usd = ?, reserved_usd = ?
                WHERE window_id = ?
                """,
                (spent, total_reserved, window_id),
            )
            return spent, total_reserved

    def release(self, window_id: str, limit_usd: float, reserved_usd: float) -> tuple[float, float]:
        """Atomically consume a reservation without accruing spend."""
        _validate_amount("reserved_usd", reserved_usd)
        with self._transaction() as connection:
            spent, total_reserved = self._load_or_create_locked(connection, window_id, limit_usd)
            total_reserved = _consume_reservation(total_reserved, reserved_usd)
            connection.execute(
                "UPDATE budget_windows SET reserved_usd = ? WHERE window_id = ?",
                (total_reserved, window_id),
            )
            return spent, total_reserved

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1_000)
        connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _load_or_create_locked(
        self, connection: sqlite3.Connection, window_id: str, limit_usd: float
    ) -> tuple[float, float]:
        _validate_window(window_id, limit_usd)
        row = connection.execute(
            """
            SELECT limit_usd, spent_usd, reserved_usd
            FROM budget_windows
            WHERE window_id = ?
            """,
            (window_id,),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO budget_windows (window_id, limit_usd, spent_usd, reserved_usd)
                VALUES (?, ?, 0.0, 0.0)
                """,
                (window_id, limit_usd),
            )
            return 0.0, 0.0
        return self._snapshot_from_row(row, window_id, limit_usd)

    @staticmethod
    def _snapshot_from_row(
        row: tuple[float, float, float], window_id: str, limit_usd: float
    ) -> tuple[float, float]:
        """Validate a stored ``(limit, spent, reserved)`` row and return its snapshot."""
        stored_limit, spent, reserved = (float(value) for value in row)
        if not math.isclose(stored_limit, limit_usd, rel_tol=0.0, abs_tol=_EPS):
            raise ValueError(
                f"stored limit_usd ({stored_limit!r}) does not match guard limit_usd "
                f"({limit_usd!r}) for window {window_id!r}"
            )
        _validate_amount("stored spent_usd", spent)
        _validate_amount("stored reserved_usd", reserved)
        return spent, reserved


def _validate_window(window_id: str, limit_usd: float) -> None:
    if not window_id:
        raise ValueError("window_id must not be empty")
    _validate_amount("limit_usd", limit_usd)


def _validate_amount(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")


def _consume_reservation(total_reserved: float, reserved_usd: float) -> float:
    if reserved_usd > total_reserved + _EPS:
        raise ValueError(
            f"reserved handle ({reserved_usd!r}) exceeds total in-flight reservations "
            f"({total_reserved!r}) — a handle must come from a matching reserve()"
        )
    return max(0.0, total_reserved - reserved_usd)


__all__ = ["SqliteStore", "StateStore"]
