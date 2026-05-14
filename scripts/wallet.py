from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Iterable


class TransactionType(str, Enum):
    SET_BALANCE = "SET_BALANCE"
    DEPOSIT = "DEPOSIT"
    PAYMENT = "PAYMENT"
    ADJUSTMENT = "ADJUSTMENT"


@dataclass(frozen=True)
class WalletSnapshot:
    balance: int
    pending_type: TransactionType | None = None
    pending_total: int = 0


@dataclass
class PendingTransaction:
    transaction_type: TransactionType
    notes: list[int] = field(default_factory=list)
    target_amount: int | None = None

    @property
    def total(self) -> int:
        if self.transaction_type == TransactionType.SET_BALANCE:
            return int(self.target_amount or 0)
        return sum(self.notes)


class WalletStore:
    """Persistent EGP wallet with explicit pending transaction confirmation."""

    def __init__(self, db_path: Path) -> None:
        self._logger = logging.getLogger("wallet")
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: PendingTransaction | None = None
        self._init_db()
        self._logger.info("WALLET_INIT db_path=%s balance=%s", self._db_path, self.get_balance())

    def get_balance(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM wallet_state WHERE key = 'balance'"
            ).fetchone()
        return int(row[0]) if row else 0

    def set_balance(self, amount: int) -> None:
        self._validate_amount(amount, allow_zero=True)
        self._pending = PendingTransaction(
            TransactionType.SET_BALANCE,
            target_amount=amount,
        )
        self._logger.info("WALLET_PENDING type=%s amount=%s", TransactionType.SET_BALANCE.value, amount)

    def begin_transaction(self, transaction_type: TransactionType) -> None:
        if transaction_type == TransactionType.SET_BALANCE:
            raise ValueError("Use set_balance(amount) for SET_BALANCE transactions")
        self._pending = PendingTransaction(transaction_type)
        self._logger.info("WALLET_PENDING type=%s", transaction_type.value)

    def add_scanned_note(self, value: int) -> None:
        self._validate_amount(value)
        if self._pending is None:
            return
        if self._pending.transaction_type not in {
            TransactionType.DEPOSIT,
            TransactionType.PAYMENT,
        }:
            return
        self._pending.notes.append(value)
        self._logger.info(
            "WALLET_PENDING_NOTE type=%s value=%s total=%s notes=%s",
            self._pending.transaction_type.value,
            value,
            self._pending.total,
            self._pending.notes,
        )

    def preview_transaction_total(self) -> int:
        return self._pending.total if self._pending else 0

    def snapshot(self) -> WalletSnapshot:
        return WalletSnapshot(
            balance=self.get_balance(),
            pending_type=self._pending.transaction_type if self._pending else None,
            pending_total=self.preview_transaction_total(),
        )

    def commit_transaction(self) -> WalletSnapshot:
        if self._pending is None:
            self._logger.info("WALLET_COMMIT_SKIPPED no_pending=True")
            return self.snapshot()

        current_balance = self.get_balance()
        transaction_type = self._pending.transaction_type
        amount = self._pending.total
        self._logger.info(
            "WALLET_COMMIT_START type=%s amount=%s current_balance=%s",
            transaction_type.value,
            amount,
            current_balance,
        )

        if transaction_type == TransactionType.PAYMENT and amount > current_balance:
            self._logger.info(
                "WALLET_COMMIT_REJECTED type=%s amount=%s current_balance=%s",
                transaction_type.value,
                amount,
                current_balance,
            )
            raise ValueError("Payment is greater than wallet balance")

        if transaction_type == TransactionType.SET_BALANCE:
            new_balance = amount
        elif transaction_type == TransactionType.DEPOSIT:
            new_balance = current_balance + amount
        elif transaction_type == TransactionType.PAYMENT:
            new_balance = current_balance - amount
        elif transaction_type == TransactionType.ADJUSTMENT:
            new_balance = current_balance + amount
        else:
            raise ValueError(f"Unsupported transaction type: {transaction_type}")

        self._write_transaction(
            transaction_type=transaction_type,
            amount=amount,
            balance_after=new_balance,
            note_values=self._pending.notes,
        )
        self._pending = None
        self._logger.info("WALLET_COMMIT_DONE type=%s balance_after=%s", transaction_type.value, new_balance)
        return self.snapshot()

    def cancel_transaction(self) -> WalletSnapshot:
        self._logger.info(
            "WALLET_PENDING_CANCEL type=%s total=%s",
            self._pending.transaction_type.value if self._pending else None,
            self._pending.total if self._pending else 0,
        )
        self._pending = None
        return self.snapshot()

    def has_pending_transaction(self) -> bool:
        return self._pending is not None

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wallet_state (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wallet_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    type TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    balance_after INTEGER NOT NULL,
                    note_values TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO wallet_state(key, value)
                VALUES ('balance', 0)
                """
            )

    def _write_transaction(
        self,
        transaction_type: TransactionType,
        amount: int,
        balance_after: int,
        note_values: Iterable[int],
    ) -> None:
        note_values_text = ",".join(str(value) for value in note_values)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE wallet_state
                SET value = ?
                WHERE key = 'balance'
                """,
                (balance_after,),
            )
            conn.execute(
                """
                INSERT INTO wallet_transactions(
                    created_at,
                    type,
                    amount,
                    balance_after,
                    note_values
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    transaction_type.value,
                    amount,
                    balance_after,
                    note_values_text,
                ),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _validate_amount(amount: int, allow_zero: bool = False) -> None:
        if amount < 0 or (amount == 0 and not allow_zero):
            raise ValueError("Amount must be positive")
