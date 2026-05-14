from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.wallet import TransactionType, WalletStore


class WalletStoreTests(unittest.TestCase):
    def test_set_balance_persists_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "wallet.db"
            wallet = WalletStore(db_path)
            wallet.set_balance(500)
            wallet.commit_transaction()

            restarted = WalletStore(db_path)
            self.assertEqual(restarted.get_balance(), 500)

    def test_deposit_requires_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wallet = WalletStore(Path(tmp) / "wallet.db")
            wallet.set_balance(100)
            wallet.commit_transaction()

            wallet.begin_transaction(TransactionType.DEPOSIT)
            wallet.add_scanned_note(50)
            self.assertEqual(wallet.get_balance(), 100)

            snapshot = wallet.commit_transaction()
            self.assertEqual(snapshot.balance, 150)

    def test_payment_requires_confirm_and_can_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wallet = WalletStore(Path(tmp) / "wallet.db")
            wallet.set_balance(200)
            wallet.commit_transaction()

            wallet.begin_transaction(TransactionType.PAYMENT)
            wallet.add_scanned_note(50)
            wallet.cancel_transaction()
            self.assertEqual(wallet.get_balance(), 200)

            wallet.begin_transaction(TransactionType.PAYMENT)
            wallet.add_scanned_note(50)
            snapshot = wallet.commit_transaction()
            self.assertEqual(snapshot.balance, 150)

    def test_payment_greater_than_balance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wallet = WalletStore(Path(tmp) / "wallet.db")
            wallet.set_balance(20)
            wallet.commit_transaction()
            wallet.begin_transaction(TransactionType.PAYMENT)
            wallet.add_scanned_note(50)

            with self.assertRaises(ValueError):
                wallet.commit_transaction()


if __name__ == "__main__":
    unittest.main()
