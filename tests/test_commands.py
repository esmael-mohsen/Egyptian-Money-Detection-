from __future__ import annotations

import unittest

from voice.command_catalog import CommandCatalog
from voice.commands import CommandId, parse_command


class CommandParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = CommandCatalog.default()

    def test_catalog_loads(self) -> None:
        self.assertGreaterEqual(len(self.catalog.specs), 18)
        self.assertIsNotNone(self.catalog.spec_for(CommandId.START_SCAN))

    def test_english_command_aliases(self) -> None:
        samples = {
            "scan money": CommandId.START_SCAN,
            "begin note reading": CommandId.START_SCAN,
            "tell me my balance": CommandId.WALLET_BALANCE,
            "how much is in my wallet": CommandId.WALLET_BALANCE,
            "complete payment": CommandId.FINISH_PAYMENT,
            "save deposit": CommandId.FINISH_DEPOSIT,
            "count stack": CommandId.START_FLIP_SCAN,
            "flip count": CommandId.START_FLIP_SCAN,
            "deposit stack": CommandId.START_FLIP_DEPOSIT,
            "finish stack deposit": CommandId.FINISH_FLIP_DEPOSIT,
            "pay from stack": CommandId.START_FLIP_PAYMENT,
            "finish stack payment": CommandId.FINISH_FLIP_PAYMENT,
            "go ahead": CommandId.CONFIRM,
            "never mind": CommandId.CANCEL,
        }
        for text, command_id in samples.items():
            with self.subTest(text=text):
                self.assertEqual(parse_command(text, self.catalog).command_id, command_id)

    def test_arabic_command_aliases(self) -> None:
        samples = {
            "ابدأ المسح": CommandId.START_SCAN,
            "عد الفلوس": CommandId.START_SCAN,
            "رصيد المحفظة": CommandId.WALLET_BALANCE,
            "المحفظة فيها كام": CommandId.WALLET_BALANCE,
            "انهاء الدفع": CommandId.FINISH_PAYMENT,
            "احسب الدفع": CommandId.FINISH_PAYMENT,
            "تمام": CommandId.CONFIRM,
            "بلاش": CommandId.CANCEL,
        }
        for text, command_id in samples.items():
            with self.subTest(text=text):
                self.assertEqual(parse_command(text, self.catalog).command_id, command_id)

    def test_fuzzy_misrecognitions(self) -> None:
        self.assertEqual(parse_command("start skin", self.catalog).command_id, CommandId.START_SCAN)
        self.assertEqual(parse_command("finish pavement", self.catalog).command_id, CommandId.FINISH_PAYMENT)
        self.assertEqual(parse_command("can firm", self.catalog).command_id, CommandId.CONFIRM)

    def test_amount_extraction(self) -> None:
        parsed = parse_command("set balance to 750", self.catalog)
        self.assertEqual(parsed.command_id, CommandId.SET_BALANCE)
        self.assertEqual(parsed.amount, 750)

        paid = parse_command("I paid fifty", self.catalog)
        self.assertEqual(paid.command_id, CommandId.START_PAYMENT)
        self.assertEqual(paid.amount, 50)

        arabic = parse_command("خمسين جنيه", self.catalog)
        self.assertIsNone(arabic.command_id)
        self.assertEqual(arabic.amount, 50)

        arabic_balance = parse_command("الرصيد يبقى خمسمية جنيه", self.catalog)
        self.assertEqual(arabic_balance.command_id, CommandId.SET_BALANCE)
        self.assertEqual(arabic_balance.amount, 500)


if __name__ == "__main__":
    unittest.main()
