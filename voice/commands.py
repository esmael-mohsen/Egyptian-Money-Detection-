from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice.command_catalog import CommandCatalog


class CommandId(str, Enum):
    START_SCAN = "START_SCAN"
    STOP_SCAN = "STOP_SCAN"
    COUNT_TOTAL = "COUNT_TOTAL"
    LAST_DETECTION = "LAST_DETECTION"
    REPEAT = "REPEAT"
    RESET_SESSION = "RESET_SESSION"
    STATUS_CHECK = "STATUS_CHECK"
    WALLET_BALANCE = "WALLET_BALANCE"
    SET_BALANCE = "SET_BALANCE"
    START_DEPOSIT = "START_DEPOSIT"
    FINISH_DEPOSIT = "FINISH_DEPOSIT"
    START_PAYMENT = "START_PAYMENT"
    FINISH_PAYMENT = "FINISH_PAYMENT"
    START_FLIP_SCAN = "START_FLIP_SCAN"
    START_FLIP_DEPOSIT = "START_FLIP_DEPOSIT"
    FINISH_FLIP_DEPOSIT = "FINISH_FLIP_DEPOSIT"
    START_FLIP_PAYMENT = "START_FLIP_PAYMENT"
    FINISH_FLIP_PAYMENT = "FINISH_FLIP_PAYMENT"
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    SWITCH_ARABIC = "SWITCH_ARABIC"
    SWITCH_ENGLISH = "SWITCH_ENGLISH"
    EXIT_APP = "EXIT_APP"


@dataclass(frozen=True)
class ParsedCommand:
    command_id: CommandId | None
    raw_text: str
    normalized_text: str
    amount: int | None = None
    confidence: float = 0.0
    matched_alias: str = ""
    requires_amount: bool = False
    confirmation_required: bool = False


_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f]")
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "صفر": 0,
    "واحد": 1,
    "واحدة": 1,
    "اتنين": 2,
    "اثنين": 2,
    "تلاتة": 3,
    "ثلاثة": 3,
    "اربعة": 4,
    "أربعة": 4,
    "خمسة": 5,
    "ستة": 6,
    "سبعة": 7,
    "تمانية": 8,
    "ثمانية": 8,
    "تسعة": 9,
    "عشرة": 10,
    "حداشر": 11,
    "احداشر": 11,
    "اتناشر": 12,
    "اثناشر": 12,
    "تلتاشر": 13,
    "اربعتاشر": 14,
    "خمستاشر": 15,
    "ستاشر": 16,
    "سبعتاشر": 17,
    "تمنتاشر": 18,
    "تسعتاشر": 19,
    "عشرين": 20,
    "تلاتين": 30,
    "ثلاثين": 30,
    "اربعين": 40,
    "أربعين": 40,
    "خمسين": 50,
    "ستين": 60,
    "سبعين": 70,
    "تمانين": 80,
    "ثمانين": 80,
    "تسعين": 90,
    "مية": 100,
    "ميه": 100,
    "مئة": 100,
    "مائه": 100,
    "ميتين": 200,
    "مائتين": 200,
    "تلتمية": 300,
    "تلتميه": 300,
    "ثلاثمية": 300,
    "ثلاثميه": 300,
    "ربعمية": 400,
    "ربعميه": 400,
    "خمسمية": 500,
    "خمسميه": 500,
    "ستمية": 600,
    "ستميه": 600,
    "سبعمية": 700,
    "سبعميه": 700,
    "تمانمية": 800,
    "تمانميه": 800,
    "تسعمية": 900,
    "تسعميه": 900,
    "الف": 1000,
    "ألف": 1000,
}
_AMOUNT_FILLERS = {
    "and",
    "و",
    "جنيه",
    "جنيهات",
    "جنية",
    "مصري",
    "pounds",
    "pound",
    "egp",
}


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s\u0600-\u06ff]", " ", text)
    return " ".join(text.split())


def parse_command(
    text: str,
    catalog: "CommandCatalog | None" = None,
    min_confidence: float = 0.72,
) -> ParsedCommand:
    normalized = normalize_text(text)
    amount = extract_amount(normalized)
    if not normalized:
        return ParsedCommand(None, text, normalized, amount)

    catalog = catalog or _default_catalog()
    best = _best_catalog_match(normalized, catalog)
    if best is None:
        return ParsedCommand(None, text, normalized, amount)

    spec, matched_alias, confidence = best
    if confidence < min_confidence:
        return ParsedCommand(
            None,
            text,
            normalized,
            amount,
            confidence=confidence,
            matched_alias=matched_alias,
        )

    return ParsedCommand(
        command_id=spec.command_id,
        raw_text=text,
        normalized_text=normalized,
        amount=amount,
        confidence=confidence,
        matched_alias=matched_alias,
        requires_amount=spec.requires_amount,
        confirmation_required=spec.confirmation_required,
    )


def extract_amount(text: str) -> int | None:
    digit_match = re.search(r"\d+", text)
    if digit_match:
        return int(digit_match.group(0))

    tokens = [token for token in text.split() if token not in _AMOUNT_FILLERS]
    total = 0
    current = 0
    found = False

    for token in tokens:
        value = _NUMBER_WORDS.get(token)
        if value is None:
            continue
        found = True
        if value in {100, 1000}:
            current = max(current, 1) * value
            if value == 1000:
                total += current
                current = 0
        else:
            current += value

    total += current
    return total if found else None


def _best_catalog_match(normalized_text: str, catalog: "CommandCatalog"):
    best = None
    for alias, spec in catalog.normalized_aliases():
        if not alias:
            continue
        confidence = _match_confidence(normalized_text, alias)
        if best is None or confidence > best[2]:
            best = (spec, alias, confidence)
    return best


def _match_confidence(text: str, alias: str) -> float:
    if text == alias:
        return 1.0
    if alias in text:
        return 0.92
    if text in alias:
        return 0.84

    text_tokens = set(text.split())
    alias_tokens = set(alias.split())
    if text_tokens and alias_tokens:
        overlap = len(text_tokens & alias_tokens) / len(alias_tokens)
    else:
        overlap = 0.0
    sequence = SequenceMatcher(None, text, alias).ratio()
    return max(sequence * 0.88, overlap * 0.82)


@lru_cache(maxsize=1)
def _default_catalog() -> "CommandCatalog":
    from voice.command_catalog import CommandCatalog

    return CommandCatalog.load(Path("voice/command_catalog.yaml"))
