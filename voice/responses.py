from __future__ import annotations


RESPONSES = {
    "ready": {"en": "Money detection is running.", "ar": "كاشف الفلوس يعمل الآن."},
    "unknown": {"en": "Say that again.", "ar": "كرر الأمر."},
    "low_confidence": {"en": "I am not sure. Repeat.", "ar": "لست متأكدا. كرر."},
    "voice_error": {"en": "Voice error.", "ar": "خطأ في الصوت."},
    "arabic_on": {"en": "Arabic enabled.", "ar": "تم تشغيل العربية."},
    "english_on": {"en": "English enabled.", "ar": "تم تشغيل الإنجليزية."},
}


def response(key: str, language: str = "en") -> str:
    values = RESPONSES.get(key, RESPONSES["unknown"])
    return values.get(language, values["en"])
