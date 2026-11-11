"""Türkçe metin normalizasyonu ve bulanık eşleştirme yardımcıları.

Whisper bazen 'ı/i', 'ş/s' gibi harfleri karıştırır veya noktalamayı farklı koyar.
Bu yüzden eşleştirmeler hem Türkçe-küçük-harf hem ASCII-katlanmış metin üzerinde yapılır.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_ASCII_FOLD = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "i": "i", "ö": "o", "ş": "s", "ü": "u",
    "â": "a", "î": "i", "û": "u",
})
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")

# Tekrar karşılaştırmasında anlam taşımayan dolgu kelimeleri
FILLERS = {
    "ee", "eee", "eh", "ehm", "hmm", "mm", "ıı", "ııı", "şey", "yani", "işte",
    "hani", "aa", "aaa", "ah", "oh", "tamam", "evet", "hıhı", "ok", "okey",
}


def tr_lower(text: str) -> str:
    return text.translate(_TR_LOWER).lower()


def normalize(text: str) -> str:
    """Küçük harf, noktalama yok, tek boşluk."""
    text = unicodedata.normalize("NFC", tr_lower(text))
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def fold(text: str) -> str:
    """Normalize + ASCII katlama (ş→s, ı→i ...)."""
    return normalize(text).translate(_ASCII_FOLD)


def tokens(text: str, drop_fillers: bool = False) -> list[str]:
    toks = fold(text).split()
    if drop_fillers:
        folded_fillers = {f.translate(_ASCII_FOLD) for f in FILLERS}
        toks = [t for t in toks if t not in folded_fillers]
    return toks


def similarity(a: list[str] | str, b: list[str] | str) -> float:
    """Kelime dizisi düzeyinde benzerlik (0..1)."""
    if isinstance(a, str):
        a = tokens(a, drop_fillers=True)
    if isinstance(b, str):
        b = tokens(b, drop_fillers=True)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def prefix_overlap(short: list[str], long: list[str]) -> float:
    """`short` dizisinin `long` dizisinin başlangıcıyla ne kadar örtüştüğü (0..1).

    Yarım kalmış cümle (false start) tespiti için: "bugün size şey" → "bugün size yeni ürünü anlatacağım".
    """
    if not short or not long:
        return 0.0
    head = long[: len(short) + 2]
    sm = SequenceMatcher(None, short, head, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(short)


def fuzzy_phrase_ratio(phrase: str, window: str) -> float:
    """Karakter düzeyinde bulanık eşleşme — kısa komut ifadeleri için."""
    return SequenceMatcher(None, fold(phrase), fold(window), autojunk=False).ratio()
