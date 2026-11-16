"""Çekim sırasında söylenen reel işaretleri: "1. reel", "ikinci reel", "reel 3", "yeni reel", "sıradaki reele geçiyoruz".

Bunlar reels sınırının en güvenilir sinyalidir:
  - Bölmede kesin sınır olarak kullanılır (Claude'a ayrıca bildirilir, sezgisel bölmede zorunlu sınırdır).
  - İşaret cümlesi/kelimeleri reel'den çıkarılır (izleyici "ikinci reel" sesini duymaz).
  - Aynı numara birden fazla geçerse aynı reel yeniden çekilmiş demektir; son tam deneme kalır.

Whisper yazımı değişkendir ("2. Reel'e", "İkinci reel", "Rill"); eşleştirme ASCII-katlanmış metinde yapılır.
Yalın "reels" kelimesi (ör. "bu reels içerikleri") işaret sayılmaz: yanında sayı ya da yeni/sıradaki gerekir.
"""
from __future__ import annotations

import re

from .detector import Cut
from .textnorm import fold

_NUM_WORDS = {
    "bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
    "birinci": 1, "ikinci": 2, "ucuncu": 3, "dorduncu": 4, "besinci": 5, "altinci": 6, "yedinci": 7,
    "sekizinci": 8, "dokuzuncu": 9, "onuncu": 10, "ilk": 1,
}
_NUM = r"(?P<num>[1-9]\d?|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")"
# "bir" sayı sayılmaz: "bir reel çekelim", "bu reel bir harika" gibi belirsiz kullanımı çok yaygın. 1 için "1."/"birinci"/"ilk".
_NUM_BEFORE = r"(?P<num>[1-9]\d?|" +"|".join(sorted((w for w in _NUM_WORDS if w != "bir"), key=len, reverse=True)) + r")"
_SUFFIX = r"(?:\s?(?:nci|inci|uncu|unci|ncu|nci|\.))?"
_REEL = r"(?:reel|riil|rill|ril|riyl)(?:s|ler|e|i|in|imiz|imize|imizi|de|den)?"
_ADJ = r"(?P<adj>yeni|siradaki|sonraki|diger|baska|son)"

PATTERNS = [
    re.compile(rf"\b{_NUM_BEFORE}{_SUFFIX}\s+(?:numarali\s+)?{_REEL}\b"),   # 2 reel, ikinci reel, 3 numaralı reel
    re.compile(rf"\b{_REEL}\s+(?:numara\s+)?{_NUM_BEFORE}\b"),              # reel 3, reel numara 2 ("reel bir" hariç)
    re.compile(rf"\b{_ADJ}\s+{_REEL}\b"),                                   # yeni reel, sıradaki reele
    re.compile(rf"\b(?P<adj>bir)\s+{_REEL}\s+daha\b"),                      # bir reel daha (çekelim)
]
# İşaretin hemen önünde/arkasında genelde söylenen ve işaretle birlikte kesilmesi gereken kelimeler
# Önde: yalnızca dolgu/hata sözleri ("ay olmadı", "pardon", "tamam şimdi"). İçerik kelimeleri kesilmez —
# "Messi en iyisi ikinci reel" cümlesinde "Messi en iyisi" önceki reel'in son sözüdür.
_LEAD = {"tamam", "evet", "simdi", "ve", "hadi", "o", "zaman", "peki", "geldik", "gecelim", "geciyoruz",
         "ay", "of", "olmadi", "olmuyor", "pardon", "dur", "bekle", "yok", "hayir", "yanlis", "karistirdim",
         "kusura", "bakma", "tekrar", "bastan", "yeniden", "bi", "bir", "daha", "saniye"}
_TRAIL = {"basliyoruz", "basliyor", "baslayalim", "basla", "geciyoruz", "gecelim", "geciyorum", "aliyoruz",
          "cekiyoruz", "cekelim", "kayit", "kayitta", "aksiyon", "motor", "hazir", "hazirim", "tamam",
          "tekrar", "tekrardan", "bastan", "yeniden", "alalim", "aliyorum"}


def _num(token: str | None) -> int | None:
    if not token:
        return None
    return int(token) if token.isdigit() else _NUM_WORDS.get(token)


def find_markers(segs: list[dict]) -> list[dict]:
    """Segmentlerdeki reel işaretleri: [{seg_id, start, end, number, label, text, whole}, ...]

    start/end: işaretin (ve çevresindeki 'tamam', 'başlıyoruz' gibi kelimelerin) zamanı.
    whole: segment neredeyse yalnızca işaretten oluşuyorsa True (segmentin tamamı reel dışı).
    """
    out = []
    for s in segs:
        words = s.get("words") or []
        folded = [fold(w["word"]) for w in words]
        # Katlanmış kelimeleri boşlukla birleştir; her kelimenin karakter aralığını tut
        spans, parts, pos = [], [], 0
        for f in folded:
            spans.append((pos, pos + len(f)))
            parts.append(f)
            pos += len(f) + 1
        text = " ".join(parts) if words else fold(s.get("text", ""))
        for pat in PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            number = _num(m.groupdict().get("num"))
            adj = m.groupdict().get("adj")
            adj = "yeni" if adj == "bir" else adj
            if not words:
                out.append({"seg_id": s["id"], "start": s["start"], "end": s["end"], "number": number,
                            "label": f"{number}. reel" if number else f"{adj} reel", "text": s.get("text", ""),
                            "whole": True})
                break
            hit = [k for k, (a, b) in enumerate(spans) if b > m.start() and a < m.end() and folded[k]]
            i, j = hit[0], hit[-1]
            while i > 0 and (folded[i - 1] in _LEAD or not folded[i - 1]):
                i -= 1
            while j + 1 < len(words):
                if folded[j + 1] in _TRAIL or not folded[j + 1]:
                    j += 1
                elif folded[j + 1] == "bir" and j + 2 < len(words) and folded[j + 2] == "daha":
                    j += 2   # "birinci reel bir daha" — tek başına "bir" içeriğin ilk kelimesi olabilir
                else:
                    break
            rest = [f for k, f in enumerate(folded) if f and not (i <= k <= j)]
            out.append({
                "seg_id": s["id"], "start": words[i]["start"], "end": words[j]["end"], "number": number,
                "prev_end": words[i - 1]["end"] if i > 0 else None,
                "next_start": words[j + 1]["start"] if j + 1 < len(words) else None,
                "label": f"{number}. reel" if number else f"{adj} reel",
                "text": " ".join(w["word"] for w in words[i:j + 1]), "whole": len(rest) <= 1,
            })
            break
    return out


def marker_cuts(markers: list[dict]) -> list[Cut]:
    """İşaret kelimelerini reel'den çıkaran kesimler."""
    cuts = []
    for n, mk in enumerate(markers, 1):
        # Kenar payı: en çok 0.08 sn, komşu kelimeye olan boşluğun yarısını geçmez (kelime kırpılmasın)
        a = mk["start"] - (min(0.08, (mk["start"] - mk["prev_end"]) / 2) if mk.get("prev_end") is not None else 0.08)
        b = mk["end"] + (min(0.08, (mk["next_start"] - mk["end"]) / 2) if mk.get("next_start") is not None else 0.08)
        cuts.append(Cut(round(max(0.0, a), 3), round(b, 3), "command",
                        f"Reel işareti: \"{mk['label']}\"", mk["text"], 0.95, source="rule",
                        id=f"m{n:03d}", seg_ids=[mk["seg_id"]]))
    return cuts


def prompt_lines(markers: list[dict]) -> str:
    if not markers:
        return "(yok)"
    return "\n".join(f"#{mk['seg_id']} t={mk['start']:.1f} sn: \"{mk['text']}\" → {mk['label']}"
                     + (" (segmentin tamamı işaret)" if mk["whole"] else "") for mk in markers)
