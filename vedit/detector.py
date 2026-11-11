"""Kesim tespiti: sözlü kurgu komutları, yeniden çekimler, tekrarlar ve sessizlikler.

Akış:
    1. detect_commands     — "burayı kes", "baştan alıyorum" vb. sözlü komutlar
    2. detect_repeats      — komut söylenmeden yapılan tekrar / yarım kalan cümleler
    3. detect_stutters     — cümle içi kelime grubu tekrarı ("bugün size bugün size")
    4. detect_silences     — uzun boşlukları kısaltma
    5. (isteğe bağlı) Claude ile doğrulama ve anlamsal tekrar / çekim arası muhabbet tespiti
    6. finalize            — çakışmaları temizler, kenarları kelime aralarına oturtur

Kurallar ve ifadeler: skills/video-edit-automation/references/cut-rules.md
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

from .config import settings
from .textnorm import fold, fuzzy_phrase_ratio, prefix_overlap, similarity, tokens

# ---------------------------------------------------------------------------
# Komut sözlüğü (ASCII-katlanmış yazılır; eşleştirme bulanıktır)
# ---------------------------------------------------------------------------
# RETAKE: önceki deneme + komut kesilir, sonrasındaki yeni deneme kalır.
RETAKE_PHRASES = [
    "bastan aliyorum", "bastan alalim", "bastan aliyoruz", "bastan al", "bastan basliyorum",
    "tekrar aliyorum", "tekrar alalim", "tekrar aliyoruz", "tekrar cekelim",
    "bir daha alalim", "bir daha aliyorum", "bir daha cekelim", "bir daha aliyoruz",
    "yeniden aliyorum", "yeniden alalim", "yeniden cekelim",
    "olmadi bastan", "olmadi tekrar", "yanlis oldu", "yanlis soyledim", "pardon bastan",
    "dur bastan", "dur dur dur", "bir daha deniyorum",
]
# CUT_THIS: hemen önceki cümle + komut kesilir.
CUT_PHRASES = [
    "burayi kes", "surayi kes", "bunu kes", "bu kismi kes", "kes bunu", "kesin bunu", "kesin burayi",
    "burayi keselim", "surayi keselim", "bunu keselim", "keselim burayi", "keselim bunu",
    "burasi kesilecek", "burasi kesilsin", "bunu cikar", "burayi cikar", "bunu cikaralim",
    "bunu sil", "burayi sil", "kurguda kes", "kurguda cikar", "montajda kes", "montajda cikar",
    "editte kes", "editte cikar", "bunu kullanmayalim", "bunu kullanmayin", "bunu at",
]
# RANGE: başlangıç ve bitiş işaretleri arası kesilir.
RANGE_START_PHRASES = [
    "buradan itibaren kes", "buradan sonrasini kes", "kesmeye basla", "kesim basliyor",
    "buradan itibaren cikar", "buradan itibaren sil",
]
RANGE_END_PHRASES = [
    "buraya kadar kes", "buraya kadar cikar", "buraya kadar sil", "kesim bitti", "kesme bitti",
    "kesmeyi bitir", "buradan devam",
]

COMMAND_MATCH_THRESHOLD = 0.86
LOOKBACK_SEC = 30.0          # retake/cut komutunda geriye bakılacak en fazla süre
MAX_BETWEEN_GAP_SEC = 25.0   # iki deneme arasında kesilebilecek en fazla ara


@dataclass
class Cut:
    start: float
    end: float
    kind: str                    # command | retake | repeat | false_start | stutter | silence | chatter | range
    reason: str
    text: str = ""
    confidence: float = 0.8
    enabled: bool = True
    source: str = "rule"         # rule | claude | user
    id: str = ""
    seg_ids: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Cut":
        allowed = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**allowed)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _seg_text(segs: list[dict], a: int, b: int) -> str:
    return " ".join(s["text"] for s in segs[a: b + 1]).strip()


def _find_phrase(seg: dict, phrases: list[str]) -> tuple[str, int, int, float] | None:
    """Segment içindeki en iyi komut eşleşmesi: (ifade, ilk_kelime, son_kelime, skor)."""
    words = seg.get("words") or []
    if not words:
        # Kelime yoksa tüm segment üzerinde kaba kontrol
        best = max(((p, fuzzy_phrase_ratio(p, seg["text"])) for p in phrases), key=lambda x: x[1])
        folded = fold(seg["text"])
        for p in phrases:
            if p in folded:
                return p, 0, 0, 1.0
        return (best[0], 0, 0, best[1]) if best[1] >= COMMAND_MATCH_THRESHOLD else None

    wtoks = [fold(w["word"]) for w in words]
    best: tuple[str, int, int, float] | None = None
    for phrase in phrases:
        n = len(phrase.split())
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(0, len(wtoks) - size + 1):
                window = " ".join(t for t in wtoks[i: i + size] if t)
                if not window:
                    continue
                score = fuzzy_phrase_ratio(phrase, window)
                if score >= COMMAND_MATCH_THRESHOLD and (best is None or score > best[3]):
                    best = (phrase, i, i + size - 1, score)
    return best


def _command_span(seg: dict, w0: int, w1: int) -> tuple[float, float]:
    """Komutun kapladığı zaman. Komutun etrafında ≤2 kelime kalıyorsa segment sınırına genişletir."""
    words = seg.get("words") or []
    if not words:
        return seg["start"], seg["end"]
    start = seg["start"] if w0 <= 2 else words[w0]["start"]
    end = seg["end"] if len(words) - 1 - w1 <= 2 else words[w1]["end"]
    return start, end


def _match_start(seg: dict, other_text: str, min_block: int = 2) -> float:
    """`seg` içinde `other_text` ile örtüşmenin başladığı kelimenin zamanı.

    Bir birimde önce farklı (korunacak) bir cümle, ardından tekrar edilen deneme varsa
    yalnızca denemenin başından kesmek için kullanılır. Örtüşme ilk 2 kelimede başlıyorsa segment başı döner.
    """
    words = seg.get("words") or []
    if len(words) < 4:
        return seg["start"]
    from difflib import SequenceMatcher
    a = [fold(w["word"]) for w in words]
    b = fold(other_text).split()
    for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        if block.size >= min_block:
            return words[block.a]["start"] if block.a > 2 else seg["start"]
    return seg["start"]


def _retake_start(segs: list[dict], cmd_idx: int, cmd_start: float) -> tuple[int, float, float]:
    """Retake komutundan önce başlayan başarısız denemenin başlangıç segmentini bulur.

    Komuttan sonraki yeni denemeyi önceki segmentlerle karşılaştırır; en benzer olanın
    başından itibaren kesilir. Benzerlik yoksa yalnızca hemen önceki segment kesilir.
    Döner: (başlangıç_segment_index, başlangıç_zamanı, güven)
    """
    after = next((s for s in segs[cmd_idx + 1:] if len(tokens(s["text"], True)) >= 2), None)
    candidates = [
        i for i in range(cmd_idx - 1, -1, -1)
        if cmd_start - segs[i]["start"] <= LOOKBACK_SEC
    ]
    if not candidates:
        return cmd_idx, cmd_start, 0.6

    if after is not None:
        after_toks = tokens(after["text"], True)
        best_i, best_score = None, 0.0
        for i in candidates:
            prev_toks = tokens(segs[i]["text"], True)
            score = max(similarity(prev_toks, after_toks), prefix_overlap(prev_toks, after_toks))
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= 0.5:
            return best_i, _match_start(segs[best_i], after["text"]), min(0.95, 0.7 + best_score / 4)

    i = candidates[0]
    return i, segs[i]["start"], 0.65


# ---------------------------------------------------------------------------
# 1) Sözlü komutlar
# ---------------------------------------------------------------------------
def detect_commands(segs: list[dict]) -> list[Cut]:
    cuts: list[Cut] = []
    open_range: tuple[int, float, str] | None = None

    for idx, seg in enumerate(segs):
        # Aralık işaretleri
        m_end = _find_phrase(seg, RANGE_END_PHRASES)
        if open_range and m_end:
            s_idx, s_time, phrase = open_range
            _, e_time = _command_span(seg, m_end[1], m_end[2])
            cuts.append(Cut(s_time, e_time, "range",
                            f"Sözlü aralık komutu: '{phrase}' … '{m_end[0]}'",
                            _seg_text(segs, s_idx, idx), 0.9, seg_ids=list(range(s_idx, idx + 1))))
            open_range = None
            continue
        m_start = _find_phrase(seg, RANGE_START_PHRASES)
        if m_start:
            s_time, _ = _command_span(seg, m_start[1], m_start[2])
            open_range = (idx, s_time, m_start[0])
            continue

        # Retake komutları
        m = _find_phrase(seg, RETAKE_PHRASES)
        if m:
            c_start, c_end = _command_span(seg, m[1], m[2])
            s_idx, s_time, conf = _retake_start(segs, idx, c_start)
            cuts.append(Cut(s_time, c_end, "retake",
                            f"Yeniden çekim komutu: '{m[0]}' — önceki deneme kesildi",
                            _seg_text(segs, s_idx, idx), conf, seg_ids=list(range(s_idx, idx + 1))))
            continue

        # "Burayı kes" komutları
        m = _find_phrase(seg, CUT_PHRASES)
        if m:
            c_start, c_end = _command_span(seg, m[1], m[2])
            prev = idx - 1
            if prev >= 0 and c_start - segs[prev]["end"] <= 6.0 and c_start - segs[prev]["start"] <= LOOKBACK_SEC:
                s_idx, s_time, conf = prev, segs[prev]["start"], 0.75
            else:
                s_idx, s_time, conf = idx, c_start, 0.7
            cuts.append(Cut(s_time, c_end, "command",
                            f"Kesim komutu: '{m[0]}' — komut ve önceki cümle kesildi",
                            _seg_text(segs, s_idx, idx), conf, seg_ids=list(range(s_idx, idx + 1))))

    if open_range:  # bitişi söylenmemiş aralık → yalnızca başlangıç segmentini işaretle, kapalı gelsin
        s_idx, s_time, phrase = open_range
        cuts.append(Cut(s_time, segs[s_idx]["end"], "range",
                        f"'{phrase}' dendi ama bitiş işareti bulunamadı — elle ayarlayın",
                        segs[s_idx]["text"], 0.4, enabled=False, seg_ids=[s_idx]))
    return cuts


# ---------------------------------------------------------------------------
# 2) Komutsuz tekrarlar ve yarım cümleler
# ---------------------------------------------------------------------------
def detect_repeats(segs: list[dict], threshold: float | None = None,
                   window_sec: float | None = None) -> list[Cut]:
    threshold = threshold or settings.repeat_similarity
    window_sec = window_sec or settings.repeat_window_sec
    toks = [tokens(s["text"], drop_fillers=True) for s in segs]
    cuts: list[Cut] = []
    marked: set[int] = set()

    for i, j in itertools.combinations(range(len(segs)), 2):
        if j - i > 8 or segs[j]["start"] - segs[i]["end"] > window_sec:
            continue
        a, b = toks[i], toks[j]
        if len(a) < 2 or not b or i in marked:
            continue

        full = similarity(a, b)
        partial = prefix_overlap(a, b) if len(a) < len(b) else 0.0
        kind = None
        if len(a) >= 3 and full >= threshold:
            kind, score = "repeat", full
        elif j - i <= 2 and len(a) >= 2 and partial >= 0.8 and len(b) - len(a) >= 2:
            kind, score = "false_start", partial
        if not kind:
            continue

        # Aradaki segmentler kısa (muhabbet/dolgu) ve süre makulse aradaki kısmı da kes
        between = list(range(i + 1, j))
        end_time = segs[i]["end"]
        end_idx = i
        if between and segs[j]["start"] - segs[i]["end"] <= MAX_BETWEEN_GAP_SEC and all(
            len(toks[k]) <= 6 or k in marked for k in between
        ):
            end_time = segs[j - 1]["end"]
            end_idx = j - 1

        distance_penalty = min(0.25, (j - i - 1) * 0.05)
        conf = round(max(0.4, min(0.95, score - distance_penalty)), 2)
        label = "Tekrar eden cümle — son deneme tutuldu" if kind == "repeat" else \
            "Yarım kalan cümle (false start) — tamamlanan hali tutuldu"
        cuts.append(Cut(_match_start(segs[i], segs[j]["text"]), end_time, kind,
                        f"{label} (benzerlik {score:.2f}, tutulan: #{j})",
                        _seg_text(segs, i, end_idx), conf, seg_ids=list(range(i, end_idx + 1))))
        marked.update(range(i, end_idx + 1))
    return cuts


# ---------------------------------------------------------------------------
# 3) Cümle içi kekeleme / kelime grubu tekrarı
# ---------------------------------------------------------------------------
def detect_stutters(segs: list[dict], min_n: int = 2, max_n: int = 5) -> list[Cut]:
    cuts: list[Cut] = []
    for seg in segs:
        words = seg.get("words") or []
        wt = [fold(w["word"]) for w in words]
        k = 0
        while k < len(wt):
            found = False
            for n in range(max_n, min_n - 1, -1):
                if k + 2 * n <= len(wt) and wt[k: k + n] == wt[k + n: k + 2 * n] and all(wt[k: k + n]):
                    cuts.append(Cut(words[k]["start"], words[k + n]["start"], "stutter",
                                    f"Arka arkaya tekrarlanan {n} kelime",
                                    " ".join(w["word"] for w in words[k: k + n]), 0.6, seg_ids=[seg["id"]]))
                    k += n
                    found = True
                    break
            if not found:
                k += 1
    return cuts


# ---------------------------------------------------------------------------
# 4) Sessizlikler
# ---------------------------------------------------------------------------
def detect_silences(segs: list[dict], duration: float, min_sec: float | None = None,
                    keep_sec: float | None = None) -> list[Cut]:
    min_sec = min_sec or settings.silence_min_sec
    keep_sec = keep_sec if keep_sec is not None else settings.silence_keep_sec
    times: list[tuple[float, float]] = []
    for s in segs:
        if s.get("words"):
            times.extend((w["start"], w["end"]) for w in s["words"])
        else:
            times.append((s["start"], s["end"]))
    if not times:
        return []
    times.sort()

    cuts: list[Cut] = []
    half = keep_sec / 2
    boundaries = [(0.0, 0.0)] + times + [(duration, duration)] if duration else [(0.0, 0.0)] + times
    for (_, prev_end), (next_start, _) in zip(boundaries, boundaries[1:]):
        gap = next_start - prev_end
        if gap >= min_sec:
            a = prev_end + (0 if prev_end == 0 else half)
            b = next_start - (0 if next_start == duration else half)
            if b - a > 0.2:
                cuts.append(Cut(round(a, 3), round(b, 3), "silence",
                                f"{gap:.1f} sn sessizlik kısaltıldı", "", 0.9))
    return cuts


# ---------------------------------------------------------------------------
# 5) Sonlandırma
# ---------------------------------------------------------------------------
_PRIORITY = {"range": 6, "retake": 5, "command": 5, "chatter": 4, "repeat": 4,
             "false_start": 3, "stutter": 2, "silence": 1}


def finalize(cuts: list[Cut], segs: list[dict], duration: float,
             min_confidence: float = 0.6) -> list[Cut]:
    """Kapsanan kesimleri atar, kenarları kelime aralarına oturtur, id verir, sıralar."""
    word_times = sorted(
        (w["start"], w["end"]) for s in segs for w in (s.get("words") or [{"start": s["start"], "end": s["end"]}])
    )

    def snap(c: Cut) -> Cut:
        prev_end = max((e for _, e in word_times if e <= c.start + 1e-3), default=0.0)
        next_start = min((s for s, _ in word_times if s >= c.end - 1e-3), default=duration or c.end)
        if c.kind != "silence":
            c.start = max(0.0, c.start - min(settings.pad_before, max(0.0, c.start - prev_end) / 2))
            c.end = c.end + min(settings.pad_after, max(0.0, next_start - c.end) / 2)
            if duration:
                c.end = min(c.end, duration)
        c.start, c.end = round(c.start, 3), round(c.end, 3)
        return c

    cuts = [snap(c) for c in cuts if c.end - c.start > 0.05]
    cuts.sort(key=lambda c: (-_PRIORITY.get(c.kind, 0), -c.duration))

    kept: list[Cut] = []
    for c in cuts:
        container = next((k for k in kept if k.start <= c.start + 0.05 and k.end >= c.end - 0.05), None)
        if container is not None:
            continue
        kept.append(c)

    kept.sort(key=lambda c: c.start)
    for n, c in enumerate(kept, 1):
        c.id = f"c{n:03d}"
        if c.source != "user" and c.kind != "silence":
            c.enabled = c.enabled and c.confidence >= min_confidence
    return kept


def keep_ranges(cuts: list[Cut], duration: float, within: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    """Etkin kesimlerin tümleyeni — videoda kalacak aralıklar."""
    lo, hi = within or (0.0, duration)
    spans = sorted((max(lo, c.start), min(hi, c.end)) for c in cuts if c.enabled and c.end > lo and c.start < hi)
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 0.02:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    keeps, cursor = [], lo
    for a, b in merged:
        if a - cursor > 0.04:
            keeps.append((round(cursor, 3), round(a, 3)))
        cursor = max(cursor, b)
    if hi - cursor > 0.04:
        keeps.append((round(cursor, 3), round(hi, 3)))
    return keeps


# ---------------------------------------------------------------------------
# Ana giriş
# ---------------------------------------------------------------------------
def detect_all(transcript: dict, *, use_silence: bool = True, use_stutter: bool = True,
               use_claude: bool = True, progress=None) -> list[Cut]:
    segs = transcript["segments"]
    duration = float(transcript.get("duration") or (segs[-1]["end"] if segs else 0.0))

    cuts = detect_commands(segs) + detect_repeats(segs)
    if use_stutter:
        cuts += detect_stutters(segs)

    if use_claude and settings.has_claude and segs:
        from .claude_review import review_cuts  # geç import: API anahtarı yoksa gerekmez
        if progress:
            progress(0.0, "Claude ile kesimler doğrulanıyor")
        cuts = review_cuts(segs, cuts, progress=progress)

    if use_silence:
        cuts += detect_silences(segs, duration)
    return finalize(cuts, segs, duration)
