"""Uzun bir çekimden birden çok reel'i ayırma.

Videoda reels'ler arasında belirgin bir işaret olmadığı varsayılır. Sınırlar üç sinyalle bulunur:
  1. Transkript: konu değişimi, giriş/kapanış kalıpları ("selam arkadaşlar", "takip etmeyi unutmayın")
  2. Duraklamalar: konuşmanın uzun süre kesildiği yerler
  3. Görüntü: sahne/kadraj/kıyafet/mekân değişimi (ffmpeg sahne tespiti + Claude vision)

Claude, numaralı transkript + aday sınırlar + zaman etiketli karelerle karar verir; segment
numarasıyla cevap verdiği için zamanlar uydurulmaz. API anahtarı yoksa sezgisel bölme yapılır.
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import media
from .claude_client import ClaudeError, call_json, image_block, text_block
from .claude_review import fmt_segment
from .config import settings

MAX_FRAMES = 48
MIN_REEL_SEC = 8.0


@dataclass
class Reel:
    start: float
    end: float
    title: str = ""
    hook: str = ""
    summary: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    confidence: float = 0.7
    notes: str = ""
    enabled: bool = True
    clean: bool = True              # reel içindeki tekrar/sessizlikleri de kes
    id: str = ""
    seg_ids: list[int] = field(default_factory=list)
    source: str = "claude"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Reel":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


SYSTEM = """Sen bir sosyal medya video editörüsün. Sana tek bir ham çekim veriliyor; içinde arka arkaya çekilmiş
BİRDEN FAZLA bağımsız reel (dikey kısa video) var. Aralarında belirgin bir ayraç yok.

Görevin her reel'in nerede başlayıp bittiğini bulmak:
- Her reel kendi içinde bütün bir konu/mesajdır; genelde bir giriş (hook) ile başlar, bir kapanış/çağrı ile biter.
- Sınır ipuçları: konu değişimi, yeni bir selamlama/giriş, uzun duraklama, kadraj/mekân/kıyafet/ışık değişimi (karelere bak).
- Aynı reel birden fazla kez baştan çekilmişse, SON TAM denemeyi seç; önceki yarım denemeleri 'discarded'a koy.
- Reels arasındaki hazırlık konuşmaları, kamera ayarı, gülüşmeler 'discarded' olur.
- Bir reel genelde 15-90 sn'dir; 8 sn'den kısa parçalar reel değildir.
- Reel içindeki küçük takılmaları dert etme; onları ayrı bir adım temizliyor.
- "S1:", "S2:" etiketleri konuşmacıyı gösterir; kamera arkasından gelen kısa yönlendirmeler reel'e dahil değildir.

Yalnızca JSON döndür."""

PROMPT = """Video süresi: {duration:.1f} sn

Transkript (#id [başlangıç-bitiş sn] metin):
{segments}

Aday sınırlar (sn) — uzun duraklama (P) ve sahne değişimi (S):
{candidates}

Ardından zaman etiketli kareler geliyor.

İstenen JSON:
{{
  "reels": [{{"start_id": <int>, "end_id": <int>, "title": "<kısa başlık>", "hook": "<ilk 3 sn'de söylenen kanca cümle>",
             "summary": "<1 cümle>", "caption": "<Instagram açıklaması, Türkçe, 1-3 cümle>",
             "hashtags": ["#..."], "confidence": <0-1>, "notes": "<sınır kararı gerekçesi>"}}],
  "discarded": [{{"start_id": <int>, "end_id": <int>, "reason": "<neden atıldı>"}}]
}}"""


def _candidates(segs: list[dict], scenes: list[float], min_gap: float = 1.5) -> list[tuple[float, str]]:
    out = []
    for a, b in zip(segs, segs[1:]):
        gap = b["start"] - a["end"]
        if gap >= min_gap:
            out.append((round((a["end"] + b["start"]) / 2, 2), f"P {gap:.1f}sn"))
    out += [(t, "S") for t in scenes]
    return sorted(out)


def _sample_times(duration: float, candidates: list[tuple[float, str]]) -> list[float]:
    step = max(6.0, duration / (MAX_FRAMES * 0.6))
    times = [round(t, 1) for t in _frange(1.0, duration - 0.5, step)]
    # Sahne değişimlerinin hemen sonrası (yeni kadrajı görmek için)
    budget = MAX_FRAMES - len(times)
    for t, kind in candidates:
        if budget <= 0:
            break
        if all(abs(t + 0.8 - x) > step / 3 for x in times):
            times.append(round(min(duration - 0.2, t + 0.8), 1))
            budget -= 1
    return sorted(times)[:MAX_FRAMES]


def _frange(a: float, b: float, step: float):
    t = a
    while t <= b:
        yield t
        t += step


def _snap(segs: list[dict], s_id: int, e_id: int, duration: float) -> tuple[float, float]:
    by_id = {s["id"]: k for k, s in enumerate(segs)}
    i, j = by_id[s_id], by_id[e_id]
    prev_end = segs[i - 1]["end"] if i > 0 else 0.0
    next_start = segs[j + 1]["start"] if j + 1 < len(segs) else duration
    start = max(prev_end, segs[i]["start"] - 0.2)
    end = min(next_start, segs[j]["end"] + 0.4)
    return round(max(0.0, start), 3), round(min(duration or end, end), 3)


def split_with_claude(video: str | Path, transcript: dict, info: dict, progress=None) -> list[Reel]:
    segs = transcript["segments"]
    duration = info["duration"]
    if progress:
        progress(0.1, "Sahne değişimleri aranıyor")
    scenes = media.scene_changes(video) if info.get("has_video") else []
    cands = _candidates(segs, scenes)

    content: list[dict] = [text_block(PROMPT.format(
        duration=duration,
        segments="\n".join(fmt_segment(s) for s in segs),
        candidates=", ".join(f"{t}({k})" for t, k in cands) or "(yok)",
    ))]

    if info.get("has_video"):
        times = _sample_times(duration, cands)
        with tempfile.TemporaryDirectory() as td:
            for n, t in enumerate(times):
                if progress:
                    progress(0.15 + 0.35 * n / max(1, len(times)), f"Kareler çıkarılıyor {n + 1}/{len(times)}")
                p = Path(td) / f"f{n:03d}.jpg"
                try:
                    media.extract_frame(video, t, p, width=384)
                except RuntimeError:
                    continue
                content += [text_block(f"Kare t={t:.1f} sn"), image_block(p)]
            if progress:
                progress(0.55, "Claude reels sınırlarını belirliyor")
            data = call_json(content, system=SYSTEM, max_tokens=max(settings.claude_max_tokens, 8000))
    else:
        data = call_json(content, system=SYSTEM)

    return reels_from_decision(segs, data, duration)


def reels_from_decision(segs: list[dict], data: dict, duration: float) -> list[Reel]:
    """Claude'un {"reels": [...]} kararını segment sınırlarına oturtulmuş Reel listesine çevirir."""
    ids = {s["id"] for s in segs}
    reels: list[Reel] = []
    for r in data.get("reels", []):
        s_id, e_id = r.get("start_id"), r.get("end_id")
        if s_id not in ids or e_id not in ids or s_id > e_id:
            continue
        start, end = _snap(segs, s_id, e_id, duration)
        if end - start < MIN_REEL_SEC:
            continue
        reels.append(Reel(start, end, r.get("title", ""), r.get("hook", ""), r.get("summary", ""),
                          r.get("caption", ""), list(r.get("hashtags", []))[:15],
                          float(r.get("confidence", 0.7)), r.get("notes", ""),
                          seg_ids=list(range(s_id, e_id + 1))))
    return _finish(reels)


def prepare_packet(video: str | Path, transcript: dict, info: dict, out_dir: str | Path) -> Path:
    """Skill modu: kareleri out_dir/frames altına çıkarır, Claude'un okuyacağı packet.md'yi yazar."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    segs = transcript["segments"]
    scenes = media.scene_changes(video) if info.get("has_video") else []
    cands = _candidates(segs, scenes)
    frame_lines = []
    if info.get("has_video"):
        for t in _sample_times(info["duration"], cands):
            p = frames_dir / f"t{t:07.1f}.jpg"
            try:
                media.extract_frame(video, t, p, width=384)
                frame_lines.append(f"- t={t:.1f} sn → frames/{p.name}")
            except RuntimeError:
                pass
    body = PROMPT.format(
        duration=info["duration"],
        segments="\n".join(fmt_segment(s) for s in segs),
        candidates=", ".join(f"{t}({k})" for t, k in cands) or "(yok)",
    ).replace("Ardından zaman etiketli kareler geliyor.", "Kareler (görmek için dosyaları açın):\n" + "\n".join(frame_lines))
    packet = out_dir / "packet.md"
    packet.write_text("# Reels bölme paketi\n\n## Talimat\n\n" + SYSTEM + "\n\n" + body, encoding="utf-8")
    return packet


def split_heuristic(video: str | Path, transcript: dict, info: dict, min_gap: float = 3.0) -> list[Reel]:
    """API anahtarı yokken: uzun duraklama + sahne değişimi birlikteyse sınır kabul eder."""
    segs = transcript["segments"]
    if not segs:
        return []
    scenes = media.scene_changes(video) if info.get("has_video") else []
    groups, cur = [], [segs[0]]
    for a, b in zip(segs, segs[1:]):
        gap = b["start"] - a["end"]
        scene_near = any(a["end"] - 0.5 <= t <= b["start"] + 0.5 for t in scenes)
        if gap >= min_gap or (gap >= 1.2 and scene_near):
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)

    # Çok kısa grupları komşusuna kat
    merged: list[list[dict]] = []
    for g in groups:
        if merged and (g[-1]["end"] - g[0]["start"] < MIN_REEL_SEC * 1.5):
            merged[-1] += g
        else:
            merged.append(g)

    reels = []
    for g in merged:
        start, end = _snap(segs, g[0]["id"], g[-1]["id"], info["duration"])
        reels.append(Reel(start, end, title=g[0]["text"][:60], hook=g[0]["text"], confidence=0.5,
                          notes="Sezgisel bölme (Claude API anahtarı yok) — sınırları kontrol edin",
                          source="heuristic", seg_ids=[s["id"] for s in g]))
    return _finish(reels)


def _finish(reels: list[Reel]) -> list[Reel]:
    reels.sort(key=lambda r: r.start)
    out: list[Reel] = []
    for r in reels:  # çakışmaları kırp
        if out and r.start < out[-1].end:
            r.start = out[-1].end
        if r.duration >= MIN_REEL_SEC:
            out.append(r)
    for n, r in enumerate(out, 1):
        r.id = f"reel{n:02d}"
    return out


def split_reels(video: str | Path, transcript: dict, info: dict, progress=None) -> tuple[list[Reel], str]:
    """Reels listesi ve kullanılan yöntem ('claude' | 'heuristic' | 'heuristic-fallback')."""
    if settings.has_claude:
        try:
            return split_with_claude(video, transcript, info, progress), "claude"
        except ClaudeError as e:
            if progress:
                progress(0.6, f"Claude hatası, sezgisel bölmeye geçildi: {e}")
            return split_heuristic(video, transcript, info), "heuristic-fallback"
    return split_heuristic(video, transcript, info), "heuristic"
