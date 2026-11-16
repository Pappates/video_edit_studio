"""Uzun bir çekimden birden çok reel'i ayırma.

Videoda reels'ler arasında belirgin bir işaret olmadığı varsayılır. Sınırlar üç sinyalle bulunur:
  1. Transkript: konu değişimi, giriş/kapanış kalıpları ("selam arkadaşlar", "takip etmeyi unutmayın")
  2. Duraklamalar: konuşmanın uzun süre kesildiği yerler
  3. Görüntü: sahne/kadraj/kıyafet/mekân değişimi (ffmpeg sahne tespiti + Claude vision)

Claude, numaralı transkript + aday sınırlar + zaman etiketli karelerle karar verir; segment
numarasıyla cevap verdiği için zamanlar uydurulmaz. API anahtarı yoksa sezgisel bölme yapılır.
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import markers, media
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
    visuals: list[dict] = field(default_factory=list)   # konuya göre görseller (vedit.visuals)
    # Reel'e özel sıkılaştırma (vedit.tighten): Cut sözlükleri ve özet
    tighten: list[dict] = field(default_factory=list)
    tight_level: str = ""
    tight_target: float | None = None
    tight_note: str = ""
    tight_before: float = 0.0
    tight_after: float = 0.0
    marker: str = ""          # çekimde söylenen işaret ("2. reel", "yeni reel")
    retake: str = ""          # aynı numara sonra yeniden çekildiyse açıklama (bu reel kapalı gelir)
    # İçerik ekle: işaretliyse konuya göre görseller (oyuncu/takım resimleri, sıra numaraları) eklenir.
    # Oyun reel'lerinde kapalı bırakılır; kapalıysa görsel aranmaz ve render'a görsel konmaz.
    enrich: bool = False
    enrich_done: bool = False      # bu reel için görsel araması yapıldı (render tekrar aramasın)
    # Dikey yerleşim: "" otomatik (reel'in kendi karesine bakılır), "talk" sohbet (yalnızca yüzler),
    # "game" ekran/oyun + yüzler. Otomatik tespitin sonucu layout_regions'ta saklanır.
    layout_kind: str = ""
    layout_regions: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Reel":
        r = cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
        if "enrich" not in d:          # eski işler: görseli bulunmuş reel'ler içerikli sayılır
            r.enrich = bool(r.visuals)
        return r


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
- OYUNLAR: Bir oyun reel'i (Frekans/Wavelength, quiz, tahmin oyunu vb.) BİRDEN FAZLA TUR içerir. Turlar
  arasında kategori değişir, "hazır mısın", "yeni kategori", "bakabilirsin" denir — bunlar YENİ REEL DEĞİLDİR,
  aynı oyunun devamıdır. Karelerde skor/tur sayacı (ör. "TUR 3", "6/10") devam ediyorsa oyun sürüyor.
  Bir tur sonucunu gösteren ekran oyunun sonu değildir. Reel, oyunun BİTİŞİYLE biter: kazanma/bitiş ekranı
  ("Başardınız", skor hedefe ulaştı), son tepkiler; ardından gelen "bir tane daha oynayalım mı" yeni
  hazırlıktır.
- REEL İŞARETLERİ: Ekip çekim sırasında "1. reel", "ikinci reel", "yeni reel", "sıradaki reele geçiyoruz" gibi
  işaretler söyler. Bunlar KESİN sınırdır: işaretten sonra yeni reel başlar. İşaret cümlesi reel'e dahil
  edilmez (start_id işaretten SONRAKİ segment olsun). Aynı numara birden fazla geçiyorsa o reel yeniden
  çekilmiştir: son tam denemeyi al, öncekileri 'discarded'a koy. İşaretler listesi aşağıda ayrıca verilir.

Yalnızca JSON döndür."""

PROMPT = """Video süresi: {duration:.1f} sn

Transkript (#id [başlangıç-bitiş sn] metin):
{segments}

Aday sınırlar (sn) — uzun duraklama (P) ve sahne değişimi (S):
{candidates}

Konuşmacının söylediği reel işaretleri (kesin sınır; işaret cümlesi reel'e girmez):
{markers}

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


def _sample_times(duration: float, candidates: list[tuple[float, str]], start: float = 0.0) -> list[float]:
    """Claude'a gösterilecek kare zamanları; [start, duration] aralığında."""
    span = max(1.0, duration - start)
    step = max(6.0, span / (MAX_FRAMES * 0.6))
    times = [round(t, 1) for t in _frange(start + 1.0, duration - 0.5, step)]
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


PARTIAL_NOTE = """
DİKKAT: Bu, uzun bir videonun yalnızca {start:.0f}–{end:.0f} sn arasıdır; video bu aralığın dışında devam ediyor.
- Aralığın başında, önceki bir reel'in sonu kalmış olabilir; reel'in başlangıcını görmüyorsan onu reel sayma.
- Aralığın sonundaki reel yarıda kesilmiş olabilir: kapanışı/çağrısı gelmeden bitiyorsa "complete": false yaz.
Her reel için "complete": true|false alanını ekle."""


def split_with_claude(video: str | Path, transcript: dict, info: dict, progress=None,
                      window: tuple[float, float] | None = None) -> list[Reel]:
    """window=(a, b): yalnızca bu aralığa bakılır (uzun videoda parça parça); yarım kalan reel atılır."""
    segs = transcript["segments"]
    duration = info["duration"]
    start, end = window or (0.0, duration)
    partial = window is not None and (start > 0.5 or end < duration - 0.5)
    if progress:
        progress(0.1, "Sahne değişimleri aranıyor")
    scenes = media.scene_changes(video, start=start, end=end if partial else None) if info.get("has_video") else []
    cands = _candidates(segs, scenes)

    prompt = PROMPT.format(
        duration=duration,
        segments="\n".join(fmt_segment(s) for s in segs),
        candidates=", ".join(f"{t}({k})" for t, k in cands) or "(yok)",
        markers=markers.prompt_lines(markers.find_markers(segs)),
    )
    if partial:
        prompt += PARTIAL_NOTE.format(start=start, end=end)
    content: list[dict] = [text_block(prompt)]

    if info.get("has_video"):
        times = _sample_times(end, cands, start=start)
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

    reels = reels_from_decision(segs, data, duration)
    if partial and end < duration - 0.5:
        # Aralığın sonuna dayanan reel'in devamı görülmedi; bir sonraki parçada yeniden bulunur
        reels = [r for r in reels if r.end < end - 1.0]
    return reels


def reels_from_decision(segs: list[dict], data: dict, duration: float) -> list[Reel]:
    """Claude'un {"reels": [...]} kararını segment sınırlarına oturtulmuş Reel listesine çevirir.
    "complete": false işaretli (parça sonunda yarım kalan) reels alınmaz."""
    ids = {s["id"] for s in segs}
    reels: list[Reel] = []
    for r in data.get("reels", []):
        s_id, e_id = r.get("start_id"), r.get("end_id")
        if s_id not in ids or e_id not in ids or s_id > e_id or r.get("complete") is False:
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
        markers=markers.prompt_lines(markers.find_markers(segs)),
    ).replace("Ardından zaman etiketli kareler geliyor.", "Kareler (görmek için dosyaları açın):\n" + "\n".join(frame_lines))
    packet = out_dir / "packet.md"
    packet.write_text("# Reels bölme paketi\n\n## Talimat\n\n" + SYSTEM + "\n\n" + body, encoding="utf-8")
    return packet


def split_heuristic(video: str | Path, transcript: dict, info: dict, min_gap: float = 3.0,
                    window: tuple[float, float] | None = None) -> list[Reel]:
    """API anahtarı yokken: uzun duraklama + sahne değişimi birlikteyse sınır kabul eder.
    window verilirse ve video devam ediyorsa, aralığın sonuna dayanan son grup yarım sayılıp atılır."""
    segs = transcript["segments"]
    if not segs:
        return []
    start, end = window or (0.0, info["duration"])
    partial = window is not None and end < info["duration"] - 0.5
    scenes = media.scene_changes(video, start=start, end=end if window else None) if info.get("has_video") else []
    marks = {m["seg_id"]: m for m in markers.find_markers(segs)}
    # (grup, işaretle mi başladı) — işaret kesin sınırdır; yalnızca işaretten oluşan segment gruba girmez
    groups: list[tuple[list[dict], bool]] = []
    cur: list[dict] = []
    hard = False
    prev = None
    for b in segs:
        mk = marks.get(b["id"])
        if prev is not None:
            gap = b["start"] - prev["end"]
            scene_near = any(prev["end"] - 0.5 <= t <= b["start"] + 0.5 for t in scenes)
            if mk or gap >= min_gap or (gap >= 1.2 and scene_near):
                if cur:
                    groups.append((cur, hard))
                cur, hard = [], False
        if mk:
            hard = True
        if not (mk and mk["whole"]):
            cur.append(b)
        prev = b
    if cur:
        groups.append((cur, hard))

    # Çok kısa grupları komşusuna kat (işaretle başlayan grup önceki reel'e katılmaz)
    merged: list[list[dict]] = []
    for g, g_hard in groups:
        if merged and not g_hard and (g[-1]["end"] - g[0]["start"] < MIN_REEL_SEC * 1.5):
            merged[-1] += g
        else:
            merged.append(g)

    if partial and len(merged) > 0 and end - merged[-1][-1]["end"] < min_gap:
        merged.pop()  # sonrası görülmedi: reel burada bitiyor mu bilinmez
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


def split_reels(video: str | Path, transcript: dict, info: dict, progress=None,
                window: tuple[float, float] | None = None) -> tuple[list[Reel], str]:
    """Reels listesi ve kullanılan yöntem ('claude' | 'heuristic' | 'heuristic-fallback').
    window=(a, b): transcript yalnızca bu aralığı kapsar; yarım kalan son reel listeye girmez."""
    if settings.has_claude:
        try:
            reels, method = split_with_claude(video, transcript, info, progress, window=window), "claude"
        except ClaudeError as e:
            if progress:
                progress(0.6, f"Claude hatası, sezgisel bölmeye geçildi: {e}")
            reels, method = split_heuristic(video, transcript, info, window=window), "heuristic-fallback"
    else:
        reels, method = split_heuristic(video, transcript, info, window=window), "heuristic"
    if method == "claude" and reels:
        limit = window[1] if window else info["duration"]
        reels = refine_ends(video, transcript["segments"], reels, info, limit, progress)
    return mark_retakes(apply_markers(reels, transcript["segments"])), method


DESCRIBE_SYSTEM = """Sen sosyal medya editörüsün. Sınırlarını ekibin elle belirlediği TEK bir reel'in transkripti
veriliyor. Reel'e başlık, kanca ve Instagram açıklaması yaz. Sınırları değiştirme. Yalnızca JSON döndür."""

DESCRIBE_PROMPT = """Transkript (#id [başlangıç-bitiş sn] metin):
{segments}

İstenen JSON: {{"title": "<kısa başlık>", "hook": "<ilk 3 sn'de söylenen kanca cümle ya da en güçlü cümle>",
  "summary": "<1 cümle>", "caption": "<Instagram açıklaması, Türkçe, 1-3 cümle>", "hashtags": ["#..."]}}"""


def describe_reel(segs: list[dict], reel: dict) -> dict:
    """Elle işaretlenen reel için başlık/kanca/açıklama/hashtag. Claude hatasında boş döner (reel yine kullanılır)."""
    inside = [s for s in segs if s["end"] > reel["start"] and s["start"] < reel["end"]]
    if not inside:
        return {}
    try:
        data = call_json(DESCRIBE_PROMPT.format(segments="\n".join(fmt_segment(s) for s in inside)),
                         system=DESCRIBE_SYSTEM)
    except ClaudeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {k: str(data.get(k) or "")[:500] for k in ("title", "hook", "summary", "caption") if data.get(k)}
    if isinstance(data.get("hashtags"), list):
        out["hashtags"] = [str(h) for h in data["hashtags"]][:15]
    out["seg_ids"] = [s["id"] for s in inside]
    return out


END_SYSTEM = """Sen video editörüsün. Bir reel'in nerede BİTMESİ gerektiğini kontrol ediyorsun. Reel bir
konu/oyun bütünüdür: erken biterse izleyici sonucu göremez, geç biterse ilgisiz konuşma girer.

- Oyunlarda reel oyunun BİTİŞİYLE biter: son turun sonucu, kazanma/bitiş ekranı ("Başardınız",
  "Kazandınız", "Oyun bitti", skor hedefe ulaştı) ve hemen ardından gelen son tepkiler. Turlar arasında
  kategori değişmesi, "hazır mısın", "yeni kategori", "bakabilirsin" gibi geçişler oyunun DEVAMIDIR.
- Konu anlatımında anlatım ve kapanış cümlesiyle biter.
- "Bir tane daha oynayalım mı", yeni kategori/konu seçme sohbeti, kamera ayarı, "ikinci reel" gibi işaretler
  SONRAKİ hazırlıktır; reel'e girmez.
- Karelere mutlaka bak: skor/tur sayacı (ör. "TUR 3", "6/10") sürüyorsa oyun bitmemiştir; bitiş ekranı
  belirleyicidir.

Yalnızca JSON döndür."""

END_PROMPT = """Reel: {title}
Özet: {summary}
Reel #{start_id} ile başlıyor; şu an #{cur_id} ile bitiyor (t={end:.1f} sn).

Bitiş çevresindeki transkript (#id [başlangıç-bitiş sn] metin):
{segments}

Ardından zaman etiketli kareler geliyor.

İstenen JSON: {{"end_id": <reel'in olması gereken son segmenti>, "changed": true|false,
  "reason": "<kısa Türkçe gerekçe: oyun/konu nerede bitiyor>"}}"""

END_LOOKAHEAD_SEC = 150.0


def refine_ends(video: str | Path, segs: list[dict], reels: list[Reel], info: dict,
                limit_end: float | None = None, progress=None) -> list[Reel]:
    """Her reel'in bitişini Claude'a bir kez daha sorar: bitişten sonraki ~2.5 dk'lık transkript ve oyun
    ekranının kareleriyle. Bölme adımı tüm videoya bir bakışta karar verdiği için turlar arası geçişi yeni
    reel sanıp oyunun son turunu/bitiş ekranını dışarıda bırakabiliyordu. Reel ancak sonraki reel'e kadar
    uzatılabilir; Claude hatasında reel olduğu gibi kalır."""
    if not segs:
        return reels
    limit_end = limit_end or info["duration"]
    ordered = sorted(reels, key=lambda r: r.start)
    by_id = {s["id"]: s for s in segs}
    order = {s["id"]: k for k, s in enumerate(segs)}
    for n, r in enumerate(ordered):
        nxt = ordered[n + 1].start if n + 1 < len(ordered) else limit_end
        hi = min(nxt, r.end + END_LOOKAHEAD_SEC, limit_end)
        lo = max(r.start, r.end - 40.0)
        win = [s for s in segs if s["end"] > lo and s["start"] < hi]
        inside = [s for s in segs if s["start"] >= r.start - 0.3 and s["end"] <= r.end + 0.5]
        if not win or not inside or not any(s["start"] >= r.end - 0.1 for s in win):
            continue   # sonrasında konuşma yok: uzatılacak bir şey yok
        if progress:
            progress(0.9, f"{r.id or 'reel'}: bitiş kontrol ediliyor")
        content: list[dict] = [text_block(END_PROMPT.format(
            title=r.title or "(başlıksız)", summary=r.summary or r.hook or "-", start_id=inside[0]["id"],
            cur_id=inside[-1]["id"], end=r.end, segments="\n".join(fmt_segment(s) for s in win)))]
        try:
            with tempfile.TemporaryDirectory(prefix="vedit_end_") as td:
                if info.get("has_video"):
                    a = max(r.start, r.end - 15.0)
                    step = max(8.0, (hi - a) / 9)
                    t, k = a, 0
                    while t <= hi and k < 10:
                        p = Path(td) / f"e{k:02d}.jpg"
                        try:
                            media.extract_frame(video, t, p, width=512)
                            content += [text_block(f"Kare t={t:.1f} sn"), image_block(p)]
                        except RuntimeError:
                            pass
                        t, k = t + step, k + 1
                data = call_json(content, system=END_SYSTEM)
        except ClaudeError:
            continue
        end_id = data.get("end_id") if isinstance(data, dict) else None
        if end_id not in by_id or order[end_id] < order[inside[0]["id"]]:
            continue
        seg = by_id[end_id]
        nxt_seg = segs[order[end_id] + 1] if order[end_id] + 1 < len(segs) else None
        new_end = round(min(seg["end"] + 0.4, nxt_seg["start"] if nxt_seg else seg["end"] + 0.4, nxt - 0.1), 3)
        if new_end - r.start < MIN_REEL_SEC or abs(new_end - r.end) < 0.5:
            continue
        if new_end > limit_end - 2.0 and limit_end < info["duration"] - 0.5:
            continue   # taranan parçanın sınırına dayandı: devamı görülmedi, sonraki parçada karar verilir
        reason = str(data.get("reason") or "")[:200]
        r.notes = (f"Bitiş {r.end:.1f} → {new_end:.1f} sn düzeltildi: {reason}. " + (r.notes or "")).strip()
        r.end = new_end
    return reels


def marker_number(reel: Reel) -> int | None:
    m = re.match(r"(\d+)\. reel", reel.marker or "")
    return int(m.group(1)) if m else None


def mark_retakes(reels: list[Reel]) -> list[Reel]:
    """Aynı numaralı işaretle birden çok reel varsa ("1. reel" bozuldu, sonra yine "1. reel"): son deneme
    seçili kalır, öncekiler silinmez ama kapalı gelir ve not düşülür; kullanıcı istediğini seçebilir."""
    last: dict[int, Reel] = {}
    for r in sorted(reels, key=lambda r: r.start):
        n = marker_number(r)
        if n is not None:
            last[n] = r
    count: dict[int, int] = {}
    for r in sorted(reels, key=lambda r: r.start):
        n = marker_number(r)
        if n is None:
            continue
        count[n] = count.get(n, 0) + 1
        if last[n] is not r:
            r.enabled = False
            r.retake = f"Önceki deneme ({count[n]}. çekim) — {r.marker} sonra yeniden çekildi; son deneme seçildi"
        elif count[n] > 1:
            r.retake = ""
    for n, r in last.items():
        if count.get(n, 0) > 1 and not r.retake and "kez çekilmiş" not in (r.notes or ""):
            r.notes = (f"{r.marker} {count[n]} kez çekilmiş; bu son deneme. " + (r.notes or "")).strip()
    return reels


def first_n(reels: list[Reel], n: int) -> list[Reel]:
    """İlk n reel (tekrar çekimlerin kapalı önceki denemeleri sayılmaz ama listede kalır)."""
    if not n:
        return reels
    chosen = [r for r in reels if r.enabled][:n]
    if not chosen:
        return []
    numbers = {marker_number(r) for r in chosen} - {None}
    last_end = max(r.end for r in chosen)
    return [r for r in reels if r in chosen or (not r.enabled and r.retake and marker_number(r) in numbers
                                                 and r.end <= last_end)]


def apply_markers(reels: list[Reel], segs: list[dict]) -> list[Reel]:
    """Söylenen reel işaretlerine göre reel kenarlarını düzeltir ve işareti reel'e yazar.

    - Reel yalnızca işaretten oluşan bir segmentle başlıyorsa başlangıç sonraki segmente kayar.
    - Reel sonraki reel'in işaretiyle bitiyorsa (Claude dahil ettiyse) bitiş işaretten öncesine çekilir.
    İşaret kelimeleri cümlenin içindeyse ayrıca kesim olarak çıkarılır (markers.marker_cuts).
    """
    marks = markers.find_markers(segs)
    if not marks:
        return reels
    whole = {m["seg_id"] for m in marks if m["whole"]}
    prev_end = float("-inf")
    for r in sorted(reels, key=lambda r: r.start):
        inside = [s for s in segs if s["end"] > r.start and s["start"] < r.end]
        while inside and inside[0]["id"] in whole and len(inside) > 1:
            nxt = inside[1]
            r.start = round(max(inside[0]["end"], nxt["start"] - 0.2), 3)
            inside = inside[1:]
        while inside and inside[-1]["id"] in whole and len(inside) > 1:
            prv = inside[-2]
            r.end = round(min(inside[-1]["start"], prv["end"] + 0.4), 3)
            inside = inside[:-1]
        # Reel'in işareti: kendisinden hemen önce (ya da ilk cümlesinde) söylenen işaret. Önceki reel'in
        # bitişinden önceki işaret o reel'e aittir — duraklamayla bölünmüş bir reel'in ikinci parçası
        # önceki numarayı devralırsa mark_retakes asıl reel'i "önceki deneme" sanıp kapatırdı.
        before = [m for m in marks if prev_end - 0.05 <= m["start"] <= r.start + 3.0 and m["end"] >= r.start - 30.0]
        if before:
            r.marker = before[-1]["label"]
        prev_end = max(prev_end, r.end)
    return reels
