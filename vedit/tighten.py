"""Reel sıkılaştırma: konu konu bölündükten sonra her reel'i kendi içinde olabildiğince kısaltır.

Genel temizlik (sessizlik, tekrar, komut, çekim dışı muhabbet) tüm ham videoya bakar ve reel'in ne
anlattığını bilmez. Burada Claude tek bir reel'in kalan transkriptini, reel'in başlığı ve kancasıyla görür
ve "hikâye bozulmadan kısalt" görevini alır: kategori yazma/değiştirme, kural açıklamasının uzayan kısmı,
bekleme, lojistik ve konu dışı laflar gider; oyunun can alıcı anları, espriler, önemli tepkiler kalır.
Ayrıca reel içindeki sessizlikler daha sıkı kesilir (kısa video tarzı hızlı geçiş).

Kesimler reel'in `tighten` listesinde Cut sözlükleri olarak saklanır; arayüzde tek tek açılıp kapatılabilir.
"""
from __future__ import annotations

from .claude_client import ClaudeError, call_json, image_block, text_block
from .claude_review import fmt_segment
from .config import settings
from .detector import Cut, keep_ranges

# silence_keep: kesilen sessizlikten geriye bırakılan süre (kelimenin sonu ve nefes payı). Whisper kelime
# bitişlerini erken verir; çok kısa bırakmak cümle sonlarını "kesilmiş" duyurur.
LEVELS: dict[str, dict] = {
    "light": {"name": "Hafif", "silence_min": 1.2, "silence_keep": 0.5,
              "rule": "Yalnızca açıkça gereksiz olanları kes: uzun bekleme, kategori/kağıt yazma-değiştirme, "
                      "ekip içi lojistik, konu dışı laflar. Oyunun/anlatının akışına dokunma; emin değilsen KESME."},
    "medium": {"name": "Orta", "silence_min": 0.8, "silence_keep": 0.35,
               "rule": "Lojistiğin hepsini kes; ayrıca uzayan açıklamaları, gereksiz tekrarlanan tahmin/yorumları ve "
                       "tempoyu düşüren ara konuşmaları kısalt. Her turun/maddenin özü ve sonucu kalsın."},
    "aggressive": {"name": "Agresif", "silence_min": 0.6, "silence_keep": 0.25,
                   "rule": "Sadece hikâyeyi anlatmak için GEREKLİ olan kalsın: kanca, her adımın en kısa hali, "
                           "doruk/sonuç anı ve en iyi tepki. İzleyici anlamaya devam ettiği sürece cesurca kes."},
}
MIN_KEEP_SEC = 0.3     # bundan kısa kalan parça "kırpık hece" gibi duyulur → atılır
MIN_CUT_SEC = 0.15     # bundan kısa kesim fark edilmez ama sesi bozar → yapılmaz

SYSTEM = """Sen kısa video (Instagram Reels / TikTok) editörüsün. Sana TEK bir reel'in transkripti veriliyor;
reel zaten konu olarak ayrılmış ve tekrar/sessizlik gibi hatalardan temizlenmiş. Görevin onu
izleyiciyi kaçırmadan OLABİLDİĞİNCE KISALTMAK.

Kesilecek tipik şeyler:
- Oyun/çekim lojistiği: kategori, kart, kağıt yazma/değiştirme ("dur şunu yazayım", "hangi kategori olsun",
  "kalem nerede", "tamam değiştirelim"), sıra belirleme, puan sayma, kuralı uzun uzun anlatma.
- Bekleme ve dolgu: "hmm", "bir saniye", "düşüneyim" ile uzayan yerler, bir şey olmasını beklerken konuşulanlar.
- Konu dışı muhabbet, iç espriler, ekip arası konuşmalar.
- Aynı şeyin ikinci kez söylenmesi, aynı tahminin tekrarı.

Kesilmeyecekler:
- Kanca (reel'in ilk cümlesi/sorusu) ve sonuç/doruk anı ile en iyi tepki — reel'in neden izlendiği.
- Bir sonraki anı anlamak için gereken bağlam (ör. hangi kategoriyle oynandığı TEK cümleyle kalmalı).
- Komik ya da duygusal güçlü anlar.
- OYUNUN KENDİSİ: ekranda bir oyun/uygulama oynanıyorsa (karelere bak) "sola doğru", "biraz daha",
  "oralar", "şurası", sayı/derece söyleme, ipucu verme, tahmin ve hamleler oyunun parçasıdır — kamera
  ayarı ya da lojistik DEĞİLDİR. Bir konuşmanın lojistik mi oyun mu olduğundan emin değilsen kesme.

Bazı segmentlerin yanında ⟨genel temizlik 'muhabbet' dedi⟩ notu olabilir: bu, tüm videoya yalnızca metinle
bakan ön temizliğin tahminidir ve oyun içi konuşmayı yanlışlıkla muhabbet sanmış olabilir. Kareye ve reel'in
konusuna bakarak SEN karar ver: gerçekten çekim dışıysa kes, oyunun/konunun parçasıysa bırak.

Segmentlerde "S1:", "S2:" konuşmacıyı gösterir. Yalnızca segment numaralarıyla cevap ver; ardışık
segmentleri tek kesimde birleştir. Yalnızca JSON döndür."""

PROMPT = """Reel: {title}
Kanca: {hook}
Şu anki süre: {cur:.0f} sn{target}{context}

Seviye — {level_name}: {rule}

Kalan transkript (#id [başlangıç-bitiş sn] metin):
{segments}

İstenen JSON:
{{"cuts": [{{"start_id": <int>, "end_id": <int>, "kind": "logistics|waiting|offtopic|redundant|slow",
            "reason": "<kısa Türkçe gerekçe>", "confidence": <0-1>}}],
  "summary": "<1 cümle: neyi kestin, ne kaldı>"}}{frames_note}"""

KIND_TR = {"logistics": "lojistik", "waiting": "bekleme", "offtopic": "konu dışı", "redundant": "tekrar",
           "slow": "yavaş kısım", "silence": "sessizlik"}


def _reel_words(segs: list[dict], start: float, end: float) -> list[dict]:
    """Reel'e değen kelimeler — kenardan taşanlar dahil (sınır elle bir kelimenin ortasına konmuş olabilir)."""
    return sorted((w for s in segs for w in (s.get("words") or [{"start": s["start"], "end": s["end"], "word": s["text"]}])
                   if w["end"] > start and w["start"] < end), key=lambda w: w["start"])


def clip_to_quiet(a: float, b: float, quiet: list[tuple[float, float]] | None,
                  min_len: float = 0.2) -> list[tuple[float, float]]:
    """[a, b] aralığının gerçekten sessiz olan parçaları. quiet None ise (ses ölçülmediyse) aralık aynen döner.
    Transkriptte kelime yok diye kesilecek yer kahkaha/tepki/oyun sesi olabilir; yalnızca sessiz kısmı kesilir."""
    if quiet is None:
        return [(a, b)] if b - a >= min_len else []
    out = []
    for qa, qb in quiet:
        x, y = max(a, qa), min(b, qb)
        if y - x >= min_len:
            out.append((round(x, 3), round(y, 3)))
    return out


def refine_silence_cuts(cuts: list[Cut], quiet: list[tuple[float, float]], lo: float = float("-inf"),
                        hi: float = float("inf")) -> list[Cut]:
    """Genel temizliğin 'silence' kesimlerini [lo, hi] içinde gerçek sessizlikle sınırlar (bölebilir)."""
    out: list[Cut] = []
    for c in cuts:
        if c.kind != "silence" or c.end <= lo or c.start >= hi:
            out.append(c)
            continue
        for n, (a, b) in enumerate(clip_to_quiet(c.start, c.end, quiet)):
            out.append(Cut(a, b, "silence", c.reason, c.text, c.confidence, c.enabled, c.source,
                           c.id if n == 0 else f"{c.id}q{n}", c.seg_ids))
    return out


def tight_silences(segs: list[dict], start: float, end: float, level: str,
                   quiet: list[tuple[float, float]] | None = None) -> list[Cut]:
    """Reel içindeki konuşmasız aralıkları seviyeye göre sıkı keser.

    - Boşluk her zaman o ana kadar söylenmiş kelimelerin EN GEÇ bitişinden ölçülür; reel sınırından taşan
      ya da üst üste binen kelimelerin ucu kırpılmaz.
    - Bırakılan payın %60'ı kelimeden sonra (Whisper bitişi erken verir, cümle sonu kesilmesin), %40'ı
      sonraki kelimeden önce.
    - quiet verilirse yalnızca gerçekten sessiz kısım kesilir (kahkaha, tepki, oyun sesi korunur)."""
    lv = LEVELS[level]
    words = _reel_words(segs, start, end)
    if not words:
        return []
    min_sec, keep = lv["silence_min"], lv["silence_keep"]
    tail, lead = keep * 0.6, keep * 0.4
    cuts: list[Cut] = []

    def gap(a_edge: float, b_edge: float, at_start: bool, at_end: bool) -> None:
        if b_edge - a_edge < min_sec:
            return
        a = a_edge + (0.05 if at_start else tail)
        b = b_edge - (0.05 if at_end else lead)
        for x, y in clip_to_quiet(a, b, quiet, min_len=max(0.2, min_sec - keep)):
            cuts.append(Cut(x, y, "silence", f"{b_edge - a_edge:.1f} sn sessizlik → ~{keep} sn", "", 0.9,
                            source="tighten"))

    prev_end, first = start, True
    for w in words:
        gap(prev_end, w["start"], first, False)   # reel başındaki ölü süre (neredeyse) tamamen gider
        prev_end, first = max(prev_end, w["end"]), False
    gap(prev_end, end, False, True)               # reel sonundaki ölü süre (neredeyse) tamamen gider
    return cuts


def _snap(segs_by_id: dict, ordered: list[dict], s_id: int, e_id: int) -> tuple[float, float]:
    """Kesimi kelime aralarına oturtur: öncekinin bitişi ile sonrakinin başlangıcının ortası."""
    idx = {s["id"]: k for k, s in enumerate(ordered)}
    i, j = idx[s_id], idx[e_id]
    a, b = segs_by_id[s_id]["start"], segs_by_id[e_id]["end"]
    if i > 0:
        a = max(ordered[i - 1]["end"], a - 0.06)
    if j + 1 < len(ordered):
        b = min(ordered[j + 1]["start"], b + 0.06)
    return round(a, 3), round(b, 3)


def claude_cuts(segs: list[dict], reel: dict, level: str, target_sec: float | None,
                base_keeps: list[tuple[float, float]], frames: list | None = None,
                context: str = "", flagged: list[Cut] | None = None) -> tuple[list[Cut], str]:
    """Claude'un reel'e özel kesimleri. base_keeps: genel temizlikten sonra kalan aralıklar.
    frames: [(t, jpg_yolu), ...] — Claude ekranda ne olduğunu görsün (oyun hamlesi ≠ lojistik).
    context: kullanıcının içerik notu (ör. "Frekans oyunu: ibreyi sağa/sola yönlendirmek oyunun parçası").
    flagged: genel temizliğin 'muhabbet' dediği yerler — segmentlerin yanında belirtilir, karar Claude'un."""
    flagged = flagged or []

    def line(s):
        f = next((c for c in flagged if c.start < s["end"] and s["start"] < c.end), None)
        return fmt_segment(s) + (f"   ⟨genel temizlik 'muhabbet' dedi: {f.reason[:60]}⟩" if f else "")
    def kept(s):
        mid = (s["start"] + s["end"]) / 2
        return any(a <= mid <= b for a, b in base_keeps)

    remaining = [s for s in segs if s["end"] > reel["start"] and s["start"] < reel["end"] and kept(s)]
    if len(remaining) < 2:
        return [], ""
    cur = sum(b - a for a, b in base_keeps)
    lv = LEVELS[level]
    target = f" — hedef: {target_sec:.0f} sn veya altı" if target_sec and target_sec < cur else ""
    ctx = f"\nİçerik notu (ekipten): {context.strip()[:600]}" if context and context.strip() else ""
    frames = frames or []
    prompt = PROMPT.format(title=reel.get("title") or "(başlıksız)", hook=reel.get("hook") or "-", cur=cur,
                           target=target, context=ctx, level_name=lv["name"], rule=lv["rule"],
                           segments="\n".join(line(s) for s in remaining),
                           frames_note="\n\nArdından reel'den zaman etiketli kareler geliyor." if frames else "")
    content: list[dict] = [text_block(prompt)]
    for t, path in frames:
        content += [text_block(f"Kare t={t:.1f} sn"), image_block(path)]
    data = call_json(content, system=SYSTEM)
    by_id = {s["id"]: s for s in remaining}
    first_id = remaining[0]["id"]
    out: list[Cut] = []
    for c in data.get("cuts", []) if isinstance(data, dict) else []:
        s_id, e_id = c.get("start_id"), c.get("end_id")
        if s_id not in by_id or e_id not in by_id or s_id > e_id:
            continue
        if s_id == first_id and e_id == remaining[-1]["id"]:
            continue   # reel'in tamamını kesmek geçersiz bir karar
        a, b = _snap(by_id, remaining, s_id, e_id)
        kind = c.get("kind") if c.get("kind") in KIND_TR else "slow"
        try:
            conf = float(c.get("confidence", 0.75))
        except (TypeError, ValueError):
            conf = 0.75
        text = " ".join(by_id[k]["text"] for k in sorted(by_id) if s_id <= k <= e_id)
        out.append(Cut(a, b, kind, str(c.get("reason") or KIND_TR[kind])[:200], text[:400], conf,
                       enabled=conf >= 0.5, source="tighten",
                       seg_ids=[k for k in sorted(by_id) if s_id <= k <= e_id]))
    return out, str(data.get("summary", "") if isinstance(data, dict) else "")[:300]


# Genel temizliğin (tüm ham videoya, yalnızca metinle bakan Claude) bu türdeki kesimleri, kısaltılmış bir
# reel'de doğrudan uygulanmaz: reel bağlamını ve kareleri gören kısaltma adımı karar verir. Oyun reel'lerinde
# genel temizlik oyun içi konuşmayı ("sola doğru", "hazır mısın", "bakmayacağım") muhabbet sanabiliyordu.
DEFER_KINDS = {"chatter"}


def reel_base_cuts(reel: dict, base_cuts: list[Cut], tightened: bool | None = None) -> list[Cut]:
    """Reel'e uygulanacak genel temizlik kesimleri. 'İçini temizle' kapalıysa hiçbiri; reel kısaltıldıysa
    (ya da kısaltılıyorsa) 'muhabbet' kesimleri kısaltma adımına bırakılır."""
    if not reel.get("clean", True):
        return []
    tightened = bool(reel.get("tight_level")) if tightened is None else tightened
    return [c for c in base_cuts if not (tightened and c.kind in DEFER_KINDS)]


def tighten_reel(transcript: dict, reel: dict, base_cuts: list[Cut], level: str = "medium",
                 target_sec: float | None = None, use_claude: bool = True,
                 quiet: list[tuple[float, float]] | None = None, frames: list | None = None,
                 context: str = "") -> dict:
    """Reel'in sıkılaştırma alanlarını hesaplar ve döner (reel sözlüğüne update edilecek).
    quiet: media.quiet_intervals() — verilirse sessizlik yalnızca gerçekten sessiz yerde kesilir."""
    level = level if level in LEVELS else "medium"
    segs = transcript["segments"]
    start, end = reel["start"], reel["end"]
    base = reel_base_cuts(reel, base_cuts, tightened=True)
    flagged = [c for c in base_cuts if c.kind in DEFER_KINDS and c.enabled and c.end > start and c.start < end] \
        if reel.get("clean", True) else []
    base_keeps = keep_ranges(base, 0.0, within=(start, end))
    before = sum(b - a for a, b in base_keeps)

    cuts = tight_silences(segs, start, end, level, quiet)
    note = ""
    if use_claude and settings.has_claude:
        try:
            cc, note = claude_cuts(segs, reel, level, target_sec, base_keeps, frames, context, flagged)
            cuts += cc
        except ClaudeError as e:
            note = f"Claude hatası, yalnızca sessizlikler sıkılaştırıldı: {e}"
    else:
        note = "Claude bağlı değil — yalnızca sessizlikler sıkılaştırıldı"
        cuts += [Cut(c.start, c.end, "offtopic", f"Genel temizlik: {c.reason}"[:200], c.text, c.confidence,
                     c.enabled, "tighten") for c in flagged]   # Claude yoksa genel temizliğin kararı geçerli

    cuts.sort(key=lambda c: c.start)
    for n, c in enumerate(cuts, 1):
        c.id = f"t{n:03d}"
    after = sum(b - a for a, b in smooth_keeps(keep_ranges(base + cuts, 0.0, within=(start, end))))
    return {"tighten": [c.to_dict() for c in cuts], "tight_level": level,
            "tight_target": target_sec or None, "tight_note": note,
            "tight_before": round(before, 1), "tight_after": round(after, 1)}


def smooth_keeps(keeps: list[tuple[float, float]], min_keep: float = MIN_KEEP_SEC,
                 min_cut: float = MIN_CUT_SEC) -> list[tuple[float, float]]:
    """Üst üste binen kesimlerin bıraktığı kırıntıları temizler:
    - iki kalan parça arasındaki kesim min_cut'tan kısaysa kesim yapılmaz (parçalar birleşir);
    - min_keep'ten kısa kalan parça atılır (0.07 sn'lik parça kırpık hece gibi duyulur)."""
    merged: list[list[float]] = []
    for a, b in keeps:
        if merged and a - merged[-1][1] < min_cut:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_keep]


def reel_keeps(reel: dict, base_cuts: list[Cut]) -> list[tuple[float, float]]:
    """Reel'in render'da kalacak aralıkları: genel temizlik ('içini temizle' açıksa) + sıkılaştırma."""
    cuts = reel_base_cuts(reel, base_cuts)
    cuts += [Cut.from_dict(c) for c in reel.get("tighten") or []]
    return smooth_keeps(keep_ranges(cuts, 0.0, within=(reel["start"], reel["end"])))
