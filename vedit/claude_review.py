"""Claude ile kural tabanlı kesim adaylarını doğrulama ve anlamsal ek kesim bulma.

Kurallar kelime benzerliğine dayanır; Claude ise şunları yakalar:
  * Farklı kelimelerle yapılan tekrarlar (aynı fikri ikinci kez anlatma)
  * Çekimler arası muhabbet ("kamera açık mı", "hazır mısın", "su verir misin")
  * Kuralların yanlış pozitifleri (ör. "bunu kesinlikle" içinde 'kes' geçmesi)

Zaman uydurmasını önlemek için Claude yalnızca segment numaralarıyla cevap verir.
"""
from __future__ import annotations

from .claude_client import ClaudeError, call_json
from .detector import Cut

CHUNK = 260
OVERLAP = 20

SYSTEM = """Sen Türkçe video kurgu asistanısın. Ham çekimin transkripti sana numaralı segmentler halinde verilir.
Görevin yayınlanacak videodan nelerin çıkarılması gerektiğine karar vermek.

Çıkarılacaklar:
1. command  : Konuşmacının kurgu komutu verdiği yer ("burayı kes", "şurayı keselim") ve komutun işaret ettiği hatalı kısım.
2. retake   : "Baştan alıyorum", "tekrar alalım" gibi denemeler. Başarısız deneme + komut çıkar, SON deneme kalır.
3. repeat   : Aynı cümlenin/fikrin birden fazla kez söylenmesi (kelimeler farklı olsa da). En iyi/son hali kalır.
4. chatter  : Çekim dışı muhabbet: kamera/ses kontrolü, "hazır mısın", "kaydediyor mu", ekip arası konuşmalar, kahkaha ve dağılmalar.

Segmentlerde "S1:", "S2:" gibi etiketler varsa konuşmacıyı gösterir. Kamera arkasındaki kişinin kısa
yönlendirmeleri ("tamam", "bir daha", "kamera açık") çoğunlukla chatter'dır; ama röportaj/sohbet
formatında karşı tarafın soruları içeriğin parçasıdır.

Çıkarılmayacaklar: konunun parçası olan doğal tekrarlar ve vurgular ("tekrar ediyorum, bu çok önemli"),
özet/kapanış cümleleri, içerik içindeki esprili ama konuyla ilgili diyaloglar. Emin değilsen confidence'ı düşük ver.

Yalnızca JSON döndür."""

PROMPT = """Segmentler (#id [başlangıç-bitiş sn] metin):
{segments}

Kural motorunun bulduğu adaylar:
{candidates}

İstenen JSON formatı:
{{
  "verdicts": [{{"candidate": "<aday id>", "keep_cut": true|false, "start_id": <int>, "end_id": <int>, "note": "<kısa gerekçe>"}}],
  "new_cuts": [{{"start_id": <int>, "end_id": <int>, "kind": "command|retake|repeat|chatter", "reason": "<kısa Türkçe gerekçe>", "confidence": <0-1>}}]
}}
- verdicts: her aday için karar. Aralık yanlışsa start_id/end_id ile düzelt.
- new_cuts: adaylarda olmayan ek kesimler. start_id..end_id dahil tüm segmentler kesilir.
- Bir segmentin yalnızca bir kısmı kesilecekse yine o segmenti ver; ince ayarı kullanıcı yapar."""


def _fmt_segments(segs: list[dict]) -> str:
    return "\n".join(fmt_segment(s) for s in segs)


def fmt_segment(s: dict) -> str:
    """#id [başlangıç-bitiş] S2: metin  (konuşmacı etiketi varsa)"""
    spk = f"{s['speaker']}: " if s.get("speaker") else ""
    return f"#{s['id']} [{s['start']:.1f}-{s['end']:.1f}] {spk}{s['text']}"


def review_cuts(segs: list[dict], cuts: list[Cut], progress=None) -> list[Cut]:
    by_id = {s["id"]: s for s in segs}
    for n, c in enumerate(cuts):
        c.id = c.id or f"r{n}"
    result: dict[str, Cut] = {c.id: c for c in cuts}
    extra: list[Cut] = []

    chunks = list(range(0, len(segs), CHUNK - OVERLAP)) or [0]
    for ci, start in enumerate(chunks):
        part = segs[start: start + CHUNK]
        if not part:
            break
        lo, hi = part[0]["start"], part[-1]["end"]
        # Kelime düzeyindeki kekeleme kesimleri segment düzeyinde incelenemez; kurala bırakılır.
        local = [c for c in cuts if c.start < hi and c.end > lo and c.kind not in ("stutter", "silence")]
        cand_txt = "\n".join(
            f"{c.id}: {c.kind} segs {c.seg_ids[:1] + c.seg_ids[-1:] if c.seg_ids else '?'} — {c.reason}"
            for c in local
        ) or "(yok)"
        try:
            data = call_json(PROMPT.format(segments=_fmt_segments(part), candidates=cand_txt), system=SYSTEM)
        except ClaudeError as e:
            if progress:
                progress((ci + 1) / len(chunks), f"Claude doğrulaması atlandı: {e}")
            return cuts

        extra += apply_decisions(segs, result, data)
        if progress:
            progress((ci + 1) / len(chunks), f"Claude doğrulaması {ci + 1}/{len(chunks)}")

    return list(result.values()) + extra


def apply_decisions(segs: list[dict], cuts_by_id: dict[str, Cut], data: dict) -> list[Cut]:
    """Claude kararlarını (verdicts/new_cuts JSON) aday kesimlere uygular, yeni kesimleri döner.

    Hem API üzerinden çağrıda hem de skill modunda (Claude'un kendisi review.md'yi okuyup
    decisions.json yazdığında) aynı fonksiyon kullanılır.
    """
    by_id = {s["id"]: s for s in segs}
    for v in data.get("verdicts", []):
        c = cuts_by_id.get(str(v.get("candidate")))
        if not c:
            continue
        if v.get("keep_cut") is False:
            c.enabled = False
            c.confidence = min(c.confidence, 0.35)
            c.reason += f" | Claude: yanlış pozitif ({v.get('note', '')})"
            continue
        s_id, e_id = v.get("start_id"), v.get("end_id")
        if s_id in by_id and e_id in by_id and s_id <= e_id and c.kind != "stutter":
            c.start, c.end = by_id[s_id]["start"], by_id[e_id]["end"]
            c.seg_ids = list(range(s_id, e_id + 1))
        c.confidence = max(c.confidence, 0.8)
        if v.get("note"):
            c.reason += f" | Claude: {v['note']}"

    extra: list[Cut] = []
    for nc in data.get("new_cuts", []):
        s_id, e_id = nc.get("start_id"), nc.get("end_id")
        if s_id not in by_id or e_id not in by_id or s_id > e_id:
            continue
        extra.append(Cut(
            by_id[s_id]["start"], by_id[e_id]["end"], nc.get("kind", "chatter"),
            f"Claude: {nc.get('reason', '')}",
            " ".join(by_id[k]["text"] for k in range(s_id, e_id + 1) if k in by_id),
            float(nc.get("confidence", 0.7)), source="claude", seg_ids=list(range(s_id, e_id + 1)),
        ))
    return extra


def review_packet(segs: list[dict], cuts: list[Cut]) -> str:
    """Skill modunda Claude'un okuyacağı inceleme paketi (API çağrısıyla aynı içerik)."""
    for n, c in enumerate(cuts):
        c.id = c.id or f"r{n}"
    cand = "\n".join(f"{c.id}: {c.kind} segs {c.seg_ids[:1] + c.seg_ids[-1:] if c.seg_ids else '?'} — {c.reason}"
                     for c in cuts) or "(yok)"
    return "# Kurgu inceleme paketi\n\n## Talimat\n\n" + SYSTEM + "\n\n" + PROMPT.format(
        segments=_fmt_segments(segs), candidates=cand)
