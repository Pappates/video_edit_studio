"""YouTube açıklaması: bölüm zaman damgaları (chapters), açıklama metni, hashtag'ler ve etiketler.

YouTube bölüm kuralları (support.google.com/youtube/answer/9884579):
  - ilk zaman damgası 00:00, en az 3 bölüm, artan sıra, her bölüm en az 10 sn
Hashtag kuralları (support.google.com/youtube/answer/6390658):
  - boşluk içeremez; 60'tan fazlaysa YouTube hepsini yok sayar; ilk 3'ü başlığın üstünde görünür

Bölüm başlangıçları KAYNAK videonun zaman çizelgesinde saklanır. Açıklama oluşturulurken kalan
aralıklara (keeps) göre kurgulanmış videonun zamanına çevrilir; böylece Video Edit işinden üretilen
açıklama, kesimler sonradan değişse bile doğru kalır.
"""
from __future__ import annotations

import re
import unicodedata

from .claude_client import call_json
from .exporters import remap

MIN_CHAPTER_SEC = 10.0
MAX_HASHTAGS = 15          # YouTube sınırı 60; kalite için daha az önerilir
MAX_DESCRIPTION = 5000
MAX_TAGS_CHARS = 500


# ---------------------------------------------------------------------------
# Zaman yardımcıları
# ---------------------------------------------------------------------------
def fmt_ts(t: float, long: bool = False) -> str:
    t = int(max(0, t))
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if long or h else f"{m:02d}:{s:02d}"


def to_output(t_src: float, keeps: list[tuple[float, float]]) -> float | None:
    """Kaynak zamanı → kurgulanmış videodaki zaman. Kesilen bölgedeyse bir sonraki kalan aralığın başı."""
    r = remap(t_src, keeps)
    if r is not None:
        return r
    acc = 0.0
    for a, b in keeps:
        if t_src < a:
            return acc
        acc += b - a
    return None


def to_source(t_out: float, keeps: list[tuple[float, float]]) -> float:
    acc = 0.0
    for a, b in keeps:
        if t_out <= acc + (b - a):
            return a + (t_out - acc)
        acc += b - a
    return keeps[-1][1] if keeps else t_out


# ---------------------------------------------------------------------------
# Transkript blokları (Claude'a giden özet zaman çizelgesi)
# ---------------------------------------------------------------------------
def build_blocks(transcript: dict, keeps: list[tuple[float, float]], speakers: dict | None = None,
                 max_len: float = 12.0) -> list[dict]:
    """Kurgulanmış videoda kalan cümleleri ~12 sn.lik bloklara toplar (bölüm başı adayları; daha küçük blok = daha hassas bölüm zamanı)."""
    names = {k: v.get("name", k) for k, v in (speakers or {}).items()}
    blocks: list[dict] = []
    for s in transcript["segments"]:
        words = s.get("words") or [{"start": s["start"], "end": s["end"], "word": s["text"]}]
        kept = [w for w in words if remap((w["start"] + w["end"]) / 2, keeps) is not None]
        if not kept:
            continue
        out_start = to_output(kept[0]["start"], keeps)
        text = " ".join(w["word"] for w in kept).strip()
        spk = s.get("speaker")
        cur = blocks[-1] if blocks else None
        if cur and out_start - cur["out_start"] < max_len and spk == cur["speaker"]:
            cur["text"] += " " + text
        else:
            blocks.append({"id": len(blocks), "src_start": kept[0]["start"], "out_start": out_start,
                           "speaker": spk, "speaker_name": names.get(spk, spk), "text": text})
    return blocks


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------
SYSTEM = """Sen deneyimli bir YouTube içerik editörüsün. Bir videonun zaman damgalı transkriptinden
YouTube açıklaması hazırlıyorsun. Videonun dilinde yaz (transkript Türkçeyse Türkçe).

Kurallar:
- Bölümler (chapters): videonun gerçek konu değişimlerini takip et. İlk bölüm blok #0'dan başlar.
  Genelde her 1-4 dakikaya bir bölüm; çok kısa videoda 3-5 bölüm yeter. Bölüm başlığı 2-6 kelime,
  merak uyandıran ama dürüst; emoji yok, numara yok, "Bölüm 1" gibi boş başlık yok.
- Açıklama: ilk 1-2 cümle aramada görünür, videonun vaadini net söyler. Toplam 2-4 kısa paragraf.
  Transkriptte olmayan bilgi, link, indirim kodu, kişi ünvanı UYDURMA.
- Hashtag: 3-8 adet, konuya özgü; ilk 3'ü en önemlileri (başlığın üstünde görünürler). Boşluk yok.
  Alakasız veya yanıltıcı hashtag kullanma.
- Etiketler (tags): arama için 8-15 anahtar ifade, hashtag işareti olmadan.
- Başlık önerileri: 3 adet, en fazla 70 karakter, clickbait değil.

Yalnızca JSON döndür."""

PROMPT = """Video süresi (yayınlanacak hali): {duration}
{speakers}
Transkript blokları (#id [zaman] konuşmacı: metin):
{blocks}
{extra}{instruction}
İstenen JSON:
{{
  "titles": ["...", "...", "..."],
  "intro": "<açıklama paragrafları, \\n\\n ile ayrılmış>",
  "chapters": [{{"block_id": 0, "title": "..."}}, ...],
  "hashtags": ["#...", ...],
  "tags": ["...", ...]
}}"""


def generate(transcript: dict, keeps: list[tuple[float, float]], speakers: dict | None = None,
             extra: str = "", instruction: str = "", previous: dict | None = None) -> dict:
    blocks = build_blocks(transcript, keeps, speakers)
    if not blocks:
        raise ValueError("Transkriptte metin yok — açıklama oluşturulamaz.")
    duration = sum(b - a for a, b in keeps)
    spk_line = ""
    if speakers and len(speakers) > 1:
        spk_line = "Konuşmacılar: " + ", ".join(v.get("name", k) for k, v in speakers.items()) + \
                   " (isimler kullanıcı tarafından verilmediyse 'Konuşmacı N' olabilir; o durumda isim kullanma)\n"
    lines = "\n".join(
        f"#{b['id']} [{fmt_ts(b['out_start'], duration >= 3600)}] "
        f"{(b['speaker_name'] + ': ') if b['speaker'] and speakers and len(speakers) > 1 else ''}{b['text']}"
        for b in blocks)
    extra_txt = (f"\nŞu sabit bilgiler açıklamanın SONUNA otomatik eklenecek; bunları intro'ya YAZMA, "
                 f"yalnızca bağlam olarak kullan:\n{extra}\n") if extra.strip() else ""
    instr = f"\nKullanıcının ek isteği: {instruction}\n" if instruction.strip() else ""
    if previous and instruction.strip():
        instr += f"\nÖnceki taslak (bunu isteğe göre düzelt):\n{compose(previous, keeps)[0][:3000]}\n"

    data = call_json(PROMPT.format(duration=fmt_ts(duration, duration >= 3600), speakers=spk_line,
                                   blocks=lines, extra=extra_txt, instruction=instr), system=SYSTEM)

    by_id = {b["id"]: b for b in blocks}
    chapters = []
    for ch in data.get("chapters", []):
        b = by_id.get(ch.get("block_id"))
        title = str(ch.get("title", "")).strip()
        if b and title:
            chapters.append({"start": round(b["src_start"], 3), "title": title[:100]})
    return {
        "titles": [str(t).strip()[:100] for t in data.get("titles", []) if str(t).strip()][:5],
        "intro": strip_extra(str(data.get("intro", "")).strip(), extra),
        "chapters": chapters,
        "hashtags": clean_hashtags(data.get("hashtags", [])),
        "tags": clean_tags(data.get("tags", [])),
        "extra": extra,
    }


# ---------------------------------------------------------------------------
# Temizlik ve doğrulama
# ---------------------------------------------------------------------------
def strip_extra(intro: str, extra: str) -> str:
    """Model sabit bilgileri yine de intro'ya yazdıysa çıkar (sonda zaten ekleniyor, tekrar olmasın)."""
    lines = {ln.strip() for ln in extra.splitlines() if ln.strip()}
    if not lines:
        return intro
    kept = [ln for ln in intro.splitlines() if ln.strip() not in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def clean_hashtags(tags) -> list[str]:
    out, seen = [], set()
    for t in tags or []:
        t = unicodedata.normalize("NFC", str(t)).strip().lstrip("#")
        t = re.sub(r"[^\w]", "", t, flags=re.UNICODE)  # boşluk ve noktalama yok
        if not t or t.isdigit():
            continue
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            out.append("#" + t)
    return out[:MAX_HASHTAGS]


def clean_tags(tags) -> list[str]:
    out, total, seen = [], 0, set()
    for t in tags or []:
        t = re.sub(r"\s+", " ", str(t).replace("#", "").replace(",", " ")).strip()
        if not t or t.casefold() in seen:
            continue
        cost = len(t) + (2 if " " in t else 0) + (1 if out else 0)  # YouTube çok kelimeli etiketi tırnakla sayar
        if total + cost > MAX_TAGS_CHARS:
            break
        seen.add(t.casefold())
        out.append(t)
        total += cost
    return out


def output_chapters(yt: dict, keeps: list[tuple[float, float]]) -> tuple[list[dict], list[str]]:
    """Bölümleri yayınlanacak videonun zamanına çevirir ve YouTube kurallarına uydurur."""
    duration = sum(b - a for a, b in keeps)
    warnings: list[str] = []
    items = []
    for ch in yt.get("chapters", []):
        t = to_output(float(ch["start"]), keeps)
        if t is None:
            warnings.append(f"'{ch['title']}' bölümü kesilen kısımda kaldı, çıkarıldı.")
            continue
        items.append({"t": t, "title": ch["title"], "start": ch["start"]})
    items.sort(key=lambda x: x["t"])
    if items:
        items[0]["t"] = 0.0  # YouTube: ilk bölüm 00:00
    fixed: list[dict] = []
    for it in items:
        if fixed and it["t"] - fixed[-1]["t"] < MIN_CHAPTER_SEC:
            warnings.append(f"'{it['title']}' önceki bölüme 10 sn'den yakın olduğu için açıklamaya eklenmedi "
                            f"(YouTube her bölümün en az 10 sn olmasını ister).")
            continue
        fixed.append(it)
    if fixed and duration - fixed[-1]["t"] < MIN_CHAPTER_SEC and len(fixed) > 1:
        warnings.append(f"Son bölüm '{fixed[-1]['title']}' 10 sn'den kısa, çıkarıldı.")
        fixed.pop()
    if 0 < len(fixed) < 3:
        warnings.append("YouTube bölüm göstermek için en az 3 bölüm ister; zaman damgaları yine de listelendi.")
    return fixed, warnings


def compose(yt: dict, keeps: list[tuple[float, float]]) -> tuple[str, list[str]]:
    """Kopyalanmaya hazır açıklama metni + uyarılar."""
    duration = sum(b - a for a, b in keeps)
    long = duration >= 3600
    chapters, warnings = output_chapters(yt, keeps)
    parts = []
    if yt.get("intro"):
        parts.append(yt["intro"].strip())
    if chapters:
        parts.append("⏱️ Bölümler\n" + "\n".join(f"{fmt_ts(c['t'], long)} {c['title']}" for c in chapters))
    if yt.get("extra", "").strip():
        parts.append(yt["extra"].strip())
    if yt.get("hashtags"):
        parts.append(" ".join(yt["hashtags"]))
    text = "\n\n".join(parts)
    if len(text) > MAX_DESCRIPTION:
        warnings.append(f"Açıklama {len(text)} karakter; YouTube sınırı {MAX_DESCRIPTION}. Kısaltın.")
    if len(yt.get("hashtags", [])) > 60:
        warnings.append("60'tan fazla hashtag varsa YouTube hepsini yok sayar.")
    return text, warnings
