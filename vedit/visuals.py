"""Konuşulan konuya göre reel'e görsel ekleme ("ilk 10 oyuncu" sayılırken her oyuncunun resmi gibi).

Akış:
  1. Claude reel'in numaralı transkriptini okur; görsel gösterilecek anları döner: hangi cümlede, hangi
     kelimede, ne gösterilecek (kişi/takım/yer/marka/nesne/kavram), liste sırası ve ne kadar kalacağı.
     Zaman uydurmasın diye segment numarası + o segmentteki kelimeyle cevap verir; zamanı kod bulur.
  2. Görsel kaynağı:
       - Wikipedia/Wikimedia Commons: belirli kişi, takım, yer, marka için. Ücretsiz, anahtar gerekmez;
         yalnızca serbest lisanslı sayfa görselleri alınır, yazar/lisans bilgisi saklanır.
       - Pexels: genel kavramlar ("stadyum", "para") için. Ücretsiz PEXELS_API_KEY gerekir.
     Arayüzde başka aday seçilebilir, arama değiştirilebilir ya da kendi dosyanız yüklenebilir.
  3. Render: ffmpeg overlay ile zamanında, yumuşak geçişle; isim etiketi ve sıra numarası ASS ile yazılır.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from difflib import SequenceMatcher
from pathlib import Path

from .claude_client import ClaudeError, call_json
from .claude_review import fmt_segment
from .config import settings
from .exporters import remap_forward
from .textnorm import fold

USER_AGENT = "VideoEditStudio/1.0 (local desktop app; python-urllib)"
KINDS = ("person", "team", "place", "brand", "work", "object", "concept")
ENTITY_KINDS = {"person", "team", "place", "brand", "work"}   # önce Wikipedia
POSITIONS = ("top", "center", "full")
MIN_HOLD, MAX_HOLD = 1.2, 7.0

SYSTEM = """Sen kısa video (reels) editörüsün. Konuşmacının anlattığı şeyi izleyicinin GÖRMESİ için hangi anda
hangi görselin ekrana geleceğine karar veriyorsun.

Kurallar:
- Belirli bir isim geçtiğinde (kişi, futbolcu, takım, şehir, ülke, marka, film, oyun, ürün) o anda onun görseli gelsin.
- Liste/sıralama anlatılıyorsa ("ilk 10", "3 numara", "birinci sırada") listedeki HER öğeye görsel ver ve "rank" yaz.
- Somut bir nesne ya da sahne anlatılıyorsa (stadyum, kupa, kahve) stok fotoğraf için "stock" sorgusu ver.
- Soyut, genel konuşma, selamlaşma, "takip edin" gibi yerlere görsel koyma.
- Görseller üst üste binmesin; aynı şeyi tekrar tekrar gösterme (ilk geçtiği yer yeterli).
- "anchor": isim o segmentte HANGİ kelimelerle geçiyorsa aynen o kelimeler (görsel o anda belirir).
- "wiki": varlığın Wikipedia makale başlığı (doğru yazımla; emin değilsen en olası başlık). Kavramlarda boş bırak.
- "stock": İngilizce, 1-3 kelimelik stok fotoğraf sorgusu (kişiler için de genel bir yedek yaz, ör. "football player").
- "hold": görselin ekranda kalacağı süre (sn), genelde 2-5.

Yalnızca JSON döndür."""

PROMPT = """Reel: {title}

Transkript (#id [başlangıç-bitiş sn] metin):
{segments}

İstenen JSON:
{{"visuals": [{{"seg_id": <int>, "anchor": "<segmentteki kelimeler>", "label": "<ekranda yazacak isim>",
  "kind": "person|team|place|brand|work|object|concept", "wiki": "<Wikipedia başlığı veya boş>",
  "wiki_lang": "tr|en", "stock": "<english query>", "rank": <int veya null>, "hold": <sn>}}]}}"""


# ---------------------------------------------------------------------------
# 1) Anları bulma
# ---------------------------------------------------------------------------
def reel_segments(transcript: dict, start: float, end: float) -> list[dict]:
    return [s for s in transcript["segments"] if s["end"] > start and s["start"] < end]


def ask_claude(title: str, segs: list[dict]) -> list[dict]:
    if not segs:
        return []
    data = call_json(PROMPT.format(title=title or "(başlıksız)", segments="\n".join(fmt_segment(s) for s in segs)),
                     system=SYSTEM)
    return list(data.get("visuals") or []) if isinstance(data, dict) else []


def _anchor_time(seg: dict, anchor: str) -> float:
    """Segmentte anchor kelimelerinin başladığı an; bulunamazsa segmentin başı."""
    words = seg.get("words") or []
    target = fold(anchor or "")
    if not words or not target:
        return seg["start"]
    n = max(1, len(target.split()))
    best, best_t = 0.0, seg["start"]
    for i in range(len(words)):
        chunk = fold(" ".join(w["word"] for w in words[i:i + n]))
        if not chunk:
            continue
        r = SequenceMatcher(None, chunk, target).ratio()
        if r > best:
            best, best_t = r, words[i]["start"]
    return best_t if best >= 0.6 else seg["start"]


def place(moments: list[dict], segs: list[dict], reel_start: float, reel_end: float) -> list[dict]:
    """Claude'un önerilerini kaynak videonun zamanına oturtur, çakışmaları kırpar."""
    by_id = {s["id"]: s for s in segs}
    out = []
    for m in moments:
        seg = by_id.get(m.get("seg_id"))
        label = str(m.get("label") or m.get("wiki") or m.get("stock") or "").strip()
        if not seg or not label:
            continue
        kind = m.get("kind") if m.get("kind") in KINDS else "concept"
        t = max(reel_start, _anchor_time(seg, str(m.get("anchor", ""))) - 0.15)
        try:
            hold = float(m.get("hold") or 3.0)
        except (TypeError, ValueError):
            hold = 3.0
        rank = m.get("rank")
        out.append(new_visual(
            start=t, end=t + min(MAX_HOLD, max(MIN_HOLD, hold)), label=label[:60], kind=kind,
            wiki=str(m.get("wiki") or "").strip(), wiki_lang=str(m.get("wiki_lang") or settings.wiki_lang),
            stock=str(m.get("stock") or "").strip(), rank=int(rank) if isinstance(rank, (int, float)) and rank else None,
        ))
    out.sort(key=lambda v: v["start"])
    kept = []
    for i, v in enumerate(out):
        nxt = out[i + 1]["start"] if i + 1 < len(out) else reel_end
        v["end"] = round(min(v["end"], nxt - 0.05, reel_end), 3)
        if kept and v["start"] < kept[-1]["end"]:
            v["start"] = kept[-1]["end"]
        if v["end"] - v["start"] >= 0.8:
            v["start"] = round(v["start"], 3)
            kept.append(v)
    return kept


def new_visual(start: float, end: float, label: str = "", kind: str = "concept", wiki: str = "",
               wiki_lang: str = "", stock: str = "", rank: int | None = None) -> dict:
    return {"id": "v" + uuid.uuid4().hex[:6], "start": start, "end": end, "label": label, "kind": kind,
            "wiki": wiki, "wiki_lang": wiki_lang or settings.wiki_lang, "stock": stock, "rank": rank,
            "enabled": True, "candidates": [], "choice": -1, "file": "", "credit": "", "page": "", "source": ""}


# ---------------------------------------------------------------------------
# 2) Görsel kaynakları
# ---------------------------------------------------------------------------
def _get_json(url: str, headers: dict | None = None, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def search_wikipedia(query: str, lang: str = "tr", limit: int = 3, width: int = 1080) -> list[dict]:
    """Arama sonucu sayfaların serbest lisanslı ana görselleri (en alakalı önce)."""
    if not query.strip():
        return []
    params = {"action": "query", "format": "json", "formatversion": "2", "generator": "search",
              "gsrsearch": query, "gsrlimit": str(limit), "gsrnamespace": "0",
              "prop": "pageimages|info|pageprops", "piprop": "thumbnail|name", "pithumbsize": str(width),
              "pilicense": "free", "inprop": "url", "ppprop": "disambiguation"}
    data = _get_json(f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params))
    pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
    cands = []
    for p in pages:
        if "disambiguation" in (p.get("pageprops") or {}) or not p.get("thumbnail"):
            continue
        cands.append({"source": "wikipedia", "url": p["thumbnail"]["source"], "title": p.get("title", ""),
                      "page": p.get("fullurl", ""), "file": p.get("pageimage", ""),
                      "credit": f"Wikipedia · {p.get('title', '')}"})
    _add_commons_credits(cands)
    return cands


def _add_commons_credits(cands: list[dict]) -> None:
    """Commons'tan yazar ve lisans bilgisi (kaynak gösterimi için)."""
    files = [c["file"] for c in cands if c.get("file")]
    if not files:
        return
    params = {"action": "query", "format": "json", "formatversion": "2", "prop": "imageinfo",
              "iiprop": "extmetadata", "iiextmetadatafilter": "Artist|LicenseShortName",
              "titles": "|".join("File:" + f for f in files)}
    try:
        data = _get_json("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params))
    except (urllib.error.URLError, OSError, ValueError):
        return
    meta = {}
    for p in data.get("query", {}).get("pages", []):
        info = (p.get("imageinfo") or [{}])[0].get("extmetadata", {})
        name = p.get("title", "").split(":", 1)[-1]
        meta[name.replace(" ", "_")] = (_strip_html(info.get("Artist", {}).get("value", "")),
                                        info.get("LicenseShortName", {}).get("value", ""))
    for c in cands:
        artist, lic = meta.get(c.get("file", "").replace(" ", "_"), ("", ""))
        if artist or lic:
            c["credit"] = " · ".join(x for x in (artist[:80], lic, "Wikimedia Commons") if x)


def search_pexels(query: str, limit: int = 6, orientation: str = "") -> list[dict]:
    if not settings.pexels_api_key or not query.strip():
        return []
    params = {"query": query, "per_page": str(limit)}
    if orientation:
        params["orientation"] = orientation
    data = _get_json("https://api.pexels.com/v1/search?" + urllib.parse.urlencode(params),
                     headers={"Authorization": settings.pexels_api_key})
    return [{"source": "pexels", "url": ph["src"].get("large2x") or ph["src"].get("large"),
             "title": ph.get("alt", "") or query, "page": ph.get("url", ""),
             "credit": f"{ph.get('photographer', '')} · Pexels"} for ph in data.get("photos", [])]


def find_candidates(v: dict, source: str = "auto", prefer_wiki: bool = False) -> tuple[list[dict], list[str]]:
    """Görsel adayları ve karşılaşılan hatalar. source: auto | wikipedia | pexels.

    prefer_wiki: kullanıcının yazdığı aramada önce Wikipedia (çoğunlukla bir isim yazılır; Pexels'te
    belirli kişiler yoktur ve rastgele stok fotoğraflar gelirdi)."""
    errors: list[str] = []

    def wiki() -> list[dict]:
        q = v.get("wiki") or v.get("label", "")
        out: list[dict] = []
        langs = [v.get("wiki_lang") or settings.wiki_lang]
        langs += [lang for lang in ("en", settings.wiki_lang) if lang not in langs]
        for lang in langs:
            try:
                out += search_wikipedia(q, lang)
            except (urllib.error.URLError, OSError, ValueError) as e:
                errors.append(f"Wikipedia ({lang}): {e}")
            if out:
                break
        return out

    def pexels() -> list[dict]:
        if not settings.pexels_api_key:
            errors.append("Pexels anahtarı yok (.env: PEXELS_API_KEY)")
            return []
        try:
            return search_pexels(v.get("stock") or v.get("label", ""))
        except (urllib.error.URLError, OSError, ValueError) as e:
            errors.append(f"Pexels: {e}")
            return []

    if source == "wikipedia":
        return wiki(), errors
    if source == "pexels":
        return pexels(), errors
    if prefer_wiki or v.get("kind") in ENTITY_KINDS or not settings.pexels_api_key:
        cands = wiki()
        cands += pexels() if settings.pexels_api_key else []
    else:
        cands = pexels() + wiki()
    seen, uniq = set(), []
    for c in cands:
        if c["url"] not in seen:
            seen.add(c["url"])
            uniq.append(c)
    return uniq, errors


_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}


def download(url: str, dst_stem: Path, timeout: int = 30, max_bytes: int = 25 * 1024 * 1024) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype not in _EXT:
            raise ValueError(f"Görsel değil: {ctype or 'bilinmeyen tür'}")
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Görsel çok büyük")
    dst = dst_stem.with_suffix(_EXT[ctype])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return dst


def choose(v: dict, index: int, image_dir: Path) -> dict:
    """v['candidates'][index]'i indirip görsele atar; indirilemeyenleri atlayarak sıradakini dener."""
    cands = v.get("candidates") or []
    last_err = None
    for k in range(len(cands)):
        i = (index + k) % len(cands)
        c = cands[i]
        try:
            path = download(c["url"], image_dir / f"{v['id']}_{i}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            last_err = e
            continue
        old = image_dir / Path(v["file"]).name if v.get("file") else None
        if old and old.exists() and old.name != path.name:
            old.unlink(missing_ok=True)
        v.update(choice=i, file=path.name, credit=c.get("credit", ""), page=c.get("page", ""), source=c["source"])
        return v
    if last_err:
        raise ValueError(f"Görsel indirilemedi: {last_err}")
    raise ValueError("Bu görsel için aday bulunamadı — aramayı değiştirin ya da dosya yükleyin")


def fill(v: dict, image_dir: Path, source: str = "auto", prefer_wiki: bool = False) -> list[str]:
    """Adayları arar ve ilkini indirir. Hata mesajlarını döner (görsel bulunamazsa da hata sayılmaz)."""
    v["candidates"], errors = find_candidates(v, source, prefer_wiki)
    v["candidates"] = v["candidates"][:8]
    v.update(choice=-1, file="", credit="", page="", source="")
    if v["candidates"]:
        try:
            choose(v, 0, image_dir)
        except ValueError as e:
            errors.append(str(e))
    return errors


# ---------------------------------------------------------------------------
# 3) Render planı
# ---------------------------------------------------------------------------
def overlay_box(position: str, width: int, height: int,
                area: tuple[int, int, int, int] | None = None) -> tuple[int, int, int, int]:
    """Görselin sığdırılacağı kutu (x, y, w, h).
    area: dikey yerleşimde oyun alanı — görsel yüzlerin üstüne binmesin diye bu alanın içine konur."""
    if area is not None:
        ax, ay, aw, ah = area
        if position == "full":
            return ax, ay, aw, ah
        w, h = int(aw * (0.84 if position == "center" else 0.86)), int(min(ah * 0.6, height * 0.38))
        y = ay + (ah - h) // 2 if position == "center" else ay + int(ah * 0.06)
        return ax + (aw - w) // 2, y, w, h
    if position == "full":
        return 0, 0, width, height
    if width >= height:  # yatay video: sağda kart
        w, h = int(width * 0.42), int(height * 0.62)
        return width - w - int(width * 0.04), int(height * 0.1), w, h
    if position == "center":
        w, h = int(width * 0.84), int(height * 0.44)
        return (width - w) // 2, (height - h) // 2 - int(height * 0.04), w, h
    w, h = int(width * 0.86), int(height * 0.38)
    return (width - w) // 2, int(height * 0.08), w, h


def _even(x: float) -> int:
    return max(2, int(x) // 2 * 2)


def fit(img_w: int, img_h: int, box: tuple[int, int, int, int], position: str, border: int) -> dict:
    """Görselin ölçeklenmiş boyutu ve konumu (çerçeve dahil)."""
    bx, by, bw, bh = box
    if position == "full":
        return {"scale": (bw, bh), "cover": True, "x": bx, "y": by, "w": bw, "h": bh, "border": 0}
    inner_w, inner_h = bw - 2 * border, bh - 2 * border
    k = min(inner_w / max(1, img_w), inner_h / max(1, img_h))
    sw, sh = _even(img_w * k), _even(img_h * k)
    w, h = sw + 2 * border, sh + 2 * border
    return {"scale": (sw, sh), "cover": False, "x": bx + (bw - w) // 2, "y": by + (bh - h) // 2,
            "w": w, "h": h, "border": border}


def overlay_plan(visuals: list[dict], keeps: list[tuple[float, float]], width: int, height: int,
                 position: str, image_dir: Path, probe_fn, skipped: list[str] | None = None,
                 area: tuple[int, int, int, int] | None = None) -> list[dict]:
    """Kaynak zamanlı görselleri çıktı videosunun zamanına ve piksel konumuna çevirir.

    skipped: verilirse, seçili olduğu halde eklenemeyen görsellerin (dosyası kayıp ya da okunamıyor)
    açıklaması buraya eklenir; render bunları kullanıcıya bildirir. Kesilen yere düşen görseller
    bilinçli olarak atlanır ve buraya yazılmaz."""
    position = position if position in POSITIONS else "top"
    total = sum(b - a for a, b in keeps)
    border = 0 if position == "full" else max(4, width // 120)
    items = []
    for v in visuals:
        if not v.get("enabled", True) or not v.get("file"):
            continue
        name = v.get("label") or v.get("id", "görsel")
        path = image_dir / Path(v["file"]).name
        if not path.exists():
            if skipped is not None:
                skipped.append(f"{name} (dosya yok)")
            continue
        a, b = remap_forward(v["start"], keeps), remap_forward(v["end"], keeps)
        b = total if b is None else b
        if a is None or b - a < 0.5:
            continue
        try:
            info = probe_fn(path)
            if not info.get("width") or not info.get("height"):
                raise ValueError("boyut okunamadı")
        except Exception as e:  # noqa: BLE001 — bozuk görsel render'ı durdurmasın, ama bildirilsin
            if skipped is not None:
                skipped.append(f"{name} (okunamadı: {str(e)[:60]})")
            continue
        geo = fit(info["width"], info["height"], overlay_box(position, width, height, area), position, border)
        items.append({"path": str(path), "start": round(a, 3), "end": round(min(b, total), 3), "position": position,
                      "label": v.get("label", "") if v.get("show_label", True) else "", "rank": v.get("rank"),
                      "box": (geo["x"], geo["y"], geo["w"], geo["h"]), "geo": geo})
    return items


def find_for_reel(transcript: dict, reel: dict, image_dir: Path, progress=None) -> tuple[list[dict], list[str]]:
    """Bir reel için anları Claude'a sorar ve görselleri indirir. Elle yüklenmiş görseller korunur."""
    segs = reel_segments(transcript, reel["start"], reel["end"])
    moments = ask_claude(reel.get("title", ""), segs)
    visuals = place(moments, segs, reel["start"], reel["end"])
    errors: list[str] = []
    for n, v in enumerate(visuals):
        if progress:
            progress(n / max(1, len(visuals)), f"{reel.get('id', '')}: görsel aranıyor — {v['label']}")
        errors += fill(v, image_dir)
    manual = [v for v in reel.get("visuals", []) if v.get("source") == "upload"]
    merged = sorted(manual + [v for v in visuals
                              if not any(m["start"] < v["end"] and v["start"] < m["end"] for m in manual)],
                    key=lambda v: v["start"])
    return merged, errors


__all__ = ["ClaudeError", "find_for_reel", "fill", "choose", "new_visual", "overlay_plan", "overlay_box", "place",
           "search_wikipedia", "search_pexels", "POSITIONS"]
