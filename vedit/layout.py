"""Dikey (9:16) yerleşim: ekran paylaşımı/oyun + kamera(lar) aynı anda görünsün.

Yatay bir kayıtta (ör. Riverside: solda ekran paylaşımı, sağda kameralar alt alta) ortadan 9:16 kırpmak
oyunun yarısını ve yüzleri keser. Burada kayıt bölgelere ayrılır:
  content — izleyicinin görmesi gereken ekran/oyun kısmı (tarayıcı çubuğu, görev çubuğu hariç)
  cams    — kişilerin kamera kutuları (her kişi ayrı)
ve 1080x1920'ye yeniden dizilir:

  faces_top  Yüzler üstte yan yana, oyun altta (önerilen: Instagram'ın alt yazı/buton alanı oyunun boş
             alt kısmına denk gelir)
  game_top   Oyun üstte, yüzler altta yan yana
  game_full  Oyun tam boy, yüzler altta küçük çerçeveli kutular
  talk       Sohbet: ekranda oyun yok, yalnızca kameralar — yüzler alt alta, ekranı ikiye bölerek;
             altyazı iki yüzün birleştiği yerde. Reel'in kendi karesine bakılarak otomatik seçilir.

Bölgeler Claude'a bir kare gösterilerek bulunur (yoksa basit görüntü analizi); arayüzde sürüklenerek
düzeltilebilir. Tek kamera varsa yüz şeridi tek geniş kutu olur; kamera yoksa oyun tam boy gösterilir.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

W, H = 1080, 1920
BG = "0x0e0b1a"
MODES = {
    "faces_top": "Yüzler üstte, oyun altta",
    "game_top": "Oyun üstte, yüzler altta",
    "game_full": "Oyun tam boy, yüzler küçük",
    "talk": "Sohbet: yüzler alt alta",
}
GAME_MODES = ("faces_top", "game_top", "game_full")
Rect = tuple[int, int, int, int]


def _even(v: float) -> int:
    """Boyut için: çift ve en az 2 (yuv420 kırpma/ölçekleme çift ister)."""
    return max(2, int(round(v)) // 2 * 2)


def _even_pos(v: float) -> int:
    """Konum için: çift, 0 olabilir."""
    return max(0, int(round(v)) // 2 * 2)


def clamp_rect(r, fw: int, fh: int, min_size: int = 16) -> Rect:
    x, y, w, h = (float(v) for v in r)
    w, h = max(min_size, min(w, fw)), max(min_size, min(h, fh))
    x, y = max(0.0, min(x, fw - w)), max(0.0, min(y, fh - h))
    w, h = _even(w), _even(h)
    return min(_even_pos(x), fw - w), min(_even_pos(y), fh - h), w, h


def fit_aspect(box, aspect: float, fw: int, fh: int) -> Rect:
    """box'ı içeren, en/boy oranı `aspect` olan en küçük dikdörtgen (box merkezli, kare içinde kalır).
    Kareye sığmazsa kare sınırında kalır (oran o zaman tutmaz; render'da kenar boşluğuyla sığdırılır)."""
    x, y, w, h = (float(v) for v in box)
    cx, cy = x + w / 2, y + h / 2
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(w, fw), min(h, fh)
    return clamp_rect((cx - w / 2, cy - h / 2, w, h), fw, fh)


def _cam_aspect(cams: list[Rect]) -> float:
    return sum(c[3] / c[2] for c in cams) / len(cams)   # boy / en


def inner_aspect(box, aspect: float, fw: int, fh: int, bias_y: float = 0.45) -> Rect:
    """box'ın İÇİNDE kalan, en/boy oranı `aspect` olan en büyük dikdörtgen (yan kameraya taşmaz).
    Dikeyde biraz yukarı kayık (bias_y): yüzler kamera karesinin üst-ortasındadır."""
    x, y, w, h = (float(v) for v in box)
    if w / h > aspect:
        nw, nh = h * aspect, h
    else:
        nw, nh = w, w / aspect
    nx = x + (w - nw) / 2
    ny = y + (h - nh) * bias_y
    return clamp_rect((nx, ny, nw, nh), fw, fh)


def _talk_plan(fw: int, fh: int, cams: list[Rect]) -> dict:
    """Sohbet: kameralar alt alta, ekranı eşit bölerek. Tek kamera → tam ekran."""
    cams = cams or [(0, 0, fw, fh)]
    n = len(cams)
    th = _even(H / n)
    tiles = []
    for i, c in enumerate(cams):
        h_i = H - th * (n - 1) if i == n - 1 else th
        tiles.append({"src": inner_aspect(c, W / h_i, fw, fh), "dst": (0, i * th, W, h_i), "border": 0})
    if n == 2:
        # Altyazı iki yüzün birleştiği çizgide (yüzleri kapatmaz); görsel kartı üstteki yüzün alanında
        game_area, caption_margin = (0, 0, W, th), H - th - int(H * 0.035)
    else:
        game_area, caption_margin = (0, 0, W, min(H, th)), int(H * 0.2)
    return {"mode": "talk", "tiles": tiles, "game_area": game_area, "caption_margin": caption_margin}


def plan(mode: str, fw: int, fh: int, regions: dict) -> dict:
    """Kaynak bölgelerinden 9:16 yerleşim planı.

    Döner: {"tiles": [{"src": Rect, "dst": Rect, "border": int}, ...]  (ilki oyun — taban katman),
            "game_area": Rect (görsellerin konacağı alan), "caption_margin": int (alttan, ASS MarginV)}
    """
    cams = [clamp_rect(c, fw, fh) for c in (regions.get("cams") or [])][:4]
    if mode == "talk":
        return _talk_plan(fw, fh, cams[:3])
    content = clamp_rect(regions.get("content") or (0, 0, fw, fh), fw, fh)
    mode = mode if mode in MODES else "faces_top"
    if not cams:
        mode = "game_full"

    if mode in ("faces_top", "game_top"):
        n = len(cams)
        tw = W // n
        th = _even(min(H * 0.42, max(H * (0.22 if n > 1 else 0.3), tw * _cam_aspect(cams))))
        gh = H - th
        game_y, face_y = (th, 0) if mode == "faces_top" else (0, gh)
        tiles = [{"src": fit_aspect(content, W / gh, fw, fh), "dst": (0, game_y, W, gh), "border": 0}]
        for i, c in enumerate(cams):
            w_i = W - tw * (n - 1) if i == n - 1 else tw     # son kutu yuvarlama artığını alır
            tiles.append({"src": fit_aspect(c, w_i / th, fw, fh), "dst": (i * tw, face_y, w_i, th), "border": 0})
        game_area = (0, game_y, W, gh)
        # faces_top: altyazı oyunun alt kısmında (Instagram'ın en alt şeridinin üstünde);
        # game_top: oyunla yüzlerin birleştiği yerin hemen üstünde
        caption_margin = int(H * 0.2) if mode == "faces_top" else th + 30
    else:
        n = len(cams)
        if n:
            # Oyun büyük (üst ~%68), yüzler altta küçük çerçeveli kutular — oyunun üstüne binmez
            gap = 40
            s = _even(min(400, (W - gap * (n + 1)) / n))
            y = _even(H * 0.68)
            gh = y - 10
            tiles = [{"src": fit_aspect(content, W / gh, fw, fh), "dst": (0, 0, W, gh), "border": 0}]
            total = n * s + (n - 1) * gap
            x0 = (W - total) // 2
            for i, c in enumerate(cams):
                tiles.append({"src": fit_aspect(c, 1.0, fw, fh), "dst": (x0 + i * (s + gap), y + 20, s, s), "border": 6})
            game_area, caption_margin = (0, 0, W, gh), H - y + 10
        else:
            tiles = [{"src": fit_aspect(content, W / H, fw, fh), "dst": (0, 0, W, H), "border": 0}]
            game_area, caption_margin = (0, 0, W, H), int(H * 0.24)
    return {"mode": mode, "tiles": tiles, "game_area": game_area, "caption_margin": caption_margin}


def graph(in_label: str, out_label: str, p: dict) -> str:
    """Plan → ffmpeg filtre zinciri. İlk karo (oyun) tuvale yerleşip taban olur, diğerleri üstüne biner."""
    tiles = p["tiles"]
    n = len(tiles)
    parts = [f"[{in_label}]split={n}" + "".join(f"[L{k}]" for k in range(n)) + ";"] if n > 1 else []
    src_lbl = [f"L{k}" for k in range(n)] if n > 1 else [in_label]
    for k, t in enumerate(tiles):
        sx, sy, sw, sh = t["src"]
        dx, dy, dw, dh = t["dst"]
        b = t.get("border", 0)
        iw, ih = dw - 2 * b, dh - 2 * b
        chain = (f"[{src_lbl[k]}]crop={sw}:{sh}:{sx}:{sy},scale={iw}:{ih}:force_original_aspect_ratio=decrease,"
                 f"pad={iw}:{ih}:(ow-iw)/2:(oh-ih)/2:color={BG},setsar=1")
        if b:
            chain += f",pad={dw}:{dh}:{b}:{b}:color=white"
        if k == 0:
            chain += f",pad={W}:{H}:{dx}:{dy}:color={BG}"
        parts.append(chain + f"[T{k}];")
    cur = "T0"
    for k in range(1, n):
        dx, dy, _, _ = tiles[k]["dst"]
        nxt = f"O{k}"
        parts.append(f"[{cur}][T{k}]overlay={dx}:{dy}[{nxt}];")
        cur = nxt
    parts.append(f"[{cur}]null[{out_label}]")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Bölge tespiti
# ---------------------------------------------------------------------------
DETECT_SYSTEM = """Bir video kaydının tek karesine bakıyorsun. Kayıt dikey (9:16) kısa videoya dönüştürülecek;
izleyicinin hem EKRANDAKİ İÇERİĞİ (oyun, uygulama, sunum) hem de KİŞİLERİN YÜZLERİNİ görmesi gerekiyor.
Bölgeleri piksel koordinatıyla ver. Yalnızca JSON döndür."""

DETECT_PROMPT = """Karenin boyutu: {w}x{h} piksel.

1. "content": izleyicinin görmesi gereken asıl ekran içeriği. Oyunun/uygulamanın TÜM önemli parçalarını
   (başlık, skor, ibre, butonlar, yazılar) içersin; tarayıcı adres/sekme çubuğu, Windows görev çubuğu ve
   içeriğin çevresindeki boş arka plan dışarıda kalsın (biraz pay bırak).
   Ekranda paylaşılan bir oyun/uygulama YOKSA (yalnızca kişilerin kamera görüntüleri varsa) content null olsun.
2. "cameras": her kişinin kamera görüntüsü ayrı bir kutu (kamera penceresinin tamamı). Kamera yoksa boş liste.

İstenen JSON: {{"content": {{"x": <int>, "y": <int>, "w": <int>, "h": <int>}} veya null,
  "cameras": [{{"x": <int>, "y": <int>, "w": <int>, "h": <int>}}], "notes": "<kısa>"}}"""


def _frame(video: str | Path, t: float, dst: Path, width: int | None = None) -> Path:
    from . import media
    vf = ["-vf", f"scale={width}:-2"] if width else []
    media._run_ffmpeg(["-ss", f"{max(0.0, t):.3f}", "-i", str(video), "-frames:v", "1", *vf, "-q:v", "3", str(dst)])
    return dst


def _busy_block(img, box, band_top: float = 0.09, band_bottom: float = 0.06, gap: int = 3):
    """box (küçük görüntü koordinatı) içinde ayrıntının (kenarların) toplandığı en büyük bloğun sınırları.
    Ekran paylaşımında oyun/uygulama kartını bulur; üst/alt şeritler (tarayıcı çubuğu, görev çubuğu) atlanır."""
    import numpy as np
    x, y, w, h = box
    sub = img[y:y + h, x:x + w]
    edges = np.zeros_like(sub, dtype=bool)
    edges[:, 1:] |= np.abs(np.diff(sub, axis=1)) > 14
    edges[1:, :] |= np.abs(np.diff(sub, axis=0)) > 14
    b = 6
    hb, wb = h // b, w // b
    if hb < 4 or wb < 4:
        return box
    dens = edges[: hb * b, : wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3)) > 0.06
    # Üst/alt şeritler kareye göre: tarayıcı sekme+yer imi çubuğu ~%8, görev çubuğu ~%5
    top = max(0, int((sh_full := img.shape[0]) * band_top) - y) // b
    bot = hb - max(0, (y + h) - int(sh_full * (1 - band_bottom))) // b
    dens[:top], dens[bot:] = False, False

    def run(profile, min_count):
        """En çok ayrıntı içeren kesintisiz bölge; `gap` bloğa kadar boşluklar (kartın içindeki aralar) köprülenir."""
        best, cur, start, last, holes = (0, 0, 0), 0, 0, -1, 0
        for i, v in enumerate(list(profile) + [0] * (gap + 1)):
            if v >= min_count:
                if cur == 0:
                    start = i
                cur, last, holes = cur + v, i, 0
            elif cur:
                holes += 1
                if holes > gap:
                    if cur > best[0]:
                        best = (cur, start, last + 1)
                    cur, holes = 0, 0
        return best[1], best[2]

    r0, r1 = run(dens.sum(axis=1), 2)
    if r1 - r0 < 3:
        return box
    c0, c1 = run(dens[r0:r1].sum(axis=0), 2)
    if c1 - c0 < 3:
        return box
    pad = 2   # pay, ama hariç tutulan üst/alt şeritlere taşmadan
    r0, r1, c0, c1 = max(top, r0 - pad), min(bot, r1 + pad), max(0, c0 - pad), min(wb, c1 + pad)
    return (x + c0 * b, y + r0 * b, (c1 - c0) * b, (r1 - r0) * b)


def detect_heuristic(video: str | Path, t: float, fw: int, fh: int) -> dict:
    """Claude yokken: karedeki en belirgin dikey ayrım çizgisini (kamera sütunu) ve sütun içindeki yatay
    ayrımı (alt alta iki kamera) bulur. Ekran paylaşımı + yan sütunda kameralar düzeni için."""
    import numpy as np
    from . import media
    sw, sh = 384, 216
    raw = subprocess.run([media._bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}",
                          "-i", str(video), "-frames:v", "1", "-vf", f"scale={sw}:{sh}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    if len(raw) < sw * sh:
        return {"content": [0, 0, fw, fh], "cams": [], "source": "default"}
    img = np.frombuffer(raw[: sw * sh], dtype=np.uint8).reshape(sh, sw).astype(np.int16)
    col = (np.abs(np.diff(img, axis=1)) > 18).mean(axis=0)          # her sütunda dikey kenar oranı
    kx, ky = fw / sw, fh / sh
    sc = lambda r: list(clamp_rect((r[0] * kx, r[1] * ky, r[2] * kx, r[3] * ky), fw, fh))  # noqa: E731
    # Sohbet düzeni: ortada keskin bir dikiş → iki kamera yan yana, paylaşılan ekran yok. Kenar sayısı
    # yerine ortalama parlaklık sıçraması ölçülür: iki odanın duvar rengi benzer olsa da dikiş çizgisi
    # genelin birkaç katıdır (gerçek kayıtta ~20-57 / ~2.5; oyun ekranında ~1).
    # Ekran paylaşımında kamera sütununun boydan boya kenarı vardır (oyun kartının kenarı ortaya denk
    # gelebilir); o kenar varsa sohbet sayılmaz.
    md = np.abs(np.diff(img, axis=1)).mean(axis=0)
    lo_c, hi_c = int(sw * .42), int(sw * .58)
    bx = lo_c + int(np.argmax(md[lo_c:hi_c]))
    side = max(col[x] for x in list(range(int(sw * .05), lo_c - 2)) + list(range(hi_c + 2, int(sw * .95))))
    if md[bx] >= max(10.0, 5.0 * float(np.median(md))) and side < 0.45:
        return {"content": None, "cams": [sc((0, 0, bx + 1, sh)), sc((bx + 1, 0, sw - bx - 1, sh))],
                "kind": "talk", "source": "heuristic"}
    cand = [(col[x], x) for x in list(range(int(sw * .05), int(sw * .45))) + list(range(int(sw * .55), int(sw * .95)))]
    score, cx = max(cand)
    if score < 0.45:
        return {"content": [0, 0, fw, fh], "cams": [], "source": "heuristic"}
    right = cx > sw / 2
    x0, x1 = (cx + 1, sw) if right else (0, cx + 1)
    strip = img[:, x0:x1]
    row = (np.abs(np.diff(strip, axis=0)) > 18).mean(axis=1)
    lo, hi = int(sh * .25), int(sh * .75)
    ry = lo + int(np.argmax(row[lo:hi]))
    cams_s = [(x0, 0, x1 - x0, ry + 1), (x0, ry + 1, x1 - x0, sh - ry - 1)] if row[ry] > 0.45 else [(x0, 0, x1 - x0, sh)]
    content_s = (0, 0, cx + 1, sh) if right else (cx + 1, 0, sw - cx - 1, sh)
    content_s = _busy_block(img, content_s)   # ekran paylaşımının içindeki oyun/uygulama kartı
    return {"content": sc(content_s), "cams": [sc(c) for c in cams_s], "kind": "game", "source": "heuristic"}


def detect_claude(video: str | Path, t: float, fw: int, fh: int) -> dict:
    from .claude_client import call_json, image_block, text_block
    with tempfile.TemporaryDirectory(prefix="vedit_layout_") as td:
        width = min(fw, 1280)
        p = _frame(video, t, Path(td) / "frame.jpg", width)
        k = fw / width
        data = call_json([text_block(DETECT_PROMPT.format(w=width, h=int(round(fh / k)))), image_block(p)],
                         system=DETECT_SYSTEM)

    def rect(d):
        return list(clamp_rect((d["x"] * k, d["y"] * k, d["w"] * k, d["h"] * k), fw, fh))
    cams = [rect(c) for c in data.get("cameras") or [] if isinstance(c, dict) and all(key in c for key in "xywh")]
    notes = str(data.get("notes", ""))[:200]
    if not isinstance(data.get("content"), dict):
        if not cams:
            raise ValueError("ne içerik ne kamera bulundu")
        return {"content": None, "cams": sorted(cams, key=lambda c: (c[1], c[0])), "kind": "talk",
                "source": "claude", "notes": notes}
    content = rect(data["content"])
    if content[2] < fw * 0.1 or content[3] < fh * 0.1:
        raise ValueError("içerik bölgesi çok küçük")
    return {"content": content, "cams": cams, "kind": "game", "source": "claude", "notes": notes}


def detect(video: str | Path, t: float, fw: int, fh: int, use_claude: bool = True) -> dict:
    """Bölgeleri bulur: önce Claude (görsel), olmazsa görüntü analizi."""
    from .config import settings
    if use_claude and settings.has_claude:
        try:
            return {**detect_claude(video, t, fw, fh), "t": round(t, 2), "w": fw, "h": fh}
        except Exception as e:  # noqa: BLE001 — Claude başarısızsa sezgisele düş
            res = detect_heuristic(video, t, fw, fh)
            return {**res, "t": round(t, 2), "w": fw, "h": fh, "notes": f"Claude ile bulunamadı ({e}); sezgisel"}
    return {**detect_heuristic(video, t, fw, fh), "t": round(t, 2), "w": fw, "h": fh}


def detect_reel(video: str | Path, start: float, end: float, fw: int, fh: int, use_claude: bool = True) -> dict:
    """Bir reel'in ekran düzeni. Reel'in üç karesine hızlı (sezgisel) bakılır: hepsinde yalnızca kameralar
    varsa sohbet ("talk"). Oyun/ekran paylaşımı görülen karede ayrıntılı tespit yapılır (Claude varsa)."""
    ts = [start + (end - start) * k for k in (0.2, 0.5, 0.8)]
    quick = [(t, detect_heuristic(video, t, fw, fh)) for t in ts]
    others = [(t, r) for t, r in quick if r.get("kind") != "talk"]
    if not others:
        t, r = quick[1]
        return {**r, "t": round(t, 2), "w": fw, "h": fh}
    return detect(video, others[len(others) // 2][0], fw, fh, use_claude)


def preview(video: str | Path, t: float, mode: str, regions: dict, fw: int, fh: int, dst: Path) -> Path:
    """Tek kare 9:16 önizleme (render etmeden yerleşimi görmek için)."""
    from . import media
    g = graph("0:v", "out", plan(mode, fw, fh, regions))
    media._run_ffmpeg(["-ss", f"{max(0.0, t):.3f}", "-i", str(video), "-frames:v", "1",
                       "-filter_complex", g, "-map", "[out]", "-q:v", "3", str(dst)])
    return dst
