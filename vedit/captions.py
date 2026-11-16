"""Reels için videoya gömülen (burn-in) altyazı: ASS formatı + ffmpeg `ass` filtresi.

Whisper'ın kelime zamanları kullanılır; söylenen kelime anında vurgulanır (kısa video altyazı stili).
Aynı ASS dosyası, görsel kartların isim etiketlerini ve "ilk 10" sıra numaralarını da taşır; böylece
ffmpeg'in drawtext filtresi için yazı tipi dosyası yolu gerekmez (libass sistem fontunu bulur).
"""
from __future__ import annotations

import re

from .config import settings
from .exporters import aligned_words, group_cues

# style: (büyük harf, vurgu rengi #RRGGBB | None, vurgulu kelimede büyütme %, satırdaki en çok kelime,
#         kenar kalınlığı, gölge, yazı boyu / video yüksekliği)
STYLES: dict[str, dict] = {
    "bold":    {"upper": True,  "hi": "#FFD400", "scale": 112, "words": 3, "outline": 7, "shadow": 0, "size": 0.037},
    "karaoke": {"upper": False, "hi": "#3BE36B", "scale": 100, "words": 5, "outline": 5, "shadow": 0, "size": 0.036},
    "minimal": {"upper": False, "hi": None,      "scale": 100, "words": 6, "outline": 3, "shadow": 2, "size": 0.032},
}
POSITIONS = ("bottom", "middle")


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB → ASS &HAABBGGRR."""
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def tr_upper(text: str) -> str:
    return text.replace("i", "İ").replace("ı", "I").upper()


_APOS = re.compile(r"[’'`]")


def _name_key(word: str) -> str:
    """'Messi'nin,' → 'messi' (yabancı isim kümesinde aramak için)."""
    return re.sub(r"[^\w]", "", _APOS.split(word, 1)[0]).lower()


def foreign_names(visuals: list[dict]) -> set[str]:
    """Yabancı isim kelimeleri: Claude'un İngilizce Wikipedia ile eşlediği görsellerin etiketleri
    (ör. "Lionel Messi" → {"lionel", "messi"}). Türkçe isimler (wiki_lang=tr) Türkçe kuralla kalır."""
    out: set[str] = set()
    for v in visuals or []:
        if v.get("wiki_lang") == "en" and v.get("label"):
            out |= {k for k in (_name_key(t) for t in v["label"].split()) if k}
    return out


def upper_word(word: str, foreign: set[str] | None = None) -> str:
    """Büyük harf: Türkçe kural (i→İ); yabancı isimlerde kök normal kuralla, eki Türkçe kuralla
    ('Messi'nin' → 'MESSI'NİN', 'Lionel' → 'LIONEL' — 'LİONEL' değil)."""
    if foreign and _name_key(word) in foreign:
        parts = _APOS.split(word, 1)
        root = parts[0].upper()
        if len(parts) == 2:
            sep = word[len(parts[0])]
            return root + sep + tr_upper(parts[1])
        return root
    return tr_upper(word)


def _esc(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def _header(width: int, height: int, style: dict, position: str, margin_v: int | None = None) -> str:
    font = settings.caption_font
    size = max(18, int(height * style["size"]))
    align = 5 if position == "middle" else 2
    if margin_v is None:
        margin_v = int(height * (0.24 if height > width else 0.08))
    label_size = max(16, int(height * 0.03))
    rank_size = max(24, int(height * 0.06))
    return "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
        f"{style['outline']},{style['shadow']},{align},{int(width * 0.08)},{int(width * 0.08)},{margin_v},1",
        f"Style: Label,{font},{label_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,3,"
        f"{max(6, label_size // 3)},0,8,20,20,20,1",
        f"Style: Rank,{font},{rank_size},{ass_color('#FFD400')},&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
        f"{max(4, rank_size // 10)},2,7,20,20,20,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])


def caption_events(words: list[tuple[float, float, str]], style: dict, foreign: set[str] | None = None) -> list[str]:
    cues = group_cues(words, max_chars=30 if style["words"] <= 3 else 34, max_dur=2.4, max_words=style["words"])
    lines = []
    for cue in cues:
        texts = [_esc(upper_word(w[2], foreign) if style["upper"] else w[2]) for w in cue]
        cue_end = max(cue[-1][1], cue[0][0] + 0.3)
        if not style["hi"]:
            lines.append(f"Dialogue: 0,{ass_time(cue[0][0])},{ass_time(cue_end)},Cap,,0,0,0,,{' '.join(texts)}")
            continue
        hi = ass_color(style["hi"])
        sc = style["scale"]
        for k, w in enumerate(cue):
            a = cue[0][0] if k == 0 else w[0]
            b = cue[k + 1][0] if k + 1 < len(cue) else cue_end
            if b - a < 0.01:
                continue
            parts = [(f"{{\\c{hi}\\fscx{sc}\\fscy{sc}}}{t}{{\\r}}" if j == k else t) for j, t in enumerate(texts)]
            lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{' '.join(parts)}")
    return lines


def visual_text_events(items: list[dict], width: int, height: int) -> list[str]:
    """Görsel kartlarının etiketi ve sıra numarası. items: overlay_plan() çıktısı (çıktı zamanıyla)."""
    lines = []
    for it in items:
        x, y, w, h = it["box"]
        a, b = ass_time(it["start"]), ass_time(it["end"])
        fade = "{\\fad(200,200)}"
        if it.get("label"):
            ly = min(height - 40, y + h + 18) if it["position"] != "full" else y + int(h * 0.06)
            # Büyük harfe çevrilmez: yabancı isimlerde Türkçe kural "LİONEL" gibi yanlış yazım üretir.
            lines.append(f"Dialogue: 1,{a},{b},Label,,0,0,0,,{fade}{{\\an8\\pos({width // 2},{ly})}}"
                         f"{_esc(it['label'])}")
        if it.get("rank"):
            lines.append(f"Dialogue: 2,{a},{b},Rank,,0,0,0,,{fade}{{\\an7\\pos({x + 14},{y + 6})}}#{int(it['rank'])}")
    return lines


def build_ass(transcript: dict | None, keeps: list[tuple[float, float]], width: int, height: int,
              style: str | None = "bold", position: str = "bottom", visual_items: list[dict] | None = None,
              foreign: set[str] | None = None, margin_v: int | None = None) -> str:
    """Tam ASS dosyası. style=None ise yalnızca görsel etiketleri yazılır.
    margin_v: alt hizalı altyazının alttan uzaklığı (dikey yerleşim planı verir: yüzlerin üstüne binmesin).
    foreign: büyük harfte Türkçe kural uygulanmayacak yabancı isimler (foreign_names())."""
    st = STYLES.get(style or "bold", STYLES["bold"])
    lines = [_header(width, height, st, position if position in POSITIONS else "bottom", margin_v)]
    if style and transcript:
        lines += caption_events(aligned_words(transcript, keeps), st, foreign)
    lines += visual_text_events(visual_items or [], width, height)
    return "\n".join(lines) + "\n"

