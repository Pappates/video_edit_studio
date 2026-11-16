"""Dışa aktarma: EDL (CMX3600), FCP7 XML (Premiere + DaVinci), SRT, JSON.

EDL ve XML 'kalacak aralıkları' kaynak videoya referansla tarif eder; böylece kesimler
Premiere Pro / DaVinci Resolve'da kayıpsız, düzenlenebilir bir timeline olarak açılır.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape


def _timebase(fps: float) -> tuple[int, bool]:
    for nominal in (24, 30, 60):
        if abs(fps - nominal * 1000 / 1001) < 0.01:
            return nominal, True
    return max(1, round(fps)), False


def _tc(seconds: float, fps: float) -> str:
    base, _ = _timebase(fps)
    frames = round(seconds * fps)
    f = frames % base
    s = frames // base
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}:{f:02d}"


# ---------------------------------------------------------------------------
def to_edl(keeps: list[tuple[float, float]], fps: float, clip_name: str, title: str = "AUTO EDIT") -> str:
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    rec = 0.0
    for n, (a, b) in enumerate(keeps, 1):
        dur = b - a
        lines.append(f"{n:03d}  AX       AA/V  C        {_tc(a, fps)} {_tc(b, fps)} {_tc(rec, fps)} {_tc(rec + dur, fps)}")
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        lines.append("")
        rec += dur
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def to_fcp7_xml(keeps: list[tuple[float, float]], media: dict, src_path: str | Path,
                title: str = "Auto Edit") -> str:
    fps = media.get("fps", 30.0)
    base, ntsc = _timebase(fps)
    rate = f"<rate><timebase>{base}</timebase><ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc></rate>"
    src_path = Path(src_path).resolve()
    name = escape(src_path.name)
    url = "file://localhost" + quote(src_path.as_posix() if src_path.as_posix().startswith("/") else "/" + src_path.as_posix())
    src_frames = round(media.get("duration", 0.0) * fps)
    w, h = media.get("width", 1920), media.get("height", 1080)

    file_full = (
        f'<file id="file-1"><name>{name}</name><pathurl>{escape(url)}</pathurl>{rate}'
        f"<duration>{src_frames}</duration><media>"
        f"<video><samplecharacteristics>{rate}<width>{w}</width><height>{h}</height></samplecharacteristics></video>"
        f"<audio><samplecharacteristics><depth>16</depth><samplerate>48000</samplerate></samplecharacteristics>"
        f"<channelcount>2</channelcount></audio></media></file>"
    )

    v_items, a_items = [], []
    rec = 0
    for n, (a, b) in enumerate(keeps, 1):
        src_in, src_out = round(a * fps), round(b * fps)
        length = src_out - src_in
        if length <= 0:
            continue
        common = (f"<name>{name}</name><duration>{src_frames}</duration>{rate}"
                  f"<start>{rec}</start><end>{rec + length}</end><in>{src_in}</in><out>{src_out}</out>")
        file_el = file_full if n == 1 else '<file id="file-1"/>'
        v_items.append(f'<clipitem id="clipitem-v{n}">{common}{file_el}</clipitem>')
        if media.get("has_audio", True):
            a_items.append(
                f'<clipitem id="clipitem-a{n}">{common}<file id="file-1"/>'
                f"<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack></clipitem>")
        rec += length

    audio = f"<audio><track>{''.join(a_items)}</track></audio>" if a_items else "<audio/>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="5">'
        f'<sequence id="sequence-1"><name>{escape(title)}</name><duration>{rec}</duration>{rate}'
        f"<media><video><format><samplecharacteristics>{rate}<width>{w}</width><height>{h}</height>"
        f"</samplecharacteristics></format><track>{''.join(v_items)}</track></video>{audio}</media>"
        "</sequence></xmeml>\n"
    )


# ---------------------------------------------------------------------------
def remap(t: float, keeps: list[tuple[float, float]]) -> float | None:
    """Kaynak zamanı → kurgulanmış videodaki zaman. Kesilen yerdeyse None."""
    acc = 0.0
    for a, b in keeps:
        if a <= t <= b:
            return acc + (t - a)
        acc += b - a
    return None


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{(ms // 60000) % 60:02d}:{(ms // 1000) % 60:02d},{ms % 1000:03d}"


def remap_forward(t: float, keeps: list[tuple[float, float]]) -> float | None:
    """remap gibi; t kesilen bir yere düşerse bir sonraki kalan aralığın başına kaydırır."""
    r = remap(t, keeps)
    if r is not None:
        return r
    acc = 0.0
    for a, b in keeps:
        if a > t:
            return acc
        acc += b - a
    return None


def aligned_words(transcript: dict, keeps: list[tuple[float, float]]) -> list[tuple[float, float, str]]:
    """Kalan kelimeler, kurgulanmış videonun zamanıyla: [(başlangıç, bitiş, kelime), ...]."""
    words = []
    for s in transcript["segments"]:
        for w in s.get("words") or [{"start": s["start"], "end": s["end"], "word": s["text"]}]:
            mid = (w["start"] + w["end"]) / 2
            if remap(mid, keeps) is None:
                continue
            ns, ne = remap(w["start"], keeps), remap(w["end"], keeps)
            ns = ns if ns is not None else remap(mid, keeps)
            ne = ne if ne is not None else remap(mid, keeps)
            text = w["word"].strip()
            if text:
                words.append((ns, ne, text))
    return words


def group_cues(words: list[tuple[float, float, str]], max_chars: int = 42, max_dur: float = 3.0,
               max_words: int = 0) -> list[list[tuple[float, float, str]]]:
    """Kelimeleri ekranda birlikte görünecek kısa satırlara böler."""
    cues, cur = [], []
    for w in words:
        text = " ".join(x[2] for x in cur + [w])
        if cur and (len(text) > max_chars or w[1] - cur[0][0] > max_dur or w[0] - cur[-1][1] > 0.6
                    or (max_words and len(cur) >= max_words)):
            cues.append(cur)
            cur = []
        cur.append(w)
    if cur:
        cues.append(cur)
    return cues


def to_srt(transcript: dict, keeps: list[tuple[float, float]], max_chars: int = 42,
           max_dur: float = 3.0) -> str:
    """Kalan kelimelerden kurgulanmış videoya hizalı altyazı üretir (reels için kısa satırlar)."""
    cues = group_cues(aligned_words(transcript, keeps), max_chars, max_dur)
    out = []
    for n, c in enumerate(cues, 1):
        out += [str(n), f"{_srt_time(c[0][0])} --> {_srt_time(max(c[-1][1], c[0][0] + 0.3))}",
                " ".join(x[2] for x in c), ""]
    return "\n".join(out)


def write_all(out_dir: str | Path, stem: str, keeps, media: dict, src_path, transcript: dict,
              extra: dict | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "edl": out_dir / f"{stem}.edl",
        "xml": out_dir / f"{stem}.xml",
        "srt": out_dir / f"{stem}.srt",
        "json": out_dir / f"{stem}.cuts.json",
    }
    files["edl"].write_text(to_edl(keeps, media["fps"], Path(src_path).name, stem.upper()), encoding="utf-8")
    files["xml"].write_text(to_fcp7_xml(keeps, media, src_path, stem), encoding="utf-8")
    files["srt"].write_text(to_srt(transcript, keeps), encoding="utf-8")
    files["json"].write_text(json.dumps({"keeps": keeps, **(extra or {})}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return files
