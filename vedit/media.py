"""ffmpeg / ffprobe yardımcıları."""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from .config import settings

ProgressFn = Callable[[float, str], None]


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} bulunamadı. ffmpeg kurulu ve PATH'te olmalı.")
    return path


def probe(path: str | Path) -> dict:
    out = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    fps = 30.0
    if v and v.get("avg_frame_rate", "0/0") != "0/0":
        num, den = v["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    return {
        "duration": float(data["format"].get("duration", 0.0)),
        "width": int(v["width"]) if v else 0,
        "height": int(v["height"]) if v else 0,
        "fps": round(fps, 3),
        "has_video": v is not None,
        "has_audio": a is not None,
        "sample_rate": int(a.get("sample_rate", 48000)) if a else 48000,
        "channels": int(a.get("channels", 2)) if a else 2,
    }


@functools.lru_cache(maxsize=1)
def _script_flag() -> str:
    """ffmpeg 7+ '-/filter_complex dosya', daha eskiler '-filter_complex_script dosya' kullanır."""
    out = subprocess.run([_bin("ffmpeg"), "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    m = re.search(r"ffmpeg version n?(\d+)", out)
    major = int(m.group(1)) if m else (8 if "version N-" in out else 6)  # git derlemeleri güncel kabul edilir
    return "-/filter_complex" if major >= 7 else "-filter_complex_script"


def _run_ffmpeg(args: list[str], total: float = 0.0, progress: ProgressFn | None = None, label: str = "",
                log_path: str | Path | None = None) -> None:
    """ffmpeg'i çalıştırır; ilerlemeyi stdout'tan okur.

    stderr bir dosyaya yazılır, boruya DEĞİL: ffmpeg kodlamadan önce stderr'e uzun bir 'Stream mapping'
    listesi yazar (her kesim parçası için bir satır). Boru okunmazsa Windows'taki ~4 KB tampon dolar ve
    ffmpeg ile uygulama birbirini bekleyerek kilitlenir (render %0'da takılır).
    """
    cmd = [_bin("ffmpeg"), "-hide_banner", "-y", "-nostdin", "-progress", "pipe:1", "-nostats", *args]
    tmp_log = log_path is None
    if tmp_log:
        fd, log_path = tempfile.mkstemp(prefix="vedit_ffmpeg_", suffix=".log")
        os.close(fd)
    log_path = Path(log_path)
    pat = re.compile(r"out_time_us=(\d+)")
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL,
                                    text=True, encoding="utf-8", errors="replace")
            assert proc.stdout is not None
            for line in proc.stdout:
                m = pat.match(line.strip())
                if m and progress and total > 0:
                    progress(min(int(m.group(1)) / 1e6 / total, 1.0), label)
            code = proc.wait()
        if code != 0:
            err = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg hatası (kod {code}):\n{err[-1500:]}")
    finally:
        if tmp_log:
            log_path.unlink(missing_ok=True)


def extract_audio(src: str | Path, dst: str | Path) -> Path:
    """Whisper için 16 kHz mono WAV."""
    _run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])
    return Path(dst)


def _encode_args(vertical: bool = False) -> list[str]:
    return ["-c:v", "libx264", "-preset", settings.video_preset, "-crf", str(settings.video_crf),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]


def vertical_filter(width: int, height: int, mode: str = "crop") -> str:
    """9:16 dönüşümü. crop: ortadan kırp; blur: bulanık arka plan üstüne sığdır."""
    if mode == "blur":
        return ("split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:2[bg];"
                "[b]scale=1080:1920:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
    if width and height and width / height <= 9 / 16 + 0.01:
        return "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    return "crop=ih*9/16:ih,scale=1080:1920"


def render_keeps(src: str | Path, keeps: list[tuple[float, float]], dst: str | Path,
                 progress: ProgressFn | None = None, vertical: str | None = None,
                 workdir: str | Path | None = None, audio_src: str | Path | None = None) -> Path:
    """Kalacak aralıkları kare-hassas kesip birleştirir (trim + concat, yeniden kodlama).

    audio_src: kaynakla aynı zaman çizelgesinde işlenmiş ses (ör. kişi bazlı ses ayarı); verilirse
    ses bu dosyadan alınır.
    """
    if not keeps:
        raise ValueError("Kalacak aralık yok — tüm video kesilmiş görünüyor.")
    info = probe(src)
    has_a, has_v = info["has_audio"], info["has_video"]
    if audio_src:
        has_a = True
    a_in = "1:a" if audio_src else "0:a"
    # İlk kalan aralığın biraz öncesine atla: ör. 20. dakikadaki bir reel için baştan çözmeye gerek yok.
    # -ss girişten önce verildiğinde ffmpeg hassas arar ve zaman damgalarını 'offset' noktasından 0'a çeker.
    offset = max(0.0, min(a for a, _ in keeps) - 2.0)
    keeps = [(a - offset, b - offset) for a, b in keeps]
    lines = []
    labels = []
    for i, (a, b) in enumerate(keeps):
        if has_v:
            lines.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}];")
        if has_a:
            # kısa yumuşatma: kesim noktalarında 'tık' sesini önler
            fade = min(0.015, (b - a) / 4)
            lines.append(
                f"[{a_in}]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:d={fade:.3f},afade=t=out:st={max(0.0, b - a - fade):.3f}:d={fade:.3f}[a{i}];")
        labels.append((f"[v{i}]" if has_v else "") + (f"[a{i}]" if has_a else ""))
    graph = "".join(lines) + "".join(labels) + \
        f"concat=n={len(keeps)}:v={int(has_v)}:a={int(has_a)}" + ("[outv]" if has_v else "") + ("[outa]" if has_a else "")
    if vertical and has_v:
        graph = graph.replace("[outv]", "[cv]") + f";[cv]{vertical_filter(info['width'], info['height'], vertical)}[outv]"

    wd = Path(workdir or Path(dst).parent)
    script = wd / (Path(dst).stem + ".filtergraph.txt")
    script.write_text(graph, encoding="utf-8")
    maps = (["-map", "[outv]"] if has_v else []) + (["-map", "[outa]"] if has_a else [])
    total = sum(b - a for a, b in keeps)
    seek = ["-ss", f"{offset:.3f}"] if offset > 0 else []
    inputs = [*seek, "-i", str(src)] + ([*seek, "-i", str(audio_src)] if audio_src else [])
    _run_ffmpeg([*inputs, _script_flag(), str(script), *maps, *_encode_args(), str(dst)],
                total=total, progress=progress, label="Render",
                log_path=wd / (Path(dst).stem + ".ffmpeg.log"))
    script.unlink(missing_ok=True)
    return Path(dst)


def extract_frame(src: str | Path, t: float, dst: str | Path, width: int = 512) -> Path:
    _run_ffmpeg(["-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1",
                 "-vf", f"scale={width}:-2", "-q:v", "4", str(dst)])
    return Path(dst)


def scene_changes(src: str | Path, threshold: float = 0.32, scale_w: int = 320) -> list[float]:
    """Sahne/kadraj değişim anları (sn). Reels sınırlarını bulmak için aday üretir."""
    cmd = [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(src), "-an",
           "-vf", f"scale={scale_w}:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    return [round(float(m), 3) for m in re.findall(r"pts_time:([0-9.]+)", err)]
