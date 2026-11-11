"""Konuşmacı bazlı ses ayarı: her kişinin sesini dB olarak kıs/aç ya da tamamen sustur.

Yöntem: konuşmacı dönemlerine (diarize.timeline_turns) göre zamanla değişen bir kazanç eğrisi kurulur;
kişi değişimlerinde ~40 ms yumuşak geçiş yapılır (tık sesi olmaz). Ses akış halinde (5 sn'lik
parçalarla) işlendiği için uzun videolarda bellek dolmaz. Yükseltmede taşmayı önlemek için yumuşak
sınırlayıcı uygulanır. Sonuç bir WAV dosyasıdır; render bu sesi görüntüyle birleştirir.

Not: Ayar zamana göre yapılır. İki kişi aynı anda konuşuyorsa o an hangi kişiye atanmışsa onun ayarı
geçerlidir; seslerin birbirinden ayrıştırılması (source separation) yapılmaz.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .media import _bin

ProgressFn = Callable[[float, str], None]


def linear_gains(speakers: dict) -> dict[str, float]:
    return {spk: 0.0 if s.get("mute") else float(10 ** (float(s.get("gain_db", 0.0)) / 20)) for spk, s in speakers.items()}


def needs_mix(speakers: dict | None) -> bool:
    return bool(speakers) and any(s.get("mute") or abs(float(s.get("gain_db", 0))) > 0.05 for s in speakers.values())


def gain_curve(turns: list[tuple[float, float, str]], gains: dict[str, float],
               ramp: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Parçalı doğrusal kazanç eğrisinin kırılma noktaları (zaman, kazanç)."""
    if not turns:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])
    ts, gs = [turns[0][0]], [gains.get(turns[0][2], 1.0)]
    for (a0, b0, s0), (a1, b1, s1) in zip(turns, turns[1:]):
        g0, g1 = gains.get(s0, 1.0), gains.get(s1, 1.0)
        if g0 == g1:
            continue
        r = min(ramp, (b0 - a0) / 2, (b1 - a1) / 2) / 2
        ts += [a1 - r, a1 + r]
        gs += [g0, g1]
    ts.append(turns[-1][1])
    gs.append(gains.get(turns[-1][2], 1.0))
    t = np.maximum.accumulate(np.asarray(ts, dtype=np.float64))
    t = t + np.arange(len(t)) * 1e-9  # kesin artan
    return t, np.asarray(gs, dtype=np.float64)


def soft_limit(x: np.ndarray, threshold: float = 0.89) -> np.ndarray:
    a = np.abs(x)
    over = a > threshold
    if not over.any():
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (threshold + (1 - threshold) * np.tanh((a[over] - threshold) / (1 - threshold)))
    return y


def mix_key(turns, speakers: dict) -> str:
    payload = json.dumps([turns, {k: [v.get("gain_db", 0), v.get("mute", False)] for k, v in sorted(speakers.items())}])
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def render_mix(src, dst, turns, speakers: dict, sample_rate: int = 48000, channels: int = 2,
               duration: float = 0.0, progress: ProgressFn | None = None, chunk_sec: float = 5.0) -> Path:
    """Kaynak sesi kişi ayarlarıyla işleyip 16-bit WAV yazar."""
    t_pts, g_pts = gain_curve(turns, linear_gains(speakers))
    sr, ch = int(sample_rate or 48000), max(1, min(int(channels or 2), 2))
    err_log = tempfile.TemporaryFile()  # stderr boruya değil dosyaya (Windows'ta kilitlenmeyi önler)
    proc = subprocess.Popen(
        [_bin("ffmpeg"), "-nostdin", "-v", "error", "-i", str(src), "-vn", "-ac", str(ch), "-ar", str(sr),
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        stdout=subprocess.PIPE, stderr=err_log, stdin=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    frame_bytes = 4 * ch
    block = int(chunk_sec * sr) * frame_bytes
    done = 0
    dst = Path(dst)
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        rest = b""
        while True:
            data = proc.stdout.read(block)
            if not data:
                break
            data = rest + data
            usable = len(data) - len(data) % frame_bytes
            data, rest = data[:usable], data[usable:]
            x = np.frombuffer(data, dtype=np.float32).reshape(-1, ch).astype(np.float64)
            t = (done + np.arange(len(x))) / sr
            g = np.interp(t, t_pts, g_pts, left=g_pts[0], right=g_pts[-1])
            y = soft_limit(x * g[:, None])
            wf.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
            done += len(x)
            if progress and duration:
                progress(min(1.0, done / sr / duration), "Ses ayarlanıyor")
    code = proc.wait()
    err_log.seek(0)
    err = err_log.read().decode("utf-8", "ignore")
    err_log.close()
    if code != 0:
        raise RuntimeError(f"Ses okunamadı: {err[-500:]}")
    return dst
