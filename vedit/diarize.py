"""Konuşmacı ayrımı (diarization): "kim, ne zaman konuştu?"

faster-whisper yalnızca NE söylendiğini çıkarır. Bu modül her kelimeye bir konuşmacı etiketi ekler.

İki arka uç:
  light    — Varsayılan. Ek kurulum gerektirmez: faster-whisper'ın zaten kurduğu onnxruntime ve
             huggingface_hub ile WeSpeaker ResNet34 ses imzası modelini (~26 MB) çalıştırır.
             Whisper'ın kelime zamanlarından konuşma parçaları çıkarılır, her parçanın ses imzası
             hesaplanır, imzalar kümelenir (ortalama bağlantılı hiyerarşik kümeleme, kosinüs).
  pyannote — En yüksek doğruluk (örtüşen konuşma, kısa araya girmeler). `pip install pyannote.audio`
             ve Hugging Face token'ı (HF_TOKEN) gerektirir; model sayfasında koşulları kabul etmek gerekir.

Çıktı: kelimelere ve segmentlere "speaker": "S1" alanı eklenmiş transkript + konuşmacı istatistikleri.
Konuşmacı değişen segmentler ikiye bölünür (her segment tek kişiye ait olur).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import settings
from .media import _bin

SR = 16000
ProgressFn = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Ses
# ---------------------------------------------------------------------------
def load_audio(src, sr: int = SR) -> np.ndarray:
    """Videonun sesini 16 kHz mono float32 [-1, 1] olarak okur."""
    proc = subprocess.run(
        [_bin("ffmpeg"), "-nostdin", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Kaldi uyumlu fbank (WeSpeaker modelleri bu özniteliklerle eğitildi)
# ---------------------------------------------------------------------------
_BANKS: dict[tuple, np.ndarray] = {}


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def mel_banks(n_mels: int = 80, n_fft: int = 512, sr: int = SR, low: float = 20.0) -> np.ndarray:
    key = (n_mels, n_fft, sr, low)
    if key in _BANKS:
        return _BANKS[key]
    high = sr / 2.0
    mlow, mhigh = _mel(low), _mel(high)
    delta = (mhigh - mlow) / (n_mels + 1)
    b = np.arange(n_mels)
    left, center, right = mlow + b * delta, mlow + (b + 1) * delta, mlow + (b + 2) * delta
    m = _mel((sr / n_fft) * np.arange(n_fft // 2))[None, :]
    up = (m - left[:, None]) / (center - left)[:, None]
    down = (right[:, None] - m) / (right - center)[:, None]
    w = np.maximum(0.0, np.minimum(up, down))
    w = np.pad(w, ((0, 0), (0, 1)))  # Nyquist sütunu (Kaldi'de sıfır)
    _BANKS[key] = w
    return w


def fbank(wave: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """25 ms pencere, 10 ms kaydırma, hamming, ön-vurgu 0.97, log-mel, ortalama normalizasyonu (CMN)."""
    x = wave.astype(np.float64) * 32768.0
    flen, shift, n_fft = 400, 160, 512
    if len(x) < flen:
        x = np.pad(x, (0, flen - len(x)))
    n = 1 + (len(x) - flen) // shift
    idx = np.arange(flen)[None, :] + shift * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - 0.97 * prev
    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(flen) / (flen - 1))
    spec = np.abs(np.fft.rfft(frames * win, n=n_fft)) ** 2
    feats = np.log(np.maximum(spec @ mel_banks(n_mels, n_fft).T, np.finfo(np.float64).eps))
    feats -= feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Ses imzası modeli
# ---------------------------------------------------------------------------
class OnnxEmbedder:
    """WeSpeaker ResNet34 (VoxCeleb) ONNX — giriş [1, T, 80] fbank, çıkış 256 boyutlu imza."""

    def __init__(self):
        import onnxruntime as ort

        path = settings.speaker_embed_path
        if not path:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("huggingface_hub yok: `pip install huggingface_hub`") from e
            path = hf_hub_download(settings.speaker_embed_repo, settings.speaker_embed_file)
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(path, providers=providers)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, wave: np.ndarray) -> np.ndarray:
        feats = fbank(wave)[None]
        return np.asarray(self.sess.run(None, {self.inp: feats})[0]).reshape(-1)


# ---------------------------------------------------------------------------
# Kümeleme
# ---------------------------------------------------------------------------
def cluster(embs: np.ndarray, n_speakers: int | None = None, min_sim: float | None = None,
            max_items: int = 700) -> np.ndarray:
    """Ortalama bağlantılı hiyerarşik kümeleme (kosinüs). Etiketler ilk görülme sırasına göre 0..k-1."""
    min_sim = settings.speaker_similarity if min_sim is None else min_sim
    X = np.asarray(embs, dtype=np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    if len(X) >= 10:  # ortak kanal/oda etkisini azaltmak için merkezle
        X = X - X.mean(axis=0)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    n = len(X)
    if n <= 1:
        return np.zeros(n, dtype=int)

    if n > max_items:  # büyük girdide alt küme üzerinde kümele, kalanları en yakın merkeze ata
        sub = np.linspace(0, n - 1, max_items).round().astype(int)
        sub_labels = cluster(embs[sub], n_speakers, min_sim, max_items)
        cents = np.stack([X[sub][sub_labels == k].mean(0) for k in range(sub_labels.max() + 1)])
        return _relabel(np.argmax(X @ cents.T, axis=1))

    D = 1.0 - X @ X.T
    np.fill_diagonal(D, np.inf)
    sizes = np.ones(n)
    labels = np.arange(n)
    active = n
    target = max(1, n_speakers or 1)
    while active > target:
        flat = int(np.argmin(D))
        i, j = divmod(flat, n)
        if not np.isfinite(D[i, j]):
            break
        if n_speakers is None and 1.0 - D[i, j] < min_sim:
            break
        new = (sizes[i] * D[i] + sizes[j] * D[j]) / (sizes[i] + sizes[j])
        D[i, :] = new
        D[:, i] = new
        D[j, :] = np.inf
        D[:, j] = np.inf
        D[i, i] = np.inf
        sizes[i] += sizes[j]
        labels[labels == j] = i
        active -= 1
    return _relabel(labels)


def _relabel(labels: np.ndarray) -> np.ndarray:
    order: dict[int, int] = {}
    return np.array([order.setdefault(int(v), len(order)) for v in labels], dtype=int)


# ---------------------------------------------------------------------------
# Hafif arka uç: Whisper kelime zamanları + ses imzası
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    start: float
    end: float
    words: list[tuple[int, int]]  # (segment index, word index)


def speech_chunks(transcript: dict, target: float = 2.5, max_len: float = 4.0) -> list[Chunk]:
    """Her segmenti en fazla ~4 sn'lik kelime gruplarına böler (uzun segmentte konuşmacı değişebilir)."""
    chunks: list[Chunk] = []
    for si, seg in enumerate(transcript["segments"]):
        words = seg.get("words") or [{"start": seg["start"], "end": seg["end"], "word": seg["text"]}]
        cur: list[int] = []
        for wi, w in enumerate(words):
            if cur and (w["end"] - words[cur[0]]["start"] > max_len or
                        (words[cur[-1]]["end"] - words[cur[0]]["start"] >= target and w["start"] - words[cur[-1]]["end"] > 0.15)):
                chunks.append(Chunk(words[cur[0]]["start"], words[cur[-1]]["end"], [(si, k) for k in cur]))
                cur = []
            cur.append(wi)
        if cur:
            chunks.append(Chunk(words[cur[0]]["start"], words[cur[-1]]["end"], [(si, k) for k in cur]))
    return chunks


def diarize_light(audio: np.ndarray, transcript: dict, n_speakers: int | None = None,
                  embedder=None, progress: ProgressFn | None = None, min_chunk: float = 0.7) -> list[list[str]]:
    """Her kelime için konuşmacı etiketi döner: labels[segment][word]."""
    embedder = embedder or OnnxEmbedder()
    chunks = speech_chunks(transcript)
    usable = [c for c in chunks if c.end - c.start >= min_chunk]
    if not usable:
        usable = chunks

    embs = []
    for k, c in enumerate(usable):
        a, b = int(max(0, c.start - 0.05) * SR), int((c.end + 0.05) * SR)
        embs.append(embedder(audio[a:b]))
        if progress and k % 10 == 0:
            progress(k / max(1, len(usable)), f"Ses imzaları {k}/{len(usable)}")
    labels = cluster(np.stack(embs), n_speakers) if embs else np.zeros(0, dtype=int)

    if n_speakers is None and len(labels):
        labels = _absorb_tiny(np.stack(embs), labels, [c.end - c.start for c in usable])

    chunk_label: dict[int, int] = {id(c): int(l) for c, l in zip(usable, labels)}
    # Kısa parçalara zamanca en yakın etiketli parçanın konuşmacısını ver
    labeled = [(c.start, c.end, chunk_label[id(c)]) for c in usable]
    out = [[""] * len(seg.get("words") or [None]) for seg in transcript["segments"]]
    for c in chunks:
        if id(c) in chunk_label:
            lab = chunk_label[id(c)]
        else:
            mid = (c.start + c.end) / 2
            lab = min(labeled, key=lambda t: 0 if t[0] <= mid <= t[1] else min(abs(mid - t[0]), abs(mid - t[1])))[2] \
                if labeled else 0
        for si, wi in c.words:
            out[si][wi] = f"S{lab + 1}"
    return out


def _absorb_tiny(embs: np.ndarray, labels: np.ndarray, durs: list[float],
                 max_dur: float = 1.5) -> np.ndarray:
    """Otomatik modda yalnızca tek parçadan oluşan ve 1.5 sn'den kısa 'kümeleri' (öksürük, gülme,
    tek kelime) en yakın kümeye katar. Az konuşan gerçek kişiler (ör. iki kısa soru soran
    röportajcı) birden fazla parça ürettiği için korunur."""
    durs_arr = np.asarray(durs)
    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    while True:
        ks = np.unique(labels)
        if len(ks) <= 1:
            return _relabel(labels)
        per = {k: durs_arr[labels == k].sum() for k in ks}
        cnt = {k: int((labels == k).sum()) for k in ks}
        small = [k for k in ks if cnt[k] == 1 and per[k] < max_dur]
        if not small:
            return _relabel(labels)
        k = min(small, key=lambda q: per[q])
        cents = {q: X[labels == q].mean(0) for q in ks if q != k}
        ck = X[labels == k].mean(0)
        target = max(cents, key=lambda q: float(ck @ cents[q]))
        labels = np.where(labels == k, target, labels)


# ---------------------------------------------------------------------------
# pyannote arka ucu
# ---------------------------------------------------------------------------
def pyannote_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
        return bool(settings.hf_token)
    except Exception:
        return False


def diarize_pyannote(audio: np.ndarray, n_speakers: int | None = None) -> list[tuple[float, float, str]]:
    import torch
    from pyannote.audio import Pipeline

    try:
        pipe = Pipeline.from_pretrained(settings.pyannote_model, token=settings.hf_token)
    except TypeError:  # pyannote.audio 3.x
        pipe = Pipeline.from_pretrained(settings.pyannote_model, use_auth_token=settings.hf_token)
    if pipe is None:
        raise RuntimeError(f"{settings.pyannote_model} yüklenemedi: Hugging Face'te model koşullarını kabul edin.")
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    kw = {"num_speakers": n_speakers} if n_speakers else {}
    out = pipe({"waveform": torch.from_numpy(audio)[None], "sample_rate": SR}, **kw)
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", None) or out
    return [(float(t.start), float(t.end), str(s)) for t, _, s in ann.itertracks(yield_label=True)]


def labels_from_turns(transcript: dict, turns: list[tuple[float, float, str]]) -> list[list[str]]:
    """Kelime başına en çok örtüşen konuşma dönemi; örtüşme yoksa zamanca en yakını."""
    names: dict[str, str] = {}
    for _, _, s in sorted(turns):
        names.setdefault(s, f"S{len(names) + 1}")
    out = []
    for seg in transcript["segments"]:
        row = []
        for w in seg.get("words") or [{"start": seg["start"], "end": seg["end"]}]:
            best, best_ov = None, 0.0
            for a, b, s in turns:
                ov = min(b, w["end"]) - max(a, w["start"])
                if ov > best_ov:
                    best, best_ov = s, ov
            if best is None and turns:
                mid = (w["start"] + w["end"]) / 2
                best = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]
            row.append(names.get(best, "S1"))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Transkripte uygulama, istatistik, zaman çizelgesi
# ---------------------------------------------------------------------------
def apply_labels(transcript: dict, labels: list[list[str]]) -> dict:
    """Etiketleri kelimelere yazar; konuşmacı değişen segmentleri böler ve yeniden numaralar."""
    new_segs = []
    for seg, row in zip(transcript["segments"], labels):
        words = seg.get("words") or []
        if not words:
            new_segs.append({**seg, "speaker": row[0] if row else "S1"})
            continue
        for w, lab in zip(words, row):
            w["speaker"] = lab or "S1"
        part = [words[0]]
        for w in words[1:]:
            if w["speaker"] != part[-1]["speaker"]:
                new_segs.append(_seg_from(part))
                part = []
            part.append(w)
        new_segs.append(_seg_from(part))
    for i, s in enumerate(new_segs):
        s["id"] = i
    return {**transcript, "segments": new_segs}


def _seg_from(words: list[dict]) -> dict:
    return {"start": words[0]["start"], "end": words[-1]["end"], "speaker": words[0]["speaker"],
            "text": " ".join(w["word"] for w in words).strip(), "words": words}


def set_segment_speaker(transcript: dict, seg_id: int, speaker: str) -> dict:
    """Kullanıcı düzeltmesi: bir segmentin tüm kelimelerini başka konuşmacıya ata."""
    for s in transcript["segments"]:
        if s["id"] == seg_id:
            s["speaker"] = speaker
            for w in s.get("words") or []:
                w["speaker"] = speaker
    return transcript


def speaker_stats(transcript: dict, audio: np.ndarray | None, previous: dict | None = None) -> dict:
    """Konuşmacı başına süre, ortalama ses seviyesi (dBFS), örnek aralık. Önceki isim/ayarlar korunur."""
    previous = previous or {}
    spans: dict[str, list[tuple[float, float]]] = {}
    for s in transcript["segments"]:
        spans.setdefault(s.get("speaker", "S1"), []).append((s["start"], s["end"]))
    out = {}
    for n, spk in enumerate(sorted(spans, key=lambda k: int(k[1:]) if k[1:].isdigit() else 99)):
        sp = spans[spk]
        dur = sum(b - a for a, b in sp)
        rms_db = None
        if audio is not None and len(audio):
            pieces, acc = [], 0.0
            for a, b in sorted(sp, key=lambda t: t[0] - t[1]):  # uzun parçalardan başla
                pieces.append(audio[int(a * SR): int(b * SR)])
                acc += b - a
                if acc > 120:
                    break
            cat = np.concatenate(pieces) if pieces else np.zeros(1)
            rms = float(np.sqrt(np.mean(np.square(cat)))) if len(cat) else 0.0
            rms_db = round(20 * np.log10(max(rms, 1e-6)), 1)
        longest = max(sp, key=lambda t: t[1] - t[0])
        prev = previous.get(spk, {})
        out[spk] = {
            "name": prev.get("name", f"Konuşmacı {spk[1:]}"),
            "gain_db": float(prev.get("gain_db", 0.0)),
            "mute": bool(prev.get("mute", False)),
            "duration": round(dur, 1),
            "segments": len(sp),
            "rms_db": rms_db if rms_db is not None else prev.get("rms_db"),
            "sample": [round(longest[0], 2), round(min(longest[1], longest[0] + 8), 2)],
            "color": prev.get("color", n),
        }
    return out


def timeline_turns(transcript: dict, duration: float) -> list[tuple[float, float, str]]:
    """Tüm süreyi konuşmacı dönemlerine böler. Farklı kişiler arasındaki boşluk ortadan ayrılır;
    aynı kişinin sessizlikleri o kişiye aittir. Ses ayarı bu dönemlere uygulanır."""
    words = sorted(
        (w["start"], w["end"], w.get("speaker", s.get("speaker", "S1")))
        for s in transcript["segments"] for w in (s.get("words") or [{"start": s["start"], "end": s["end"]}])
    )
    if not words:
        return []
    runs: list[list] = [[words[0][0], words[0][1], words[0][2]]]
    for a, b, spk in words[1:]:
        if spk == runs[-1][2]:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b, spk])
    turns = []
    for k, (a, b, spk) in enumerate(runs):
        start = 0.0 if k == 0 else (runs[k - 1][1] + a) / 2
        end = duration if k == len(runs) - 1 else (b + runs[k + 1][0]) / 2
        if end > start:
            turns.append((round(start, 3), round(max(start, end), 3), spk))
    return turns


# ---------------------------------------------------------------------------
# Ana giriş
# ---------------------------------------------------------------------------
def choose_backend() -> str:
    b = settings.diarization_backend
    if b in ("light", "pyannote", "off"):
        return b
    return "pyannote" if pyannote_available() else "light"


def diarize(media_path, transcript: dict, n_speakers: int | None = None, progress: ProgressFn | None = None,
            embedder=None, previous: dict | None = None) -> dict:
    """Döner: {"backend", "note", "transcript", "speakers"}"""
    audio = load_audio(media_path)
    backend = choose_backend() if embedder is None else "light"
    note = ""
    if n_speakers == 1 or not transcript["segments"]:
        labels = [["S1"] * len(s.get("words") or [None]) for s in transcript["segments"]]
        backend = "single"
    else:
        labels = None
        if backend == "pyannote":
            try:
                if progress:
                    progress(0.1, "pyannote ile konuşmacılar ayrılıyor")
                labels = labels_from_turns(transcript, diarize_pyannote(audio, n_speakers))
            except Exception as e:  # noqa: BLE001
                note = f"pyannote kullanılamadı ({e}); hafif modele geçildi."
                backend = "light"
        if labels is None:
            labels = diarize_light(audio, transcript, n_speakers, embedder=embedder, progress=progress)
    tr = apply_labels(transcript, labels)
    return {"backend": backend, "note": note, "transcript": tr,
            "speakers": speaker_stats(tr, audio, previous)}
