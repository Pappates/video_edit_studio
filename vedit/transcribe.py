"""faster-whisper ile kelime düzeyinde zaman damgalı transkripsiyon.

Çıktı formatı (transcript.json):
{
  "language": "tr",
  "duration": 312.4,
  "segments": [
    {"id": 0, "start": 0.52, "end": 4.10, "text": "Merhaba arkadaşlar",
     "words": [{"start": 0.52, "end": 1.1, "word": "Merhaba", "prob": 0.98}, ...]}
  ]
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .config import settings

ProgressFn = Callable[[float, str], None]

# Kesim komutlarının Whisper tarafından doğru yazılma olasılığını artırır.
INITIAL_PROMPT = (
    "Türkçe video çekimi. Konuşmacılar bazen 'burayı kes', 'şurayı kes', 'baştan alıyorum', "
    "'tekrar alalım', 'bir daha çekelim', 'kurguda çıkar' gibi kurgu komutları söyler."
)

_model_cache: dict[tuple, object] = {}


def _load_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - ortam bağımlı
        raise RuntimeError(
            "faster-whisper kurulu değil. `pip install faster-whisper` ile kurun."
        ) from exc

    device = settings.whisper_device
    compute = settings.whisper_compute_type
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"

    key = (settings.whisper_model, device, compute)
    if key not in _model_cache:
        _model_cache[key] = WhisperModel(settings.whisper_model, device=device, compute_type=compute)
    return _model_cache[key]


def transcribe(media_path: str | Path, out_json: str | Path | None = None,
               progress: ProgressFn | None = None) -> dict:
    """Videoyu/sesi yazıya döker. `out_json` varsa ve doluysa önbellekten okur."""
    if out_json and Path(out_json).exists():
        return json.loads(Path(out_json).read_text(encoding="utf-8"))

    model = _load_model()
    seg_iter, info = model.transcribe(
        str(media_path),
        language=settings.language or None,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=False,  # tekrarlarda halüsinasyonu azaltır
        beam_size=5,
    )
    duration = float(getattr(info, "duration", 0.0) or 0.0)

    segments = []
    for i, seg in enumerate(seg_iter):
        words = [
            {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip(),
             "prob": round(getattr(w, "probability", 1.0), 3)}
            for w in (seg.words or [])
        ]
        segments.append({
            "id": i,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words,
        })
        if progress and duration:
            progress(min(seg.end / duration, 1.0), f"Transkripsiyon {seg.end:.0f}/{duration:.0f} sn")

    result = {"language": info.language, "duration": duration, "segments": segments}
    result = resegment(result)
    if out_json:
        Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def resegment(transcript: dict, max_gap: float = 0.7, max_len: float = 12.0) -> dict:
    """Whisper segmentlerini kelime aralarındaki duraklamalara göre cümle benzeri birimlere böler.

    Tekrar tespiti için segmentlerin 'bir deneme = bir birim' olması önemlidir.
    Kelime bilgisi yoksa transkripti olduğu gibi döner.
    """
    words = [w for s in transcript["segments"] for w in s.get("words", [])]
    if not words:
        return transcript

    units: list[list[dict]] = [[words[0]]]
    for prev, cur in zip(words, words[1:]):
        unit = units[-1]
        gap = cur["start"] - prev["end"]
        ends_sentence = prev["word"].rstrip().endswith((".", "?", "!"))
        too_long = cur["end"] - unit[0]["start"] > max_len
        # Whisper noktalaması güvenilir: cümle sonu her zaman yeni birim başlatır (tekrar tespiti için önemli)
        if gap >= max_gap or (ends_sentence and len(unit) >= 2) or too_long:
            units.append([cur])
        else:
            unit.append(cur)

    segments = [
        {
            "id": i,
            "start": u[0]["start"],
            "end": u[-1]["end"],
            "text": " ".join(w["word"] for w in u).strip(),
            "words": u,
        }
        for i, u in enumerate(units)
    ]
    return {**transcript, "segments": segments}
