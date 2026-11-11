"""Merkezi ayarlar. Değerler ortam değişkenlerinden (.env) okunur."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Bağımlılık eklemeden basit .env okuyucu (KEY=VALUE satırları)."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # --- Claude ---
    # auto: API anahtarı varsa API, yoksa kurulu Claude Code (abonelik), o da yoksa kapalı
    backend_pref: str = field(default_factory=lambda: _env("CLAUDE_BACKEND", "auto").lower())
    claude_cli_path: str = field(default_factory=lambda: _env("CLAUDE_CLI_PATH", "claude"))
    claude_cli_model: str = field(default_factory=lambda: _env("CLAUDE_CLI_MODEL", "sonnet"))
    claude_cli_timeout: int = field(default_factory=lambda: int(_env("CLAUDE_CLI_TIMEOUT", "900")))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    claude_model: str = field(default_factory=lambda: _env("CLAUDE_MODEL", "claude-sonnet-5-5"))
    claude_max_tokens: int = field(default_factory=lambda: int(_env("CLAUDE_MAX_TOKENS", "8000")))

    # --- Whisper ---
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "large-v3"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "auto"))  # auto|cuda|cpu
    whisper_compute_type: str = field(default_factory=lambda: _env("WHISPER_COMPUTE_TYPE", "auto"))
    language: str = field(default_factory=lambda: _env("LANGUAGE", "tr"))

    # --- Kesim ayarları ---
    pad_before: float = field(default_factory=lambda: float(_env("CUT_PAD_BEFORE", "0.08")))
    pad_after: float = field(default_factory=lambda: float(_env("CUT_PAD_AFTER", "0.08")))
    repeat_similarity: float = field(default_factory=lambda: float(_env("REPEAT_SIMILARITY", "0.72")))
    repeat_window_sec: float = field(default_factory=lambda: float(_env("REPEAT_WINDOW_SEC", "120")))
    silence_min_sec: float = field(default_factory=lambda: float(_env("SILENCE_MIN_SEC", "1.2")))
    silence_keep_sec: float = field(default_factory=lambda: float(_env("SILENCE_KEEP_SEC", "0.35")))

    # --- Konuşmacı ayrımı ---
    # auto: pyannote kurulu ve HF_TOKEN varsa pyannote, yoksa hafif ONNX (ek kurulum gerektirmez)
    diarization_backend: str = field(default_factory=lambda: _env("DIARIZATION_BACKEND", "auto").lower())
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))
    pyannote_model: str = field(default_factory=lambda: _env("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"))
    speaker_embed_repo: str = field(default_factory=lambda: _env("SPEAKER_EMBED_REPO", "Wespeaker/wespeaker-voxceleb-resnet34-LM"))
    speaker_embed_file: str = field(default_factory=lambda: _env("SPEAKER_EMBED_FILE", "voxceleb_resnet34_LM.onnx"))
    speaker_embed_path: str = field(default_factory=lambda: _env("SPEAKER_EMBED_PATH", ""))  # yerel .onnx (isteğe bağlı)
    # Kişi sayısı 'otomatik' iken iki kümenin aynı kişi sayılacağı en düşük kosinüs benzerliği
    speaker_similarity: float = field(default_factory=lambda: float(_env("SPEAKER_SIMILARITY", "0.45")))

    # --- Render ---
    video_crf: int = field(default_factory=lambda: int(_env("VIDEO_CRF", "18")))
    video_preset: str = field(default_factory=lambda: _env("VIDEO_PRESET", "veryfast"))

    # --- Depolama ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")).resolve())

    @property
    def claude_backend(self) -> str:
        pref = self.backend_pref
        if pref in ("api", "cli", "off"):
            return pref
        if self.anthropic_api_key:
            return "api"
        if shutil.which(self.claude_cli_path) or os.path.isfile(self.claude_cli_path):
            return "cli"
        return "off"

    @property
    def has_claude(self) -> bool:
        return self.claude_backend != "off"


settings = Settings()
