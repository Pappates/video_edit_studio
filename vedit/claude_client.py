"""Claude istemcisi — iki arka uç, aynı arayüz. Metin + görsel (VLM) destekler.

    api : Claude Messages API (ANTHROPIC_API_KEY ile, kullanım başına ücretli)
    cli : Bilgisayardaki Claude Code (`claude -p`) — Pro/Max aboneliğinin limitlerinden düşer,
          API anahtarı gerekmez. Görseller dosya yolu olarak verilir, Claude bunları Read aracıyla açar.

Seçim .env içindeki CLAUDE_BACKEND ile yapılır (auto | api | cli | off).
"""
from __future__ import annotations

import base64
import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import settings

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class ClaudeError(RuntimeError):
    pass


def image_block(path: str | Path, media_type: str = "image/jpeg") -> dict:
    """Görsel bloğu. API base64 kullanır; CLI arka ucu `_path` üzerinden dosyayı Read ile açar."""
    path = Path(path).resolve()
    return {"type": "image", "_path": str(path),
            "source": {"type": "base64", "media_type": media_type, "path": str(path)}}


def _materialize(block: dict) -> dict:
    """API'ye gitmeden hemen önce görseli base64'e çevirir (bellekte gereksiz kopya tutmamak için)."""
    if block.get("type") != "image":
        return block
    data = base64.b64encode(Path(block["_path"]).read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": block["source"]["media_type"],
                                        "data": data}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def call(content: list[dict] | str, system: str = "", max_tokens: int | None = None,
         temperature: float = 0.0, retries: int = 3, timeout: int | None = None) -> str:
    """Tek tur istek atar, yanıt metnini döner. Arka uç settings.claude_backend ile seçilir."""
    if isinstance(content, str):
        content = [text_block(content)]
    backend = settings.claude_backend
    if backend == "cli":
        return _call_cli(content, system, retries=retries, timeout=timeout or settings.claude_cli_timeout)
    if backend == "api":
        return _call_api(content, system, max_tokens, temperature, retries, timeout or 300)
    raise ClaudeError("Claude kapalı: .env içinde ANTHROPIC_API_KEY verin ya da Claude Code'u kurup "
                      "`claude` ile aboneliğinizle giriş yapın (CLAUDE_BACKEND=cli).")


def _call_api(content: list[dict], system: str, max_tokens: int | None, temperature: float,
              retries: int, timeout: int) -> str:
    """Claude Messages API. 429/5xx durumlarında üstel bekleme ile tekrar dener."""
    if not settings.anthropic_api_key:
        raise ClaudeError("ANTHROPIC_API_KEY tanımlı değil (.env dosyasına ekleyin).")
    content = [_materialize(b) for b in content]

    body = {
        "model": settings.claude_model,
        "max_tokens": max_tokens or settings.claude_max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }
    if system:
        body["system"] = system
    payload = json.dumps(body).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=payload, method="POST", headers={
            "content-type": "application/json",
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": API_VERSION,
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:500]
            last_err = ClaudeError(f"Claude API hatası {e.code}: {detail}")
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise last_err from e
        except urllib.error.URLError as e:
            last_err = ClaudeError(f"Claude API'ye ulaşılamadı: {e.reason}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
    raise last_err or ClaudeError("Bilinmeyen hata")


# ---------------------------------------------------------------------------
# Claude Code (abonelik) arka ucu
# ---------------------------------------------------------------------------
def cli_path() -> str | None:
    return shutil.which(settings.claude_cli_path) or (
        settings.claude_cli_path if Path(settings.claude_cli_path).is_file() else None)


@functools.lru_cache(maxsize=1)
def _cli_supports(flag: str) -> bool:
    exe = cli_path()
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return flag in out


def _call_cli(content: list[dict], system: str, retries: int, timeout: int) -> str:
    exe = cli_path()
    if not exe:
        raise ClaudeError("Claude Code bulunamadı. Kurun (https://claude.com/claude-code) ve terminalde "
                          "bir kez `claude` yazıp Claude hesabınızla giriş yapın.")

    images = [b["_path"] for b in content if b.get("type") == "image"]
    parts: list[str] = []
    for b in content:
        if b.get("type") == "text":
            parts.append(b["text"])
        elif b.get("type") == "image":
            parts.append(f"[GÖRSEL: {b['_path']}]")
    prompt = "\n\n".join(parts)
    if images:
        prompt = ("Bu istekte [GÖRSEL: yol] ile işaretlenmiş görsel dosyaları var. Karar vermeden önce "
                  "hepsini Read aracıyla aç ve incele. Başka araç kullanma, dosya yazma.\n\n" + prompt)
    prompt += "\n\nYanıt olarak yalnızca istenen JSON'u yaz; açıklama ekleme."

    cmd = [exe, "-p", "--output-format", "json", "--model", settings.claude_cli_model]
    sys_file = None
    if system:
        # Çok satırlı Türkçe metni argüman olarak geçirmek (özellikle Windows'ta claude.cmd ile) kırılgan;
        # destekleniyorsa dosyadan verilir.
        if _cli_supports("--system-prompt-file") or _cli_supports("system-prompt[-file]"):
            fd, sys_file = tempfile.mkstemp(suffix=".txt", prefix="vedit_sys_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(system)
            cmd += ["--system-prompt-file", sys_file]
        else:
            cmd += ["--system-prompt", system]
    if images:
        dirs = sorted({str(Path(p).parent) for p in images})
        cmd += ["--allowedTools", "Read", "--add-dir", *dirs]
        if _cli_supports("--tools"):
            cmd += ["--tools", "Read"]
    elif _cli_supports("--tools"):
        cmd += ["--tools", ""]
    if _cli_supports("--permission-mode"):
        cmd += ["--permission-mode", "dontAsk"]

    # API anahtarı ortamdaysa Claude Code onu kullanır; abonelikle gitmesi için çıkarılır.
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    cwd = str(Path(images[0]).parent) if images else None

    try:
        return _run_cli(cmd, prompt, env, cwd, retries, timeout)
    finally:
        if sys_file:
            Path(sys_file).unlink(missing_ok=True)


def _run_cli(cmd: list[str], prompt: str, env: dict, cwd: str | None, retries: int, timeout: int) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=timeout, env=env, cwd=cwd)
        except subprocess.TimeoutExpired:
            last = ClaudeError(f"Claude Code {timeout} sn içinde yanıt vermedi (CLAUDE_CLI_TIMEOUT artırılabilir).")
            continue
        out = proc.stdout.strip()
        try:
            data = json.loads(out.splitlines()[-1]) if out else {}
        except json.JSONDecodeError:
            data = {}
        if proc.returncode == 0 and data and not data.get("is_error"):
            return str(data.get("result", ""))

        detail = (data.get("result") or proc.stderr or out or "").strip()[:600]
        low = detail.lower()
        if any(k in low for k in ("login", "log in", "not authenticated", "/login", "unauthorized")):
            raise ClaudeError("Claude Code'da oturum açık değil. Terminalde `claude` yazıp /login ile "
                              f"aboneliğinizle giriş yapın. Ayrıntı: {detail}")
        last = ClaudeError(f"Claude Code hatası (kod {proc.returncode}): {detail}")
        if any(k in low for k in ("limit", "rate", "overloaded", "529", "429")) and attempt < retries - 1:
            time.sleep(20 * (attempt + 1))
            continue
        if attempt < retries - 1:
            time.sleep(3)
    raise last or ClaudeError("Claude Code yanıt vermedi")


def backend_status() -> dict:
    """Arayüz ve sağlık kontrolü için: hangi arka uç kullanılıyor."""
    b = settings.claude_backend
    return {
        "backend": b,
        "label": {"api": f"Claude API · {settings.claude_model}",
                  "cli": f"Claude aboneliği · {settings.claude_cli_model}",
                  "off": "Claude kapalı"}[b],
        "cli_found": bool(cli_path()),
        "api_key": bool(settings.anthropic_api_key),
    }


def extract_json(text: str):
    """Yanıttaki ilk JSON nesnesini/dizisini ayıklar (```json bloklarını da destekler)."""
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ClaudeError(f"Yanıtta JSON bulunamadı: {text[:200]}")
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for k in range(start, len(text)):
        ch = text[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start: k + 1])
    raise ClaudeError("JSON yanıtı yarım kaldı (max_tokens artırılabilir).")


def call_json(content, system: str = "", **kw):
    return extract_json(call(content, system=system, **kw))
