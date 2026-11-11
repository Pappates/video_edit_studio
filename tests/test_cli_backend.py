"""Claude Code (abonelik) arka ucu: sahte bir `claude` çalıştırılabiliri ile test edilir."""
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedit import claude_client  # noqa: E402
from vedit.claude_client import ClaudeError, call_json, image_block, text_block  # noqa: E402
from vedit.config import settings  # noqa: E402

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
if "--help" in sys.argv:
    print("--tools --permission-mode --add-dir --allowedTools --system-prompt-file"); sys.exit(0)
log = os.environ["FAKE_LOG"]
prompt = sys.stdin.read()
sp = sys.argv[sys.argv.index("--system-prompt-file") + 1] if "--system-prompt-file" in sys.argv else None
json.dump({"system": open(sp, encoding="utf-8").read() if sp else None, "argv": sys.argv[1:], "prompt": prompt, "has_key": "ANTHROPIC_API_KEY" in os.environ,
           "cwd": os.getcwd()}, open(log, "w"))
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "login":
    print(json.dumps({"type": "result", "is_error": True, "result": "Not logged in · Please run /login"}))
    sys.exit(1)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": "Tamam:\n```json\n{\"reels\": [{\"start_id\": 0, \"end_id\": 3}]}\n```"}))
'''


def _setup(tmp_path, monkeypatch, mode="ok"):
    exe = tmp_path / "claude"
    exe.write_text(FAKE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "log.json"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_MODE", mode)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-gizli")
    monkeypatch.setattr(settings, "claude_cli_path", str(exe))
    monkeypatch.setattr(settings, "backend_pref", "cli")
    claude_client._cli_supports.cache_clear()
    return log


def test_auto_backend_prefers_api_then_cli(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "backend_pref", "auto")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert settings.claude_backend == "cli"
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-x")
    assert settings.claude_backend == "api"
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "claude_cli_path", str(tmp_path / "yok"))
    assert settings.claude_backend == "off"


def test_cli_text_and_images(tmp_path, monkeypatch):
    log = _setup(tmp_path, monkeypatch)
    frames = tmp_path / "frames"
    frames.mkdir()
    img = frames / "t0012.0.jpg"
    img.write_bytes(b"\xff\xd8fakejpeg")

    data = call_json([text_block("Transkript burada"), text_block("Kare t=12.0 sn"), image_block(img)],
                     system="Sen editörsün")
    assert data == {"reels": [{"start_id": 0, "end_id": 3}]}

    rec = json.loads(log.read_text())
    argv = rec["argv"]
    assert argv[:3] == ["-p", "--output-format", "json"]
    assert rec["system"] == "Sen editörsün"
    assert argv[argv.index("--tools") + 1] == "Read"
    assert str(frames) in argv[argv.index("--add-dir") + 1:]
    assert f"[GÖRSEL: {img}]" in rec["prompt"] and "Read aracıyla" in rec["prompt"]
    assert rec["has_key"] is False  # abonelikle gitmesi için API anahtarı gizlenir
    assert os.path.realpath(rec["cwd"]) == os.path.realpath(frames)


def test_cli_text_only_disables_tools(tmp_path, monkeypatch):
    log = _setup(tmp_path, monkeypatch)
    call_json("sadece metin")
    argv = json.loads(log.read_text())["argv"]
    assert argv[argv.index("--tools") + 1] == "" and "--add-dir" not in argv


def test_cli_login_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, mode="login")
    try:
        call_json("x")
    except ClaudeError as e:
        assert "/login" in str(e)
    else:
        raise AssertionError("ClaudeError bekleniyordu")
