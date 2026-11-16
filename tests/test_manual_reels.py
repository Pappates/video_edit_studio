"""Elle reel işaretleme: kurgu şeridi (S ile böl), kayıt, hazırlama (başlık + kısaltma), küçük resim şeridi."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from fixtures import make_transcript  # noqa: E402
from vedit import pipeline, reels  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"


def _job(tmp_path, monkeypatch, opts=None):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    return pipeline.create_job(src, "reels", options=opts or {"manual": True})


def test_save_manual_reels_keeps_existing_and_numbers_new(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch)
    job.set(reels=[{"id": "reel01", "start": 10.0, "end": 40.0, "title": "Frekans", "tight_level": "light",
                    "visuals": [{"id": "v1"}], "tighten": [{"start": 20, "end": 22, "kind": "slow", "reason": "x"}]}])
    out = pipeline.save_manual_reels(job.id, [
        {"start": 100.0, "end": 160.0},                       # yeni
        {"start": 10.0, "end": 40.0, "id": "reel01"},         # aynen
        {"start": 200.0, "end": 200.5},                       # 1 sn'den kısa → atılır
    ]).data["reels"]
    assert [r["id"] for r in out] == ["reel01", "reel02"]
    assert out[0]["title"] == "Frekans" and out[0]["visuals"] == [{"id": "v1"}] and out[0]["tighten"]
    assert out[1]["source"] == "user" and (out[1]["start"], out[1]["end"]) == (100.0, 160.0)
    # Sınırı değişen reel'in başlığı/görselleri korunur, kısaltma notu düşülür
    out = pipeline.save_manual_reels(job.id, [{"start": 8.0, "end": 45.0, "id": "reel01"}]).data["reels"]
    assert len(out) == 1 and out[0]["end"] == 45.0 and out[0]["title"] == "Frekans"
    assert "Sınır değişti" in out[0]["tight_note"]


def test_manual_mode_analysis_skips_auto_split(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch, {"manual": True, "claude": False, "diarize": False, "max_reels": 1})
    tr = make_transcript([(0.5, "birinci reel başlıyoruz bugün kahve"), (5.0, "kahve çok güzel bir içecek"),
                          (30.0, "ikinci reel çay anlatıyorum"), (35.0, "çay da güzel")])
    monkeypatch.setattr(pipeline.media, "probe", lambda p: {"duration": 50.0, "has_video": False, "has_audio": False,
                                                           "width": 0, "height": 0, "fps": 30})
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: tr)
    called = []
    monkeypatch.setattr(pipeline, "split_reels", lambda *a, **k: called.append(1) or ([], "claude"))
    monkeypatch.setattr(pipeline, "_reels_incremental", lambda *a, **k: called.append("inc"))
    # Kullanıcı analiz sürerken işaretlemiş olsun
    pipeline.save_manual_reels(job.id, [{"start": 0.3, "end": 25.0}])
    d = pipeline.analyze(job.id).data
    assert d["status"] == "ready" and not called
    assert [(r["id"], r["start"]) for r in d["reels"]] == [("reel01", 0.3)]
    assert "1 reel işaretli" in d["message"]


def test_prepare_reels_describes_and_tightens(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch, {"manual": True, "claude": True})
    tr = make_transcript([(0.5, "bugün frekans oynuyoruz"), (5.0, "sola doğru biraz daha"), (12.0, "başardık güzel")])
    (job.dir / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    job.set(status="ready", media={"duration": 20.0, "has_video": False, "has_audio": False}, cuts=[])
    pipeline.save_manual_reels(job.id, [{"start": 0.3, "end": 14.0}])
    monkeypatch.setattr(settings, "backend_pref", "api")
    monkeypatch.setattr(reels, "call_json", lambda c, system="", **k: {
        "title": "Frekans", "hook": "bugün frekans oynuyoruz", "summary": "oyun", "caption": "Kim kazandı?",
        "hashtags": ["#frekans"]})
    monkeypatch.setattr(pipeline.tighten, "call_json", lambda c, system="", **k: {"cuts": [], "summary": "tamam"})
    monkeypatch.setattr(pipeline.media, "quiet_intervals", lambda *a, **k: [])
    d = pipeline.prepare_reels(job.id, None, "light").data
    r = d["reels"][0]
    assert d["status"] == "ready" and "hazırlandı" in d["message"]
    assert r["title"] == "Frekans" and r["caption"] == "Kim kazandı?" and r["hashtags"] == ["#frekans"]
    assert r["tight_level"] == "light" and (r["start"], r["end"]) == (0.3, 14.0)   # sınırlar korunur


def test_manual_endpoint_allowed_during_transcription_not_during_tighten(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch)
    app = importlib.import_module("server.app")
    from fastapi import HTTPException
    job.set(status="running", stage="transcribe")
    res = app.put_manual_reels(job.id, [{"start": 1.0, "end": 9.0}])
    assert res["reels"][0]["id"] == "reel01"
    job.set(status="running", stage="tighten")
    with pytest.raises(HTTPException) as e:
        app.put_manual_reels(job.id, [{"start": 1.0, "end": 9.0}])
    assert e.value.status_code == 409


def test_timeline_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=s=640x360:r=30:d=12", "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-c:a", "aac", "-shortest", str(src)], check=True)
    job = pipeline.create_job(src, "reels")
    tl = pipeline.build_timeline_assets(job.id).data["timeline"]
    assert tl["state"] == "ready" and tl["wave"] and tl["tiles"]
    from vedit import media
    info = media.probe(job.dir / "timeline" / tl["tiles"][0])
    assert info["height"] == pipeline.THUMB_H and info["width"] % pipeline.THUMB_W == 0
    assert info["width"] // pipeline.THUMB_W >= 2          # 12 sn / 5 sn → en az 2-3 küçük resim
