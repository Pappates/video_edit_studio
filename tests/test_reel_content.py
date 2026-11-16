"""Reel başına: 'İçerik ekle' (görseller yalnızca işaretli reels'e) ve otomatik yerleşim (sohbet / oyun)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from vedit import layout, media, pipeline  # noqa: E402
from vedit.config import settings  # noqa: E402
from vedit.reels import Reel  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"

MEDIA = {"duration": 100.0, "has_video": True, "has_audio": True, "width": 1920, "height": 1080, "fps": 30}
TALK = {"content": None, "cams": [[0, 0, 960, 1080], [960, 0, 960, 1080]], "kind": "talk", "source": "heuristic"}
GAME = {"content": [450, 90, 510, 660], "cams": [[1400, 0, 520, 540], [1400, 540, 520, 540]], "kind": "game",
        "source": "heuristic"}


def _job(tmp_path, monkeypatch, reels):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    job = pipeline.create_job(src, "reels")
    job.set(status="ready", media=MEDIA, cuts=[], reels=reels)
    return job


def test_old_jobs_with_visuals_count_as_enriched():
    assert Reel.from_dict({"start": 0, "end": 5, "visuals": [{"id": "v1"}]}).enrich is True
    assert Reel.from_dict({"start": 0, "end": 5}).enrich is False
    assert Reel.from_dict({"start": 0, "end": 5, "visuals": [{"id": "v1"}], "enrich": False}).enrich is False


def test_find_visuals_only_for_enriched_reels(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch, [{"id": "reel01", "start": 0, "end": 10, "enrich": False},
                                       {"id": "reel02", "start": 20, "end": 30, "enrich": True}])
    monkeypatch.setattr(settings, "backend_pref", "api")
    monkeypatch.setattr(pipeline, "Job", pipeline.Job)
    asked = []
    monkeypatch.setattr(pipeline.visuals, "find_for_reel", lambda tr, r, d, progress=None: asked.append(r["id"]) or ([], []))
    monkeypatch.setattr(pipeline.Job, "transcript", property(lambda self: {"segments": []}))
    d = pipeline.find_visuals(job.id, ["reel01", "reel02"]).data
    assert asked == ["reel02"] and d["status"] == "ready"
    assert [bool(r.get("enrich_done")) for r in d["reels"]] == [False, True]
    # Hiçbiri işaretli değilse açık bir mesaj
    job.set(reels=[{"id": "reel01", "start": 0, "end": 10, "enrich": False}])
    assert "İçerik ekle" in pipeline.find_visuals(job.id).data["message"]


def test_reel_layout_detects_per_reel_and_caches(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch, [{"id": "reel01", "start": 0, "end": 10}, {"id": "reel02", "start": 50, "end": 60}])
    calls = []

    def fake(video, a, b, fw, fh, use_claude=True):
        calls.append(a)
        return dict(TALK if a >= 50 else GAME)
    monkeypatch.setattr(layout, "detect_reel", fake)
    r1, r2 = (r.to_dict() for r in pipeline.Job(job.id).reels)
    assert pipeline.reel_layout(job, r1, "faces_top", False)[0] == "faces_top"
    mode, reg = pipeline.reel_layout(job, r2, "faces_top", False)
    assert mode == "talk" and reg["kind"] == "talk"
    # Sonuç reel'e yazıldı; aynı sınırlarla tekrar tespit edilmez
    r2 = next(r for r in pipeline.Job(job.id).data["reels"] if r["id"] == "reel02")
    pipeline.reel_layout(job, r2, "faces_top", False)
    assert calls == [0, 50]
    # Elle seçim otomatiği geçer
    assert pipeline.reel_layout(job, {**r2, "layout_kind": "game"}, "game_top", False)[0] == "game_top"
    r1 = next(r for r in pipeline.Job(job.id).data["reels"] if r["id"] == "reel01")
    assert pipeline.reel_layout(job, {**r1, "layout_kind": "talk"}, "faces_top", False)[0] == "talk"
    # "Hepsi: yüzler alt alta"
    assert pipeline.reel_layout(job, r1, "talk", False)[0] == "talk"


def test_render_uses_layout_and_visuals_per_reel(tmp_path, monkeypatch):
    vis = [{"id": "v1", "start": 2, "end": 5, "label": "X", "file": "v1_0.png", "enabled": True}]
    job = _job(tmp_path, monkeypatch, [
        {"id": "reel01", "start": 0, "end": 10, "enrich": False, "visuals": vis},       # oyun reel'i: görsel yok
        {"id": "reel02", "start": 50, "end": 60, "enrich": True},                        # içerikli, henüz aranmadı
    ])
    monkeypatch.setattr(settings, "backend_pref", "api")
    monkeypatch.setattr(layout, "detect_reel", lambda v, a, b, fw, fh, use_claude=True: dict(TALK if a >= 50 else GAME))
    monkeypatch.setattr(pipeline, "_mixed_audio", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.Job, "transcript", property(lambda self: None))
    monkeypatch.setattr(pipeline.exporters, "to_srt", lambda *a, **k: "")
    searched = []

    def fake_find(job_, ids, progress=None):
        searched.extend(ids)
        pipeline._patch_reel(job_, "reel02", lambda r: r.update(enrich_done=True, visuals=vis))
        return 1, []
    monkeypatch.setattr(pipeline, "_find_visuals_for", fake_find)
    plans = {}
    monkeypatch.setattr(pipeline.visuals, "overlay_plan", lambda v, *a, **k: [{"n": len(v)}])
    monkeypatch.setattr(pipeline.captions, "build_ass", lambda *a, **k: "")

    def fake_render(src, keeps, dst, **kw):
        plans[dst.stem[-6:]] = (kw["layout_plan"]["mode"], kw["overlays"])
        dst.write_bytes(b"x")
    monkeypatch.setattr(media, "render_keeps", fake_render)

    d = pipeline.render_reels(job.id, None, "faces_top", {"captions": None, "visuals": True}).data
    assert d["status"] == "done", d.get("error")
    assert searched == ["reel02"]                                   # yalnızca içerikli ve aranmamış olan
    assert plans["reel01"] == ("faces_top", [])                     # oyun: yüzler üstte, görsel konmaz
    assert plans["reel02"] == ("talk", [{"n": 1}])                  # sohbet: yüzler alt alta, görselli
