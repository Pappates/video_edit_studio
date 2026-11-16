"""Kod incelemesinde bulunan hataların düzeltmeleri için testler."""
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
from vedit import captions, markers, media, pipeline, reels, tighten, visuals  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"


def _job(tmp_path, monkeypatch, mode="reels"):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    return pipeline.create_job(src, mode)


# 1) Duraklamayla bölünen reel'in ikinci parçası önceki numarayı devralmamalı
def test_split_reel_part_does_not_inherit_marker():
    shoot = [(0.5, "birinci reel başlıyoruz")]
    shoot += [(3.0 + j * 4, f"kahve anlatıyorum bölüm {j} bir iki üç") for j in range(5)]      # 3–22 sn
    shoot += [(27.0 + j * 4, f"devamında süt anlatıyorum bölüm {j} bir iki") for j in range(4)]  # 5 sn duraklama sonrası
    tr = make_transcript(shoot)
    out, _ = reels.split_reels("x.mp4", tr, {"duration": 50.0, "has_video": False})
    assert len(out) == 2
    assert out[0].marker == "1. reel" and out[0].enabled and not out[0].retake
    assert out[1].marker == "" and out[1].enabled


# 2) Arka plan işi sürerken reels/görsel düzenlemesi reddedilir
def test_reel_and_visual_edits_rejected_while_running(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch)
    app = importlib.import_module("server.app")       # data_dir artık geçici klasör
    from fastapi import HTTPException
    job.set(status="running", reels=[{"id": "reel01", "start": 0, "end": 10, "visuals": [{"id": "v1"}]}])
    for call in (lambda: app.put_reels(job.id, []),
                 lambda: app.visual_next(job.id, "reel01", "v1"),
                 lambda: app.visual_search(job.id, "reel01", "v1", "x", "auto")):
        with pytest.raises(HTTPException) as e:
            call()
        assert e.value.status_code == 409
    assert pipeline.Job(job.id).data["reels"][0]["visuals"] == [{"id": "v1"}]


def test_background_endpoints_mark_busy_before_thread_starts(tmp_path, monkeypatch):
    """Arayüzün ilk sorgusu iş parçacığından önce gelse de iş 'meşgul' görünmeli (yoksa takip bırakılır)."""
    job = _job(tmp_path, monkeypatch)
    app = importlib.import_module("server.app")
    job.set(status="done", reels=[{"id": "reel01", "start": 0, "end": 10}])
    monkeypatch.setattr(pipeline, "run_in_thread", lambda *a, **k: None)   # iş parçacığı hiç başlamasın
    app.render_reels(job.id, ["reel01"], "crop", None)
    d = pipeline.Job(job.id).data
    assert d["status"] == "rendering" and d["resume_status"] == "done"
    # Bir sonraki iş (ör. görsel bulma) bittiğinde 'done'a döner, 'rendering'de kalmaz
    job.set(status="done")
    monkeypatch.setattr(settings, "backend_pref", "off")
    app.find_visuals(job.id, ["reel01"])
    assert pipeline.Job(job.id).data["status"] == "running"
    pipeline.find_visuals(job.id, ["reel01"])          # Claude yok → hata mesajıyla biter
    assert pipeline.Job(job.id).data["status"] == "done"


def test_reel_edits_allowed_while_rendering(tmp_path, monkeypatch):
    """Render reels'i başta okur, geri yazmaz: render sürerken başlık/kutucuk düzenlemesi kaydedilir (409 değil)."""
    job = _job(tmp_path, monkeypatch)
    app = importlib.import_module("server.app")
    job.set(status="rendering", stage="render", reels=[{"id": "reel01", "start": 0, "end": 10, "title": "a"}])
    out = app.put_reels(job.id, [{"id": "reel01", "start": 0, "end": 10, "title": "b"}])
    assert out["reels"][0]["title"] == "b" and out["status"] == "rendering"


# 3) Elle yazılan aramada önce Wikipedia; varlıkların stok sorgusu korunur
def test_manual_search_prefers_wikipedia(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "pexels_api_key", "k")
    calls = []
    monkeypatch.setattr(visuals, "search_wikipedia", lambda q, lang="tr", **k: calls.append(("wiki", q)) or
                        [{"source": "wikipedia", "url": "https://w/1.jpg"}])
    monkeypatch.setattr(visuals, "search_pexels", lambda q, **k: calls.append(("pexels", q)) or
                        [{"source": "pexels", "url": "https://p/1.jpg"}])
    v = visuals.new_visual(0, 2, kind="concept")
    v["wiki"] = v["stock"] = "Arda Güler"
    cands, _ = visuals.find_candidates(v, prefer_wiki=True)
    assert cands[0]["source"] == "wikipedia" and calls[0][0] == "wiki"

    job = _job(tmp_path, monkeypatch)
    person = {**visuals.new_visual(0, 2, label="Messi", kind="person", wiki="Lionel Messi", stock="football player"),
              "id": "v1"}
    job.set(reels=[{"id": "reel01", "start": 0, "end": 10, "visuals": [person]}])
    monkeypatch.setattr(visuals, "choose", lambda v, i, d: v.update(file="", choice=i))
    calls.clear()
    pipeline.visual_search(job.id, "reel01", "v1", "Leo Messi", "auto")
    saved = pipeline.Job(job.id).data["reels"][0]["visuals"][0]
    assert saved["wiki"] == "Leo Messi" and saved["stock"] == "football player"
    assert calls == [("wiki", "Leo Messi"), ("pexels", "football player")]


# 4) İşaretten önceki içerik kelimeleri kesilmez; dolgu sözleri kesilir
def test_marker_keeps_content_words_before_it():
    tr = make_transcript([(0.0, "Messi en iyisi ikinci reel"), (10.0, "ay olmadı birinci reel tekrar")])
    m = {x["seg_id"]: x for x in markers.find_markers(tr["segments"])}
    assert m[0]["text"] == "ikinci reel" and not m[0]["whole"]
    assert m[1]["text"] == "ay olmadı birinci reel tekrar" and m[1]["whole"]


# 5) Dosya kilidi: okuma hiç başarısız olmaz (500 hatasının sebebi buydu). Windows dosyayı kapandıktan
#    kısa süre sonra bıraktığı için yazmanın replace adımı nadiren yeniden dener — bu beklenen yedek yol.
def test_json_lock_reads_never_retry(tmp_path, monkeypatch):
    import traceback
    p = tmp_path / "x.json"
    pipeline.write_json(p, {"a": 1})
    sleeps = []
    real_sleep = pipeline.time.sleep
    monkeypatch.setattr(pipeline.time, "sleep",
                        lambda s: (sleeps.append(traceback.extract_stack()[-2].name), real_sleep(s)))
    import threading
    stop = threading.Event()

    def writer():
        for i in range(200):
            pipeline.write_json(p, {"a": i, "pad": "x" * 20000})
        stop.set()

    t = threading.Thread(target=writer)
    t.start()
    while not stop.is_set():
        assert "a" in pipeline.read_json(p)
    t.join()
    assert "read_json" not in sleeps                   # okuma hiç yeniden denemedi
    assert pipeline.read_json(p)["a"] == 199            # tüm yazmalar sonunda başarılı


# 6) Görseller render'dan önce bir kez ölçeklenir; geçici dosyalar silinir
def test_overlays_are_prescaled_once(tmp_path):
    img = tmp_path / "card.png"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=1600x900",
                    "-frames:v", "1", str(img)], check=True)
    items = visuals.overlay_plan([{"file": "card.png", "start": 0.0, "end": 2.0, "label": "x"}], [(0.0, 3.0)],
                                 1080, 1920, "top", tmp_path, media.probe)
    pre = media.prescale_overlays(items, tmp_path)
    assert pre[0]["prescaled"]
    info = media.probe(pre[0]["path"])
    assert (info["width"], info["height"]) == (items[0]["geo"]["w"], items[0]["geo"]["h"])
    chain, _ = media.overlay_graph("base", pre, 1)
    assert "scale=" not in chain and "pad=" not in chain


# 7) Eklenemeyen görseller bildirilir
def test_skipped_visuals_are_reported(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    skipped = []
    items = visuals.overlay_plan([{"file": "yok.png", "start": 0, "end": 2, "label": "Kayıp"},
                                  {"file": "bad.png", "start": 0, "end": 2, "label": "Bozuk"},
                                  {"file": "", "start": 0, "end": 2, "label": "Bulunamadı"}],
                                 [(0.0, 3.0)], 1080, 1920, "top", tmp_path, media.probe, skipped=skipped)
    assert items == []
    assert any("Kayıp (dosya yok)" in s for s in skipped) and any(s.startswith("Bozuk (okunamadı") for s in skipped)
    assert len(skipped) == 2


# 8) Kullanılmayan görsel dosyaları silinir
def test_orphan_visual_files_cleaned(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch)
    for n in ("a.jpg", "b.jpg", "c_up.png"):
        (job.visuals_dir / n).write_bytes(b"x")
    job.set(reels=[{"id": "reel01", "start": 0, "end": 10, "visuals": [{"id": "v1", "file": "a.jpg"}]},
                   {"id": "reel02", "start": 10, "end": 20, "visuals": [{"id": "v2", "file": "c_up.png"}]}])
    assert pipeline.cleanup_visual_files(job) == 1
    assert sorted(p.name for p in job.visuals_dir.iterdir()) == ["a.jpg", "c_up.png"]


# 9) Sıkı sessizlik kesimi sınırdan taşan kelimeyi kırpmaz
def test_tight_silence_does_not_clip_straddling_word():
    segs = [{"id": 0, "start": 12.10, "end": 12.60, "text": "kelime",
             "words": [{"start": 12.10, "end": 12.60, "word": "kelime"}]},
            {"id": 1, "start": 14.00, "end": 14.50, "text": "sonra",
             "words": [{"start": 14.00, "end": 14.50, "word": "sonra"}]}]
    cuts = tighten.tight_silences(segs, 12.30, 16.0, "aggressive")
    assert cuts and all(c.start >= 12.60 for c in cuts)   # kesim kelimenin bitişinden önce başlamaz


# 10) Büyük harfte yabancı isimler doğru yazılır
def test_caption_upper_foreign_names():
    foreign = captions.foreign_names([{"label": "Lionel Messi", "wiki_lang": "en"},
                                      {"label": "Arda Güler", "wiki_lang": "tr"}])
    assert foreign == {"lionel", "messi"}
    assert captions.upper_word("Lionel", foreign) == "LIONEL"
    assert captions.upper_word("Messi'nin,", foreign) == "MESSI'NİN,"
    assert captions.upper_word("istanbul", foreign) == "İSTANBUL"
    assert captions.upper_word("Güler", foreign) == "GÜLER"
    tr = make_transcript([(0.5, "Lionel Messi bir efsane")])
    ass = captions.build_ass(tr, [(0, 5)], 1080, 1920, style="bold", foreign=foreign)
    assert "LIONEL" in ass and "LİONEL" not in ass and "EFSANE" in ass
