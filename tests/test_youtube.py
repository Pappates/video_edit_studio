import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import make_transcript  # noqa: E402
from vedit import youtube  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.backend_pref = "off"

TALK = [(0.5 + i * 15, f"Konu {i // 4} hakkında cümle {i} burada anlatılıyor") for i in range(16)]  # ~240 sn


def test_time_mapping():
    keeps = [(0, 10), (20, 30), (40, 50)]
    assert youtube.to_output(25, keeps) == 15
    assert youtube.to_output(15, keeps) == 10      # kesilen bölge → sonraki aralığın başı
    assert youtube.to_output(55, keeps) is None
    assert youtube.to_source(15, keeps) == 25
    assert youtube.fmt_ts(65) == "01:05" and youtube.fmt_ts(3725) == "1:02:05"
    assert youtube.fmt_ts(65, long=True) == "0:01:05"


def test_clean_hashtags_and_tags():
    h = youtube.clean_hashtags(["#Kahve Tarifi", "kahvetarifi", "#çay!", "#2024", "  ", "#Barista"] + [f"#t{i}" for i in range(30)])
    assert h[:3] == ["#KahveTarifi", "#çay", "#Barista"]
    assert len(h) == youtube.MAX_HASHTAGS
    tags = youtube.clean_tags([f"anahtar kelime {i}" for i in range(100)])
    assert sum(len(t) + 3 for t in tags) <= 500 + 3 and "#" not in "".join(tags)


def test_chapter_rules():
    yt = {"chapters": [{"start": 5.0, "title": "Giriş"}, {"start": 9.0, "title": "Çok yakın"},
                       {"start": 60.0, "title": "Kurulum"}, {"start": 120.0, "title": "Sonuç"},
                       {"start": 236.0, "title": "Son saniyeler"}]}
    keeps = [(0.0, 240.0)]
    chs, warns = youtube.output_chapters(yt, keeps)
    assert [c["title"] for c in chs] == ["Giriş", "Kurulum", "Sonuç"]
    assert chs[0]["t"] == 0.0
    assert any("10 sn" in w and "eklenmedi" in w for w in warns)
    text, _ = youtube.compose({**yt, "intro": "Açıklama.", "hashtags": ["#a", "#b"]}, keeps)
    assert "00:00 Giriş\n01:00 Kurulum\n02:00 Sonuç" in text and text.endswith("#a #b")
    few, w2 = youtube.output_chapters({"chapters": [{"start": 0, "title": "A"}, {"start": 50, "title": "B"}]}, keeps)
    assert len(few) == 2 and any("en az 3" in w for w in w2)


def _fake_response(content, system="", **kw):
    text = content if isinstance(content, str) else content[0]["text"]
    assert "#0 [00:00]" in text and "Konu 0" in text
    return {"titles": ["Başlık bir", "Başlık iki", "Başlık üç"],
            "intro": "Bu videoda dört konu var.\n\nİzlediğiniz için teşekkürler.",
            "chapters": [{"block_id": 0, "title": "Giriş"}, {"block_id": 4, "title": "İkinci konu"},
                         {"block_id": 8, "title": "Üçüncü konu"}, {"block_id": 12, "title": "Kapanış"},
                         {"block_id": 999, "title": "uydurma"}],
            "hashtags": ["#Kahve", "#Barista Eğitimi", "#evde kahve"],
            "tags": ["kahve", "barista", "#espresso"]}


def test_generate_maps_blocks(monkeypatch):
    monkeypatch.setattr(youtube, "call_json", _fake_response)
    tr = make_transcript(TALK)
    yt = youtube.generate(tr, [(0.0, tr["duration"])])
    assert [c["title"] for c in yt["chapters"]] == ["Giriş", "İkinci konu", "Üçüncü konu", "Kapanış"]
    assert yt["chapters"][1]["start"] == tr["segments"][4]["start"]  # blok 4 = segment 4 (her biri 15 sn)
    assert yt["hashtags"] == ["#Kahve", "#BaristaEğitimi", "#evdekahve"] and yt["tags"][-1] == "espresso"


def test_pipeline_youtube_mode_and_edit_bridge(monkeypatch, tmp_path):
    import subprocess
    from vedit import pipeline

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(youtube, "call_json", _fake_response)
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x90:r=10:d=242",
                    "-f", "lavfi", "-i", "sine=d=242", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                    "-shortest", str(vid)], check=True)
    tr = make_transcript(TALK, duration=242)

    # 1) YouTube sekmesi: yüklenen video olduğu gibi
    job = pipeline.create_job(vid, "youtube", "Bölüm 5.mp4", {"diarize": False})
    (job.dir / "transcript.json").write_text(json.dumps(tr), encoding="utf-8")
    pipeline.analyze(job.id)
    view = pipeline.youtube_view(job.id)
    assert pipeline.Job(job.id).data["status"] == "ready"
    assert view["youtube"]["chapters"][1]["t"] == tr["segments"][4]["start"]
    assert view["text"].count("\n0") >= 3 and "00:00 Giriş" in view["text"]

    # Kullanıcı bir bölümün yayın zamanını ve hashtag'leri düzenler
    chs = view["youtube"]["chapters"]
    chs[1]["t"] = 70.0
    v2 = pipeline.update_youtube(job.id, {"chapters": chs, "hashtags": ["#yeni etiket", "#Kahve"]})
    assert "01:10 İkinci konu" in v2["text"] and v2["text"].endswith("#yenietiket #Kahve")

    # 2) Video Edit işinden: kesimler değişince bölüm zamanları kendiliğinden kayar
    ej = pipeline.create_job(vid, "edit", "Ham.mp4", {"claude": False, "silence": False, "diarize": False})
    (ej.dir / "transcript.json").write_text(json.dumps(tr), encoding="utf-8")
    pipeline.analyze(ej.id)
    pipeline.update_cuts(ej.id, [])
    pipeline.generate_youtube(ej.id)
    before = pipeline.youtube_view(ej.id)["youtube"]["chapters"][2]["t"]
    seg = tr["segments"]
    pipeline.update_cuts(ej.id, [{"start": seg[1]["start"], "end": seg[3]["end"], "kind": "repeat",
                                  "reason": "test", "enabled": True}])
    after_view = pipeline.youtube_view(ej.id)
    removed = seg[3]["end"] - seg[1]["start"]
    assert abs(before - after_view["youtube"]["chapters"][2]["t"] - removed) < 0.01
    assert pipeline.Job(ej.id).data["status"] == "ready"


def test_youtube_mode_survives_claude_failure(monkeypatch, tmp_path):
    import subprocess
    from vedit import pipeline
    from vedit.claude_client import ClaudeError

    def boom(*a, **k):
        raise ClaudeError("Claude kapalı")

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(youtube, "call_json", boom)
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x90:r=10:d=20",
                    "-c:v", "libx264", "-preset", "ultrafast", str(vid)], check=True)
    job = pipeline.create_job(vid, "youtube", "x.mp4", {"diarize": False})
    (job.dir / "transcript.json").write_text(json.dumps(make_transcript(TALK[:1])), encoding="utf-8")
    pipeline.analyze(job.id)
    j = pipeline.Job(job.id)
    assert j.data["status"] == "ready" and "Claude kapalı" in j.data["youtube_error"]


def test_extra_not_duplicated(monkeypatch):
    def fake(content, system="", **kw):
        r = _fake_response(content)
        r["intro"] = "Bu videoda kahve var.\n\n📸 Instagram: @evdebarista\n\nAbone olun."
        return r
    monkeypatch.setattr(youtube, "call_json", fake)
    tr = make_transcript(TALK)
    yt = youtube.generate(tr, [(0.0, tr["duration"])], extra="📸 Instagram: @evdebarista")
    text, _ = youtube.compose(yt, [(0.0, tr["duration"])])
    assert text.count("@evdebarista") == 1 and "Abone olun." in text
