import os
import subprocess
import sys
import xml.dom.minidom
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from fixtures import SHOOT, make_transcript  # noqa: E402
from vedit import detector, exporters, media, reels  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"  # testler gerçek Claude çağırmasın


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """50 sn test videosu: 25. sn'de belirgin sahne değişimi (renk), sesli."""
    p = tmp_path_factory.mktemp("v") / "shoot.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=navy:s=640x360:r=30:d=25",
        "-f", "lavfi", "-i", "color=c=orange:s=640x360:r=30:d=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=50",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(p),
    ], check=True)
    return p


def test_probe(video):
    info = media.probe(video)
    assert abs(info["duration"] - 50) < 0.2 and info["width"] == 640 and info["has_audio"]


def test_render_keeps_duration(video, tmp_path):
    keeps = [(1.0, 5.0), (10.0, 12.5), (30.0, 33.0)]
    out = media.render_keeps(video, keeps, tmp_path / "edit.mp4", workdir=tmp_path)
    assert abs(media.probe(out)["duration"] - 9.5) < 0.15


def test_render_vertical(video, tmp_path):
    out = media.render_keeps(video, [(0, 3)], tmp_path / "v.mp4", vertical="crop", workdir=tmp_path)
    info = media.probe(out)
    assert (info["width"], info["height"]) == (1080, 1920)
    out2 = media.render_keeps(video, [(0, 2)], tmp_path / "b.mp4", vertical="blur", workdir=tmp_path)
    assert media.probe(out2)["height"] == 1920


def test_scene_and_frame(video, tmp_path):
    scenes = media.scene_changes(video)
    assert any(24 < t < 26 for t in scenes), scenes
    f = media.extract_frame(video, 10, tmp_path / "f.jpg")
    assert f.stat().st_size > 500


def test_exporters(video, tmp_path):
    tr = make_transcript(SHOOT, duration=50)
    cuts = detector.detect_all(tr, use_claude=False)
    keeps = detector.keep_ranges(cuts, 50)
    info = media.probe(video)
    files = exporters.write_all(tmp_path, "shoot_edit", keeps, info, video, tr)

    edl = files["edl"].read_text()
    assert edl.count("* FROM CLIP NAME") == len(keeps)
    doc = xml.dom.minidom.parse(str(files["xml"]))
    assert len(doc.getElementsByTagName("clipitem")) == 2 * len(keeps)
    srt = files["srt"].read_text()
    assert "baştan" not in srt.lower() and "şurayı" not in srt.lower()
    assert "Merhaba" in srt
    # Zaman eşlemesi: ikinci kalan aralığın başı = birinci aralığın uzunluğu
    a0, b0 = keeps[0]
    assert abs(exporters.remap(keeps[1][0], keeps) - (b0 - a0)) < 1e-6


def test_heuristic_reels(video):
    tr = make_transcript([
        (1.0, "Selam arkadaşlar bugün size kahve demlemenin üç sırrını anlatacağım."),
        (6.0, "Birincisi taze çekilmiş kahve kullanın."),
        (11.0, "İkincisi su sıcaklığı doksan iki derece olsun."),
        (16.0, "Üçüncüsü demleme süresine dikkat edin takip etmeyi unutmayın."),
        (27.0, "Merhaba bugün evde soğuk kahve nasıl yapılır onu gösteriyorum."),
        (32.0, "Kahveyi soğuk suyla karıştırıp on iki saat bekletin."),
        (38.0, "Sonra süzün ve buzla servis edin afiyet olsun."),
    ], duration=50)
    out = reels.split_heuristic(video, tr, media.probe(video))
    assert len(out) == 2
    assert out[0].end <= 26 and out[1].start >= 20


def test_pipeline_end_to_end(video, tmp_path, monkeypatch):
    """Transkript önbellekten okunur (Whisper gerekmez); analiz → onay → render → reels."""
    import json
    from vedit import pipeline
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    job = pipeline.create_job(video, "reels", "Çekim 1.mp4", {"claude": False})
    (job.dir / "transcript.json").write_text(json.dumps(make_transcript(SHOOT, duration=50)), encoding="utf-8")

    pipeline.analyze(job.id)
    j = pipeline.Job(job.id)
    assert j.data["status"] == "ready", j.data.get("error")
    assert j.data["cuts"] and j.data["reels"]

    # kullanıcı bir kesimi kapatır
    cuts = j.data["cuts"]
    cuts[0]["enabled"] = False
    pipeline.update_cuts(job.id, cuts)
    assert pipeline.Job(job.id).cuts[0].enabled is False

    pipeline.render_edit(job.id)
    j = pipeline.Job(job.id)
    assert j.data["status"] == "done", j.data.get("error")
    assert (j.out_dir / j.data["outputs"]["video"]).exists()
    assert {"edl", "xml", "srt", "json"} <= set(j.data["outputs"])

    pipeline.render_reels(job.id, vertical="crop")
    j = pipeline.Job(job.id)
    assert j.data["status"] == "done", j.data.get("error")
    first = next(iter(j.data["outputs"]["reels"].values()))
    assert media.probe(j.out_dir / first["video"])["height"] == 1920
