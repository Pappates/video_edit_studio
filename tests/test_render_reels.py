"""render_reels uçtan uca: kısaltılmış reel + görsel + gömülü altyazı + Instagram açıklama dosyası.

(Bir isim çakışması — yerel 'captions' listesinin captions modülünü gölgelemesi — yalnızca bu akışta
ortaya çıkmıştı; birim testleri yakalamamıştı.)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from fixtures import make_transcript  # noqa: E402
from vedit import media, pipeline, tighten  # noqa: E402
from vedit.config import settings  # noqa: E402
from vedit.detector import Cut  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"


def test_render_reels_with_captions_visuals_and_tighten(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    src = tmp_path / "shoot.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=navy:s=640x360:r=30:d=20",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    job = pipeline.create_job(src, "reels")
    tr = make_transcript([(1.0, "birinci cümle burada"), (5.0, "kategori yazıyorum bekle"),
                          (9.0, "ikinci cümle Messi geliyor"), (13.0, "son cümle bitti")], duration=20.0)
    (job.dir / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")

    vis_dir = job.dir / "visuals"
    vis_dir.mkdir()
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=300x400",
                    "-frames:v", "1", str(vis_dir / "v1_0.png")], check=True)
    reel = {"id": "reel01", "start": 0.5, "end": 16.0, "title": "Test", "caption": "Açıklama", "hashtags": ["#test"],
            "enabled": True, "clean": True,
            "tighten": [Cut(4.9, 8.9, "logistics", "kategori yazma").to_dict()],
            "visuals": [{"id": "v1", "start": 9.0, "end": 11.5, "label": "Lionel Messi", "rank": 1,
                         "file": "v1_0.png", "enabled": True}]}
    job.set(status="ready", media=media.probe(src), cuts=[], reels=[reel])

    pipeline.render_reels(job.id, vertical="crop",
                          style={"captions": "bold", "caption_pos": "bottom", "visuals": True, "visual_pos": "top"})

    done = pipeline.Job(job.id).data
    assert done["status"] == "done", done.get("error") or done.get("message")
    out = job.out_dir / done["outputs"]["reels"]["reel01"]["video"]
    info = media.probe(out)
    expected = sum(b - a for a, b in tighten.reel_keeps(reel, []))
    assert (info["width"], info["height"]) == (1080, 1920)
    assert abs(info["duration"] - expected) < 0.2           # kısaltma uygulandı (15.5 → 11.5 sn)
    md = (job.out_dir / done["outputs"]["captions"]).read_text(encoding="utf-8")
    assert "reel01" in md and "#test" in md
