"""Reels altyazısı (ASS) ve konuya göre görseller. Ağ ve Claude gerektirmez (sahte yanıtlar)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from fixtures import make_transcript  # noqa: E402
from vedit import captions, exporters, media, visuals  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"

TOP = [
    (0.5, "Bugün tüm zamanların en iyi üç futbolcusunu sayıyorum."),
    (4.0, "Üçüncü sırada Cristiano Ronaldo var."),
    (7.5, "İkinci sırada Lionel Messi geliyor."),
    (11.0, "Birinci sırada ise Pele var takip etmeyi unutmayın."),
]


# --- altyazı -----------------------------------------------------------------
def test_ass_time_and_color():
    assert captions.ass_time(3661.234) == "1:01:01.23"
    assert captions.ass_color("#FFD400") == "&H0000D4FF"
    assert captions.tr_upper("istanbul ılık") == "İSTANBUL ILIK"


def test_bold_captions_highlight_every_word():
    tr = make_transcript([(0.5, "bir iki üç dört beş")])
    ass = captions.build_ass(tr, [(0.0, 10.0)], 1080, 1920, style="bold")
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(events) == 5                              # her kelime için bir vurgulu olay
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert all(r"\c&H0000D4FF" in e for e in events)
    assert "{\\c&H0000D4FF\\fscx112\\fscy112}ÜÇ{\\r}" in events[2]
    # 3 kelimelik satırlar: 4. kelimeyle yeni satır başlar
    assert "DÖRT" not in events[0] and "DÖRT" in events[3]


def test_captions_follow_cut_timeline():
    tr = make_transcript([(0.5, "kalan kelime"), (5.0, "kesilen kısım"), (9.0, "son söz")])
    keeps = [(0.0, 2.0), (8.5, 11.0)]
    ass = captions.build_ass(tr, keeps, 1080, 1920, style="minimal")
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert not any("kesilen" in e for e in events)
    last = events[-1].split(",")
    assert last[1] == captions.ass_time(2.0 + (9.0 - 8.5))   # 9.0 sn → 2.5 sn


def test_no_caption_style_only_visual_labels():
    items = [{"start": 1.0, "end": 3.0, "position": "top", "label": "Lionel Messi", "rank": 2, "box": (70, 150, 940, 700)}]
    ass = captions.build_ass(make_transcript([(0.5, "merhaba dünya")]), [(0, 5)], 1080, 1920, style=None,
                             visual_items=items)
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(events) == 2
    assert "Lionel Messi" in events[0] and "#2" in events[1]


# --- anları yerleştirme -----------------------------------------------------
def test_place_anchors_to_spoken_word_and_trims_overlaps():
    tr = make_transcript(TOP)
    segs = tr["segments"]
    moments = [
        {"seg_id": 1, "anchor": "Cristiano Ronaldo", "label": "Cristiano Ronaldo", "kind": "person",
         "wiki": "Cristiano Ronaldo", "rank": 3, "hold": 6},
        {"seg_id": 2, "anchor": "Lionel Messi", "label": "Lionel Messi", "kind": "person", "rank": 2, "hold": 3},
        {"seg_id": 3, "anchor": "Pele", "label": "Pelé", "kind": "person", "rank": 1, "hold": 3},
        {"seg_id": 99, "anchor": "yok", "label": "Geçersiz"},
        {"seg_id": 0, "anchor": "", "label": ""},
    ]
    vs = visuals.place(moments, segs, 0.0, 15.0)
    assert [v["rank"] for v in vs] == [3, 2, 1]
    ronaldo_word = next(w for w in segs[1]["words"] if w["word"] == "Cristiano")
    assert abs(vs[0]["start"] - (ronaldo_word["start"] - 0.15)) < 1e-6
    assert vs[0]["end"] <= vs[1]["start"]               # 6 sn'lik görsel bir sonrakinden önce biter
    pele_word = next(w for w in segs[3]["words"] if w["word"] == "Pele")
    assert abs(vs[2]["start"] - (pele_word["start"] - 0.15)) < 1e-6
    assert all(v["id"].startswith("v") and v["enabled"] for v in vs)


def test_overlay_plan_remaps_and_skips_missing(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    vs = [
        {"file": "a.jpg", "start": 10.5, "end": 12.0, "label": "A", "rank": 1},
        {"file": "b.jpg", "start": 3.0, "end": 4.0, "label": "B"},       # tamamen kesilen yerde
        {"file": "", "start": 11.0, "end": 12.0},                          # dosya yok
        {"file": "a.jpg", "start": 11.0, "end": 12.0, "enabled": False},
        {"file": "../../etc/passwd", "start": 10.0, "end": 12.0},          # klasör dışına çıkamaz
    ]
    keeps = [(0.0, 2.0), (10.0, 14.0)]
    items = visuals.overlay_plan(vs, keeps, 1080, 1920, "top", tmp_path,
                                 probe_fn=lambda p: {"width": 1600, "height": 900})
    assert len(items) == 1
    it = items[0]
    assert (it["start"], it["end"]) == (2.5, 4.0)
    x, y, w, h = it["box"]
    assert w <= 1080 and x >= 0 and (1080 - w) // 2 == x      # yatayda ortalı
    assert it["geo"]["scale"][0] % 2 == 0


def test_overlay_box_positions():
    assert visuals.overlay_box("full", 1080, 1920) == (0, 0, 1080, 1920)
    x, y, w, h = visuals.overlay_box("top", 1080, 1920)
    assert y + h < 1920 * 0.5
    x, y, w, h = visuals.overlay_box("top", 1920, 1080)       # yatay video: sağda kart
    assert x > 960


# --- kaynaklar -----------------------------------------------------------------
def test_search_wikipedia_parses_and_skips_disambiguation(monkeypatch):
    calls = []

    def fake(url, headers=None, timeout=20):
        calls.append(url)
        if "commons.wikimedia.org" in url:
            return {"query": {"pages": [{"title": "File:Messi 2023.jpg", "imageinfo": [{"extmetadata": {
                "Artist": {"value": "<a href='x'>Foto Grafçı</a>"}, "LicenseShortName": {"value": "CC BY-SA 4.0"}}}]}]}}
        return {"query": {"pages": [
            {"index": 2, "title": "Messi (anlam ayrımı)", "pageprops": {"disambiguation": ""},
             "thumbnail": {"source": "https://upload.example/d.jpg"}},
            {"index": 1, "title": "Lionel Messi", "fullurl": "https://tr.wikipedia.org/wiki/Lionel_Messi",
             "pageimage": "Messi_2023.jpg", "thumbnail": {"source": "https://upload.example/m.jpg"}},
            {"index": 3, "title": "Görselsiz sayfa"},
        ]}}

    monkeypatch.setattr(visuals, "_get_json", fake)
    c = visuals.search_wikipedia("Lionel Messi", "tr")
    assert len(c) == 1 and c[0]["url"] == "https://upload.example/m.jpg"
    assert c[0]["credit"] == "Foto Grafçı · CC BY-SA 4.0 · Wikimedia Commons"
    assert "pilicense=free" in calls[0] and calls[0].startswith("https://tr.wikipedia.org/")


def test_find_candidates_falls_back_to_english_and_skips_pexels_without_key(monkeypatch):
    monkeypatch.setattr(settings, "pexels_api_key", "")
    seen = []

    def fake_wiki(q, lang="tr", limit=3, width=1080):
        seen.append(lang)
        return [{"source": "wikipedia", "url": f"https://x/{lang}.jpg"}] if lang == "en" else []

    monkeypatch.setattr(visuals, "search_wikipedia", fake_wiki)
    v = visuals.new_visual(0, 2, label="Some Player", kind="person", wiki_lang="tr")
    cands, errors = visuals.find_candidates(v)
    assert seen == ["tr", "en"] and cands[0]["url"] == "https://x/en.jpg" and not errors
    _, errors = visuals.find_candidates(v, "pexels")
    assert errors and "PEXELS_API_KEY" in errors[0]


def test_choose_skips_failed_downloads(monkeypatch, tmp_path):
    def fake_download(url, stem, **kw):
        if "bad" in url:
            raise ValueError("Görsel değil")
        p = stem.with_suffix(".jpg")
        p.write_bytes(b"img")
        return p

    monkeypatch.setattr(visuals, "download", fake_download)
    v = visuals.new_visual(0, 2, label="X")
    v["candidates"] = [{"source": "wikipedia", "url": "https://bad", "credit": "a"},
                       {"source": "pexels", "url": "https://ok", "credit": "b · Pexels", "page": "https://p"}]
    visuals.choose(v, 0, tmp_path)
    assert v["choice"] == 1 and v["source"] == "pexels" and (tmp_path / v["file"]).exists()


# --- render (ffmpeg) -----------------------------------------------------------
@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("vis")
    video, img = d / "clip.mp4", d / "card.png"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=navy:s=640x360:r=30:d=8",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(video)], check=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=red:s=400x300", "-frames:v", "1", str(img)], check=True)
    return video, img


def _pixel(path, t, x, y):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", str(path),
                          "-frames:v", "1", "-vf", f"crop=2:2:{x // 2 * 2}:{y // 2 * 2}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return tuple(out[:3])


def test_render_with_overlay_and_burned_captions(clip, tmp_path):
    video, img = clip
    keeps = [(1.0, 6.0)]
    tr = make_transcript([(1.5, "bu bir altyazı denemesi")])
    vs = [{"file": img.name, "start": 2.0, "end": 4.5, "label": "Kırmızı Kart", "rank": 1}]
    items = visuals.overlay_plan(vs, keeps, 1080, 1920, "top", img.parent, media.probe)
    assert len(items) == 1
    ass = captions.build_ass(tr, keeps, 1080, 1920, style="bold", visual_items=items)
    out = media.render_keeps(video, keeps, tmp_path / "reel.mp4", vertical="crop", workdir=tmp_path,
                             overlays=items, ass_text=ass)
    info = media.probe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and abs(info["duration"] - 5.0) < 0.2
    x, y, w, h = items[0]["box"]
    cx, cy = x + w // 2, y + h // 2
    r, g, b = _pixel(out, 2.0, cx, cy)           # görsel ekranda (çıktıda 1.0–3.5 sn)
    assert r > 180 and g < 80 and b < 80
    r, g, b = _pixel(out, 4.6, cx, cy)           # görsel bitti → lacivert arka plan
    assert r < 60 and b > 60
    assert not list(tmp_path.glob("subs_*.ass"))  # geçici altyazı dosyası silindi


def test_remap_forward():
    keeps = [(0.0, 2.0), (10.0, 14.0)]
    assert exporters.remap_forward(1.0, keeps) == 1.0
    assert exporters.remap_forward(5.0, keeps) == 2.0     # kesilen yer → sonraki aralığın başı
    assert exporters.remap_forward(20.0, keeps) is None
