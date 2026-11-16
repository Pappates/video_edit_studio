"""Dikey yerleşim: oyun/ekran + kameralar 9:16'ya dizilir (sentetik Riverside düzeniyle)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ANTHROPIC_API_KEY"] = ""

from vedit import captions, layout, media, visuals  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"

FW, FH = 1920, 1080
# Riverside benzeri: solda ekran (koyu mor) ve ortasında oyun kartı (kırmızı), sağda iki kamera alt alta
CONTENT = [440, 80, 520, 654]          # oyun kartı bölgesi
CAMS = [[1400, 0, 520, 540], [1400, 540, 520, 540]]
REGIONS = {"content": CONTENT, "cams": CAMS}


@pytest.fixture(scope="module")
def riverside(tmp_path_factory):
    p = tmp_path_factory.mktemp("lay") / "rv.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=0x1a1033:s={FW}x{FH}:r=30:d=3",
        "-f", "lavfi", "-i", "color=c=red:s=400x560:r=30:d=3",
        "-f", "lavfi", "-i", "color=c=0x00c000:s=520x540:r=30:d=3",
        "-f", "lavfi", "-i", "color=c=0x0040ff:s=520x540:r=30:d=3",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=3",
        "-filter_complex", "[0][1]overlay=500:120[a];[a][2]overlay=1400:0[b];[b][3]overlay=1400:540[v]",
        "-map", "[v]", "-map", "4:a", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(p),
    ], check=True)
    return p


def _px(path, t, x, y):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", str(path),
                          "-frames:v", "1", "-vf", f"crop=2:2:{x // 2 * 2}:{y // 2 * 2}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return tuple(out[:3])


def test_fit_aspect_contains_box_and_stays_in_frame():
    r = layout.fit_aspect(CONTENT, 1080 / 1359, FW, FH)
    x, y, w, h = r
    assert x <= CONTENT[0] and y <= CONTENT[1] and x + w >= CONTENT[0] + CONTENT[2] - 2 and y + h >= CONTENT[1] + CONTENT[3] - 2
    assert abs(w / h - 1080 / 1359) < 0.02 and x >= 0 and y >= 0 and x + w <= FW and y + h <= FH


@pytest.mark.parametrize("mode", ["faces_top", "game_top", "game_full"])
def test_plan_tiles_cover_canvas(mode):
    p = layout.plan(mode, FW, FH, REGIONS)
    assert p["mode"] == mode and len(p["tiles"]) == 3
    gx, gy, gw, gh = p["tiles"][0]["dst"]
    for t in p["tiles"]:
        x, y, w, h = t["dst"]
        assert 0 <= x and x + w <= 1080 and 0 <= y and y + h <= 1920
    if mode == "faces_top":
        assert gy > 0 and gy + gh == 1920 and all(t["dst"][1] == 0 for t in p["tiles"][1:])
        assert p["tiles"][1]["dst"][2] + p["tiles"][2]["dst"][2] == 1080          # yüzler yan yana tam genişlik
    if mode == "game_top":
        assert gy == 0 and all(t["dst"][1] == gh for t in p["tiles"][1:])
    if mode == "game_full":
        # Oyun büyük (üstte), yüzler altta çerçeveli küçük kutular — oyunun üstüne binmez
        assert gw == 1080 and gh > 1920 * 0.6 and all(t["border"] for t in p["tiles"][1:])
        assert all(t["dst"][1] >= gy + gh for t in p["tiles"][1:])
    # Altyazı ve görsel alanı yüzlerin üstüne binmez
    face_tops = [t["dst"][1] for t in p["tiles"][1:]]
    cap_y = 1920 - p["caption_margin"]
    assert not any(t["dst"][1] < cap_y < t["dst"][1] + t["dst"][3] for t in p["tiles"][1:]) or mode == "faces_top"
    assert p["game_area"][1] >= (min(face_tops) + p["tiles"][1]["dst"][3] if mode == "faces_top" else 0)


def test_no_cameras_falls_back_to_full_game():
    p = layout.plan("faces_top", FW, FH, {"content": CONTENT, "cams": []})
    assert p["mode"] == "game_full" and len(p["tiles"]) == 1


def test_one_camera_makes_wide_strip():
    p = layout.plan("faces_top", FW, FH, {"content": CONTENT, "cams": CAMS[:1]})
    assert p["tiles"][1]["dst"][2] == 1080 and p["tiles"][1]["dst"][3] <= 1920 * 0.42 + 2


def test_render_faces_top_shows_both_faces_and_game(riverside, tmp_path):
    p = layout.plan("faces_top", FW, FH, REGIONS)
    out = media.render_keeps(riverside, [(0.0, 2.0)], tmp_path / "r.mp4", vertical="faces_top",
                             workdir=tmp_path, layout_plan=p)
    info = media.probe(out)
    assert (info["width"], info["height"]) == (1080, 1920)
    th = p["tiles"][1]["dst"][3]
    r, g, b = _px(out, 1.0, 270, th // 2)            # sol üst: 1. kamera (yeşil)
    assert g > 150 and r < 80 and b < 80
    r, g, b = _px(out, 1.0, 810, th // 2)            # sağ üst: 2. kamera (mavi)
    assert b > 150 and r < 80
    gx, gy, gw, gh = p["game_area"]
    r, g, b = _px(out, 1.0, 540, gy + gh // 2)       # alt: oyun kartı (kırmızı) ortada
    assert r > 150 and g < 80 and b < 80


def test_render_game_full_with_pips(riverside, tmp_path):
    p = layout.plan("game_full", FW, FH, REGIONS)
    out = media.render_keeps(riverside, [(0.0, 2.0)], tmp_path / "f.mp4", vertical="game_full",
                             workdir=tmp_path, layout_plan=p)
    x, y, w, h = p["tiles"][1]["dst"]
    r, g, b = _px(out, 1.0, x + w // 2, y + h // 2)
    assert g > 150 and r < 80
    r, g, b = _px(out, 1.0, x + 1, y + 1)            # beyaz çerçeve
    assert min(r, g, b) > 200


def test_heuristic_detects_camera_column(riverside):
    res = layout.detect_heuristic(riverside, 1.0, FW, FH)
    assert len(res["cams"]) == 2
    (x1, y1, w1, h1), (x2, y2, w2, h2) = res["cams"]
    assert abs(x1 - 1400) < 20 and abs(y2 - 540) < 20 and abs(h1 - 540) < 20
    # İçerik: ekran paylaşımının tamamı değil, içindeki ayrıntılı kart (kırmızı 500,120 400x560) — paylı
    cx, cy, cw, ch = res["content"]
    assert cx <= 500 and cy <= 120 and cx + cw >= 900 and cy + ch >= 680
    assert cw < 1000 and cx + cw <= 1400


def test_detect_without_claude_uses_heuristic(riverside):
    res = layout.detect(riverside, 1.0, FW, FH, use_claude=True)   # backend off → sezgisel
    assert res["source"] == "heuristic" and res["w"] == FW and len(res["cams"]) == 2


def test_visuals_and_captions_follow_layout():
    p = layout.plan("faces_top", FW, FH, REGIONS)
    ax, ay, aw, ah = p["game_area"]
    bx, by, bw, bh = visuals.overlay_box("top", 1080, 1920, p["game_area"])
    assert by >= ay and by + bh <= ay + ah                 # görsel kartı yüzlerin üstüne binmez
    ass = captions.build_ass(None, [(0, 1)], 1080, 1920, style="bold", margin_v=p["caption_margin"])
    assert f",2,{int(1080 * 0.08)},{int(1080 * 0.08)},{p['caption_margin']},1" in ass


# ---------------------------------------------------------------------------
# Sohbet reel'leri: ekranda oyun yok, iki kamera yan yana → 9:16'da yüzler alt alta
# ---------------------------------------------------------------------------
TALK_CAMS = [[0, 0, 960, 1080], [960, 0, 960, 1080]]


@pytest.fixture(scope="module")
def side_by_side(tmp_path_factory):
    """Riverside sohbet görünümü: iki kamera yan yana (duvar renkleri birbirine yakın)."""
    p = tmp_path_factory.mktemp("talk") / "talk.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=0xb8b4ae:s=960x{FH}:r=30:d=3",
        "-f", "lavfi", "-i", f"color=c=0xd6d3cc:s=960x{FH}:r=30:d=3",
        "-f", "lavfi", "-i", "color=c=0x00c000:s=300x380:r=30:d=3",
        "-f", "lavfi", "-i", "color=c=0x0040ff:s=300x380:r=30:d=3",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=3",
        "-filter_complex", "[0][2]overlay=330:250[a];[1][3]overlay=330:250[b];[a][b]hstack[v]",
        "-map", "[v]", "-map", "4:a", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(p),
    ], check=True)
    return p


def test_inner_aspect_stays_inside_camera():
    x, y, w, h = layout.inner_aspect(TALK_CAMS[0], 1080 / 960, FW, FH)
    assert x >= 0 and x + w <= 960 and y >= 0 and y + h <= FH          # yandaki kameraya taşmaz
    assert abs(w / h - 1080 / 960) < 0.02 and w == 960


def test_talk_plan_stacks_faces_and_puts_captions_on_the_seam():
    p = layout.plan("talk", FW, FH, {"content": None, "cams": TALK_CAMS})
    assert p["mode"] == "talk" and len(p["tiles"]) == 2
    (a, b) = (t["dst"] for t in p["tiles"])
    assert a == (0, 0, 1080, 960) and b == (0, 960, 1080, 960)          # alt alta, ekranı eşit böler
    assert abs((1920 - p["caption_margin"]) - 960) < 100                # altyazı iki yüzün birleştiği yerde
    assert p["tiles"][0]["src"][0] + p["tiles"][0]["src"][2] <= 960     # üstteki: soldaki kamera


def test_heuristic_tells_talk_from_game(riverside, side_by_side):
    talk = layout.detect_heuristic(side_by_side, 1.0, FW, FH)
    assert talk["kind"] == "talk" and talk["content"] is None and len(talk["cams"]) == 2
    assert abs(talk["cams"][0][2] - 960) < 20
    game = layout.detect_heuristic(riverside, 1.0, FW, FH)
    assert game["kind"] == "game" and game["content"]                   # oyun ekranı sohbet sanılmaz
    assert layout.detect_reel(side_by_side, 0.2, 2.8, FW, FH, use_claude=False)["kind"] == "talk"
    assert layout.detect_reel(riverside, 0.2, 2.8, FW, FH, use_claude=False)["kind"] == "game"


def test_render_talk_shows_both_faces_stacked(side_by_side, tmp_path):
    dst = tmp_path / "talk.mp4"
    p = layout.plan("talk", FW, FH, {"content": None, "cams": TALK_CAMS})
    media.render_keeps(side_by_side, [(0.0, 2.0)], dst, vertical="talk", workdir=tmp_path, layout_plan=p)
    top, bottom = _px(dst, 1.0, 540, 480), _px(dst, 1.0, 540, 1440)
    assert top[1] > 150 and top[2] < 80            # üstte yeşil (sol kamera)
    assert bottom[2] > 150 and bottom[1] < 120     # altta mavi (sağ kamera)
