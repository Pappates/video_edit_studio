"""Claude API çağrılarını taklit ederek eşleme mantığını test eder (ağ gerektirmez)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import SHOOT, make_transcript  # noqa: E402
from vedit import claude_review, detector, reels  # noqa: E402
from vedit.claude_client import extract_json  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.backend_pref = "off"


def test_extract_json():
    assert extract_json('Tamam:\n```json\n{"a": [1, {"b": "}"}]}\n```') == {"a": [1, {"b": "}"}]}
    assert extract_json('önce metin {"x": 1} sonra') == {"x": 1}


def test_review_maps_ids(monkeypatch):
    tr = make_transcript(SHOOT)
    segs = tr["segments"]
    cuts = detector.detect_commands(segs) + detector.detect_repeats(segs)
    first = cuts[0]
    first.id = "r0"

    def fake(prompt, system=""):
        assert "#7" in prompt and "r0" in prompt
        return {
            "verdicts": [{"candidate": "r0", "keep_cut": True, "start_id": 1, "end_id": 2, "note": "doğru"}],
            "new_cuts": [{"start_id": 10, "end_id": 10, "kind": "chatter", "reason": "test", "confidence": 0.7},
                         {"start_id": 999, "end_id": 1000, "kind": "chatter", "reason": "uydurma id"}],
        }

    monkeypatch.setattr(claude_review, "call_json", fake)
    out = claude_review.review_cuts(segs, cuts)
    assert any(c.source == "claude" and c.seg_ids == [10] for c in out)
    assert not any(999 in c.seg_ids for c in out)
    r0 = next(c for c in out if c.id == "r0")
    assert r0.start == segs[1]["start"] and r0.confidence >= 0.8


def test_reels_with_claude(monkeypatch, tmp_path):
    settings.backend_pref = "api"
    tr = make_transcript([
        (1.0, "Selam bugün kahve demlemenin üç sırrı."),
        (6.0, "Taze kahve, doğru sıcaklık ve doğru süre."),
        (11.0, "Takip etmeyi unutmayın."),
        (16.0, "Hazır mısın kamera çekiyor mu."),
        (20.0, "Merhaba soğuk kahve tarifi geliyor."),
        (25.0, "On iki saat bekletin ve buzla servis edin."),
    ], duration=30)

    def fake(content, system="", **kw):
        return {"reels": [
            {"start_id": 0, "end_id": 2, "title": "Kahve sırları", "caption": "c", "hashtags": ["#kahve"]},
            {"start_id": 4, "end_id": 5, "title": "Soğuk kahve"},
            {"start_id": 3, "end_id": 3, "title": "çok kısa"},
        ], "discarded": [{"start_id": 3, "end_id": 3, "reason": "muhabbet"}]}

    monkeypatch.setattr(reels, "call_json", fake)
    monkeypatch.setattr(reels.media, "scene_changes", lambda v: [])
    out = reels.split_with_claude("yok.mp4", tr, {"duration": 30, "has_video": False})
    settings.backend_pref = "off"
    assert [r.title for r in out] == ["Kahve sırları", "Soğuk kahve"]
    assert out[0].id == "reel01" and out[0].start < 1.0 and out[0].end <= tr["segments"][3]["start"]
