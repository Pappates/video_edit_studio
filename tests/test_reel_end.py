"""Reel bitiş kontrolü: çok turlu oyunda tur geçişi yeni reel sanılıp son tur/bitiş ekranı kesilmesin."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from fixtures import make_transcript  # noqa: E402
from vedit import reels  # noqa: E402
from vedit.claude_client import ClaudeError  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"

# Frekans benzeri: tur 1-2 (0-40), tur 3 kategori geçişi (44), tur 3 (48-64), kazanma (68), "bir tane daha" (80)
GAME = [(1.0, "bugün frekans oynuyoruz iyi şutör kötü şutör")]
GAME += [(5.0 + j * 4, f"tur bir tahmin {j} sola doğru biraz daha") for j in range(9)]
GAME += [(44.0, "hazır mısın tamamdır iyi smaçör kötü smaçör")]                     # tur geçişi
GAME += [(48.0 + j * 4, f"tur üç tahmin {j} kötü tarafına yetmiş derece") for j in range(5)]
GAME += [(68.0, "yes başardık güzel abi ya")]                                       # bitiş
GAME += [(80.0, "bir tane daha oynayalım mı olur abi"), (84.0, "futbol kategorisi yapalım")]
INFO = {"duration": 95.0, "has_video": False}


def _ids(tr, needle):
    return next(s["id"] for s in tr["segments"] if needle in s["text"])


def test_split_then_end_check_extends_to_game_end(monkeypatch):
    monkeypatch.setattr(settings, "backend_pref", "api")
    tr = make_transcript(GAME)
    turn3 = _ids(tr, "hazır mısın")
    win = _ids(tr, "başardık")
    calls = []

    def fake(content, system="", **kw):
        calls.append(system)
        if "BİTMESİ" in system:                       # bitiş kontrolü
            text = content[0]["text"]
            assert f"#{win}" in text and "bir tane daha" in text   # bitişten sonrası da gösterildi
            return {"end_id": win, "changed": True, "reason": "Oyun 3. turdaki 'başardık' ile bitiyor"}
        # bölme: Claude tur geçişinde reel'i yanlışlıkla bitirsin
        return {"reels": [{"start_id": 0, "end_id": turn3 - 1, "title": "Frekans"}]}

    monkeypatch.setattr(reels, "call_json", fake)
    out, method = reels.split_reels("x.mp4", tr, INFO)
    assert method == "claude" and len(out) == 1
    seg_win = tr["segments"][win]
    nxt = tr["segments"][win + 1]
    assert seg_win["end"] <= out[0].end <= nxt["start"]          # kazanma dahil, "bir tane daha" hariç
    assert "Bitiş" in out[0].notes and "başardık" in out[0].notes
    assert any("OYUNLAR" in s for s in calls)                       # bölme talimatı turları biliyor


def test_end_check_never_crosses_next_reel(monkeypatch):
    monkeypatch.setattr(settings, "backend_pref", "api")
    tr = make_transcript(GAME)
    last = len(tr["segments"]) - 1
    turn3 = _ids(tr, "hazır mısın")

    def fake(content, system="", **kw):
        if "BİTMESİ" in system:
            return {"end_id": last, "reason": "hepsi"}
        return {"reels": [{"start_id": 0, "end_id": turn3 - 1, "title": "A"},
                          {"start_id": turn3, "end_id": last, "title": "B"}]}
    monkeypatch.setattr(reels, "call_json", fake)
    out, _ = reels.split_reels("x.mp4", tr, INFO)
    assert len(out) == 2 and out[0].end <= out[1].start


def test_end_check_failure_keeps_reel(monkeypatch):
    monkeypatch.setattr(settings, "backend_pref", "api")
    tr = make_transcript(GAME)
    turn3 = _ids(tr, "hazır mısın")

    def fake(content, system="", **kw):
        if "BİTMESİ" in system:
            raise ClaudeError("limit")
        return {"reels": [{"start_id": 0, "end_id": turn3 - 1, "title": "A"}]}
    monkeypatch.setattr(reels, "call_json", fake)
    out, _ = reels.split_reels("x.mp4", tr, INFO)
    assert len(out) == 1 and out[0].end < tr["segments"][turn3]["start"] + 0.5


def test_end_check_does_not_stretch_to_window_edge(monkeypatch):
    """'Kaç reel?' modunda taranan parçanın sınırına kadar uzatılmaz (devamı görülmedi)."""
    monkeypatch.setattr(settings, "backend_pref", "api")
    tr = make_transcript(GAME)
    turn3 = _ids(tr, "hazır mısın")
    part_end = tr["segments"][_ids(tr, "tur üç tahmin 2")]["end"] + 0.5
    segs = [s for s in tr["segments"] if s["end"] <= part_end]

    def fake(content, system="", **kw):
        if "BİTMESİ" in system:
            return {"end_id": segs[-1]["id"], "reason": "devam ediyor"}
        return {"reels": [{"start_id": 0, "end_id": turn3 - 1, "title": "A", "complete": True}]}
    monkeypatch.setattr(reels, "call_json", fake)
    out, _ = reels.split_reels("x.mp4", {**tr, "segments": segs}, INFO, window=(0.0, part_end))
    assert out and out[0].end < tr["segments"][turn3]["start"] + 0.5


def test_check_reel_end_updates_job(tmp_path, monkeypatch):
    """Arayüzdeki '⏹ Bitişi kontrol et': mevcut işte reel'in bitişini düzeltir, kısaltma notu düşer."""
    import json
    from vedit import pipeline
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "backend_pref", "api")
    tr = make_transcript(GAME)
    turn3, win = _ids(tr, "hazır mısın"), _ids(tr, "başardık")
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    job = pipeline.create_job(src, "reels")
    (job.dir / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    end0 = tr["segments"][turn3 - 1]["end"] + 0.4
    job.set(status="ready", media=INFO, reels=[{"id": "reel01", "start": 0.8, "end": end0, "enabled": True,
                                                "tight_level": "light", "tight_note": ""}])
    monkeypatch.setattr(reels, "call_json", lambda c, system="", **k: {"end_id": win, "reason": "bitiş ekranı"})
    out = pipeline.check_reel_end(job.id, "reel01").data
    r = out["reels"][0]
    assert r["end"] > tr["segments"][win]["end"] - 0.01 and "bitiş" in out["message"]
    assert "✂ Kısalt" in r["tight_note"]
