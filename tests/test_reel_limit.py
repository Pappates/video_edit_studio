"""Uzun videoda 'ilk N reel': Whisper parça parça okunur, N tam reel bulununca durur."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from fixtures import make_transcript  # noqa: E402
from vedit import pipeline, reels  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"   # sezgisel bölme: Claude çağrılmaz

REEL_LEN, REEL_GAP, N_REELS = 32.0, 8.0, 8      # 8 reel, her biri ~32 sn, aralarında 8 sn sessizlik
DURATION = N_REELS * (REEL_LEN + REEL_GAP)


def _fake_stream():
    sentences = []
    for k in range(N_REELS):
        base = k * (REEL_LEN + REEL_GAP)
        sentences += [(base + j * 4.0, f"parça {k} cümle {j} bir iki üç dört beş") for j in range(8)]
    segs = make_transcript(sentences)["segments"]
    consumed = {"n": 0, "last_end": 0.0}

    def gen():
        for s in segs:
            consumed["n"] += 1
            consumed["last_end"] = s["end"]
            yield {k: v for k, v in s.items() if k != "id"}
    return segs, consumed, gen()


def _job(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    return pipeline.create_job(src, "reels", options={"max_reels": 2})


def test_stops_after_n_complete_reels(tmp_path, monkeypatch):
    segs, consumed, stream = _fake_stream()
    monkeypatch.setattr(pipeline, "transcribe_stream", lambda p: ("tr", DURATION, stream))
    job = _job(tmp_path, monkeypatch)
    info = {"duration": DURATION, "has_video": False, "has_audio": True}

    tr, found, method = pipeline._reels_incremental(job, info, 2, window=60.0)

    assert method == "heuristic"
    assert [r.id for r in found] == ["reel01", "reel02"]
    assert found[0].start < 1.0 and abs(found[0].end - REEL_LEN) < 2.0
    assert abs(found[1].start - (REEL_LEN + REEL_GAP)) < 1.0
    # Whisper videonun sonuna kadar okunmadı: 2. reel'in bitişi + yeniden çekim kontrolü (120 sn) kadar okundu
    limit = found[1].end + pipeline.RETAKE_LOOKAHEAD_SEC + 20   # + bir cümle/sessizlik payı
    assert consumed["n"] < len(segs) and found[1].end + pipeline.RETAKE_LOOKAHEAD_SEC <= consumed["last_end"] < limit
    assert tr["partial_until"] < limit
    saved = json.loads((job.dir / "transcript.json").read_text(encoding="utf-8"))
    assert saved["partial_until"] == tr["partial_until"]


def test_half_reel_at_window_edge_is_not_counted(tmp_path, monkeypatch):
    """Pencere ikinci reel'in ortasında biterse o reel sayılmaz, bir sonraki parçada bütün olarak bulunur."""
    segs, consumed, stream = _fake_stream()
    monkeypatch.setattr(pipeline, "transcribe_stream", lambda p: ("tr", DURATION, stream))
    job = _job(tmp_path, monkeypatch)
    info = {"duration": DURATION, "has_video": False, "has_audio": True}
    # 55 sn: 2. reel 40 sn'de başlıyor, 72 sn'de bitiyor → ilk parçada yarım
    tr, found, _ = pipeline._reels_incremental(job, info, 2, window=55.0)
    assert len(found) == 2
    assert abs(found[1].end - (REEL_LEN + REEL_GAP + REEL_LEN)) < 2.0


def test_whole_video_when_limit_exceeds_reel_count(tmp_path, monkeypatch):
    segs, consumed, stream = _fake_stream()
    monkeypatch.setattr(pipeline, "transcribe_stream", lambda p: ("tr", DURATION, stream))
    job = _job(tmp_path, monkeypatch)
    info = {"duration": DURATION, "has_video": False, "has_audio": True}
    tr, found, _ = pipeline._reels_incremental(job, info, 20, window=60.0)
    assert consumed["n"] == len(segs) and "partial_until" not in tr
    assert len(found) == N_REELS


def test_claude_incomplete_flag_is_dropped():
    segs = make_transcript([(0.5, "bir iki üç dört beş altı yedi sekiz dokuz on"),
                            (8.0, "on bir on iki on üç on dört on beş on altı"),
                            (20.0, "yirmi yirmi bir yirmi iki yirmi üç yirmi dört"),
                            (28.0, "otuz otuz bir otuz iki otuz üç otuz dört")])["segments"]
    data = {"reels": [{"start_id": 0, "end_id": 1, "title": "a", "complete": True},
                      {"start_id": 2, "end_id": 3, "title": "b", "complete": False}]}
    out = reels.reels_from_decision(segs, data, 40.0)
    assert [r.title for r in out] == ["a"]
