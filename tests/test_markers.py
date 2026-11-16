"""Çekimde söylenen reel işaretleri ("1. reel", "ikinci reel", "yeni reel")."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from fixtures import make_transcript  # noqa: E402
from vedit import markers, reels  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"


def _labels(texts):
    tr = make_transcript([(i * 10.0, t) for i, t in enumerate(texts)])
    return {m["seg_id"]: m for m in markers.find_markers(tr["segments"])}


def test_marker_phrasings():
    found = _labels([
        "2. Reel'e geçiyoruz", "İkinci reel", "Tamam yeni reel başlıyoruz", "Reel 3", "Üçüncü Rill",
        "evet 1. reel motor", "on numaralı reel", "sıradaki reele geçelim arkadaşlar bugün çok güzel bir konu",
        "bir reel daha çekelim",
    ])
    assert [found[i]["label"] for i in range(9)] == [
        "2. reel", "2. reel", "yeni reel", "3. reel", "3. reel", "1. reel", "10. reel", "siradaki reel", "yeni reel"]
    assert found[2]["whole"] and found[5]["whole"]
    assert not found[7]["whole"] and found[7]["text"] == "sıradaki reele geçelim"


def test_not_markers():
    assert _labels([
        "Real Madrid çok iyi oynadı", "Ben Mustafa abiye direkt bu Reels içeriklerini de çektirelim",
        "bu reelde size anlatacağım", "bir reel çekelim mi", "bu reel bir harika oldu",
    ]) == {}


def test_inline_marker_cut_covers_only_marker_words():
    tr = make_transcript([(0.0, "sıradaki reele geçelim arkadaşlar bugün çok güzel bir konu")])
    mk = markers.find_markers(tr["segments"])
    cut = markers.marker_cuts(mk)[0]
    words = tr["segments"][0]["words"]
    assert cut.start <= words[0]["start"] and words[2]["end"] <= cut.end < words[3]["start"]
    assert cut.kind == "command" and "Reel işareti" in cut.reason


# Üç reel, aralarında uzun sessizlik YOK; yalnızca söylenen işaretler ayırıyor
SHOOT = []
for k, (mark, topic) in enumerate([("birinci reel", "kahve"), ("ikinci reel", "çay"), ("üçüncü reel", "su")]):
    base = k * 30.0
    SHOOT.append((base + 0.5, f"tamam {mark} başlıyoruz"))
    SHOOT += [(base + 3.0 + j * 4.0, f"bugün {topic} hakkında konuşuyoruz bölüm {j} bir iki üç") for j in range(6)]


def test_heuristic_split_on_markers_without_pauses():
    tr = make_transcript(SHOOT)
    info = {"duration": 92.0, "has_video": False}
    out, method = reels.split_reels("x.mp4", tr, info)
    assert method == "heuristic" and len(out) == 3
    assert [r.marker for r in out] == ["1. reel", "2. reel", "3. reel"]
    for k, r in enumerate(out):
        first = next(s for s in tr["segments"] if s["end"] > r.start)
        assert "reel" not in first["text"]                        # işaret cümlesi reel'e girmedi
        assert abs(r.start - (k * 30.0 + 3.0)) < 0.3


def test_claude_reel_starting_on_marker_is_trimmed(monkeypatch):
    monkeypatch.setattr(settings, "backend_pref", "api")
    tr = make_transcript(SHOOT)
    segs = tr["segments"]
    seen = {}

    def fake(content, system="", **kw):
        if "BİTMESİ" in system:          # bitiş kontrolü: değişiklik yok
            return {"end_id": None}
        seen["prompt"] = content[0]["text"]
        seen["system"] = system
        # Claude işaret segmentini (#0, #7) reel'e dahil etmiş olsun
        return {"reels": [{"start_id": 0, "end_id": 6, "title": "Kahve"}, {"start_id": 7, "end_id": 14, "title": "Çay"}]}

    monkeypatch.setattr(reels, "call_json", fake)
    out, method = reels.split_reels("x.mp4", tr, {"duration": 92.0, "has_video": False})
    assert method == "claude"
    assert "#0 t=0.5 sn" in seen["prompt"] and "1. reel" in seen["prompt"] and "REEL İŞARETLERİ" in seen["system"]
    assert out[0].start >= segs[0]["end"] and abs(out[0].start - (segs[1]["start"] - 0.2)) < 0.01
    assert out[1].marker == "2. reel" and out[1].start > segs[7]["end"] - 0.01
    # İlk reel bir sonraki reel'in işaretini (#7) içermiyor
    assert out[0].end <= segs[7]["start"]


# --- Yeniden çekilen reel ("1. reel" bozuldu, sonra yine "1. reel") ---------------
RETAKE = [(0.5, "tamam birinci reel başlıyoruz")]
RETAKE += [(3.0 + j * 4, f"bugün kahve anlatıyorum bölüm {j} bir iki üç") for j in range(4)]       # bozuk deneme
RETAKE += [(20.0, "ay pardon karıştırdım"), (24.0, "tamam birinci reel tekrar")]
RETAKE += [(27.0 + j * 4, f"bugün kahve anlatıyorum bölüm {j} bir iki üç dört") for j in range(6)]   # asıl deneme
RETAKE += [(52.0, "ikinci reel")]
RETAKE += [(55.0 + j * 4, f"şimdi çay anlatıyorum bölüm {j} bir iki üç") for j in range(5)]


def test_retake_keeps_last_attempt_and_disables_earlier():
    tr = make_transcript(RETAKE)
    out, _ = reels.split_reels("x.mp4", tr, {"duration": 80.0, "has_video": False})
    assert [(r.marker, r.enabled) for r in out] == [("1. reel", False), ("1. reel", True), ("2. reel", True)]
    assert "yeniden çekildi" in out[0].retake and not out[1].retake
    assert "2 kez çekilmiş" in out[1].notes
    # İlk 1 reel: asıl deneme + kapalı önceki deneme listede; 2. reel yok
    first = reels.first_n(out, 1)
    assert [(r.marker, r.enabled) for r in first] == [("1. reel", False), ("1. reel", True)]
    assert len(reels.first_n(out, 2)) == 3


def test_retake_across_windows_in_reel_limit_mode(tmp_path, monkeypatch):
    """Bozuk deneme ilk parçada tamam görünür; yeniden çekim sonraki parçada. 'İlk 1 reel' asıl denemeyi vermeli."""
    from vedit import pipeline
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    shoot = [(0.5, "birinci reel")]
    shoot += [(3.0 + j * 4, f"bugün kahve anlatıyorum bölüm {j} bir iki üç") for j in range(5)]      # 3–23 sn
    shoot += [(40.0, "ay olmadı birinci reel tekrar")]                                                 # 2. parçada
    shoot += [(43.0 + j * 4, f"bugün kahve anlatıyorum bölüm {j} bir iki üç dört") for j in range(6)]
    shoot += [(80.0, "ikinci reel")] + [(83.0 + j * 4, f"çay bölüm {j} bir iki üç dört") for j in range(5)]
    shoot += [(300.0, "son söz")]
    segs = make_transcript(shoot)["segments"]

    def stream():
        for s in segs:
            yield {k: v for k, v in s.items() if k != "id"}
    monkeypatch.setattr(pipeline, "transcribe_stream", lambda p: ("tr", 320.0, stream()))
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    job = pipeline.create_job(src, "reels", options={"max_reels": 1})
    tr, found, _ = pipeline._reels_incremental(job, {"duration": 320.0, "has_video": False}, 1, window=30.0)
    active = [r for r in found if r.enabled]
    assert len(active) == 1 and active[0].marker == "1. reel" and active[0].start > 40.0
    assert any(not r.enabled and r.retake and r.start < 5 for r in found)
    assert all(r.marker != "2. reel" for r in found)
