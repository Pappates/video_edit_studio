"""Reel sıkılaştırma: sıkı sessizlik + Claude'un reel'e özel kesimleri (sahte Claude yanıtıyla)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""

from fixtures import make_transcript  # noqa: E402
from vedit import tighten  # noqa: E402
from vedit.config import settings  # noqa: E402
from vedit.detector import Cut  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"

# Wavelength benzeri reel: kanca, kategori yazma lojistiği, oyun, sonuç
GAME = [
    (0.5, "Bugün Wavelength oynuyoruz bakalım kim kazanacak"),       # 0 kanca
    (4.5, "dur şu kategoriyi yazayım kalem nerede"),                   # 1 lojistik
    (8.5, "tamam değiştirelim başka kategori olsun"),                  # 2 lojistik
    (14.0, "kategori sıcak soğuk ipucu olarak çorba diyorum"),        # 3 oyun (2 sn sessizlikten sonra)
    (18.0, "bence tam ortası değil biraz sıcak tarafı"),               # 4 oyun
    (22.0, "ve cevap doğru tam isabet inanılmaz"),                     # 5 sonuç
]
REEL = {"id": "reel01", "start": 0.3, "end": 26.0, "title": "Wavelength", "hook": "Bugün Wavelength oynuyoruz",
        "clean": True}


def test_tight_silences_inside_reel_only():
    tr = make_transcript(GAME)
    cuts = tighten.tight_silences(tr["segments"], REEL["start"], REEL["end"], "medium")
    # 3. ve 4. cümle arası (segs[2] bitişi → 14.0) uzun sessizlik kısaltılır, reel dışına taşmaz
    gap_start = tr["segments"][2]["end"]
    long_gap = [c for c in cuts if c.start > gap_start and c.end < 14.1]
    assert long_gap and all(c.kind == "silence" and c.source == "tighten" for c in cuts)
    assert all(REEL["start"] <= c.start and c.end <= REEL["end"] for c in cuts)
    kept_gap = 14.0 - gap_start - long_gap[0].duration
    assert abs(kept_gap - tighten.LEVELS["medium"]["silence_keep"]) < 0.05


def test_claude_cuts_map_segments_and_skip_already_cut(monkeypatch):
    monkeypatch.setattr(settings, "backend_pref", "api")
    tr = make_transcript(GAME)
    segs = tr["segments"]
    seen = {}

    def fake_call_json(content, system="", **kw):
        seen["prompt"] = content[0]["text"]
        seen["system"] = system
        seen["images"] = [b for b in content if b.get("type") == "image"]
        return {"cuts": [{"start_id": 1, "end_id": 2, "kind": "logistics", "reason": "Kategori yazma", "confidence": 0.9},
                         {"start_id": 0, "end_id": 5, "kind": "slow", "reason": "hepsi"},           # geçersiz: tüm reel
                         {"start_id": 42, "end_id": 43, "kind": "slow", "reason": "olmayan"},
                         {"start_id": 4, "end_id": 4, "kind": "slow", "reason": "zaten kesilmiş segment"},
                         {"start_id": 3, "end_id": 3, "kind": "redundant", "reason": "emin değil", "confidence": 0.3}],
                "summary": "Lojistik kesildi"}

    monkeypatch.setattr(tighten, "call_json", fake_call_json)
    # Genel temizlik 4. segmenti zaten kesmiş olsun → Claude'a gösterilmemeli
    base = [Cut(segs[4]["start"] - 0.05, segs[4]["end"] + 0.05, "repeat", "tekrar")]
    res = tighten.tighten_reel(tr, REEL, base, "medium")
    assert "#4" not in seen["prompt"] and "#1" in seen["prompt"] and "Orta" in seen["prompt"]
    content = [c for c in res["tighten"] if c["kind"] != "silence"]
    assert len(content) == 2                                    # tüm-reel, olmayan id ve kesilmiş segment atıldı
    logi = content[0]
    assert logi["kind"] == "logistics" and logi["enabled"] and "kalem" in logi["text"]
    assert segs[0]["end"] <= logi["start"] <= segs[1]["start"] and segs[2]["end"] <= logi["end"] <= segs[3]["start"]
    assert content[1]["enabled"] is False                       # düşük güven: kapalı gelir
    assert res["tight_note"] == "Lojistik kesildi" and res["tight_level"] == "medium"
    assert res["tight_after"] < res["tight_before"] - 6


def test_without_claude_only_silences():
    tr = make_transcript(GAME)
    res = tighten.tighten_reel(tr, REEL, [], "aggressive")
    assert res["tighten"] and all(c["kind"] == "silence" for c in res["tighten"])
    assert "Claude bağlı değil" in res["tight_note"]


def test_reel_keeps_respects_clean_flag_and_toggles():
    base = [Cut(2.0, 3.0, "silence", "x")]
    reel = {"start": 0.0, "end": 10.0, "clean": True,
            "tighten": [Cut(5.0, 6.0, "logistics", "y").to_dict(), {**Cut(7.0, 8.0, "slow", "z").to_dict(), "enabled": False}]}
    assert tighten.reel_keeps(reel, base) == [(0.0, 2.0), (3.0, 5.0), (6.0, 10.0)]
    reel["clean"] = False
    assert tighten.reel_keeps(reel, base) == [(0.0, 5.0), (6.0, 10.0)]


# --- Ses seviyesiyle doğrulanan sessizlik, kırıntı temizliği, kareler ve içerik notu --------------
def test_tight_silence_spares_loud_gaps():
    """Kelime yok ama ses var (kahkaha/tepki): yalnızca gerçekten sessiz kısım kesilir."""
    tr = make_transcript(GAME)
    segs = tr["segments"]
    gap_a, gap_b = segs[2]["end"], segs[3]["start"]            # ~10.4 → 14.0
    quiet = [(gap_a + 2.0, gap_b)]                              # ilk 2 sn kahkaha, sonrası sessiz
    cuts = tighten.tight_silences(segs, REEL["start"], REEL["end"], "light", quiet)
    in_gap = [c for c in cuts if gap_a <= c.start < gap_b]
    assert in_gap and all(c.start >= gap_a + 2.0 for c in in_gap)   # kahkaha kesilmedi
    # Ses ölçümü yoksa (quiet=None) eski davranış: tüm boşluk adaydır
    assert any(c.start < gap_a + 1.0 for c in tighten.tight_silences(segs, REEL["start"], REEL["end"], "light"))


def test_refine_base_silence_cuts_splits_on_noise():
    base = [Cut(10.0, 16.0, "silence", "6 sn"), Cut(20.0, 21.0, "repeat", "tekrar")]
    out = tighten.refine_silence_cuts(base, [(10.0, 12.0), (14.0, 16.0)])
    assert [(c.start, c.end, c.kind) for c in out] == [(20.0, 21.0, "repeat"), (10.0, 12.0, "silence"),
                                                        (14.0, 16.0, "silence")] or \
           sorted((c.start, c.end) for c in out) == [(10.0, 12.0), (14.0, 16.0), (20.0, 21.0)]


def test_smooth_keeps_removes_slivers():
    keeps = [(0.0, 3.0), (3.05, 5.0), (5.5, 5.57), (6.0, 9.0)]
    assert tighten.smooth_keeps(keeps) == [(0.0, 5.0), (6.0, 9.0)]   # 0.05 sn'lik kesim yok, 0.07 sn'lik parça atıldı


def test_light_level_keeps_more_breath():
    lv = tighten.LEVELS
    assert lv["light"]["silence_keep"] >= 0.5 and lv["light"]["silence_min"] >= 1.2
    assert lv["light"]["silence_keep"] > lv["medium"]["silence_keep"] > lv["aggressive"]["silence_keep"]


def test_frames_and_context_reach_claude(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "backend_pref", "api")
    seen = {}

    def fake(content, system="", **kw):
        seen["content"], seen["system"] = content, system
        return {"cuts": [], "summary": ""}
    monkeypatch.setattr(tighten, "call_json", fake)
    img = tmp_path / "f.jpg"
    img.write_bytes(b"x")
    tighten.tighten_reel(make_transcript(GAME), REEL, [], "light", frames=[(5.0, img)],
                         context="Frekans oyunu: ibreyi yönlendirmek oyunun parçası")
    text = seen["content"][0]["text"]
    assert "İçerik notu (ekipten): Frekans oyunu" in text and "kareler geliyor" in text
    assert any(b.get("type") == "image" for b in seen["content"])
    assert "OYUNUN KENDİSİ" in seen["system"] and "sola doğru" in seen["system"]


# --- Genel temizliğin 'muhabbet' kesimleri kısaltılmış reel'de kısaltma adımına bırakılır -------------
def test_chatter_deferred_to_reel_tighten(monkeypatch):
    monkeypatch.setattr(settings, "backend_pref", "api")
    tr = make_transcript(GAME)
    segs = tr["segments"]
    # Genel temizlik 1. ve 2. segmenti (oyun içi lojistik/yön tarifi) "muhabbet" sanmış olsun
    chat = Cut(segs[1]["start"] - 0.05, segs[2]["end"] + 0.05, "chatter", "Claude: kamera/konum ayarı")
    seen = {}

    def fake(content, system="", **kw):
        seen["text"] = content[0]["text"]
        return {"cuts": [], "summary": "oyun konuşması, kesilmedi"}
    monkeypatch.setattr(tighten, "call_json", fake)
    res = tighten.tighten_reel(tr, REEL, [chat], "light")
    assert f"#1 " in seen["text"] and "genel temizlik 'muhabbet' dedi" in seen["text"]   # Claude görüyor
    reel = {**REEL, **res}
    keeps = tighten.reel_keeps(reel, [chat])
    mid = (segs[1]["start"] + segs[1]["end"]) / 2
    assert any(a <= mid <= b for a, b in keeps)                       # Claude kesmedi → reel'de kaldı
    # Kısaltılmamış reel'de genel temizliğin kararı eskisi gibi geçerli
    assert not any(a <= mid <= b for a, b in tighten.reel_keeps({**REEL, "tight_level": ""}, [chat]))


def test_chatter_kept_as_cut_when_claude_off():
    tr = make_transcript(GAME)
    segs = tr["segments"]
    chat = Cut(segs[1]["start"] - 0.05, segs[2]["end"] + 0.05, "chatter", "Claude: muhabbet")
    res = tighten.tighten_reel(tr, REEL, [chat], "light", use_claude=False)
    mid = (segs[1]["start"] + segs[1]["end"]) / 2
    assert not any(a <= mid <= b for a, b in tighten.reel_keeps({**REEL, **res}, [chat]))
