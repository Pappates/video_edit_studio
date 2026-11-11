import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["ANTHROPIC_API_KEY"] = ""  # testler API'ye gitmesin

from fixtures import SHOOT, make_transcript  # noqa: E402
from vedit import detector  # noqa: E402
from vedit.config import settings  # noqa: E402
from vedit.textnorm import fold, prefix_overlap, similarity, tokens  # noqa: E402

settings.anthropic_api_key = ""
settings.backend_pref = "off"  # testler gerçek Claude çağırmasın


def _cut_covering(cuts, seg, kinds=None):
    mid = (seg["start"] + seg["end"]) / 2
    return next((c for c in cuts if c.enabled and c.start <= mid <= c.end and (not kinds or c.kind in kinds)), None)


def test_textnorm():
    assert fold("Şurayı KES!") == "surayi kes"
    assert fold("İstanbul'da IŞIK") == "istanbul da isik"
    assert tokens("ee yani şey bugün", drop_fillers=True) == ["bugun"]
    assert similarity("bu makine on beş bar", "Bu makine, on beş bar!") == 1.0
    assert prefix_overlap(tokens("kutusundan çıkan"), tokens("kutusundan çıkan aksesuarlar şunlar")) == 1.0


def test_full_shoot():
    tr = make_transcript(SHOOT)
    segs = tr["segments"]
    cuts = detector.detect_all(tr, use_claude=False)

    # 1-2) "baştan alıyorum": ilk deneme (#1) ve komut (#2) kesilir, yeni deneme (#3) kalır
    assert _cut_covering(cuts, segs[1], {"retake", "repeat"})
    assert _cut_covering(cuts, segs[2], {"retake"})
    assert not _cut_covering(cuts, segs[3])

    # 3) Yarım cümle: #5 kesilir, tamamlanmış #6 kalır
    assert _cut_covering(cuts, segs[5], {"false_start", "repeat"})
    assert not _cut_covering(cuts, segs[6])

    # 4) "şurayı kes": önceki cümle (#7) ve komut (#8) kesilir, #9 kalır
    assert _cut_covering(cuts, segs[7], {"command", "repeat", "false_start"})
    assert _cut_covering(cuts, segs[8], {"command"})
    assert not _cut_covering(cuts, segs[9])

    # 5) Kekeleme "gördüğünüz gibi gördüğünüz gibi"
    st = [c for c in cuts if c.kind == "stutter"]
    assert st and st[0].text.lower().startswith("gördüğünüz")

    # 6) 37→45 arası uzun sessizlik kısaltılır
    assert any(c.kind == "silence" and c.start > 38 and c.end < 45 for c in cuts)

    # Normal cümleler korunur
    for k in (0, 3, 4, 6, 9, 11):
        assert not _cut_covering(cuts, segs[k], {"retake", "command", "repeat", "false_start"}), k


def test_no_false_positive_on_kesinlikle():
    tr = make_transcript([
        (0.5, "Bu yöntemi kesinlikle denemelisiniz çünkü çok hızlı."),
        (5.0, "Peynirleri ince ince kesin ve tavaya koyun."),
        (10.0, "Sonra beş dakika pişirin."),
    ])
    cuts = detector.detect_all(tr, use_claude=False, use_silence=False)
    assert not [c for c in cuts if c.kind in ("command", "retake")], [c.reason for c in cuts]


def test_range_markers():
    tr = make_transcript([
        (0.5, "Giriş cümlesi burada."),
        (3.0, "Buradan itibaren kes."),
        (5.0, "Kamera ayarı yapıyoruz bir dakika."),
        (8.0, "Tamam buraya kadar kes."),
        (10.0, "Asıl konuya geçelim."),
    ])
    cuts = detector.detect_all(tr, use_claude=False, use_silence=False)
    rng = [c for c in cuts if c.kind == "range"]
    assert rng and rng[0].start <= 3.0 and rng[0].end >= 9.0 and rng[0].enabled
    assert not _cut_covering(cuts, tr["segments"][4])


def test_keep_ranges():
    cuts = [detector.Cut(2, 4, "x", ""), detector.Cut(3.5, 5, "x", ""), detector.Cut(8, 9, "x", "", enabled=False)]
    assert detector.keep_ranges(cuts, 10) == [(0.0, 2.0), (5.0, 10.0)]
    assert detector.keep_ranges(cuts, 10, within=(1, 6)) == [(1.0, 2.0), (5.0, 6.0)]


def test_resegmented_shoot_keeps_intro():
    """Whisper cümleleri birleştirse bile (kısa aralık) giriş cümlesi korunmalı."""
    from vedit.transcribe import resegment
    tr = resegment(make_transcript(SHOOT))
    assert len(tr["segments"]) == len(SHOOT)
    cuts = detector.detect_all(tr, use_claude=False)
    intro = tr["segments"][0]
    assert not _cut_covering(cuts, intro, {"retake", "repeat", "command", "false_start"})


def test_retake_inside_merged_unit():
    """Noktalama yok: giriş + başarısız deneme tek birimde. Yalnızca deneme kısmı kesilmeli."""
    tr = make_transcript([
        (0.5, "merhaba arkadaşlar bugün kahve makinesini tanıtacağım bu makine on beş bar basınçla çalışıyor"),
        (7.0, "baştan alıyorum"),
        (8.5, "bu makine on beş bar basınçla çalışıyor ve süt köpürtücüsü var"),
    ])
    cuts = [c for c in detector.detect_all(tr, use_claude=False, use_silence=False) if c.kind == "retake"]
    words = tr["segments"][0]["words"]
    assert cuts
    assert cuts[0].start > words[5]["end"]           # "tanıtacağım" sonrası
    assert cuts[0].start <= words[6]["start"] + 0.01  # "bu makine..." başı
