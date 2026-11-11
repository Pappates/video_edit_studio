"""Konuşmacı ayrımı ve kişi bazlı ses ayarı testleri.

Gerçek ses imzası modeli ağ gerektirdiği için, imza olarak ortalama fbank vektörü kullanan bir
sahte embedder ile uçtan uca akış test edilir. İki 'konuşmacı' farklı tınıda sentetik seslerdir.
"""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import make_transcript  # noqa: E402
from vedit import audiomix, diarize, media  # noqa: E402
from vedit.config import settings  # noqa: E402

settings.backend_pref = "off"
SR = 48000

# (başlangıç, metin, konuşmacı) — A ve B sırayla konuşuyor
SCRIPT = [
    (0.5, "merhaba bugün ürünü tanıtıyorum", "A"),
    (4.0, "harika ben de merak ediyorum", "B"),
    (7.5, "önce kutuyu açalım ve bakalım", "A"),
    (11.0, "kutunun içinde neler var", "B"),
    (14.5, "bir kablo ve şarj aleti var", "A"),
]


def _voice(kind: str, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    if kind == "A":  # alçak, harmonik zengin
        f = 120
        sig = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 12))
    else:  # yüksek, parlak + hafif gürültü
        f = 310
        sig = sum(np.sin(2 * np.pi * f * k * t) / (k ** 0.5) for k in range(1, 8))
        sig = sig + 0.3 * np.random.default_rng(1).standard_normal(n)
    return 0.12 * sig / np.max(np.abs(sig))


@pytest.fixture(scope="module")
def two_speakers(tmp_path_factory):
    d = tmp_path_factory.mktemp("spk")
    tr = make_transcript([(s, t) for s, t, _ in SCRIPT], duration=18.0)
    total = int(18.0 * SR)
    audio = np.zeros(total)
    for seg, (_, _, who) in zip(tr["segments"], SCRIPT):
        a, b = int(seg["start"] * SR), int(seg["end"] * SR)
        audio[a:b] = _voice(who, b - a)
    stereo = np.stack([audio, audio], axis=1)
    path = d / "dialog.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((stereo * 32767).astype("<i2").tobytes())
    return path, tr


def _fbank_embedder(wave16k):
    """Sahte ses imzası: kaba log-spektrum (tınıyı yakalar). Gerçek model ağ gerektirdiği için."""
    return np.log(np.maximum(np.abs(np.fft.rfft(wave16k, 4096))[:400], 1e-6)).reshape(40, 10).mean(1)


def test_fbank_shape_and_banks():
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    f = diarize.fbank(x)
    assert f.shape == (98, 80) and np.isfinite(f).all()
    assert abs(float(f.mean())) < 1e-4  # CMN
    banks = diarize.mel_banks()
    assert banks.shape == (80, 257) and (banks.sum(1) > 0).all()


def test_cluster_auto_and_fixed():
    rng = np.random.default_rng(3)
    centers = rng.standard_normal((3, 64))
    X = np.concatenate([c + 0.15 * rng.standard_normal((12, 64)) for c in centers])
    auto = diarize.cluster(X, None, min_sim=0.5)
    assert len(set(auto)) == 3
    assert len(set(auto[:12])) == 1 and len(set(auto[12:24])) == 1
    two = diarize.cluster(X, 2)
    assert len(set(two)) == 2
    assert auto[0] == 0  # ilk görülme sırası


def test_light_diarization_end_to_end(two_speakers):
    path, tr = two_speakers
    res = diarize.diarize(path, {**tr, "segments": [dict(s) for s in tr["segments"]]},
                          n_speakers=2, embedder=_fbank_embedder)
    segs = res["transcript"]["segments"]
    got = [s["speaker"] for s in segs]
    assert got == ["S1", "S2", "S1", "S2", "S1"], got
    sp = res["speakers"]
    assert set(sp) == {"S1", "S2"} and sp["S1"]["duration"] > sp["S2"]["duration"]
    assert sp["S1"]["rms_db"] is not None and sp["S1"]["name"] == "Konuşmacı 1"


def test_apply_labels_splits_mixed_segment():
    tr = make_transcript([(0.5, "bir iki üç dört beş altı")])
    out = diarize.apply_labels(tr, [["S1", "S1", "S1", "S2", "S2", "S2"]])
    assert [s["speaker"] for s in out["segments"]] == ["S1", "S2"]
    assert out["segments"][1]["text"] == "dört beş altı" and out["segments"][1]["id"] == 1


def test_timeline_turns_partition(two_speakers):
    _, tr = two_speakers
    labs = [[w] * len(s["words"]) for s, w in zip(tr["segments"], ["S1", "S2", "S1", "S2", "S1"])]
    t = diarize.apply_labels({**tr, "segments": [dict(s) for s in tr["segments"]]}, labs)
    turns = diarize.timeline_turns(t, 18.0)
    assert [x[2] for x in turns] == ["S1", "S2", "S1", "S2", "S1"]
    assert turns[0][0] == 0.0 and turns[-1][1] == 18.0
    assert all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(turns, turns[1:]))  # boşluksuz


def _rms(path, a, b):
    with wave.open(str(path)) as wf:
        sr, ch = wf.getframerate(), wf.getnchannels()
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").reshape(-1, ch) / 32768
    seg = x[int(a * sr): int(b * sr)]
    return float(np.sqrt(np.mean(seg ** 2)))


def test_render_mix_gain_and_mute(two_speakers, tmp_path):
    path, tr = two_speakers
    labs = [[w] * len(s["words"]) for s, w in zip(tr["segments"], ["S1", "S2", "S1", "S2", "S1"])]
    t = diarize.apply_labels({**tr, "segments": [dict(s) for s in tr["segments"]]}, labs)
    turns = diarize.timeline_turns(t, 18.0)
    s0, s1 = t["segments"][0], t["segments"][1]
    a0, b0 = s0["start"] + 0.3, s0["end"] - 0.3
    a1, b1 = s1["start"] + 0.3, s1["end"] - 0.3

    # S1 sessiz, S2 +6 dB
    out = audiomix.render_mix(path, tmp_path / "mix.wav", turns,
                              {"S1": {"mute": True}, "S2": {"gain_db": 6.0}}, SR, 2, 18.0)
    assert _rms(out, a0, b0) < 1e-4
    ratio = _rms(out, a1, b1) / _rms(path, a1, b1)
    assert 1.85 < ratio < 2.05, ratio  # +6 dB ≈ x2

    # S1 −12 dB
    out2 = audiomix.render_mix(path, tmp_path / "mix2.wav", turns, {"S1": {"gain_db": -12.0}}, SR, 2, 18.0)
    r = _rms(out2, a0, b0) / _rms(path, a0, b0)
    assert 0.23 < r < 0.27, r  # −12 dB ≈ x0.25
    assert abs(_rms(out2, a1, b1) / _rms(path, a1, b1) - 1) < 0.01  # S2 değişmedi


def test_no_click_at_boundary(two_speakers, tmp_path):
    """Geçişte kazanç eğrisi rampalı olmalı (ani sıçrama yok)."""
    t, g = audiomix.gain_curve([(0, 5, "S1"), (5, 10, "S2")], {"S1": 1.0, "S2": 0.0})
    vals = np.interp(np.linspace(4.9, 5.1, 2001), t, g)
    assert np.max(np.abs(np.diff(vals))) < 0.01


def test_soft_limit():
    x = np.array([0.5, 0.95, 1.8, -2.5])
    y = audiomix.soft_limit(x)
    assert y[0] == 0.5 and np.all(np.abs(y) < 1.0) and y[2] > y[1] > 0.89


def test_render_keeps_with_mixed_audio(two_speakers, tmp_path):
    path, _ = two_speakers
    vid = tmp_path / "v.mp4"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=25:d=18",
                    "-i", str(path), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(vid)], check=True)
    mix = tmp_path / "silent.wav"
    audiomix.render_mix(vid, mix, [(0, 18, "S1")], {"S1": {"mute": True}}, 48000, 2, 18)
    out = media.render_keeps(vid, [(1, 4), (8, 10)], tmp_path / "o.mp4", workdir=tmp_path, audio_src=mix)
    info = media.probe(out)
    assert abs(info["duration"] - 5) < 0.15 and info["has_audio"]
    wav = tmp_path / "o.wav"
    media.extract_audio(out, wav)
    with wave.open(str(wav)) as wf:
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    assert np.abs(x).max() < 50  # sessize alınmış ses kullanıldı


def test_pipeline_speakers_flow(two_speakers, tmp_path, monkeypatch):
    import json
    import subprocess
    from vedit import pipeline

    path, tr = two_speakers
    vid = tmp_path / "dialog.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=25:d=18",
                    "-i", str(path), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(vid)], check=True)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(diarize, "OnnxEmbedder", lambda: _fbank_embedder)
    monkeypatch.setattr(settings, "diarization_backend", "light")

    job = pipeline.create_job(vid, "edit", "Röportaj.mp4", {"claude": False, "speakers": 2, "silence": False})
    (job.dir / "transcript.json").write_text(json.dumps(tr), encoding="utf-8")
    pipeline.analyze(job.id)
    j = pipeline.Job(job.id)
    assert j.data["status"] == "ready", j.data.get("error")
    assert set(j.data["speakers"]) == {"S1", "S2"} and j.data["diarization"]["backend"] == "light"
    assert [s["speaker"] for s in j.transcript["segments"]] == ["S1", "S2", "S1", "S2", "S1"]

    pipeline.update_speakers(job.id, {"S1": {"name": "Ahmet", "gain_db": 99}, "S2": {"mute": True}})
    sp = pipeline.Job(job.id).data["speakers"]
    assert sp["S1"]["name"] == "Ahmet" and sp["S1"]["gain_db"] == 20.0 and sp["S2"]["mute"]

    pipeline.set_segment_speaker(job.id, 4, "new")
    j = pipeline.Job(job.id)
    assert "S3" in j.data["speakers"] and j.transcript["segments"][4]["speaker"] == "S3"
    assert j.data["speakers"]["S1"]["name"] == "Ahmet"  # isim korunur

    pipeline.equalize_speakers(job.id)
    gains = {k: v["gain_db"] for k, v in pipeline.Job(job.id).data["speakers"].items()}
    assert all(-12 <= g <= 12 for g in gains.values())

    pipeline.update_speakers(job.id, {"S1": {"gain_db": 0}, "S3": {"gain_db": 0}})
    pipeline.render_edit(job.id)
    j = pipeline.Job(job.id)
    assert j.data["status"] == "done", j.data.get("error")
    assert len(list(j.dir.glob("mix_*.wav"))) == 1
    out = j.out_dir / j.data["outputs"]["video"]
    wav = tmp_path / "out.wav"
    media.extract_audio(out, wav)
    with wave.open(str(wav)) as wf:
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(float) / 32768
    s2 = j.transcript["segments"][1]
    a, b = int((s2["start"] + 0.4) * 16000), int((s2["end"] - 0.4) * 16000)
    s1 = j.transcript["segments"][0]
    c, d = int((s1["start"] + 0.4) * 16000), int((s1["end"] - 0.4) * 16000)
    assert np.sqrt(np.mean(x[a:b] ** 2)) < 0.002   # S2 sessiz (kesim yok, zaman aynı)
    assert np.sqrt(np.mean(x[c:d] ** 2)) > 0.02    # S1 duyuluyor


def test_short_speaker_is_kept_in_auto_mode():
    """İki kısa soru soran ikinci kişi (toplam ~4 sn) otomatik modda yutulmamalı."""
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal(32), rng.standard_normal(32)
    embs = np.stack([a, b, a, a, a, b, a, a]) + 0.05 * rng.standard_normal((8, 32))
    durs = [2.3, 2.3, 2.7, 0.7, 3.9, 1.5, 1.5, 1.5]
    labels = diarize.cluster(embs, None)
    labels = diarize._absorb_tiny(embs, labels, durs)
    assert len(set(labels)) == 2 and labels[1] == labels[5] != labels[0]
    # Tek parça, 0.6 sn'lik bir 'öksürük' kümesi ise yutulur
    c = rng.standard_normal(32)
    embs2 = np.stack([a, a, a, c, a]) + 0.05 * rng.standard_normal((5, 32))
    lab2 = diarize._absorb_tiny(embs2, diarize.cluster(embs2, None), [2, 2, 2, 0.6, 2])
    assert len(set(lab2)) == 1
