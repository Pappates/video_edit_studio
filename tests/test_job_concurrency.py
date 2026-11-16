"""job.json'a aynı anda yazma (arka plan iş parçacığı) ve okuma (arayüzün durum sorgusu).

Windows'ta replace sırasında dosya okunursa ya da okunurken replace edilirse PermissionError oluşur;
eskiden bu, sunucuda 500 hatası ve analiz iş parçacığının çökmesi demekti.
"""
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ANTHROPIC_API_KEY"] = ""

from vedit import pipeline  # noqa: E402
from vedit.config import settings  # noqa: E402


def test_concurrent_read_write_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    job = pipeline.create_job(src, "reels")
    big = [{"id": f"reel{i:02d}", "start": i, "end": i + 1, "text": "x" * 400} for i in range(200)]
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer():
        try:
            for i in range(120):
                job.set(progress=i / 120, message=f"adım {i}", reels=big)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                d = pipeline.Job(job.id).data
                assert d.get("mode") == "reels"
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, repr(errors[:3])
    assert pipeline.Job(job.id).data["message"] == "adım 119"
    assert not list(job.dir.glob("*.tmp"))          # geçici dosya kalmadı


def test_progress_is_throttled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    job = pipeline.create_job(src, "reels")
    writes = []
    monkeypatch.setattr(pipeline.Job, "save", lambda self: writes.append(1))
    prog = job.progress("transcribe", 0.0, 1.0)
    for i in range(500):
        prog(i / 1000, "x")
    prog(1.0, "bitti")                                 # son adım her zaman yazılır
    assert 2 <= len(writes) <= 5
