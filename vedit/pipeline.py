"""İş (job) yönetimi ve uçtan uca akışlar. Hem web sunucusu hem CLI bunu kullanır.

Klasör yapısı:
    data/jobs/<job_id>/
        source.mp4          yüklenen orijinal
        transcript.json     whisper çıktısı (önbellek)
        job.json            durum, kesimler, reels, çıktılar
        output/             render edilmiş videolar ve dışa aktarımlar
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import audiomix, captions, diarize, exporters, layout, markers, media, tighten, visuals, youtube
from .config import settings
from .detector import Cut, detect_all, keep_ranges
from .reels import Reel, first_n, mark_retakes, split_reels
from .reels import _finish as finish_reels
from .transcribe import resegment, transcribe, transcribe_stream

_locks: dict[str, threading.Lock] = {}


def jobs_root() -> Path:
    p = settings.data_dir / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    """Aynı JSON dosyasını okuyan ve yazan herkes (API istekleri + arka plan iş parçacıkları aynı süreçte)
    bu kilidi alır. Windows'ta okuma ile replace aynı anda olursa PermissionError oluşuyordu; kilit bunu
    kökten önler. Yeniden deneme yalnızca ayrı bir süreç (ör. CLI) aynı dosyaya dokunursa devreye girer."""
    key = str(Path(path).resolve()).lower()
    with _file_locks_guard:
        return _file_locks.setdefault(key, threading.Lock())


def read_json(path: Path, default=None, tries: int = 12):
    """JSON okur (dosya kilidiyle). Yoksa default döner."""
    last: Exception | None = None
    for k in range(tries):
        with _file_lock(path):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return default
            except (PermissionError, json.JSONDecodeError) as e:   # yalnızca başka bir süreç yazarken
                last = e
        time.sleep(0.02 * (k + 1))
    raise last or RuntimeError(f"{path} okunamadı")


def write_json(path: Path, data, tries: int = 25) -> None:
    """Atomik yazma (dosya kilidiyle): geçici dosyaya yazar, sonra yerine koyar.
    Windows dosyayı okuma kapandıktan kısa süre sonra bırakır (antivirüs/dizinleyici); yük altında bu
    birkaç yüz ms sürebildiği için en çok ~5 sn yeniden denenir."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        for k in range(tries):
            with _file_lock(path):
                try:
                    tmp.replace(path)
                    return
                except PermissionError:   # Windows dosyayı henüz bırakmadı
                    if k == tries - 1:
                        raise
            time.sleep(min(0.25, 0.02 * (k + 1)))
    finally:
        tmp.unlink(missing_ok=True)


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.dir = jobs_root() / job_id
        self.file = self.dir / "job.json"
        self.data: dict = read_json(self.file, default={}) or {}

    # --- kalıcılık ---
    def save(self) -> None:
        self.data["updated_at"] = time.time()
        write_json(self.file, self.data)

    def set(self, **kw) -> None:
        with _locks.setdefault(self.id, threading.Lock()):
            # Diskteki güncel hali temel al: başka bir iş parçacığının yazdıklarını ezme
            fresh = read_json(self.file, default=None)
            fresh = dict(self.data) if fresh is None else fresh
            fresh.update(kw)
            self.data = fresh
            self.save()

    def progress(self, stage: str, base: float, span: float, min_interval: float = 0.5):
        """İlerleme yazıcısı. Diske en fazla min_interval saniyede bir yazar (her Whisper cümlesinde
        job.json'u yeniden yazmak gereksiz ve Windows'ta okuma/yazma çakışmasını artırır)."""
        last = [0.0]

        def fn(frac: float, msg: str = "") -> None:
            now = time.monotonic()
            if frac < 1.0 and now - last[0] < min_interval:
                return
            last[0] = now
            self.set(stage=stage, progress=round(base + span * frac, 3), message=msg or stage)
        return fn

    # --- kısayollar ---
    @property
    def source(self) -> Path:
        return self.dir / self.data["source_name"]

    @property
    def out_dir(self) -> Path:
        p = self.dir / "output"
        p.mkdir(exist_ok=True)
        return p

    @property
    def transcript(self) -> dict:
        data = read_json(self.dir / "transcript.json")
        if data is None:
            raise FileNotFoundError(self.dir / "transcript.json")
        return data

    @property
    def cuts(self) -> list[Cut]:
        return [Cut.from_dict(c) for c in self.data.get("cuts", [])]

    @property
    def reels(self) -> list[Reel]:
        return [Reel.from_dict(r) for r in self.data.get("reels", [])]

    @property
    def visuals_dir(self) -> Path:
        p = self.dir / "visuals"
        p.mkdir(exist_ok=True)
        return p

    def public(self) -> dict:
        d = dict(self.data)
        d["id"] = self.id
        return d


def create_job(src: str | Path, mode: str, original_name: str | None = None, options: dict | None = None,
               move: bool = False) -> Job:
    if mode not in ("edit", "reels", "youtube"):
        raise ValueError("mode 'edit', 'reels' veya 'youtube' olmalı")
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = Job(job_id)
    job.dir.mkdir(parents=True)
    name = original_name or Path(src).name
    suffix = Path(name).suffix.lower() or ".mp4"
    dst = job.dir / f"source{suffix}"
    (shutil.move if move else shutil.copy2)(str(src), dst)
    job.data = {
        "mode": mode, "status": "queued", "stage": "queued", "progress": 0.0, "message": "Sırada",
        "original_name": name, "source_name": dst.name, "options": options or {},
        "created_at": time.time(), "outputs": {},
    }
    job.save()
    return job


def recover_stale_jobs() -> int:
    """Sunucu açılırken çağrılır: çalışırken sunucu kapandıysa (ya da takıldıysa) 'running/rendering'
    durumunda kalmış işleri yeniden başlatılabilir hale getirir."""
    n = 0
    for d in jobs_root().iterdir():
        if not (d / "job.json").exists():
            continue
        job = Job(d.name)
        st = job.data.get("status")
        if st in ("queued", "running", "rendering"):
            ready = bool(job.data.get("cuts") is not None and job.data.get("media"))
            job.set(status="ready" if ready else "error", stage="ready" if ready else "error", progress=0.0,
                    message=("Önceki render yarım kaldı — tekrar 'oluştur'a basın" if st == "rendering" and ready
                             else "Analiz yarım kaldı — videoyu yeniden yükleyin" if not ready
                             else "İşlem yarım kaldı — tekrar deneyin"))
            n += 1
    return n


def list_jobs() -> list[dict]:
    out = []
    for d in sorted(jobs_root().iterdir(), reverse=True):
        if (d / "job.json").exists():
            j = Job(d.name)
            out.append({k: j.data.get(k) for k in ("mode", "status", "original_name", "created_at", "progress")}
                       | {"id": j.id})
    return out


# ---------------------------------------------------------------------------
# Analiz
# ---------------------------------------------------------------------------
def analyze(job_id: str) -> Job:
    job = Job(job_id)
    opts = job.data.get("options", {})
    try:
        job.set(status="running", stage="probe", progress=0.01, message="Video inceleniyor", error=None)
        info = media.probe(job.source)
        job.set(media=info)

        manual = job.data["mode"] == "reels" and opts.get("manual")   # reel sınırlarını kullanıcı işaretler
        max_reels = int(opts.get("max_reels") or 0) if job.data["mode"] == "reels" and not manual else 0
        early_reels, early_method = None, ""
        if max_reels and not (job.dir / "transcript.json").exists():
            tr, early_reels, early_method = _reels_incremental(job, info, max_reels)
        else:
            tr = transcribe(job.source, job.dir / "transcript.json", progress=job.progress("transcribe", 0.02, 0.48))
        if not tr.get("duration"):
            tr["duration"] = info["duration"]

        if opts.get("diarize", True) and info.get("has_audio") and tr["segments"]:
            tr = _run_diarization(job, tr, opts.get("speakers") or None, job.progress("diarize", 0.5, 0.1))
        job.set(segment_count=len(tr["segments"]))

        if job.data["mode"] == "youtube":
            # Yayınlanacak (hazır) video: kesim yapılmaz, doğrudan açıklama üretilir
            job.set(cuts=[], stage="youtube", progress=0.65, message="YouTube açıklaması hazırlanıyor")
            try:
                _generate_youtube(job, opts.get("instruction", ""))
                msg = "Açıklama hazır — gözden geçirin"
            except Exception as e:  # noqa: BLE001 — transkript korunur, kullanıcı yeniden deneyebilir
                job.set(youtube_error=str(e))
                msg = f"Transkript hazır ama açıklama oluşturulamadı: {e}"
            job.set(status="ready", stage="ready", progress=1.0, message=msg)
            return job

        cut_progress = job.progress("detect", 0.6, 0.15 if job.data["mode"] == "reels" else 0.38)
        cuts = detect_all(
            tr,
            use_silence=opts.get("silence", True),
            use_stutter=opts.get("stutter", True),
            use_claude=opts.get("claude", True),
            progress=cut_progress,
        )
        if job.data["mode"] == "reels":
            # "2. reel", "yeni reel" gibi söylenen işaretler reel'den çıkarılır
            cuts += markers.marker_cuts(markers.find_markers(tr["segments"]))
            cuts.sort(key=lambda c: c.start)
        # Transkriptte kelime olmayan her yer sessiz değildir (kahkaha, tepki, oyun sesi): sessizlik
        # kesimleri gerçek ses seviyesiyle doğrulanır. Tek geçiş, uzun videoda da birkaç saniye sürer.
        quiet_all = None
        if info.get("has_audio") and any(c.kind == "silence" for c in cuts):
            cut_progress(0.95, "Sessizlikler ses seviyesiyle doğrulanıyor")
            quiet_all = media.quiet_intervals(job.source, 0.0, tr.get("partial_until") or None)
            cuts = tighten.refine_silence_cuts(cuts, quiet_all)
        job.set(cuts=[c.to_dict() for c in cuts])

        msg = "Analiz tamamlandı — gözden geçirin"
        if manual:
            # Otomatik bölme yok: kullanıcı kurgu şeridinde işaretler (analiz sürerken de işaretleyebilir),
            # sonra "Reels'i hazırla" başlık/açıklama üretir ve kısaltır.
            n = len(Job(job_id).data.get("reels") or [])
            msg = (f"Transkript hazır — {n} reel işaretli; '📝 Reels'i hazırla' ile başlık üretip kısaltın" if n
                   else "Transkript hazır — kurgu şeridinde S ile reel sınırlarını işaretleyin")
        elif job.data["mode"] == "reels":
            if early_reels is not None:
                reels, method = early_reels, early_method
            else:
                until = tr.get("partial_until")
                reels, method = split_reels(job.source, tr, info, progress=job.progress("reels", 0.75, 0.23),
                                            window=(0.0, until) if until else None)
                if max_reels:
                    reels = first_n(reels, max_reels)
            reel_dicts = [r.to_dict() for r in reels]
            level = opts.get("tighten", "medium")
            if level in tighten.LEVELS and reel_dicts:
                prog = job.progress("tighten", 0.9, 0.09)
                for n, rd in enumerate(reel_dicts):
                    if not rd.get("enabled", True):
                        continue   # kapalı (ör. yeniden çekilmiş önceki deneme): seçilirse "✂ Kısalt" ile yapılır
                    prog(n / len(reel_dicts), f"{rd['id']} sıkılaştırılıyor ({n + 1}/{len(reel_dicts)})")
                    res, cuts = _tighten_one(job, tr, rd, cuts, level, None, opts.get("claude", True), quiet_all)
                    rd.update(res)
                job.set(cuts=[c.to_dict() for c in cuts])
            job.set(reels=reel_dicts, reels_method=method)
            # Yatay kayıtta dikey yerleşim bölgelerini (oyun + kameralar) şimdiden bul
            if info.get("has_video") and info.get("width", 0) > info.get("height", 0) and reel_dicts:
                job.set(stage="layout", message="Dikey yerleşim için oyun/kamera bölgeleri bulunuyor")
                try:
                    job_layout(job, opts.get("claude", True), force=True)
                except Exception as e:  # noqa: BLE001 — yerleşim bulunamazsa analiz yine biter
                    job.set(layout_error=str(e)[:300])
            if tr.get("partial_until"):
                msg = (f"Analiz tamamlandı — {len(reels)} reel; videonun ilk {_mmss(tr['partial_until'])} "
                       f"/ {_mmss(info['duration'])} kısmı tarandı")

        job.set(status="ready", stage="ready", progress=1.0, message=msg)
    except Exception as e:  # noqa: BLE001
        job.set(status="error", stage="error", message=str(e), error=traceback.format_exc()[-3000:])
    return job


REEL_WINDOW_SEC = 600.0   # "ilk N reel" modunda her seferinde yazıya dökülüp incelenen parça
RETAKE_LOOKAHEAD_SEC = 120.0  # durmadan önce son reel'in arkasına bakılan süre (yeniden çekim var mı)


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _reels_incremental(job: Job, info: dict, max_reels: int, window: float = REEL_WINDOW_SEC):
    """Uzun videoda yalnızca ilk N reel: video parça parça yazıya dökülür ve her yeni parça reels'e bölünür;
    N tam reel bulununca Whisper durdurulur, videonun geri kalanı hiç işlenmez.

    Döner: (transkript — yalnızca taranan kısım, 'partial_until' ile), reels, yöntem
    """
    language, duration, stream = transcribe_stream(job.source)
    duration = duration or info["duration"]
    raw: list[dict] = []
    found: list[Reel] = []
    methods: list[str] = []
    cursor, exhausted = 0.0, False
    target = min(duration, window)
    tr: dict = {"language": language, "duration": duration, "segments": []}
    tprog = job.progress("transcribe", 0.02, 0.55)   # en çok 0.5 sn'de bir diske yazar

    while True:
        while not exhausted and (not raw or raw[-1]["end"] < target):
            try:
                raw.append(next(stream))
            except StopIteration:
                exhausted = True
                break
            tprog(min(1.0, raw[-1]["end"] / duration) * 0.999,
                  f"Transkripsiyon {_mmss(raw[-1]['end'])} / {_mmss(duration)} — "
                  f"{sum(1 for r in found if r.enabled)}/{max_reels} reel bulundu")
        covered = duration if exhausted else (raw[-1]["end"] if raw else target)
        tr = resegment({"language": language, "duration": duration,
                        "segments": [{"id": i, **s} for i, s in enumerate(raw)]})
        part = [s for s in tr["segments"] if s["start"] >= cursor - 0.01]
        if part:
            job.set(stage="reels", message=f"{_mmss(cursor)}–{_mmss(covered)} arası reels'e bölünüyor "
                                           f"({len(found)}/{max_reels} bulundu)")
            reels, method = split_reels(job.source, {**tr, "segments": part}, info, window=(cursor, covered))
            methods.append(method)
            new = [r for r in reels if r.start >= cursor - 0.5]
            if new:
                cursor = max(r.end for r in new)
                # Parçalar arası: "1. reel" önceki parçada bozulup bu parçada yeniden çekilmiş olabilir
                found = mark_retakes(finish_reels(found + new))
            elif covered - cursor > 2 * window:
                cursor = covered - 120.0   # uzun süre reel yoksa (muhabbet vb.) pencereyi büyütmeye devam etme
        if exhausted:
            break
        active = [r for r in found if r.enabled]
        if len(active) >= max_reels:
            # Durmadan önce son reel'in biraz arkasına bak: aynı numara yeniden çekildiyse onu da görmeliyiz
            need = active[max_reels - 1].end + RETAKE_LOOKAHEAD_SEC
            if covered >= need:
                break
            target = min(duration, need)
        else:
            target = min(duration, covered + window)

    if not exhausted:
        tr["partial_until"] = round(covered, 2)
    write_json(job.dir / "transcript.json", tr)
    method = next((m for m in methods if m != "claude"), "claude") if methods else "heuristic"
    return tr, first_n(found, max_reels), method


def _run_diarization(job: Job, tr: dict, n_speakers: int | None, progress) -> dict:
    """Konuşmacıları ayırır; transkripti (konuşmacı etiketli) ve konuşmacı listesini kaydeder.
    Hata olursa analizi durdurmaz, tek konuşmacı varsayar ve sebebi not eder."""
    progress(0.0, "Konuşmacılar ayrılıyor")
    try:
        res = diarize.diarize(job.source, tr, n_speakers, progress=progress, previous=job.data.get("speakers"))
        tr, note, backend = res["transcript"], res["note"], res["backend"]
        speakers = res["speakers"]
    except Exception as e:  # noqa: BLE001
        labels = [["S1"] * len(s.get("words") or [None]) for s in tr["segments"]]
        tr = diarize.apply_labels(tr, labels)
        speakers = diarize.speaker_stats(tr, None, job.data.get("speakers"))
        backend, note = "error", f"Konuşmacı ayrımı yapılamadı, tek kişi varsayıldı: {e}"
    write_json(job.dir / "transcript.json", tr)
    job.set(speakers=speakers, diarization={"backend": backend, "note": note, "requested": n_speakers})
    return tr


def rediarize(job_id: str, n_speakers: int | None) -> Job:
    """Yalnızca konuşmacı ayrımını yeniden yapar (ör. kişi sayısını elle verince). Kesimler korunur."""
    job = Job(job_id)
    try:
        job.set(status="running", stage="diarize", progress=0.0, message="Konuşmacılar yeniden ayrılıyor")
        _run_diarization(job, job.transcript, n_speakers, job.progress("diarize", 0.0, 1.0))
        job.set(status="ready", stage="ready", progress=1.0, message="Konuşmacılar güncellendi")
    except Exception as e:  # noqa: BLE001
        job.set(status="error", stage="error", message=str(e), error=traceback.format_exc()[-3000:])
    return job


# ---------------------------------------------------------------------------
# Kullanıcı düzenlemeleri
# ---------------------------------------------------------------------------
def update_speakers(job_id: str, changes: dict) -> Job:
    """İsim, ses (dB, −30..+20) ve sessize alma ayarlarını günceller."""
    job = Job(job_id)
    speakers = dict(job.data.get("speakers") or {})
    for spk, ch in changes.items():
        if spk not in speakers:
            continue
        cur = dict(speakers[spk])
        if "name" in ch:
            cur["name"] = str(ch["name"]).strip()[:40] or cur["name"]
        if "gain_db" in ch:
            cur["gain_db"] = round(max(-30.0, min(20.0, float(ch["gain_db"]))), 1)
        if "mute" in ch:
            cur["mute"] = bool(ch["mute"])
        speakers[spk] = cur
    job.set(speakers=speakers)
    return job


def equalize_speakers(job_id: str) -> Job:
    """Konuşmacıların ortalama ses seviyelerini birbirine yaklaştırır (en fazla ±12 dB düzeltme)."""
    job = Job(job_id)
    speakers = dict(job.data.get("speakers") or {})
    levels = [s["rms_db"] for s in speakers.values() if s.get("rms_db") is not None and s.get("duration", 0) > 1]
    if len(levels) < 2:
        return job
    target = sorted(levels)[len(levels) // 2]  # orta değer: kimseyi aşırı yükseltmemek için
    for spk, s in speakers.items():
        if s.get("rms_db") is not None:
            speakers[spk] = {**s, "gain_db": round(max(-12.0, min(12.0, target - s["rms_db"])), 1)}
    job.set(speakers=speakers)
    return job


def set_segment_speaker(job_id: str, seg_id: int, speaker: str) -> Job:
    """Konuşmacı düzeltmesi. speaker == 'new' ise yeni bir konuşmacı oluşturur."""
    job = Job(job_id)
    speakers = dict(job.data.get("speakers") or {})
    if speaker == "new":
        nums = [int(k[1:]) for k in speakers if k[1:].isdigit()]
        speaker = f"S{max(nums, default=0) + 1}"
    tr = diarize.set_segment_speaker(job.transcript, seg_id, speaker)
    write_json(job.dir / "transcript.json", tr)
    job.set(speakers=diarize.speaker_stats(tr, None, speakers))
    return job


# ---------------------------------------------------------------------------
# YouTube açıklaması
# ---------------------------------------------------------------------------
def _keeps(job: Job) -> list[tuple[float, float]]:
    return keep_ranges(job.cuts, job.data["media"]["duration"])


def _generate_youtube(job: Job, instruction: str = "") -> None:
    opts = job.data.get("options", {})
    prev = job.data.get("youtube") or {}
    extra = prev.get("extra", opts.get("yt_extra", ""))
    yt = youtube.generate(job.transcript, _keeps(job), job.data.get("speakers"), extra=extra,
                          instruction=instruction, previous=prev or None)
    job.set(youtube=yt, youtube_error=None)


def generate_youtube(job_id: str, instruction: str = "") -> Job:
    """Arayüzden: açıklamayı (yeniden) üretir. Video Edit işinde kurgulanmış videonun zamanlarını kullanır."""
    job = Job(job_id)
    prev_status = job.data.get("status")
    if prev_status in BUSY:
        prev_status = job.data.get("resume_status", "ready")
    try:
        job.set(status="running", stage="youtube", progress=0.5, message="YouTube açıklaması hazırlanıyor")
        _generate_youtube(job, instruction)
        job.set(status="ready" if prev_status != "done" else "done", stage="ready", progress=1.0,
                message="YouTube açıklaması hazır")
    except Exception as e:  # noqa: BLE001
        job.set(status="ready" if prev_status in ("ready", "done") else "error", stage="ready", progress=1.0,
                message=f"Açıklama oluşturulamadı: {e}", youtube_error=str(e))
    return job


def youtube_view(job_id: str) -> dict:
    """Arayüz için: düzenlenebilir bölümler (yayın zamanıyla), birleştirilmiş metin, uyarılar."""
    job = Job(job_id)
    yt = job.data.get("youtube")
    if not yt:
        return {"youtube": None, "error": job.data.get("youtube_error")}
    keeps = _keeps(job)
    text, warnings = youtube.compose(yt, keeps)
    chapters = [{**c, "t": youtube.to_output(float(c["start"]), keeps)} for c in yt.get("chapters", [])]
    return {"youtube": {**yt, "chapters": chapters}, "text": text, "warnings": warnings,
            "duration": round(sum(b - a for a, b in keeps), 2), "error": job.data.get("youtube_error")}


def update_youtube(job_id: str, data: dict) -> dict:
    """Kullanıcı düzenlemeleri. Bölümler 't' (yayın zamanı) ya da 'start' (kaynak zamanı) ile gelebilir."""
    job = Job(job_id)
    yt = dict(job.data.get("youtube") or {})
    keeps = _keeps(job)
    if "chapters" in data:
        chs = []
        for c in data["chapters"]:
            title = str(c.get("title", "")).strip()
            if not title:
                continue
            start = youtube.to_source(float(c["t"]), keeps) if c.get("t") is not None else float(c.get("start", 0))
            chs.append({"start": round(start, 3), "title": title[:100]})
        yt["chapters"] = sorted(chs, key=lambda c: c["start"])
    if "hashtags" in data:
        yt["hashtags"] = youtube.clean_hashtags(data["hashtags"])
    if "tags" in data:
        yt["tags"] = youtube.clean_tags(data["tags"])
    for k in ("intro", "extra"):
        if k in data:
            yt[k] = str(data[k])
    if "titles" in data:
        yt["titles"] = [str(t)[:100] for t in data["titles"] if str(t).strip()]
    job.set(youtube=yt)
    return youtube_view(job_id)


def _mixed_audio(job: Job, progress=None) -> Path | None:
    """Kişi bazlı ses ayarı varsa işlenmiş sesi üretir (önbellekli), yoksa None."""
    speakers = job.data.get("speakers")
    if not audiomix.needs_mix(speakers) or not job.data["media"].get("has_audio"):
        return None
    info = job.data["media"]
    turns = diarize.timeline_turns(job.transcript, info["duration"])
    path = job.dir / f"mix_{audiomix.mix_key(turns, speakers)}.wav"
    if not path.exists():
        for old in job.dir.glob("mix_*.wav"):
            old.unlink(missing_ok=True)
        audiomix.render_mix(job.source, path, turns, speakers, info.get("sample_rate", 48000),
                            info.get("channels", 2), info["duration"], progress=progress)
    return path
def update_cuts(job_id: str, cuts: list[dict]) -> Job:
    job = Job(job_id)
    clean = []
    for n, c in enumerate(cuts, 1):
        cut = Cut.from_dict(c)
        if cut.end <= cut.start:
            continue
        cut.id = cut.id or f"u{n:03d}"
        clean.append(cut.to_dict())
    job.set(cuts=sorted(clean, key=lambda c: c["start"]))
    return job


def update_reels(job_id: str, reels: list[dict]) -> Job:
    job = Job(job_id)
    out = []
    for n, r in enumerate(reels, 1):
        reel = Reel.from_dict(r)
        if reel.end > reel.start:
            reel.id = reel.id or f"reel{n:02d}"
            out.append(reel.to_dict())
    job.set(reels=sorted(out, key=lambda r: r["start"]))
    if job.data.get("status") not in ("running", "rendering"):
        cleanup_visual_files(job)   # ✕ ile silinen görsellerin dosyaları
    return job


# ---------------------------------------------------------------------------
# Reels görselleri
# ---------------------------------------------------------------------------
def _patch_reel(job: Job, reel_id: str, fn) -> dict:
    """Diskteki güncel reels listesinde tek bir reel'i değiştirir (arayüzün eş zamanlı kaydını ezmez)."""
    with _locks.setdefault(job.id, threading.Lock()):
        fresh = read_json(job.file, default={})
        reels = fresh.get("reels", [])
        reel = next((r for r in reels if r.get("id") == reel_id), None)
        if reel is None:
            raise KeyError(f"Reel yok: {reel_id}")
        result = fn(reel)
        fresh["reels"] = reels
        job.data = fresh
        job.save()
    return result


def cleanup_visual_files(job: Job) -> int:
    """Hiçbir reel'in kullanmadığı görsel dosyalarını siler (yeniden arama, ↻, ✕, yeniden bulma sonrası).
    Arka plan işi sürerken çağrılmamalı: indirilen ama henüz kaydedilmemiş dosyaları silebilir."""
    data = read_json(job.file, default={}) or {}
    used = {Path(v["file"]).name for r in data.get("reels", []) for v in (r.get("visuals") or []) if v.get("file")}
    removed = 0
    for p in job.visuals_dir.iterdir():
        if p.is_file() and p.name not in used:
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def _visual(reel: dict, vid: str) -> dict:
    v = next((v for v in reel.get("visuals", []) if v.get("id") == vid), None)
    if v is None:
        raise KeyError(f"Görsel yok: {vid}")
    return v


# ---------------------------------------------------------------------------
# Elle reel işaretleme (kurgu şeridi)
# ---------------------------------------------------------------------------
THUMB_STEP, THUMB_W, THUMB_H, THUMBS_PER_TILE = 5.0, 96, 54, 100
WAVE_W, WAVE_H = 8000, 80


def build_timeline_assets(job_id: str) -> Job:
    """Kurgu şeridi için küçük resim şeritleri ve ses dalga formu üretir (bir kez; uzun videoda 1-2 dk).
    Yalnızca anahtar kareler çözülür (-skip_frame nokey) — hızlıdır, küçük resim zamanı birkaç sn kayabilir."""
    job = Job(job_id)
    d = job.dir / "timeline"
    d.mkdir(exist_ok=True)
    try:
        job.set(timeline={"state": "building"})
        src = str(job.source.resolve())
        info = job.data.get("media") or media.probe(job.source)
        if info.get("has_video"):
            media._run_ffmpeg(["-skip_frame", "nokey", "-i", src, "-an", "-vf",
                               f"fps=1/{THUMB_STEP},scale={THUMB_W}:{THUMB_H},tile={THUMBS_PER_TILE}x1",
                               "-fps_mode", "vfr", "-q:v", "5", str(d / "thumbs_%03d.jpg")])
        if info.get("has_audio"):
            media._run_ffmpeg(["-i", src, "-vn", "-filter_complex",
                               f"aformat=channel_layouts=mono,showwavespic=s={WAVE_W}x{WAVE_H}:colors=0x4fa3d1:scale=sqrt",
                               "-frames:v", "1", str(d / "wave.png")])
        tiles = sorted(p.name for p in d.glob("thumbs_*.jpg"))
        job.set(timeline={"state": "ready", "step": THUMB_STEP, "tiles": tiles, "per_tile": THUMBS_PER_TILE,
                          "thumb_w": THUMB_W, "thumb_h": THUMB_H, "wave": (d / "wave.png").exists(),
                          "duration": info["duration"]})
    except Exception as e:  # noqa: BLE001 — şerit olmadan da işaretleme yapılabilir
        job.set(timeline={"state": "error", "error": str(e)[:300]})
    return Job(job_id)


def save_manual_reels(job_id: str, spans: list[dict]) -> Job:
    """Kurgu şeridinde işaretlenen reel aralıklarını kaydeder. Aynı kimlikteki reel'in başlığı, görselleri,
    kısaltması korunur; sınırı değişen reel'in kısaltması yeni sınıra göre geçersiz kalabilir (not düşülür)."""
    job = Job(job_id)
    old = {r["id"]: r for r in job.data.get("reels") or []}
    used = set()
    out = []
    for s in sorted(spans, key=lambda s: float(s["start"])):
        a, b = round(float(s["start"]), 3), round(float(s["end"]), 3)
        if b - a < 1.0:
            continue
        rid = s.get("id") if s.get("id") in old and s.get("id") not in used else ""
        if rid:
            r = dict(old[rid])
            if abs(r["start"] - a) > 0.05 or abs(r["end"] - b) > 0.05:
                if r.get("tight_level"):
                    r["tight_note"] = "Sınır değişti — ✂ Kısalt'ı yeniden çalıştırın. " + (r.get("tight_note") or "")
            r.update(start=a, end=b)
        else:
            r = Reel(a, b, source="user", confidence=1.0).to_dict()
        used.add(rid)
        out.append(r)
    taken = {r["id"] for r in out if r.get("id")}
    n = 1
    for r in out:
        if not r.get("id"):
            while f"reel{n:02d}" in taken:
                n += 1
            r["id"] = f"reel{n:02d}"
            taken.add(r["id"])
    job.set(reels=[Reel.from_dict(r).to_dict() for r in out])
    return Job(job_id)


def prepare_reels(job_id: str, reel_ids: list[str] | None = None, level: str | None = "light") -> Job:
    """Elle işaretlenen reels için: başlık/kanca/açıklama (Claude) + kısaltma. Arka planda çalışır."""
    from .reels import describe_reel
    job = Job(job_id)
    prev = resume_status(job)
    try:
        tr, cuts = job.transcript, job.cuts
        use_claude = job.data.get("options", {}).get("claude", True)
        targets = [r.to_dict() for r in job.reels if r.enabled and (not reel_ids or r.id in reel_ids)]
        if not targets:
            raise ValueError("Hazırlanacak reel yok — önce kurgu şeridinde işaretleyin")
        for n, r in enumerate(targets):
            job.set(status="running", stage="prepare", progress=n / len(targets),
                    message=f"{r['id']}: başlık ve açıklama yazılıyor ({n + 1}/{len(targets)})")
            fields = describe_reel(tr["segments"], r) if (use_claude and settings.has_claude) else {}
            if level in tighten.LEVELS:
                job.set(message=f"{r['id']}: kısaltılıyor ({n + 1}/{len(targets)})")
                res, cuts = _tighten_one(job, tr, r, cuts, level, None, use_claude)
                fields.update(res)
                job.set(cuts=[c.to_dict() for c in cuts])
            _patch_reel(job, r["id"], lambda reel, f=fields: reel.update(f))
        job.set(status=prev, stage=prev, progress=1.0, message=f"{len(targets)} reel hazırlandı — indirmek için \"Seçili reels'leri oluştur\" butonuna bas")
    except Exception as e:  # noqa: BLE001
        job.set(status=prev, stage=prev, progress=0.0, message=f"Hazırlama hatası: {e}",
                prepare_error=traceback.format_exc()[-2000:])
    return Job(job_id)


def job_layout(job: Job, use_claude: bool = True, force: bool = False, t: float | None = None) -> dict:
    """İşin dikey yerleşim bölgeleri (oyun/ekran + kameralar). Yoksa bir kareden bulunur ve saklanır.
    Kare, ilk reel'in içinden seçilir (çekim başındaki hazırlık anı değil, oyun ekrandayken)."""
    regions = job.data.get("layout_regions")
    if regions and not force:
        return regions
    info = job.data["media"]
    if t is None:
        rs = job.data.get("reels") or []
        t = (rs[0]["start"] + 0.4 * (rs[0]["end"] - rs[0]["start"])) if rs else info["duration"] * 0.2
    regions = layout.detect(job.source, t, info["width"], info["height"], use_claude=use_claude)
    job.set(layout_regions=regions)
    return regions


def reel_layout(job: Job, reel: dict, mode: str, use_claude: bool = True) -> tuple[str, dict]:
    """Bir reel'in dikey yerleşimi: (mod, bölgeler). Her reel kendi karesine bakılarak tespit edilir:
    ekranda yalnızca yüzler varsa sohbet ("talk": yüzler alt alta), oyun/ekran varsa seçilen oyun modu.
    reel["layout_kind"] ("talk"/"game") otomatik kararı geçersiz kılar. Arayüzde elle düzeltilmiş
    bölgeler (source=user) oyun reel'lerinde kullanılır. Tespit reel'e yazılır (sınır değişince yenilenir)."""
    info = job.data["media"]
    span = [round(reel["start"], 2), round(reel["end"], 2)]
    det = reel.get("layout_regions") or {}
    if det.get("span") != span:
        det = layout.detect_reel(job.source, reel["start"], reel["end"], info["width"], info["height"], use_claude)
        det["span"] = span
        _patch_reel(job, reel["id"], lambda r, d=det: r.__setitem__("layout_regions", d))
    kind = reel.get("layout_kind") or ("talk" if mode == "talk" or det.get("kind") == "talk" else "game")
    if kind == "talk":
        return "talk", det
    mode = mode if mode in layout.GAME_MODES else "faces_top"
    user = job.data.get("layout_regions") or {}
    if user.get("source") == "user":
        return mode, user
    if det.get("kind") == "game":
        return mode, det
    return mode, job_layout(job, use_claude)   # bu reel'de oyun görülmedi ama elle "oyun" seçildi


def _tighten_one(job: Job, tr: dict, reel: dict, cuts: list[Cut], level: str, target_sec: float | None,
                 use_claude: bool, quiet_all: list[tuple[float, float]] | None = None) -> tuple[dict, list[Cut]]:
    """Bir reel'i kısaltır. Önce reel aralığındaki gerçek sessizliği ölçer ve genel temizliğin 'sessizlik'
    kesimlerini sesle doğrular (kahkaha/tepki kesilmesin); Claude'a reel'den 3 kare verir.
    Döner: (reel'e yazılacak alanlar, güncellenmiş genel kesimler)."""
    a, b = reel["start"], reel["end"]
    if quiet_all is not None:
        quiet = [q for q in quiet_all if q[1] > a and q[0] < b]
    else:
        quiet = media.quiet_intervals(job.source, a, b)
    cuts = tighten.refine_silence_cuts(cuts, quiet, a, b)
    context = job.data.get("reel_context", "")
    with tempfile.TemporaryDirectory(prefix="vedit_tight_") as td:
        frames = []
        if use_claude and settings.has_claude and job.data.get("media", {}).get("has_video"):
            for k, f in enumerate((0.2, 0.5, 0.8)):
                t = a + f * (b - a)
                p = Path(td) / f"t{k}.jpg"
                try:
                    media.extract_frame(job.source, t, p, width=640)
                    frames.append((t, p))
                except RuntimeError:
                    pass
        res = tighten.tighten_reel(tr, reel, cuts, level, target_sec, use_claude=use_claude, quiet=quiet,
                                   frames=frames, context=context)
    return res, cuts


def check_reel_end(job_id: str, reel_id: str) -> Job:
    """Tek bir reel'in bitişini Claude'la yeniden kontrol eder (oyun/konu gerçekten bitti mi). Reel en çok
    sonraki reel'in başına kadar uzar. Bitiş değişirse kısaltma kesimleri yeni kısmı kapsamaz; not düşülür."""
    from .reels import refine_ends
    job = Job(job_id)
    if not settings.has_claude:
        raise ValueError("Bitiş kontrolü için Claude bağlantısı gerekiyor")
    all_reels = sorted(job.reels, key=lambda r: r.start)
    target = next((r for r in all_reels if r.id == reel_id), None)
    if target is None:
        raise KeyError(f"Reel yok: {reel_id}")
    others = [r for r in all_reels if r.enabled or r.id == reel_id]
    old_end = target.end
    refine_ends(job.source, job.transcript["segments"], others, job.data["media"])
    if abs(target.end - old_end) < 0.01:
        job.set(message=f"{reel_id}: bitiş doğru görünüyor ({old_end:.1f} sn)")
        return Job(job_id)

    def fn(reel):
        reel.update(end=target.end, notes=target.notes)
        if reel.get("tight_level"):
            reel["tight_note"] = ("Bitiş değişti — yeni kısım için ✂ Kısalt'ı yeniden çalıştırın. "
                                  + (reel.get("tight_note") or "")).strip()
    _patch_reel(job, reel_id, fn)
    job.set(message=f"{reel_id}: bitiş {old_end:.1f} → {target.end:.1f} sn")
    return Job(job_id)


def tighten_reels(job_id: str, reel_ids: list[str] | None = None, level: str = "medium",
                  target_sec: float | None = None) -> Job:
    """Seçili reels'i kendi içinde sıkılaştırır (arka planda). Önceki sıkılaştırma kesimleri değiştirilir."""
    job = Job(job_id)
    prev = resume_status(job)
    try:
        tr, cuts = job.transcript, job.cuts
        targets = [r.to_dict() for r in job.reels if (not reel_ids or r.id in reel_ids)]
        if not targets:
            raise ValueError("Sıkılaştırılacak reel seçilmedi")
        saved = 0.0
        for n, r in enumerate(targets):
            job.set(status="running", stage="tighten", progress=n / len(targets),
                    message=f"{r['id']} sıkılaştırılıyor ({n + 1}/{len(targets)})")
            res, cuts = _tighten_one(job, tr, r, cuts, level, target_sec,
                                     use_claude=job.data.get("options", {}).get("claude", True))
            saved += res["tight_before"] - res["tight_after"]
            _patch_reel(job, r["id"], lambda reel, res=res: reel.update(res))
            job.set(cuts=[c.to_dict() for c in cuts])   # sesle doğrulanmış sessizlik kesimleri
        job.set(status=prev, stage=prev, progress=1.0,
                message=f"{len(targets)} reel sıkılaştırıldı — toplam {saved:.0f} sn kısaldı")
    except Exception as e:  # noqa: BLE001
        job.set(status=prev, stage=prev, progress=0.0, message=f"Sıkılaştırma hatası: {e}",
                tighten_error=traceback.format_exc()[-2000:])
    return Job(job_id)


def _find_visuals_for(job: Job, ids: list[str], progress=None) -> tuple[int, list[str]]:
    """Verilen reels için Claude'la görsel anlarını bulur, görselleri indirir, reel'e yazar.
    Döner: (bulunan görsel sayısı, benzersiz uyarılar)."""
    tr = job.transcript
    targets = [r.to_dict() for r in job.reels if r.id in ids]
    all_errors: list[str] = []
    total = 0
    for n, r in enumerate(targets):
        job.set(message=f"{r['id']}: Claude görsel anlarını belirliyor ({n + 1}/{len(targets)})")
        sub = (lambda f, m, n=n: progress((n + f) / len(targets), m)) if progress else None
        found, errors = visuals.find_for_reel(tr, r, job.visuals_dir, progress=sub)
        all_errors += errors
        total += sum(1 for v in found if v.get("file"))

        def put(reel, f=found):
            reel["visuals"] = f
            reel["enrich_done"] = True
        _patch_reel(job, r["id"], put)
    cleanup_visual_files(job)   # önceki aramanın artık kullanılmayan görselleri
    return total, list(dict.fromkeys(all_errors))


def find_visuals(job_id: str, reel_ids: list[str] | None = None) -> Job:
    """Seçili reels için Claude'la görsel anlarını bulur ve görselleri indirir (arka planda çalışır)."""
    job = Job(job_id)
    prev = resume_status(job)
    try:
        if not settings.has_claude:
            raise ValueError("Görsel anlarını bulmak için Claude bağlantısı gerekiyor (README > Claude bağlantısı). "
                             "Claude olmadan da '+ Görsel' ile elle ekleyebilirsiniz.")
        ids = [r.id for r in job.reels if r.enabled and r.enrich and (not reel_ids or r.id in reel_ids)]
        if not ids:
            raise ValueError("'İçerik ekle' işaretli seçili reel yok — görsel istediğin reel'de kutucuğu işaretle")
        job.set(status="running", stage="visuals", progress=0.0, message="Görseller aranıyor")
        total, uniq = _find_visuals_for(job, ids, job.progress("visuals", 0.0, 1.0))
        job.set(status=prev, stage=prev, progress=1.0,
                message=f"{total} görsel bulundu" + (f" — uyarılar: {'; '.join(uniq[:3])}" if uniq else ""))
    except Exception as e:  # noqa: BLE001
        job.set(status=prev, stage=prev, progress=0.0, message=f"Görsel bulma hatası: {e}",
                visuals_error=traceback.format_exc()[-2000:])
    return Job(job_id)


def visual_next(job_id: str, reel_id: str, vid: str) -> Job:
    """Sıradaki aday görsele geçer."""
    job = Job(job_id)

    def fn(reel):
        v = _visual(reel, vid)
        if len(v.get("candidates") or []) < 2:
            raise ValueError("Başka aday yok — aramayı değiştirin ya da dosya yükleyin")
        visuals.choose(v, v.get("choice", -1) + 1, job.visuals_dir)
    _patch_reel(job, reel_id, fn)
    cleanup_visual_files(job)
    return job


def visual_search(job_id: str, reel_id: str, vid: str, query: str, source: str = "auto") -> tuple[Job, list[str]]:
    """Görseli yeni bir aramayla yeniden bulur. vid boşsa sorgu metniyle yeni görsel yaratılmaz (önce ekleyin)."""
    job = Job(job_id)
    errors: list[str] = []

    def fn(reel):
        v = _visual(reel, vid)
        q = query.strip()
        src = source if source in ("auto", "wikipedia", "pexels") else "auto"
        if q:
            v.update(wiki=q, label=v.get("label") or q)
            # Kişi/takım gibi varlıklarda Claude'un verdiği genel stok sorgusu ("football player") korunur;
            # kullanıcı Pexels'i açıkça seçtiyse ya da stok sorgusu yoksa yazdığı arama kullanılır.
            if src == "pexels" or not v.get("stock") or v.get("kind") not in visuals.ENTITY_KINDS:
                v["stock"] = q
        errors.extend(visuals.fill(v, job.visuals_dir, src, prefer_wiki=bool(q)))
    _patch_reel(job, reel_id, fn)
    cleanup_visual_files(job)
    return job, errors


def visual_upload(job_id: str, reel_id: str, vid: str, src: Path, suffix: str) -> Job:
    """Görselin yerine kullanıcının dosyasını koyar."""
    job = Job(job_id)
    if suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        raise ValueError("Desteklenen görsel türleri: jpg, png, webp")

    def fn(reel):
        v = _visual(reel, vid)
        dst = job.visuals_dir / f"{v['id']}_up{suffix.lower()}"
        shutil.move(str(src), dst)
        if v.get("file") and v["file"] != dst.name:
            (job.visuals_dir / Path(v["file"]).name).unlink(missing_ok=True)
        v.update(file=dst.name, source="upload", credit="Kendi dosyanız", page="", choice=-1)
    _patch_reel(job, reel_id, fn)
    cleanup_visual_files(job)
    return job


# ---------------------------------------------------------------------------
# Render ve dışa aktarma
# ---------------------------------------------------------------------------
def _stem(job: Job) -> str:
    return Path(job.data["original_name"]).stem.replace(" ", "_")[:60] or "video"


def export_edit(job_id: str) -> dict[str, Path]:
    job = Job(job_id)
    info = job.data["media"]
    keeps = keep_ranges(job.cuts, info["duration"])
    files = exporters.write_all(job.out_dir, f"{_stem(job)}_edit", keeps, info, job.source, job.transcript,
                                extra={"cuts": job.data.get("cuts", [])})
    job.set(outputs={**job.data.get("outputs", {}), **{k: v.name for k, v in files.items()}})
    return files


def render_edit(job_id: str, vertical: str | None = None) -> Job:
    job = Job(job_id)
    try:
        job.set(status="rendering", stage="render", progress=0.0, message="Kurgulanmış video oluşturuluyor")
        info = job.data["media"]
        keeps = keep_ranges(job.cuts, info["duration"])
        files = export_edit(job_id)
        mix = _mixed_audio(job, job.progress("render", 0.0, 0.15))
        dst = job.out_dir / f"{_stem(job)}_edit.mp4"
        media.render_keeps(job.source, keeps, dst, progress=job.progress("render", 0.15 if mix else 0.0, 0.85 if mix else 1.0),
                           vertical=vertical, workdir=job.dir, audio_src=mix)
        outputs = {**Job(job_id).data.get("outputs", {}), "video": dst.name}
        removed = info["duration"] - sum(b - a for a, b in keeps)
        job.set(status="done", stage="done", progress=1.0, outputs=outputs,
                message=f"Hazır — {removed:.1f} sn kesildi, {len(files)} dışa aktarım dosyası üretildi")
    except Exception as e:  # noqa: BLE001
        job.set(status="error", stage="error", message=str(e), error=traceback.format_exc()[-3000:])
    return job


def render_reels(job_id: str, reel_ids: list[str] | None = None, vertical: str | None = "crop",
                 style: dict | None = None) -> Job:
    """style: {"captions": "bold"|"karaoke"|"minimal"|None, "caption_pos": "bottom"|"middle",
               "visuals": bool, "visual_pos": "top"|"center"|"full"}"""
    job = Job(job_id)
    try:
        reels = [r for r in job.reels if r.enabled and (not reel_ids or r.id in reel_ids)]
        if not reels:
            raise ValueError("Render edilecek reel seçilmedi")
        st = {**job.data.get("reel_style", {}), **(style or {})}
        job.set(reel_style=st)
        info, tr, cuts = job.data["media"], job.transcript, job.cuts
        out_w, out_h = media.output_size(info, vertical)
        use_claude = job.data.get("options", {}).get("claude", True)
        # İçerik ekle işaretli ama henüz görseli aranmamış reels: önce görselleri bul
        pending = [r.id for r in reels if r.enrich and not r.enrich_done]
        if pending and st.get("visuals", True) and settings.has_claude:
            job.set(status="rendering", stage="visuals", progress=0.0,
                    message=f"İçerik ekle: {len(pending)} reel için görseller aranıyor")
            _find_visuals_for(job, pending, job.progress("visuals", 0.0, 1.0))
            reels = [r for r in Job(job_id).reels if r.enabled and (not reel_ids or r.id in reel_ids)]
        outputs = dict(job.data.get("outputs", {}))
        reel_outputs = dict(outputs.get("reels", {}))
        insta_lines = []   # Instagram açıklamaları (captions modülüyle karışmasın)
        skipped: list[str] = []   # seçili olduğu halde eklenemeyen görseller
        mix = _mixed_audio(job, job.progress("render", 0.0, 0.05))
        for n, r in enumerate(reels):
            job.set(status="rendering", stage="render", progress=n / len(reels),
                    message=f"{r.id} render ediliyor ({n + 1}/{len(reels)})")
            keeps = tighten.reel_keeps(r.to_dict(), cuts)
            base = f"{_stem(job)}_{r.id}"
            mp4 = job.out_dir / f"{base}.mp4"
            lay = None
            if vertical in layout.MODES and info.get("has_video"):
                # Her reel kendi düzenine göre: sohbet reel'i yüzler alt alta, oyun reel'i seçilen mod
                job.set(message=f"{r.id}: ekran düzeni belirleniyor")
                rmode, regions = reel_layout(job, r.to_dict(), vertical, use_claude)
                lay = layout.plan(rmode, info["width"], info["height"], regions)
                job.set(message=f"{r.id} render ediliyor ({n + 1}/{len(reels)}) — "
                                f"{'sohbet: yüzler alt alta' if lay['mode'] == 'talk' else 'oyun + yüzler'}")
            reel_skipped: list[str] = []
            items = visuals.overlay_plan(r.visuals, keeps, out_w, out_h, st.get("visual_pos", "top"),
                                         job.visuals_dir, media.probe, skipped=reel_skipped,
                                         area=lay["game_area"] if lay else None) \
                if st.get("visuals", True) and r.enrich else []
            skipped += [f"{r.id}: {s}" for s in reel_skipped]
            ass = captions.build_ass(tr, keeps, out_w, out_h, style=st.get("captions"),
                                     position=st.get("caption_pos", "bottom"), visual_items=items,
                                     foreign=captions.foreign_names(r.visuals),
                                     margin_v=lay["caption_margin"] if lay else None) \
                if (st.get("captions") or items) else None
            media.render_keeps(job.source, keeps, mp4, vertical=vertical, workdir=job.dir, audio_src=mix,
                               overlays=items, ass_text=ass, layout_plan=lay,
                               progress=job.progress("render", n / len(reels), 1 / len(reels)))
            srt = job.out_dir / f"{base}.srt"
            srt.write_text(exporters.to_srt(tr, keeps, max_chars=28, max_dur=2.2), encoding="utf-8")
            reel_outputs[r.id] = {"video": mp4.name, "srt": srt.name}
            insta_lines.append(f"### {r.id} — {r.title}\n{r.caption}\n{' '.join(r.hashtags)}\n")
        cap = job.out_dir / f"{_stem(job)}_captions.md"
        cap.write_text("\n".join(insta_lines), encoding="utf-8")
        outputs.update(reels=reel_outputs, captions=cap.name)
        msg = f"{len(reels)} reel hazır"
        if skipped:
            msg += f" — eklenemeyen görseller ({len(skipped)}): " + "; ".join(skipped[:5]) + \
                   (" …" if len(skipped) > 5 else "") + ". Bunları ⬆ ile yeniden yükleyin ya da 🔍 ile arayın."
        job.set(status="done", stage="done", progress=1.0, outputs=outputs, message=msg)
    except Exception as e:  # noqa: BLE001
        job.set(status="error", stage="error", message=str(e), error=traceback.format_exc()[-3000:])
    return job


BUSY = ("queued", "running", "rendering")


def mark_started(job_id: str, status: str, stage: str, message: str) -> None:
    """Arka plan işini başlatmadan ÖNCE işi meşgul işaretler. Böyle olmazsa arayüzün ilk durum sorgusu
    iş parçacığı daha durumu güncellemeden gelir, eski 'hazır' durumunu görür ve takibi bırakırdı
    (render bitse de ekran 'oluşturuluyor'da kalırdı). İş bitince dönülecek durum saklanır."""
    job = Job(job_id)
    cur = job.data.get("status", "ready")
    job.set(status=status, stage=stage, progress=0.0, message=message,
            resume_status=cur if cur in ("ready", "done", "error") else job.data.get("resume_status", "ready"))


def resume_status(job: Job) -> str:
    """İş bitince dönülecek durum (ready/done): mark_started ile işaretlenmişse onun sakladığı."""
    s = job.data.get("status", "ready")
    if s in BUSY:
        s = job.data.get("resume_status", "ready")
    return s if s in ("ready", "done") else "ready"


def run_in_thread(fn, *args, **kw) -> threading.Thread:
    t = threading.Thread(target=fn, args=args, kwargs=kw, daemon=True)
    t.start()
    return t
