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
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import audiomix, diarize, exporters, media, youtube
from .config import settings
from .detector import Cut, detect_all, keep_ranges
from .reels import Reel, split_reels
from .transcribe import transcribe

_locks: dict[str, threading.Lock] = {}


def jobs_root() -> Path:
    p = settings.data_dir / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.dir = jobs_root() / job_id
        self.file = self.dir / "job.json"
        self.data: dict = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}

    # --- kalıcılık ---
    def save(self) -> None:
        self.data["updated_at"] = time.time()
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.file)

    def set(self, **kw) -> None:
        with _locks.setdefault(self.id, threading.Lock()):
            # Diskteki güncel hali temel al: başka bir iş parçacığının yazdıklarını ezme
            fresh = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else dict(self.data)
            fresh.update(kw)
            self.data = fresh
            self.save()

    def progress(self, stage: str, base: float, span: float):
        def fn(frac: float, msg: str = "") -> None:
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
        return json.loads((self.dir / "transcript.json").read_text(encoding="utf-8"))

    @property
    def cuts(self) -> list[Cut]:
        return [Cut.from_dict(c) for c in self.data.get("cuts", [])]

    @property
    def reels(self) -> list[Reel]:
        return [Reel.from_dict(r) for r in self.data.get("reels", [])]

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
        job.set(cuts=[c.to_dict() for c in cuts])

        if job.data["mode"] == "reels":
            reels, method = split_reels(job.source, tr, info, progress=job.progress("reels", 0.75, 0.23))
            job.set(reels=[r.to_dict() for r in reels], reels_method=method)

        job.set(status="ready", stage="ready", progress=1.0, message="Analiz tamamlandı — gözden geçirin")
    except Exception as e:  # noqa: BLE001
        job.set(status="error", stage="error", message=str(e), error=traceback.format_exc()[-3000:])
    return job


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
    (job.dir / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
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
    (job.dir / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
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


def render_reels(job_id: str, reel_ids: list[str] | None = None, vertical: str | None = "crop") -> Job:
    job = Job(job_id)
    try:
        reels = [r for r in job.reels if r.enabled and (not reel_ids or r.id in reel_ids)]
        if not reels:
            raise ValueError("Render edilecek reel seçilmedi")
        info, tr, cuts = job.data["media"], job.transcript, job.cuts
        outputs = dict(job.data.get("outputs", {}))
        reel_outputs = dict(outputs.get("reels", {}))
        captions = []
        mix = _mixed_audio(job, job.progress("render", 0.0, 0.05))
        for n, r in enumerate(reels):
            job.set(status="rendering", stage="render", progress=n / len(reels),
                    message=f"{r.id} render ediliyor ({n + 1}/{len(reels)})")
            keeps = keep_ranges(cuts, info["duration"], within=(r.start, r.end)) if r.clean else [(r.start, r.end)]
            base = f"{_stem(job)}_{r.id}"
            mp4 = job.out_dir / f"{base}.mp4"
            media.render_keeps(job.source, keeps, mp4, vertical=vertical, workdir=job.dir, audio_src=mix,
                               progress=job.progress("render", n / len(reels), 1 / len(reels)))
            srt = job.out_dir / f"{base}.srt"
            srt.write_text(exporters.to_srt(tr, keeps, max_chars=28, max_dur=2.2), encoding="utf-8")
            reel_outputs[r.id] = {"video": mp4.name, "srt": srt.name}
            captions.append(f"### {r.id} — {r.title}\n{r.caption}\n{' '.join(r.hashtags)}\n")
        cap = job.out_dir / f"{_stem(job)}_captions.md"
        cap.write_text("\n".join(captions), encoding="utf-8")
        outputs.update(reels=reel_outputs, captions=cap.name)
        job.set(status="done", stage="done", progress=1.0, outputs=outputs,
                message=f"{len(reels)} reel hazır")
    except Exception as e:  # noqa: BLE001
        job.set(status="error", stage="error", message=str(e), error=traceback.format_exc()[-3000:])
    return job


def run_in_thread(fn, *args, **kw) -> threading.Thread:
    t = threading.Thread(target=fn, args=args, kwargs=kw, daemon=True)
    t.start()
    return t
