"""Web arayüzü ve REST API.

Çalıştırma:  python -m uvicorn server.app:app --port 8000   (veya run.sh / run.bat)
Arayüz:      http://localhost:8000
"""
from __future__ import annotations

import shutil
import sys
from urllib.parse import quote
import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedit import exporters, pipeline  # noqa: E402
from vedit.claude_client import backend_status  # noqa: E402
from vedit.config import settings  # noqa: E402
from vedit.detector import keep_ranges  # noqa: E402

STATIC = Path(__file__).parent / "static"
settings.data_dir.mkdir(parents=True, exist_ok=True)
pipeline.recover_stale_jobs()  # önceki oturumda takılı/yarım kalan işleri kurtar
app = FastAPI(title="Video Edit Studio", version="1.0.0")


def _job(job_id: str) -> pipeline.Job:
    job = pipeline.Job(job_id)
    if not job.data:
        raise HTTPException(404, "İş bulunamadı")
    return job


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    ok = {}
    for mod in ("faster_whisper",):
        try:
            __import__(mod)
            ok[mod] = True
        except ImportError:
            ok[mod] = False
    cb = backend_status()
    from vedit.diarize import choose_backend
    db = choose_backend()
    return {"diarization": {"pyannote": "pyannote", "light": "hafif model", "off": "kapalı"}.get(db, db),"ffmpeg": bool(shutil.which("ffmpeg")), "claude": settings.has_claude,
            "claude_label": cb["label"], "claude_backend": cb["backend"],
            "whisper_model": settings.whisper_model, **ok}


@app.get("/api/jobs")
def jobs():
    return pipeline.list_jobs()


@app.post("/api/jobs")
async def create(file: UploadFile = File(...), mode: str = Form("edit"),
                 claude: bool = Form(True), silence: bool = Form(True), stutter: bool = Form(True),
                 diarize: bool = Form(True), speakers: int = Form(0), yt_extra: str = Form("")):
    if mode not in ("edit", "reels", "youtube"):
        raise HTTPException(400, "mode edit|reels|youtube olmalı")
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.data_dir) as tmp:
        shutil.copyfileobj(file.file, tmp, length=16 * 1024 * 1024)
    job = pipeline.create_job(tmp.name, mode, file.filename,
                              {"claude": claude, "silence": silence, "stutter": stutter,
                               "diarize": diarize, "speakers": speakers if 1 <= speakers <= 10 else None,
                               "yt_extra": yt_extra[:2000]},
                              move=True)
    pipeline.run_in_thread(pipeline.analyze, job.id)
    return job.public()


@app.get("/api/jobs/{job_id}")
def get(job_id: str):
    job = _job(job_id)
    d = job.public()
    if job.data.get("media") and job.data.get("cuts") is not None:
        keeps = keep_ranges(job.cuts, job.data["media"]["duration"])
        d["kept_duration"] = round(sum(b - a for a, b in keeps), 2)
    return d


@app.get("/api/jobs/{job_id}/transcript")
def transcript(job_id: str):
    job = _job(job_id)
    try:
        return job.transcript
    except FileNotFoundError:
        raise HTTPException(404, "Transkript henüz hazır değil")


@app.put("/api/jobs/{job_id}/cuts")
def put_cuts(job_id: str, cuts: list[dict] = Body(...)):
    _job(job_id)
    return pipeline.update_cuts(job_id, cuts).public()


@app.put("/api/jobs/{job_id}/reels")
def put_reels(job_id: str, reels: list[dict] = Body(...)):
    _job(job_id)
    return pipeline.update_reels(job_id, reels).public()


@app.put("/api/jobs/{job_id}/speakers")
def put_speakers(job_id: str, changes: dict = Body(...)):
    """{"S1": {"name": "Ahmet", "gain_db": -6, "mute": false}, ...}"""
    _job(job_id)
    return pipeline.update_speakers(job_id, changes).public()


@app.post("/api/jobs/{job_id}/speakers/equalize")
def equalize(job_id: str):
    _job(job_id)
    return pipeline.equalize_speakers(job_id).public()


@app.post("/api/jobs/{job_id}/speakers/rediarize")
def rediarize(job_id: str, speakers: int = Body(0, embed=True)):
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "İş zaten çalışıyor")
    pipeline.run_in_thread(pipeline.rediarize, job_id, speakers if 1 <= speakers <= 10 else None)
    return {"ok": True}


@app.put("/api/jobs/{job_id}/segments/{seg_id}/speaker")
def put_segment_speaker(job_id: str, seg_id: int, speaker: str = Body(..., embed=True)):
    """Bir cümlenin konuşmacısını düzeltir. speaker: "S2" ya da yeni kişi için "new"."""
    _job(job_id)
    return pipeline.set_segment_speaker(job_id, seg_id, speaker).public()


@app.get("/api/jobs/{job_id}/youtube")
def get_youtube(job_id: str):
    """Düzenlenebilir bölümler (yayın zamanıyla), birleştirilmiş açıklama metni ve uyarılar."""
    _job(job_id)
    return pipeline.youtube_view(job_id)


@app.put("/api/jobs/{job_id}/youtube")
def put_youtube(job_id: str, data: dict = Body(...)):
    _job(job_id)
    return pipeline.update_youtube(job_id, data)


@app.post("/api/jobs/{job_id}/youtube/generate")
def gen_youtube(job_id: str, instruction: str = Body("", embed=True)):
    """Açıklamayı (yeniden) üretir; instruction ör. 'daha kısa', 'daha samimi', 'İngilizce'."""
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "İş zaten çalışıyor")
    pipeline.run_in_thread(pipeline.generate_youtube, job_id, instruction[:500])
    return {"ok": True}


@app.get("/api/jobs/{job_id}/youtube/download", response_class=PlainTextResponse)
def download_youtube(job_id: str):
    job = _job(job_id)
    view = pipeline.youtube_view(job_id)
    if not view.get("youtube"):
        raise HTTPException(404, "Açıklama henüz yok")
    yt = view["youtube"]
    titles = "\n".join(f"- {t}" for t in yt.get("titles", []))
    body = (f"BAŞLIK ÖNERİLERİ\n{titles}\n\nAÇIKLAMA\n{view['text']}\n\n"
            f"ETİKETLER (Studio > Etiketler alanına)\n{', '.join(yt.get('tags', []))}\n")
    name = Path(job.data["original_name"]).stem + "_youtube.txt"
    return PlainTextResponse(body, headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})


@app.post("/api/jobs/{job_id}/reanalyze")
def reanalyze(job_id: str, claude: bool = Body(True, embed=True)):
    job = _job(job_id)
    job.set(options={**job.data.get("options", {}), "claude": claude})
    pipeline.run_in_thread(pipeline.analyze, job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/render")
def render(job_id: str, vertical: str | None = Body(None, embed=True)):
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "İş zaten çalışıyor")
    pipeline.run_in_thread(pipeline.render_edit, job_id, vertical if vertical in ("crop", "blur") else None)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/reels/render")
def render_reels(job_id: str, ids: list[str] | None = Body(None, embed=True),
                 vertical: str | None = Body("crop", embed=True)):
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "İş zaten çalışıyor")
    pipeline.run_in_thread(pipeline.render_reels, job_id, ids,
                           vertical if vertical in ("crop", "blur") else None)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/export/{fmt}")
def export(job_id: str, fmt: str):
    job = _job(job_id)
    if fmt not in ("edl", "xml", "srt", "json"):
        raise HTTPException(400, "Format: edl | xml | srt | json")
    files = pipeline.export_edit(job_id)
    return FileResponse(files[fmt], filename=files[fmt].name)


@app.get("/api/jobs/{job_id}/source")
def source(job_id: str):
    return FileResponse(_job(job_id).source)


@app.get("/api/jobs/{job_id}/files/{name}")
def output_file(job_id: str, name: str):
    job = _job(job_id)
    p = (job.out_dir / name).resolve()
    if job.out_dir.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "Dosya yok")
    return FileResponse(p, filename=p.name)


@app.get("/api/jobs/{job_id}/reels/{reel_id}/srt", response_class=PlainTextResponse)
def reel_srt_preview(job_id: str, reel_id: str):
    job = _job(job_id)
    reel = next((r for r in job.reels if r.id == reel_id), None)
    if not reel:
        raise HTTPException(404, "Reel yok")
    keeps = keep_ranges(job.cuts, job.data["media"]["duration"], within=(reel.start, reel.end)) \
        if reel.clean else [(reel.start, reel.end)]
    return exporters.to_srt(job.transcript, keeps, max_chars=28, max_dur=2.2)


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    job = _job(job_id)
    shutil.rmtree(job.dir)
    return {"ok": True}
