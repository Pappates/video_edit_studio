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

from vedit import captions, exporters, layout, pipeline, tighten, visuals  # noqa: E402
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
            "whisper_model": settings.whisper_model, "pexels": bool(settings.pexels_api_key), **ok}


@app.get("/api/jobs")
def jobs():
    return pipeline.list_jobs()


@app.post("/api/jobs")
async def create(file: UploadFile = File(...), mode: str = Form("edit"),
                 claude: bool = Form(True), silence: bool = Form(True), stutter: bool = Form(True),
                 diarize: bool = Form(True), speakers: int = Form(0), yt_extra: str = Form(""),
                 max_reels: int = Form(0), tighten_level: str = Form("medium"), split: str = Form("auto")):
    if mode not in ("edit", "reels", "youtube"):
        raise HTTPException(400, "mode edit|reels|youtube olmalı")
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.data_dir) as tmp:
        shutil.copyfileobj(file.file, tmp, length=16 * 1024 * 1024)
    job = pipeline.create_job(tmp.name, mode, file.filename,
                              {"claude": claude, "silence": silence, "stutter": stutter,
                               "diarize": diarize, "speakers": speakers if 1 <= speakers <= 10 else None,
                               "yt_extra": yt_extra[:2000],
                               # reels: ilk N reel bulununca dur (0 = videonun hepsi)
                               "max_reels": max_reels if 1 <= max_reels <= 50 else 0,
                               # reels: bölündükten sonra her reel'i kendi içinde kısalt ("" = kapalı)
                               "tighten": tighten_level if tighten_level in tighten.LEVELS else "",
                               # reels: "manual" → sınırları kullanıcı kurgu şeridinde işaretler
                               "manual": mode == "reels" and split == "manual"},
                              move=True)
    pipeline.run_in_thread(pipeline.analyze, job.id)
    if mode == "reels":
        pipeline.run_in_thread(pipeline.build_timeline_assets, job.id)   # kurgu şeridi küçük resimleri
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
    job = _job(job_id)
    if job.data.get("status") == "running":
        # Arka planda görsel bulma / kısaltma / hazırlama reels'i tek tek güncelliyor; tüm listeyi
        # yazmak onların sonuçlarını silerdi (ya da onlar bu düzenlemeyi ezerdi).
        # Render ise reels'i başta okur ve geri yazmaz: render sürerken düzenleme serbest
        # (değişiklik bir sonraki render'da geçerli olur).
        raise HTTPException(409, "Arka planda bir işlem sürüyor — bitince tekrar deneyin")
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
    pipeline.mark_started(job_id, "running", "diarize", "Konuşmacılar yeniden ayrılıyor")
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
    pipeline.mark_started(job_id, "running", "youtube", "YouTube açıklaması hazırlanıyor")
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
    pipeline.mark_started(job_id, "running", "analyze", "Yeniden analiz başlıyor")
    pipeline.run_in_thread(pipeline.analyze, job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/render")
def render(job_id: str, vertical: str | None = Body(None, embed=True)):
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "İş zaten çalışıyor")
    pipeline.mark_started(job_id, "rendering", "render", "Render başlıyor")
    pipeline.run_in_thread(pipeline.render_edit, job_id, vertical if vertical in ("crop", "blur") else None)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/reels/render")
def render_reels(job_id: str, ids: list[str] | None = Body(None, embed=True),
                 vertical: str | None = Body("crop", embed=True), style: dict | None = Body(None, embed=True)):
    """style: {"captions": "bold|karaoke|minimal"|null, "caption_pos": "bottom|middle",
               "visuals": bool, "visual_pos": "top|center|full"}"""
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "İş zaten çalışıyor")
    st = None
    if style is not None:
        st = {"captions": style.get("captions") if style.get("captions") in captions.STYLES else None,
              "caption_pos": style.get("caption_pos") if style.get("caption_pos") in captions.POSITIONS else "bottom",
              "visuals": bool(style.get("visuals", True)),
              "visual_pos": style.get("visual_pos") if style.get("visual_pos") in visuals.POSITIONS else "top"}
    pipeline.mark_started(job_id, "rendering", "render", "Render başlıyor")
    pipeline.run_in_thread(pipeline.render_reels, job_id, ids,
                           vertical if vertical in ("crop", "blur", *layout.MODES) else None, st)
    return {"ok": True}


# --- Dikey yerleşim (oyun + kameralar) ---
def _clean_regions(job: pipeline.Job, data: dict) -> dict:
    info = job.data.get("media") or {}
    fw, fh = info.get("width") or 1920, info.get("height") or 1080
    try:
        content = list(layout.clamp_rect(data["content"], fw, fh))
        cams = [list(layout.clamp_rect(c, fw, fh)) for c in (data.get("cams") or [])][:4]
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "Bölgeler: {content: [x,y,w,h], cams: [[x,y,w,h], ...]}")
    return {"content": content, "cams": cams, "source": "user", "w": fw, "h": fh,
            "t": (job.data.get("layout_regions") or {}).get("t")}


@app.get("/api/jobs/{job_id}/frame")
def frame(job_id: str, t: float = 0.0):
    """Bölge düzenleyici için kare (960 px genişlik)."""
    job = _job(job_id)
    d = job.dir / "previews"
    d.mkdir(exist_ok=True)
    p = d / f"frame_{int(t * 10)}.jpg"
    if not p.exists():
        layout._frame(job.source, t, p, 960)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/layout")
def get_layout(job_id: str):
    job = _job(job_id)
    return {"regions": job.data.get("layout_regions"), "modes": layout.MODES, "media": job.data.get("media")}


@app.post("/api/jobs/{job_id}/layout/detect")
def detect_layout(job_id: str, t: float | None = Body(None, embed=True)):
    """Bölgeleri yeniden bulur (Claude görsel analizi, yoksa görüntü analizi). Birkaç saniye–1 dk sürebilir."""
    job = _idle_job(job_id)
    regions = pipeline.job_layout(job, job.data.get("options", {}).get("claude", True), force=True, t=t)
    return {"regions": regions}


@app.put("/api/jobs/{job_id}/layout")
def put_layout(job_id: str, data: dict = Body(...)):
    job = _idle_job(job_id)
    regions = _clean_regions(job, data)
    job.set(layout_regions=regions)
    return {"regions": regions}


@app.get("/api/jobs/{job_id}/layout/preview")
def layout_preview(job_id: str, mode: str = "faces_top", t: float | None = None):
    """Seçili yerleşimle tek kare 9:16 önizleme."""
    job = _job(job_id)
    info = job.data.get("media") or {}
    regions = job.data.get("layout_regions")
    if not regions:
        raise HTTPException(404, "Önce bölgeleri bulun")
    if t is None:
        t = regions.get("t") or 0.0
    d = job.dir / "previews"
    d.mkdir(exist_ok=True)
    p = d / f"layout_{mode if mode in layout.MODES else 'x'}.jpg"
    layout.preview(job.source, t, mode, regions, info["width"], info["height"], p)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# --- Kurgu şeridi: elle reel işaretleme ---
@app.get("/api/jobs/{job_id}/timeline")
def timeline(job_id: str):
    """Kurgu şeridi bilgisi (küçük resim şeritleri, dalga formu). Eski işlerde yoksa üretimi başlatır."""
    job = _job(job_id)
    tl = job.data.get("timeline")
    if not tl or tl.get("state") == "error" and not tl.get("retried"):
        job.set(timeline={"state": "building", "retried": bool(tl)})
        pipeline.run_in_thread(pipeline.build_timeline_assets, job_id)
        tl = {"state": "building"}
    return tl


@app.get("/api/jobs/{job_id}/timeline/{name}")
def timeline_file(job_id: str, name: str):
    job = _job(job_id)
    d = (job.dir / "timeline").resolve()
    p = (d / Path(name).name).resolve()
    if p.parent != d or not p.exists():
        raise HTTPException(404, "Yok")
    return FileResponse(p, headers={"Cache-Control": "max-age=86400"})


@app.put("/api/jobs/{job_id}/reels/manual")
def put_manual_reels(job_id: str, reels: list[dict] = Body(..., embed=True)):
    """Kurgu şeridinde işaretlenen reel aralıkları [{start, end, id?}]. Transkripsiyon sürerken de kaydedilebilir;
    yalnızca reels'i değiştiren bir arka plan işi (kısaltma, görsel, hazırlama, render) sürerken reddedilir."""
    job = _job(job_id)
    if job.data.get("status") == "rendering" or (job.data.get("status") == "running" and
                                                 job.data.get("stage") in ("tighten", "visuals", "prepare")):
        raise HTTPException(409, "Arka planda reels üzerinde bir işlem sürüyor — bitince tekrar deneyin")
    try:
        return pipeline.save_manual_reels(job_id, reels).public()
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"Geçersiz reel aralığı: {e}")


@app.post("/api/jobs/{job_id}/reels/prepare")
def prepare_reels(job_id: str, ids: list[str] | None = Body(None, embed=True),
                  level: str | None = Body("light", embed=True), context: str | None = Body(None, embed=True)):
    """Elle işaretlenen reels: başlık/açıklama üret + kısalt (arka planda)."""
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "Analiz/işlem sürüyor — transkript bitince tekrar deneyin")
    if not (job.dir / "transcript.json").exists():
        raise HTTPException(409, "Transkript henüz hazır değil")
    if context is not None:
        job.set(reel_context=context.strip()[:600])
    pipeline.mark_started(job_id, "running", "prepare", "Reels hazırlanıyor")
    pipeline.run_in_thread(pipeline.prepare_reels, job_id, ids, level if level in tighten.LEVELS else None)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/reels/{reel_id}/check-end")
def check_reel_end(job_id: str, reel_id: str):
    """Reel'in bitişini Claude'la kontrol eder (oyun/konu gerçekten bitti mi); gerekirse uzatır/kısaltır."""
    _idle_job(job_id)
    return _visual_call(pipeline.check_reel_end, job_id, reel_id).public()


# --- Reels sıkılaştırma ---
@app.post("/api/jobs/{job_id}/reels/tighten")
def tighten_reels(job_id: str, ids: list[str] | None = Body(None, embed=True),
                  level: str = Body("medium", embed=True), target_sec: float | None = Body(None, embed=True),
                  context: str | None = Body(None, embed=True)):
    """Seçili reels'i kendi içinde kısaltır (Claude + sıkı sessizlik kesimi), arka planda.
    context: içerik notu (ör. "Frekans oyunu oynuyoruz; ibreyi yönlendirmek oyunun parçası"), işte saklanır."""
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "İş zaten çalışıyor")
    if level not in tighten.LEVELS:
        raise HTTPException(400, "level: light | medium | aggressive")
    if context is not None:
        job.set(reel_context=context.strip()[:600])
    target = target_sec if target_sec and 5 <= target_sec <= 600 else None
    pipeline.mark_started(job_id, "running", "tighten", "Kısaltma başlıyor")
    pipeline.run_in_thread(pipeline.tighten_reels, job_id, ids, level, target)
    return {"ok": True}


# --- Reels görselleri ---
@app.post("/api/jobs/{job_id}/reels/visuals")
def find_visuals(job_id: str, ids: list[str] | None = Body(None, embed=True)):
    """Seçili reels için konuya göre görselleri bulur (Claude + Wikipedia/Pexels), arka planda."""
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "İş zaten çalışıyor")
    pipeline.mark_started(job_id, "running", "visuals", "Görsel arama başlıyor")
    pipeline.run_in_thread(pipeline.find_visuals, job_id, ids)
    return {"ok": True}


def _idle_job(job_id: str) -> pipeline.Job:
    """Görsel düzenlemeleri arka plan işi sürerken yapılamaz (aynı reel'i ikisi birden değiştirirdi)."""
    job = _job(job_id)
    if job.data.get("status") in ("running", "rendering"):
        raise HTTPException(409, "Arka planda bir işlem sürüyor — bitince tekrar deneyin")
    return job


def _visual_call(fn, *args):
    try:
        return fn(*args)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'"))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/jobs/{job_id}/reels/{reel_id}/visuals/{vid}/next")
def visual_next(job_id: str, reel_id: str, vid: str):
    _idle_job(job_id)
    return _visual_call(pipeline.visual_next, job_id, reel_id, vid).public()


@app.post("/api/jobs/{job_id}/reels/{reel_id}/visuals/{vid}/search")
def visual_search(job_id: str, reel_id: str, vid: str, query: str = Body("", embed=True),
                  source: str = Body("auto", embed=True)):
    _idle_job(job_id)
    job, errors = _visual_call(pipeline.visual_search, job_id, reel_id, vid, query[:200], source)
    return {**job.public(), "visual_errors": errors}


@app.post("/api/jobs/{job_id}/reels/{reel_id}/visuals/{vid}/upload")
async def visual_upload(job_id: str, reel_id: str, vid: str, file: UploadFile = File(...)):
    job = _idle_job(job_id)
    suffix = Path(file.filename or "x.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=job.dir) as tmp:
        shutil.copyfileobj(file.file, tmp, length=4 * 1024 * 1024)
    try:
        return _visual_call(pipeline.visual_upload, job_id, reel_id, vid, Path(tmp.name), suffix).public()
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.get("/api/jobs/{job_id}/visuals/{name}")
def visual_file(job_id: str, name: str):
    job = _job(job_id)
    p = (job.visuals_dir / Path(name).name).resolve()
    if job.visuals_dir.resolve() != p.parent or not p.exists():
        raise HTTPException(404, "Görsel yok")
    return FileResponse(p)


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
    keeps = tighten.reel_keeps(reel.to_dict(), job.cuts)
    return exporters.to_srt(job.transcript, keeps, max_chars=28, max_dur=2.2)


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    job = _job(job_id)
    shutil.rmtree(job.dir)
    return {"ok": True}
