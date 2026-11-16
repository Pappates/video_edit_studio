"""Komut satırı.

    python -m vedit edit  video.mp4 [--render] [--no-claude] [--no-silence]
    python -m vedit reels video.mp4 [--render] [--vertical crop|blur|none] [--captions bold] [--visuals]
    python -m vedit detect transcript.json           # sadece transkriptten kesim listesi

Sonuçlar data/jobs/<id>/ altına yazılır; arayüzde de aynı iş görünür.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import pipeline
from .detector import detect_all, keep_ranges


def _printer(job_id: str):
    def show():
        j = pipeline.Job(job_id).data
        print(f"[{j.get('status')}] {j.get('message', '')}")
    return show


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vedit", description="Konuşma tabanlı video edit ve reels bölme")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("edit", "reels"):
        p = sub.add_parser(name)
        p.add_argument("video")
        p.add_argument("--render", action="store_true", help="Analizden sonra onaysız render et")
        p.add_argument("--no-claude", action="store_true")
        p.add_argument("--no-silence", action="store_true")
        p.add_argument("--vertical", default="faces_top" if name == "reels" else "none",
                       choices=(["faces_top", "game_top", "game_full"] if name == "reels" else []) + ["crop", "blur", "none"],
                       help="faces_top: yüzler üstte, oyun altta · game_top: oyun üstte · game_full: oyun büyük, "
                            "yüzler küçük · crop: ortadan kırp · blur: bulanık arka plan")
        if name == "reels":
            p.add_argument("--captions", default="none", choices=["bold", "karaoke", "minimal", "none"],
                           help="Videoya gömülü, kelime kelime vurgulu altyazı")
            p.add_argument("--visuals", action="store_true",
                           help="Konuya göre görselleri bul (Claude + Wikipedia/Pexels) ve videoya ekle")
            p.add_argument("--visual-pos", default="top", choices=["top", "center", "full"])
            p.add_argument("--tighten", default="medium", choices=["light", "medium", "aggressive", "off"],
                           help="Bölündükten sonra her reel'i kendi içinde kısalt")
            p.add_argument("--max-reels", type=int, default=0,
                           help="İlk N reel bulununca dur (uzun videoda hızlı test). 0 = videonun hepsi")

    d = sub.add_parser("detect")
    d.add_argument("transcript")
    d.add_argument("--no-claude", action="store_true")

    a = ap.parse_args(argv)

    if a.cmd == "detect":
        tr = json.load(open(a.transcript, encoding="utf-8"))
        cuts = detect_all(tr, use_claude=not a.no_claude)
        for c in cuts:
            flag = "x" if c.enabled else " "
            print(f"[{flag}] {c.id} {c.start:8.2f}-{c.end:8.2f} {c.kind:11s} {c.confidence:.2f}  {c.reason}")
        dur = tr.get("duration") or (tr["segments"][-1]["end"] if tr["segments"] else 0)
        kept = sum(b - a for a, b in keep_ranges(cuts, dur))
        print(f"\nToplam: {dur:.1f} sn → {kept:.1f} sn")
        return 0

    opts = {"claude": not a.no_claude, "silence": not a.no_silence,
            "max_reels": max(0, getattr(a, "max_reels", 0)),
            "tighten": "" if getattr(a, "tighten", "off") == "off" else a.tighten}
    job = pipeline.create_job(a.video, a.cmd, options=opts)
    print(f"İş: {job.id}")
    pipeline.analyze(job.id)
    _printer(job.id)()
    job = pipeline.Job(job.id)
    if job.data["status"] != "ready":
        print(job.data.get("error", ""), file=sys.stderr)
        return 1

    if a.cmd == "edit":
        for c in job.cuts:
            print(f"  [{'x' if c.enabled else ' '}] {c.start:8.2f}-{c.end:8.2f} {c.kind:11s} {c.reason}")
        files = pipeline.export_edit(job.id)
        print("Dışa aktarım:", ", ".join(str(f) for f in files.values()))
        if a.render:
            pipeline.render_edit(job.id, vertical=None if a.vertical == "none" else a.vertical)
    else:
        for r in job.reels:
            print(f"  {r.id} {r.start:8.2f}-{r.end:8.2f} ({r.duration:.0f} sn) {r.title}")
        if a.visuals:
            pipeline.find_visuals(job.id)
            _printer(job.id)()
            for r in pipeline.Job(job.id).reels:
                for v in r.visuals:
                    print(f"    {r.id} {v['start']:8.2f} {('#' + str(v['rank'])) if v.get('rank') else '  '} "
                          f"{v['label']} — {v.get('source') or 'bulunamadı'}")
        if a.render:
            pipeline.render_reels(job.id, vertical=None if a.vertical == "none" else a.vertical,
                                  style={"captions": None if a.captions == "none" else a.captions,
                                         "visuals": a.visuals, "visual_pos": a.visual_pos})
    _printer(job.id)()
    print(f"Çıktılar: {job.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
