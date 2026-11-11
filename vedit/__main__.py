"""Komut satırı.

    python -m vedit edit  video.mp4 [--render] [--no-claude] [--no-silence]
    python -m vedit reels video.mp4 [--render] [--vertical crop|blur|none]
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
        p.add_argument("--vertical", default="crop" if name == "reels" else "none",
                       choices=["crop", "blur", "none"])

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

    opts = {"claude": not a.no_claude, "silence": not a.no_silence}
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
        if a.render:
            pipeline.render_reels(job.id, vertical=None if a.vertical == "none" else a.vertical)
    _printer(job.id)()
    print(f"Çıktılar: {job.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
