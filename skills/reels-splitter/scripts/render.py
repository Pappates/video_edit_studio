#!/usr/bin/env python3
"""Adım 2 — Reel kararlarını uygular ve her reel'i ayrı dikey video olarak üretir.

Kullanım:
    python scripts/render.py VIDEO --out ÇIKTI_KLASÖRÜ --reels decision.json
                             [--vertical crop|blur|none] [--no-clean] [--only reel01,reel03]

decision.json iki biçimden biri olabilir:
  a) Claude kararı (segment numaralı):  {"reels": [{"start_id": 0, "end_id": 14, "title": ...}]}
  b) prepare.py --api çıktısı (saniyeli): {"reels": [{"start": 0.3, "end": 41.2, "title": ...}]}

Her reel için: <ad>_reelNN.mp4, <ad>_reelNN.srt ve tüm açıklamalar için <ad>_captions.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vedit import exporters, media  # noqa: E402
from vedit.detector import Cut, keep_ranges  # noqa: E402
from vedit.reels import Reel, reels_from_decision  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reels", required=True)
    ap.add_argument("--vertical", default="crop", choices=["crop", "blur", "none"])
    ap.add_argument("--no-clean", action="store_true", help="Reel içindeki tekrar/sessizlikleri kesme")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    out = Path(a.out)
    tr = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    info = json.loads((out / "media.json").read_text(encoding="utf-8"))
    cuts = [Cut.from_dict(c) for c in json.loads((out / "cuts.json").read_text(encoding="utf-8"))]
    data = json.loads(Path(a.reels).read_text(encoding="utf-8"))

    raw = data.get("reels", [])
    if raw and "start_id" in raw[0]:
        reels = reels_from_decision(tr["segments"], data, info["duration"])
    else:
        reels = [Reel.from_dict(r) for r in raw]
        for n, r in enumerate(reels, 1):
            r.id = r.id or f"reel{n:02d}"
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    reels = [r for r in reels if r.enabled and (not only or r.id in only)]
    if not reels:
        print("Render edilecek reel yok.", file=sys.stderr)
        return 1

    stem = Path(a.video).stem.replace(" ", "_")
    vertical = None if a.vertical == "none" else a.vertical
    captions = []
    for r in reels:
        keeps = [(r.start, r.end)] if a.no_clean else keep_ranges(cuts, info["duration"], within=(r.start, r.end))
        mp4 = out / f"{stem}_{r.id}.mp4"
        media.render_keeps(a.video, keeps, mp4, vertical=vertical)
        (out / f"{stem}_{r.id}.srt").write_text(exporters.to_srt(tr, keeps, max_chars=28, max_dur=2.2),
                                                 encoding="utf-8")
        final = sum(b - a_ for a_, b in keeps)
        print(f"{r.id}: {mp4.name} ({final:.0f} sn) — {r.title}")
        captions.append(f"### {r.id} — {r.title}\n{r.caption}\n{' '.join(r.hashtags)}\n")
    (out / f"{stem}_captions.md").write_text("\n".join(captions), encoding="utf-8")
    (out / "reels.final.json").write_text(json.dumps([r.to_dict() for r in reels], ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
