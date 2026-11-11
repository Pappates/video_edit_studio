#!/usr/bin/env python3
"""Adım 2 — İnceleme kararlarını uygular, dışa aktarır ve isteğe bağlı render eder.

Kullanım:
    python scripts/apply.py VIDEO --out ÇIKTI_KLASÖRÜ [--decisions decisions.json]
                            [--disable c003,c007] [--render] [--vertical crop|blur]

decisions.json şeması review.md'nin sonunda yazar:
    {"verdicts": [{"candidate": "c003", "keep_cut": false, ...}], "new_cuts": [{"start_id": 12, "end_id": 13, ...}]}

Üretilenler: <ad>_edit.edl, <ad>_edit.xml (Premiere/DaVinci), <ad>_edit.srt, <ad>_edit.cuts.json
ve --render ile <ad>_edit.mp4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vedit import exporters, media  # noqa: E402
from vedit.claude_review import apply_decisions  # noqa: E402
from vedit.detector import Cut, finalize, keep_ranges  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decisions")
    ap.add_argument("--disable", default="", help="Kapatılacak kesim id'leri, virgülle")
    ap.add_argument("--enable", default="", help="Açılacak kesim id'leri, virgülle")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--vertical", choices=["crop", "blur"])
    a = ap.parse_args()

    out = Path(a.out)
    tr = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    info = json.loads((out / "media.json").read_text(encoding="utf-8"))
    cuts = [Cut.from_dict(c) for c in json.loads((out / "cuts.json").read_text(encoding="utf-8"))]

    if a.decisions:
        data = json.loads(Path(a.decisions).read_text(encoding="utf-8"))
        by_id = {c.id: c for c in cuts}
        extra = apply_decisions(tr["segments"], by_id, data)
        cuts = finalize(list(by_id.values()) + extra, tr["segments"], info["duration"], min_confidence=0.5)

    for cid in filter(None, a.disable.split(",")):
        for c in cuts:
            if c.id == cid.strip():
                c.enabled, c.source = False, "user"
    for cid in filter(None, a.enable.split(",")):
        for c in cuts:
            if c.id == cid.strip():
                c.enabled, c.source = True, "user"

    (out / "cuts.json").write_text(json.dumps([c.to_dict() for c in cuts], ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    keeps = keep_ranges(cuts, info["duration"])
    stem = Path(a.video).stem.replace(" ", "_") + "_edit"
    files = exporters.write_all(out, stem, keeps, info, a.video, tr, extra={"cuts": [c.to_dict() for c in cuts]})
    print(f"Kalan süre: {sum(b - a for a, b in keeps):.1f} / {info['duration']:.1f} sn, {len(keeps)} parça")
    for f in files.values():
        print(" ", f)

    if a.render:
        dst = out / f"{stem}.mp4"
        media.render_keeps(a.video, keeps, dst, vertical=a.vertical,
                           progress=lambda f, m: print(f"\rRender %{f * 100:.0f}", end="", flush=True))
        print(f"\nVideo: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
