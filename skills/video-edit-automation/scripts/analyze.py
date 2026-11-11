#!/usr/bin/env python3
"""Adım 1 — Videoyu yazıya döker ve kural tabanlı kesim adaylarını çıkarır.

Kullanım:
    python scripts/analyze.py VIDEO --out ÇIKTI_KLASÖRÜ [--transcript hazir.json] [--api]

Üretilenler (ÇIKTI_KLASÖRÜ içinde):
    transcript.json  kelime zaman damgalı transkript
    media.json       süre, fps, çözünürlük
    cuts.json        kesim adayları
    review.md        Claude'un inceleyeceği paket (segment numaralı transkript + adaylar + JSON şeması)

--api verilirse ve ANTHROPIC_API_KEY tanımlıysa inceleme Claude API ile otomatik yapılır,
review.md yine de yazılır.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vedit import media  # noqa: E402
from vedit.claude_review import review_packet  # noqa: E402
from vedit.detector import detect_all, keep_ranges  # noqa: E402
from vedit.transcribe import resegment, transcribe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcript", help="Hazır transkript JSON (Whisper'ı atlar)")
    ap.add_argument("--api", action="store_true", help="İncelemeyi Claude API ile yap")
    ap.add_argument("--no-silence", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = media.probe(a.video)
    (out / "media.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    if a.transcript:
        tr = resegment(json.loads(Path(a.transcript).read_text(encoding="utf-8")))
        (out / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        tr = transcribe(a.video, out / "transcript.json",
                        progress=lambda f, m: print(f"\r{m}", end="", flush=True))
        print()
    tr["duration"] = tr.get("duration") or info["duration"]

    segs = tr["segments"]
    cuts = detect_all(tr, use_claude=a.api, use_silence=not a.no_silence)
    (out / "cuts.json").write_text(json.dumps([c.to_dict() for c in cuts], ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    # İnceleme paketi: sessizlik hariç adaylar (sessizlik kararı mekaniktir)
    (out / "review.md").write_text(review_packet(segs, [c for c in cuts if c.kind not in ("silence", "stutter")]),
                                   encoding="utf-8")

    kept = sum(b - a for a, b in keep_ranges(cuts, info["duration"]))
    print(f"{len(segs)} segment, {len(cuts)} kesim adayı. Süre {info['duration']:.1f} → {kept:.1f} sn")
    for c in cuts:
        if c.kind != "silence":
            print(f"  [{'x' if c.enabled else ' '}] {c.id} {c.start:7.2f}-{c.end:7.2f} {c.kind:11s} {c.text[:70]}")
    print(f"İnceleme paketi: {out / 'review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
