#!/usr/bin/env python3
"""Adım 1 — Reels bölme için transkript, kareler ve inceleme paketi hazırlar.

Kullanım:
    python scripts/prepare.py VIDEO --out ÇIKTI_KLASÖRÜ [--transcript hazir.json] [--api]

Üretilenler:
    transcript.json, media.json, cuts.json (reel içi temizlik için)
    frames/t0012.0.jpg ...   zaman etiketli kareler (görsel sınır ipuçları)
    packet.md                Claude'un okuyacağı paket: numaralı transkript + aday sınırlar + kare listesi
    reels.json               yalnızca --api ile: Claude API'nin kararı (doğrudan render edilebilir)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vedit import media  # noqa: E402
from vedit.detector import detect_all  # noqa: E402
from vedit.reels import prepare_packet, split_reels  # noqa: E402
from vedit.transcribe import resegment, transcribe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcript")
    ap.add_argument("--api", action="store_true", help="Sınırları Claude API (vision) ile belirle")
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

    cuts = detect_all(tr, use_claude=False)
    (out / "cuts.json").write_text(json.dumps([c.to_dict() for c in cuts], ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    packet = prepare_packet(a.video, tr, info, out)
    print(f"{len(tr['segments'])} segment. Paket: {packet}")

    if a.api:
        reels, method = split_reels(a.video, tr, info)
        (out / "reels.json").write_text(json.dumps({"reels": [r.to_dict() for r in reels], "method": method},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
        for r in reels:
            print(f"  {r.id} {r.start:7.1f}-{r.end:7.1f} ({r.duration:.0f} sn) {r.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
