#!/usr/bin/env python3
"""Skill paketlerini doğrular ve yüklemeye hazır .zip dosyaları üretir.

    python build_skills.py

1. Ortak çekirdeği (vedit/) her skill'in scripts/ klasörüne kopyalar (tek kaynak, taşınabilir skill).
2. "The Complete Guide to Building Skills for Claude" kurallarına göre doğrular:
   - klasör adı kebab-case, SKILL.md tam bu adla mevcut, skill klasöründe README.md yok
   - frontmatter --- ile sınırlı; name kebab-case ve klasör adıyla aynı; "claude"/"anthropic" içermez
   - description var, 1024 karakteri aşmıyor, ne yaptığını VE ne zaman kullanılacağını söylüyor
   - frontmatter'da XML açılı ayraç yok; compatibility 1-500 karakter; SKILL.md 5000 kelimeyi aşmıyor
3. dist/<skill>.zip üretir (Claude.ai > Ayarlar > Skills > Upload).
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILLS = ROOT / "skills"
DIST = ROOT / "dist"
KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md '---' ile başlamalı")
    end = text.find("\n---", 4)
    if end == -1:
        raise ValueError("frontmatter kapanış '---' bulunamadı")
    raw, body = text[4:end], text[end + 4:]
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        m = re.match(r"^([a-z_-]+):\s*(.*)$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return {"_raw": raw, **fields}, body


def validate(skill_dir: Path) -> list[str]:
    errs = []
    name = skill_dir.name
    if not KEBAB.match(name):
        errs.append(f"klasör adı kebab-case değil: {name}")
    md = skill_dir / "SKILL.md"
    if not md.exists():
        return errs + ["SKILL.md yok (büyük/küçük harf duyarlı)"]
    if any(p.name.lower() == "readme.md" for p in skill_dir.iterdir()):
        errs.append("skill klasöründe README.md olmamalı")
    try:
        fm, body = parse_frontmatter(md.read_text(encoding="utf-8"))
    except ValueError as e:
        return errs + [str(e)]

    if fm.get("name") != name:
        errs.append(f"name ({fm.get('name')}) klasör adıyla aynı olmalı")
    if fm.get("name") and not KEBAB.match(fm["name"]):
        errs.append("name kebab-case olmalı")
    if re.search(r"claude|anthropic", fm.get("name", ""), re.I):
        errs.append("name 'claude' veya 'anthropic' içeremez")
    desc = fm.get("description", "")
    if not desc:
        errs.append("description zorunlu")
    if len(desc) > 1024:
        errs.append(f"description {len(desc)} karakter (en fazla 1024)")
    if not re.search(r"kullan|use when", desc, re.I):
        errs.append("description ne zaman kullanılacağını (tetikleyici) söylemeli")
    if "<" in fm["_raw"] or ">" in fm["_raw"]:
        errs.append("frontmatter'da < veya > olamaz")
    comp = fm.get("compatibility")
    if comp is not None and not (1 <= len(comp) <= 500):
        errs.append("compatibility 1-500 karakter olmalı")
    words = len(body.split())
    if words > 5000:
        errs.append(f"SKILL.md gövdesi {words} kelime (en fazla 5000, ayrıntıyı references/ altına taşıyın)")
    for ref in re.findall(r"`((?:references|scripts)/[^`\s]+)`", body):
        if not (skill_dir / ref).exists():
            errs.append(f"SKILL.md'de geçen dosya yok: {ref}")
    return errs


def build() -> int:
    DIST.mkdir(exist_ok=True)
    ok = True
    for skill_dir in sorted(p for p in SKILLS.iterdir() if p.is_dir()):
        target = skill_dir / "scripts" / "vedit"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ROOT / "vedit", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        errs = validate(skill_dir)
        if errs:
            ok = False
            print(f"✗ {skill_dir.name}")
            for e in errs:
                print(f"    - {e}")
            continue

        zpath = DIST / f"{skill_dir.name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(skill_dir.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, f.relative_to(SKILLS))
        print(f"✓ {skill_dir.name} → {zpath.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(build())
