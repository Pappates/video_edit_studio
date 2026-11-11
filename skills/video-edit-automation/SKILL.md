---
name: video-edit-automation
description: Ham Türkçe video çekimini konuşmaya göre otomatik kurgular. Whisper ile yazıya döker; "burayı kes", "şurayı keselim", "baştan alıyorum", "tekrar alalım" gibi sözlü kurgu komutlarını, komut söylenmeden tekrar edilen cümleleri, yarım kalan cümleleri, çekim arası muhabbeti ve uzun sessizlikleri bulup keser. Kesilmiş MP4, Premiere/DaVinci XML, EDL ve SRT üretir. Kullanıcı bir video yükleyip "editle", "kurgula", "tekrarları kes", "kes dediğimiz yerleri çıkar", "ham çekimi temizle" dediğinde kullan. Birden fazla reel içeren videoyu parçalara bölmek için KULLANMA (reels-splitter skill'ini kullan).
license: MIT
compatibility: Python 3.10+, ffmpeg ve faster-whisper gerektirir (pip install faster-whisper). Whisper modeli ilk çalıştırmada indirilir; ağ erişimi yoksa hazır transkript JSON ile --transcript kullanılabilir. Claude API isteğe bağlıdır.
metadata:
  author: Video Edit Studio
  version: 1.0.0
  category: video
  tags: [video, kurgu, whisper, turkce, premiere, davinci]
---

# Video Edit Automation

Ham çekimden kurgu komutlarını, tekrarları ve sessizlikleri temizleyen iki adımlı iş akışı.

## Kritik kurallar

- Kesim kararlarında HER ZAMAN son denemeyi tut, önceki başarısız denemeleri kes.
- Zamanları asla tahmin etme. Kararlarını yalnızca `review.md` içindeki segment numaralarıyla (`#12`) ver; script zamanlara kendisi çevirir.
- Kullanıcı onayı olmadan render etme: önce kesim listesini göster, sonra `--render` ile üret.
- Konunun parçası olan bilinçli tekrarları kesme ("tekrar ediyorum, bu çok önemli", kapanış özeti).
- Kelimenin içinde "kes" geçmesi komut değildir ("kesinlikle", "peynirleri kesin"). Bu tür adayları reddet.

## Adım 1: Analiz

```bash
pip install faster-whisper   # bir kez
python scripts/analyze.py "VIDEO.mp4" --out work/
```

Ağ yoksa veya transkript hazırsa: `--transcript hazir.json` (Whisper formatı, bkz. `references/cut-rules.md`).

Beklenen çıktı: `work/transcript.json`, `work/cuts.json`, `work/review.md` ve konsolda kesim adayları listesi.

## Adım 2: İnceleme (Claude olarak sen yaparsın)

1. `work/review.md` dosyasını oku. Talimat, numaralı transkript, aday listesi ve JSON şeması içinde.
2. Her aday için karar ver (doğru mu, aralık doğru mu) ve adaylarda olmayan kesimleri ekle:
   - farklı kelimelerle tekrar anlatılan fikir (`repeat`)
   - çekim arası konuşma: "kamera açık mı", "hazır mısın", gülüşme, dağılma (`chatter`)
3. Kararını `work/decisions.json` olarak yaz. Kural ayrıntıları: `references/cut-rules.md`.
4. Kullanıcıya kısa bir kesim tablosu göster: id, zaman, tür, kesilen metin. Onay iste.

API anahtarı varsa bu adım otomatik yapılabilir: Adım 1'i `--api` ile çalıştır.

## Adım 3: Uygula ve üret

```bash
python scripts/apply.py "VIDEO.mp4" --out work/ --decisions work/decisions.json
# kullanıcı düzeltmeleri:  --disable c004,c009   --enable c012
python scripts/apply.py "VIDEO.mp4" --out work/ --render            # MP4 üret
python scripts/apply.py "VIDEO.mp4" --out work/ --render --vertical crop   # 9:16
```

Çıktılar: `*_edit.mp4`, `*_edit.xml` (Premiere Pro ve DaVinci Resolve içe aktarma), `*_edit.edl`, `*_edit.srt`.
İçe aktarma adımları için: `references/export-formats.md`.

## Örnekler

**Örnek 1 — komutlu yeniden çekim**
Kullanıcı: "Bu çekimi editle, baştan aldığımız yerler var."
Transkript: `#1 Bu makine on beş bar...` → `#2 Ay pardon baştan alıyorum.` → `#3 Bu makine on beş bar ... otomatik süt köpürtücüsü var.`
Karar: #1-#2 kesilir (retake), #3 kalır.

**Örnek 2 — komutsuz tekrar**
`#5 Kutusundan çıkan aksesuarlar şunlar.` → `#6 Kutusundan çıkan aksesuarlar şunlar bir ölçü kaşığı...`
Karar: #5 kesilir (false_start), #6 kalır.

**Örnek 3 — "şurayı kes"**
`#7 Şimdi ilk kahvemizi hazırlayalım ya kamera açık mı.` → `#8 Tamam şurayı kes.`
Karar: #7-#8 kesilir (command).

## Sorun giderme

**Hata: `faster-whisper kurulu değil`**
Çözüm: `pip install faster-whisper`. GPU yoksa `.env` içinde `WHISPER_MODEL=medium` ile hızlanır.

**Hata: `ffmpeg bulunamadı`**
Çözüm: ffmpeg kurup PATH'e ekle (Windows: `winget install ffmpeg`, macOS: `brew install ffmpeg`).

**Komutlar yakalanmıyor**
Neden: Whisper komutu farklı yazmış olabilir ("şurayı kesin" → "şuraya kesin").
Çözüm: `transcript.json` içinde ilgili segmenti bul, ifadeyi `vedit/detector.py` içindeki `CUT_PHRASES` veya `RETAKE_PHRASES` listesine ASCII (ş→s, ı→i) olarak ekle.

**Çok fazla kesim / yanlış tekrar**
Çözüm: `.env` içinde `REPEAT_SIMILARITY=0.8` yap; ya da `--disable` ile tek tek kapat.

**Kesim noktasında kelime kırpılıyor**
Çözüm: `CUT_PAD_BEFORE` / `CUT_PAD_AFTER` değerlerini 0.12'ye çıkar.
