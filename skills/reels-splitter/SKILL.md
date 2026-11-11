---
name: reels-splitter
description: İçinde arka arkaya çekilmiş birden fazla reel bulunan uzun bir videoyu her biri ayrı dosya olacak şekilde reels'lere böler. Aralarında işaret olmasa da sınırları transkript (konu değişimi, giriş ve kapanış cümleleri), duraklamalar ve görüntü değişimleri (kadraj, mekân, kıyafet) ile bulur; yarım denemeleri ve aradaki muhabbeti atar, her reel'in içindeki tekrar ve sessizlikleri temizler, 9:16 dikey MP4, altyazı (SRT), başlık, kanca cümlesi ve Instagram açıklaması üretir. Kullanıcı "bu videoyu reels'lere böl", "reel içeriklerini ayır", "parça parça ver", "shorts'lara kes" dediğinde kullan. Tek bir videoyu kurgulamak için KULLANMA (video-edit-automation skill'ini kullan).
license: MIT
compatibility: Python 3.10+, ffmpeg ve faster-whisper gerektirir. Görsel analiz için Claude'un görüntü okuma yeteneği veya Claude API (--api) kullanılır.
metadata:
  author: Video Edit Studio
  version: 1.0.0
  category: video
  tags: [reels, shorts, video, bolme, instagram, turkce]
---

# Reels Splitter

Tek bir uzun çekimden birden çok reel çıkaran iki adımlı iş akışı.

## Kritik kurallar

- Sınır kararlarını yalnızca `packet.md` içindeki segment numaralarıyla ver (`start_id`, `end_id`). Saniye uydurma.
- Aynı reel birden çok kez çekildiyse SON TAM denemeyi seç, öncekileri at.
- 8 sn'den kısa parçalar reel değildir; reels arasındaki hazırlık konuşmasını dahil etme.
- Karelere mutlaka bak: sesin kesintisiz olduğu ama kıyafet/mekân/kadrajın değiştiği yer genelde yeni bir reel'dir.
- Render'dan önce kullanıcıya reel listesini (başlık, süre, ilk cümle) göster ve onay al.

## Adım 1: Hazırlık

```bash
pip install faster-whisper   # bir kez
python scripts/prepare.py "VIDEO.mp4" --out work/
```

Çıktılar: `work/transcript.json`, `work/cuts.json`, `work/frames/*.jpg`, `work/packet.md`.

API anahtarı varsa bölme otomatik yapılabilir: `--api` ekle, `work/reels.json` doğrudan render edilebilir.

## Adım 2: Sınırları belirle (Claude olarak sen yaparsın)

1. `work/packet.md` dosyasını oku (talimat + numaralı transkript + aday sınırlar + kare listesi).
2. Aday sınırların (P: duraklama, S: sahne değişimi) çevresindeki kareleri aç ve karşılaştır.
3. Her reel için başlık, kanca cümlesi (ilk 3 sn), 1-3 cümlelik Türkçe açıklama ve hashtag yaz. Ayrıntılı ölçütler: `references/reels-criteria.md`.
4. Kararı `work/decision.json` olarak yaz:

```json
{"reels": [{"start_id": 0, "end_id": 14, "title": "Kahve demlemenin 3 sırrı",
            "hook": "Kahveniz neden acı oluyor biliyor musunuz?", "summary": "...",
            "caption": "...", "hashtags": ["#kahve", "#barista"], "confidence": 0.9,
            "notes": "#15'te kıyafet değişiyor, yeni giriş cümlesi"}],
 "discarded": [{"start_id": 15, "end_id": 17, "reason": "kamera ayarı muhabbeti"}]}
```

5. Kullanıcıya tablo halinde göster: reel id, başlık, başlangıç-bitiş, süre. Onay iste.

## Adım 3: Üret

```bash
python scripts/render.py "VIDEO.mp4" --out work/ --reels work/decision.json
# seçenekler: --vertical blur (yatay çekimde bulanık arka plan) | --vertical none | --no-clean | --only reel02
```

Her reel için `*_reelNN.mp4` (1080x1920) ve `*_reelNN.srt`; tüm açıklamalar `*_captions.md` içinde.

## Örnekler

**Örnek 1 — işaretsiz geçiş**
`#14 ...takip etmeyi unutmayın.` (2.8 sn duraklama, kadraj değişiyor) `#15 Merhaba bugün soğuk kahve...`
Karar: reel01 = #0-#14, reel02 = #15'ten başlar.

**Örnek 2 — baştan çekilen reel**
`#20 Selam bugün size üç...` `#21 ya olmadı` `#22 Selam bugün size üç ipucu vereceğim...` (tam)
Karar: #20-#21 discarded, reel #22'den başlar.

## Sorun giderme

**Reels yanlış yerden bölünüyor**
Neden: Konuşma kesintisiz, sahne değişimi zayıf.
Çözüm: `packet.md` aday listesinde S olmayan geçişler için kareleri daha sık incele; gerekirse arayüzde I/O ile sınırı elle düzelt.

**Dikey videoda kişi kadraj dışında kalıyor**
Çözüm: `--vertical blur` kullan (tüm kadrajı korur).

**Reel içinde kelime kırpılıyor**
Çözüm: `--no-clean` ile iç temizliği kapat veya `CUT_PAD_*` değerlerini artır.

**API kullanımında "JSON yanıtı yarım kaldı"**
Çözüm: `.env` içinde `CLAUDE_MAX_TOKENS=16000`.
