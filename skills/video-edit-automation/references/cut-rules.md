# Kesim kuralları

Bu belge kural motorunun (vedit/detector.py) ve Claude incelemesinin ortak standardıdır.

## İçindekiler
1. Kesim türleri
2. Sözlü komut sözlüğü
3. Tekrar tespiti
4. Sessizlik ve kekeleme
5. Güven ve varsayılan açık/kapalı
6. decisions.json şeması
7. Transkript formatı
8. Ayarlar (.env)

## 1. Kesim türleri

| Tür | Ne zaman | Kesilen aralık | Kalan |
|---|---|---|---|
| `retake` | "baştan alıyorum", "tekrar alalım" | Başarısız deneme başı → komut sonu | Komuttan sonraki yeni deneme |
| `command` | "burayı kes", "şurayı keselim", "bunu çıkar" | Hemen önceki cümle + komut | Sonrası |
| `range` | "buradan itibaren kes" … "buraya kadar kes" | İki işaret arası tamamı | Dışı |
| `repeat` | Aynı cümle komutsuz tekrar edildi (benzerlik ≥ 0.72) | Önceki deneme(ler) | Son deneme |
| `false_start` | Yarım cümle, ardından tamamlanmış hali | Yarım olan | Tamamlanmış |
| `stutter` | Cümle içinde 2-5 kelimelik grup arka arkaya | İlk tekrar | İkinci |
| `chatter` | Çekim dışı muhabbet (yalnızca Claude bulur) | O segmentler | — |
| `silence` | Kelimeler arası ≥ 1.2 sn boşluk | Boşluğun 0.35 sn dışında kalanı | Kısa nefes payı |

Retake'de başarısız denemenin başlangıcı: komuttan sonraki cümleye en çok benzeyen önceki segment (son 30 sn içinde). Benzer yoksa yalnızca hemen önceki segment kesilir ve güven 0.65 olur.

## 2. Sözlü komut sözlüğü

Eşleştirme ASCII-katlanmış (ş→s, ı→i, ğ→g, ç→c, ö→o, ü→u) ve bulanıktır (karakter benzerliği ≥ 0.86). Kelime içinde geçmek yetmez: "kesinlikle" komut değildir.

**Retake:** baştan alıyorum/alalım/al, tekrar alıyorum/alalım/çekelim, bir daha alalım/çekelim, yeniden alıyorum/çekelim, olmadı baştan, yanlış oldu, yanlış söyledim, pardon baştan, dur dur dur

**Cut:** burayı/şurayı/bunu kes, kes bunu, kesin bunu, burayı/şurayı keselim, burası kesilecek, bunu/burayı çıkar, bunu sil, kurguda/montajda/editte kes-çıkar, bunu kullanmayalım, bunu at

**Aralık başı:** buradan itibaren kes/çıkar/sil, buradan sonrasını kes, kesmeye başla
**Aralık sonu:** buraya kadar kes/çıkar/sil, kesim bitti, kesmeyi bitir, buradan devam

Yeni ifade eklemek için listeye ASCII hâliyle ekleyin ve `python -m pytest tests` çalıştırın.

## 3. Tekrar tespiti

- Karşılaştırma dolgu kelimeleri atılmış kelime dizileri üzerinde yapılır (ee, şey, yani, işte, hani, tamam...).
- Bakılan pencere: en fazla 8 segment ileri ve 120 sn.
- İki deneme arasındaki segmentler kısaysa (≤ 6 kelime) ve ara ≤ 25 sn ise aradaki kısım da kesilir (ör. "pardon", "bir saniye").
- Uzaklık arttıkça güven düşer (her ara segment −0.05).
- Konu içi bilinçli tekrar (vurgu, özet) Claude incelemesinde reddedilmelidir.

## 4. Sessizlik ve kekeleme

- Sessizlik kesimleri her zaman açık gelir; arayüzde tür filtresiyle topluca kapatılabilir.
- Kekeleme güveni 0.6'dır; dilsel bir vurgu olabilir ("çok çok güzel" 1 kelime olduğu için yakalanmaz).

## 5. Güven ve varsayılan

- Güven ≥ 0.6 olan kesimler açık gelir, altı kapalı gelir ve kullanıcı onayına bırakılır.
- Claude bir adayı reddederse güven 0.35'e düşer ve kesim kapanır.
- Claude onaylarsa güven en az 0.8 olur.

## 6. decisions.json şeması

```json
{
  "verdicts": [
    {"candidate": "c003", "keep_cut": true, "start_id": 5, "end_id": 6, "note": "yarım cümle"},
    {"candidate": "c007", "keep_cut": false, "note": "kesinlikle kelimesi, komut değil"}
  ],
  "new_cuts": [
    {"start_id": 21, "end_id": 22, "kind": "chatter", "reason": "kamera kontrolü", "confidence": 0.8}
  ]
}
```

`start_id`/`end_id` dahil aralıktır. Segment numarası dışında zaman yazılmaz.

## 7. Transkript formatı

```json
{"language": "tr", "duration": 312.4,
 "segments": [{"id": 0, "start": 0.52, "end": 4.10, "text": "Merhaba arkadaşlar",
               "words": [{"start": 0.52, "end": 1.10, "word": "Merhaba", "prob": 0.98}]}]}
```

Segmentler kelime aralarındaki ≥ 0.7 sn duraklamalara göre yeniden bölünür; böylece her deneme ayrı bir birim olur.

## 8. Ayarlar (.env)

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `WHISPER_MODEL` | large-v3 | tiny/base/small/medium/large-v3 |
| `WHISPER_DEVICE` | auto | cuda / cpu |
| `REPEAT_SIMILARITY` | 0.72 | Tekrar eşiği |
| `SILENCE_MIN_SEC` | 1.2 | Kısaltılacak en kısa sessizlik |
| `SILENCE_KEEP_SEC` | 0.35 | Sessizlikten bırakılan pay |
| `CUT_PAD_BEFORE/AFTER` | 0.08 | Kesim kenarı payı |
| `CLAUDE_MODEL` | claude-sonnet-5-5 | İnceleme modeli |
