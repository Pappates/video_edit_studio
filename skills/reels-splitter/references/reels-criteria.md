# Reels bölme ölçütleri

## İçindekiler
1. Sınır sinyalleri
2. Atılacak parçalar
3. Süre kuralları
4. Başlık, kanca ve açıklama standardı
5. Dikey format seçimi

## 1. Sınır sinyalleri (güçlüden zayıfa)

| Sinyal | Örnek | Ağırlık |
|---|---|---|
| Yeni giriş cümlesi | "Selam arkadaşlar", "Bugün size…", "Merhaba" | Çok güçlü |
| Kapanış / çağrı | "takip etmeyi unutmayın", "yorumlara yazın", "görüşmek üzere" | Çok güçlü (reel burada biter) |
| Görsel değişim | Kıyafet, mekân, ışık, kamera açısı, ürün değişimi | Güçlü |
| Uzun duraklama | ≥ 2.5 sn sessizlik (packet'te P) | Orta |
| Konu değişimi | Farklı ürün/tarif/konu | Orta |
| Sahne değişimi | Ani kadraj sıçraması (packet'te S) | Orta — kurgu içi kesmelerde de olur |

İki güçlü sinyal birlikteyse sınır kesindir. Tek orta sinyal yetmez.

## 2. Atılacak parçalar (`discarded`)

- Yarım kalmış ve sonra baştan çekilmiş reel denemeleri
- Kamera/ses kontrolü, "hazır mısın", "kayıtta mı", sayım ("3, 2, 1")
- Reels arası gülüşme, mola, ekip konuşması
- Kaydın en başı ve en sonundaki boşluk

## 3. Süre kuralları

- En kısa reel 8 sn; tipik aralık 15-90 sn.
- 90 sn'yi aşan bir parça muhtemelen iki reel'dir; ikinci bir giriş cümlesi arayın.
- Reel başlangıcı ilk anlamlı cümleden 0.2 sn önce, bitişi son cümleden 0.4 sn sonradır (script uygular).

## 4. Başlık, kanca ve açıklama standardı

- **title:** 3-7 kelime, dosya adına uygun, içerik ne ise o ("Soğuk kahve tarifi").
- **hook:** Videonun ilk 3 saniyesinde gerçekten söylenen cümle. Uydurma; transkriptten al.
- **caption:** Türkçe, 1-3 cümle, ilk cümle merak uyandırsın, sonda soru veya çağrı olabilir. Emoji en fazla 2.
- **hashtags:** 4-8 adet, Türkçe ve konuya özel; genel etiketlerden (#reels, #keşfet) en fazla 2.

## 5. Dikey format seçimi

| Kaynak | Seçenek |
|---|---|
| Zaten 9:16 çekim | `crop` (dokunmaz, yalnızca 1080x1920'ye ölçekler) |
| Yatay, konuşmacı ortada | `crop` |
| Yatay, iki kişi veya geniş kadraj | `blur` |
| Orijinal oran istenirse | `none` |
