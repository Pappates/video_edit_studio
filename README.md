# Video Edit Studio

Konuşmaya dayalı otomatik video kurgu ve reels bölme aracı. Kendi bilgisayarınızda çalışan bir web arayüzü
ve Claude için iki skill paketinden oluşur.

| Kısım | Ne yapar |
|---|---|
| **1 · Video Edit** | Ham çekimi Whisper ile yazıya döker; "burayı kes", "şurayı keselim", "baştan alıyorum" gibi sözlü komutları, komutsuz tekrarlanan ve yarım kalan cümleleri, kelime tekrarlarını, çekim arası muhabbeti ve uzun sessizlikleri bulur. Kesimleri arayüzde onaylarsınız; kesilmiş MP4 + Premiere/DaVinci XML + EDL + SRT üretir. |
| **3 · YouTube Açıklaması** | Yayınlayacağın videoyu yazıya döker; Claude bölüm zaman damgaları (YouTube kurallarına uygun: ilk 00:00, en az 3 bölüm, her biri ≥10 sn), açıklama metni, hashtag'ler, Studio etiketleri ve başlık önerileri hazırlar. Hepsi düzenlenebilir, tek tıkla kopyalanır. Video Edit işinden de üretilebilir; o zaman zamanlar kurgulanmış videoya göre hesaplanır ve kesimler değişince kendiliğinden güncellenir. |
| **2 · Reels Üretimi** | İçinde arka arkaya çekilmiş birden çok reel olan videoyu transkript, duraklamalar ve görüntü değişimleriyle (Claude vision) reel'lere böler; yarım denemeleri ve aradaki muhabbeti atar, her reel'in içini temizler; 9:16 MP4, altyazı, başlık, kanca ve Instagram açıklaması üretir. |

## Kurulum

Gereksinimler: Python 3.10+, [ffmpeg](https://ffmpeg.org) (PATH'te), mümkünse NVIDIA GPU.

```bash
# macOS / Linux
./run.sh
# Windows
run.bat
```

İlk çalıştırmada sanal ortam kurulur, `.env` oluşturulur.

### Claude bağlantısı — iki yol

**A) Claude aboneliğinle (Pro/Max, API anahtarı gerekmez — önerilen)**
1. Claude Code'u kur: https://claude.com/claude-code (Windows / macOS / Linux kurulum komutu sayfada).
2. Terminalde bir kez `claude` yaz, açılan ekranda **Claude hesabınla** giriş yap (`/login`), sonra çık.
3. Uygulamayı başlat. Üstteki etiket "Claude aboneliği · sonnet" görünür.

İstekler `claude -p` ile gider ve aboneliğinin kullanım limitlerinden düşer. Reels karelerini Claude, Read
aracıyla dosyadan açarak inceler. `.env` içinde `CLAUDE_BACKEND=cli` ile bu yolu sabitleyebilirsin.
Not: `ANTHROPIC_API_KEY` ortam değişkeni tanımlıysa uygulama onu Claude Code'a geçirmez, böylece istek
yanlışlıkla API'den faturalanmaz.

**B) Claude API anahtarıyla** — `.env` içine `ANTHROPIC_API_KEY` yaz (kullanım başına ücretli; Max/Team
planlarındaki aylık API kredisi de kullanılabilir).

Claude bağlı değilse de uygulama çalışır; doğrulama ve görsel analiz kapanır, reels sezgisel bölünür.
Whisper modeli ilk transkripsiyonda indirilir (large-v3 ≈ 3 GB). GPU yoksa `.env` içinde `WHISPER_MODEL=medium` önerilir.

Arayüz: http://localhost:8000

## Arayüz kullanımı

1. Sekme seçin (Video Edit / Reels), videoyu sürükleyin.
2. Analiz bitince kesimler / reels listelenir. Zaman çizelgesinde renkli bloklar kesimlerdir.
3. **▶** ile kesilen yeri dinleyin, onay kutusuyla açıp kapatın, zamanları elle düzeltin.
4. **Kesilmiş önizleme** açıkken oynatıcı kesimleri atlayarak oynatır — sonucu render etmeden görürsünüz.
5. Eksik kesim için oynatırken **I** (giriş) ve **O** (çıkış) tuşlayıp **+ Kesim ekle**.
6. **Videoyu oluştur** veya **Premiere/DaVinci XML** / **EDL** / **SRT** indirin.

Reels sekmesinde her reel'in başlığı, sınırları, açıklaması düzenlenebilir; "içini temizle" reel içindeki
tekrar ve sessizlikleri de keser. Dikey format: ortadan kırp veya bulanık arka plan.

### Reel sınırlarını elle işaretleme (kurgu şeridi)

Yüklerken **Reel sınırları: Ben işaretleyeceğim** (varsayılan) seçilirse otomatik bölme yapılmaz; oynatıcının
altındaki Shotcut benzeri **kurgu şeridinde** (zaman cetveli, küçük resimler, ses dalgası) sınırları sen
belirlersin. Video yazıya dökülürken de işaretlemeye başlayabilirsin.

| Tuş | İş |
|---|---|
| **S** | Oynatıcının konumundan böl. Art arda iki S: aradaki parça reel olur (reel başı → reel sonu) |
| **R** | Seçili parçayı reel yap / reel'likten çıkar |
| **Delete** | Seçili parçayı at (reel değil) |
| **, / .** | 1 kare geri / ileri · **+ / −** yakınlaştır |

Parçaya tıkla: seç ve oraya git · kırmızı çizgiyi sürükle: sınırı kaydır · **⇥ Birleştir**: çizgiyi kaldır.
İşaretler anında kaydedilir. **📝 Reels'i hazırla**: işaretli reels için Claude başlık/kanca/Instagram
açıklaması yazar ve seçilen seviyede kısaltır (sınırlara dokunmaz). Sonrası aynı: görseller, yerleşim, render.
"Otomatik (Claude)" seçilirse eski davranış (Claude böler); şerit yine görünür, sınırlar elle düzeltilebilir.

**Bitiş kontrolü:** Bölmeden sonra Claude her reel'in bitişine bir kez daha bakar (bitişten sonraki ~2,5 dk'lık
transkript + oyun ekranının kareleri): oyunlarda turlar arası kategori değişimi yeni reel sayılmaz, reel
bitiş ekranıyla ("Başardınız", skor tamamlandı) ve son tepkilerle biter; "bir tane daha oynayalım mı" dışarıda
kalır. Mevcut bir işte her reel'deki **⏹ Bitişi kontrol et** ile aynı kontrol tek tıkla yapılır.

**Çekimde söylenen işaretler:** "1. reel", "ikinci reel", "reel 3", "yeni reel", "sıradaki reele geçiyoruz",
"bir reel daha" gibi işaretler kesin reel sınırı sayılır ve işaret sözleri reel'den kesilir (önündeki
"ay olmadı", "tamam şimdi" ve arkasındaki "tekrar", "baştan alıyorum" gibi sözlerle birlikte).

**Yeniden çekilen reel:** Aynı numara tekrar söylenirse ("1. reel" bozuldu → "1. reel tekrar") son deneme
seçili gelir; önceki denemeler silinmez, **kapalı** gelir ve "⟲ Önceki deneme" notu düşülür — istersen
kutucuğu işaretleyip onu kullanırsın. "Kaç reel?" modunda önceki denemeler sayılmaz; durmadan önce son
reel'in 2 dakika arkasına da bakılır ki yeniden çekim kaçmasın. Numarasız "yeni reel" işaretlerinde
tekrarı ancak Claude içerikten anlayabilir. Tek başına "bir reel çekelim",
"bu reels içerikleri", "Real Madrid" işaret sayılmaz (1 için "1. reel", "birinci reel" ya da "ilk reel" deyin).

**Uzun videoda hızlı deneme — "Kaç reel?":** Reels sekmesinde yüklemeden önce "İlk 2 reel" gibi bir sayı
seçersen video baştan ~10 dakikalık parçalar halinde yazıya dökülür ve her parça reels'e bölünür; o kadar
tam reel bulununca Whisper durur, videonun geri kalanı işlenmez. Parça sonunda yarım kalan reel sayılmaz,
bir sonraki parçada bütün olarak bulunur. "Videonun hepsine bak" eskisi gibi tüm videoyu işler.
Komut satırında: `--max-reels 2`.

### Reels: dikey yerleşim (oyun + yüzler)

Yatay kayıtta (ör. Riverside: solda ekran paylaşımı, sağda kameralar) ortadan 9:16 kırpmak oyunun yarısını
ve yüzleri keser. Bunun yerine kayıt bölgelere ayrılıp yeniden dizilir:

| Seçenek | Görünüm |
|---|---|
| **Yüzler üstte, oyun altta** (önerilen) | Kameralar üstte yan yana, oyun altta; Instagram'ın alt yazı/buton alanı oyunun boş alt kısmına denk gelir |
| Oyun üstte, yüzler altta | Oyun büyük üstte, kameralar altta yan yana |
| Oyun tam boy, yüzler küçük | Oyun ekranın ~%70'i, kameralar altta çerçeveli kutular |

Bölgeler analizde otomatik bulunur: Claude bir kareye bakıp oyunu (tarayıcı/görev çubuğu hariç) ve her
kamerayı işaretler; Claude yoksa görüntü analizi kamera sütununu ve ekrandaki oyun kartını bulur.
**⚙ Bölgeler ve önizleme** ile kutuları görüp sürükleyerek düzeltebilir, 9:16 önizlemeyi render etmeden
görebilirsin. Altyazı ve görsel kartları yerleşime göre konur (yüzlerin üstüne binmez). Tek kamera varsa
yüz şeridi tek geniş kutu olur. Komut satırı: `--vertical faces_top|game_top|game_full|crop|blur|none`.

### Reels: kendi içinde kısaltma

Konu konu bölündükten sonra her reel ayrıca kısaltılır (yüklemede "Reel'i kısalt": Kapalı · Hafif · **Orta** ·
Agresif). Claude reel'in kalan transkriptini başlığı, kancası ve reel'den 3 kareyle okur (ekranda oyun
varsa yön tarifleri/hamleler oyunun parçası sayılır; isteğe bağlı **içerik notu** ile bağlam verilebilir),
hikâyeyi bozmadan keser: kategori
yazma/değiştirme, kural açıklamasının uzayan kısmı, beklemeler, lojistik, konu dışı laflar, aynı tahminin
tekrarı. Kanca, oyunun can alıcı anları, sonuç ve önemli tepkiler kalır. Reel içindeki sessizlikler de
kısaltılır (Hafif: 1,2 sn'den uzunlar ~0,5 sn'ye, Orta: 0,8→0,35, Agresif: 0,6→0,25). "Sessizlik" gerçek ses
seviyesiyle doğrulanır: transkriptte kelime olmayan ama kahkaha/tepki/oyun sesi olan yer kesilmez.
0,3 sn'den kısa kalan kırıntı parçalar atılır (kesik hece gibi duyulmasın).

- Her reel'de "⏱ 92 sn → 48 sn" görünür; **Kısaltma kesimleri** listesinden her parçayı ▶ ile dinleyip tek tek
  geri açabilirsin. Düşük güvenli kararlar kapalı gelir.
- **✂ Seçili reels'i kısalt**: farklı seviye ve isteğe bağlı hedef süreyle ("60 sn altı") yeniden yapar.
- "Kesilmiş önizleme" açıkken oynatıcı reel'i kısaltılmış haliyle oynatır.
- Komut satırında: `--tighten light|medium|aggressive|off`.

### Reels: altyazı ve konuya göre görseller

- **Altyazı:** Videoya gömülü, söylenen kelimeyi anında vurgulayan altyazı (Kalın · Karaoke · Sade; altta veya
  ortada). Whisper'ın kelime zamanlarıyla ASS dosyası üretilir, ffmpeg (libass) ile yazılır. Yazı tipi:
  `.env` → `CAPTION_FONT`.
- **Görseller:** **🖼 Konuya göre görselleri bul** → Claude reel'in transkriptini okur; belirli bir isim
  (futbolcu, takım, şehir, marka, film…) geçtiği anda onun görselini koyar. "İlk 10" gibi listelerde her
  öğeye sıra numarası (#3) verir. Görseller **Wikipedia/Wikimedia Commons**'tan (yalnızca serbest lisanslı,
  yazar/lisans bilgisiyle) ve genel kavramlar için **Pexels**'ten gelir (`PEXELS_API_KEY`, ücretsiz).
- Her görsel için: zamanı ve sıra numarasını düzelt, ekranda yazacak ismi değiştir, aramayı değiştirip 🔍,
  **↻** ile başka aday, **⬆** ile kendi dosyanı yükle, ✕ ile sil. **+ Görsel** oynatıcının konumuna ekler.
  Oynatıcı, görselleri render etmeden videonun üstünde gösterir.
- Görsel yeri: üstte kart, ortada kart veya tam ekran. Kesilen yerlere denk gelen görseller otomatik kaydırılır.
- Telif: Wikimedia ve Pexels görselleri yeniden kullanıma uygundur; Wikimedia görsellerinde açıklamada yazar
  ve lisansı belirtin (her görselin altında yazar). Kendi yüklediğin görsellerin hakları sende olmalı.

## YouTube açıklaması

- **3 · YouTube Açıklaması** sekmesine yayınlayacağın (kurgusu bitmiş) videoyu sürükle. Her açıklamanın
  sonuna eklenecek sabit bilgileri (linkler, sosyal medya) yükleme ekranında bir kez yaz; tarayıcı hatırlar.
- Ya da **Video Edit** işinde **YouTube açıklaması oluştur**'a bas: bölüm zamanları kurgulanmış videoya göre
  hesaplanır, kesimleri değiştirirsen zamanlar kendiliğinden güncellenir.
- Bölümleri düzenle (zaman/başlık), oynatıcıda istediğin yere gelip **+ bölüm ekle**. YouTube kurallarına
  uymayan bölüm kırmızı işaretlenir ve açıklamaya konmaz; sebebi altta yazar.
- Beğenmezsen alttaki kutuya "daha kısa", "daha samimi", "İngilizce yaz" gibi istek yazıp **Yeniden oluştur**.
- **Açıklamayı kopyala** → YouTube Studio açıklama alanına yapıştır; etiketleri ayrıca kopyala. `.txt indir`
  başlık önerileri + açıklama + etiketleri tek dosyada verir.

## Konuşmacı ayrımı ve kişi bazlı ses

Her iki sekmede de video yüklenirken **Konuşmacıları ayır** seçiliyse kimin ne zaman konuştuğu bulunur
(kişi sayısını biliyorsanız seçin, bilmiyorsanız Otomatik). Analizden sonra **Konuşmacılar** panelinde:

- Her kişiye isim verin, ▶ ile örnek dinleyin.
- Kaydırıcıyla sesini −24 dB ile +12 dB arasında ayarlayın, 🔊 ile tamamen susturun.
- **Sesleri eşitle**: herkesin ortalama seviyesini birbirine yaklaştırır (uzaktaki mikrofon vb.).
- Önizlemede ayarlar anında duyulur; **Videoyu oluştur** / reels render'ında uygulanır.
- Bir cümle yanlış kişiye atandıysa transkriptte başındaki renkli etikete tıklayın (Shift+tık: yeni kişi).
- Bütünüyle yanlışsa kişi sayısını seçip **Yeniden ayır**.

Nasıl çalışır: faster-whisper konuşmacı ayırmaz; uygulama Whisper'ın kelime zamanlarından konuşma
parçaları çıkarır, her parçanın ses imzasını WeSpeaker modeliyle (ONNX, ~26 MB, ilk kullanımda
indirilir, ek kurulum gerekmez) hesaplar ve kümeler. Daha zor kayıtlar (çok kişi, sık söz kesme) için
`pip install pyannote.audio` + `.env` içinde `HF_TOKEN` ile pyannote kullanılır.

Sınırlama: ses ayarı zamana göre uygulanır. İki kişi aynı anda konuşursa o an atanmış kişinin ayarı
geçerlidir; üst üste binen sesler birbirinden ayrıştırılmaz. Konuşmacı etiketleri Claude'a da
gönderilir; kamera arkasından gelen yönlendirmeleri ayırt etmesine yardım eder.

## Komut satırı

```bash
python -m vedit edit  "cekim.mp4" --render
python -m vedit reels "reels.mp4" --render --vertical blur
python -m vedit reels "liste.mp4" --render --captions bold --visuals   # altyazı + konuya göre görseller
python -m vedit detect transcript.json --no-claude     # yalnızca kesim listesini gör
```

## Nasıl çalışır

```
video ─► ffprobe ─► faster-whisper (kelime zaman damgası, Türkçe, VAD)
                          │
                          ▼
         cümle birimlerine yeniden bölme (duraklama + noktalama)
                          │
        ┌─────────────────┼──────────────────────────┐
        ▼                 ▼                          ▼
  sözlü komutlar     tekrar / yarım cümle      kekeleme, sessizlik
  (bulanık eşleşme)  (kelime dizisi benzerliği)
        └─────────────────┼──────────────────────────┘
                          ▼
       Claude doğrulaması — abonelik (claude -p) veya API
       (segment numarasıyla karar; yanlış pozitif ret,
       anlamsal tekrar ve çekim arası muhabbet ekleme)
                          ▼
         kullanıcı onayı (arayüz) ─► ffmpeg trim+concat ─► MP4 / XML / EDL / SRT

Reels: transkript + duraklamalar + ffmpeg sahne tespiti + zaman etiketli kareler
       ─► Claude (vision) ─► reel sınırları, başlık, kanca, açıklama ─► 9:16 render
```

Claude zaman uydurmasın diye yalnızca segment numaralarıyla cevap verir; zamanlara kod çevirir.
Kesim kuralları ve komut sözlüğü: `skills/video-edit-automation/references/cut-rules.md`.

## Claude skill'leri

`skills/` klasöründeki iki skill, eklenen "The Complete Guide to Building Skills for Claude" standardına göre
hazırlanmıştır (kebab-case klasör, `SKILL.md`, YAML frontmatter'da ne yaptığı + ne zaman kullanılacağı,
`scripts/` ve `references/` ile kademeli yükleme, skill içinde README yok).

```bash
python build_skills.py      # doğrular ve dist/*.zip üretir
```

`dist/video-edit-automation.zip` ve `dist/reels-splitter.zip` dosyalarını Claude > Ayarlar > Skills > Upload ile
yükleyin ya da Claude Code skills klasörüne koyun. Skill modunda API anahtarı gerekmez: inceleme ve reels
kararını Claude'un kendisi `review.md` / `packet.md` dosyalarını okuyarak verir.

## Proje yapısı

```
vedit/            çekirdek kütüphane (web, CLI ve skill'ler ortak kullanır)
  transcribe.py   faster-whisper
  detector.py     komut / tekrar / kekeleme / sessizlik tespiti
  claude_review.py Claude doğrulaması
  reels.py        reels bölme (transkript + sahne + vision)
  diarize.py      konuşmacı ayrımı (WeSpeaker ONNX veya pyannote)
  audiomix.py     kişi bazlı ses ayarı
  youtube.py      YouTube bölümleri, açıklama, hashtag, etiket
  captions.py     reels için gömülü, kelime vurgulu altyazı (ASS)
  visuals.py      konuya göre görseller (Claude + Wikipedia/Pexels) ve yerleşim
  media.py        ffmpeg
  exporters.py    EDL, FCP7 XML, SRT
  pipeline.py     iş yönetimi
server/           FastAPI + tek sayfa arayüz
skills/           video-edit-automation, reels-splitter
tests/            pytest (Whisper ve API gerektirmez)
```

## Test

```bash
pip install pytest && python -m pytest tests
```

## Ayar ipuçları

| Sorun | Çözüm (.env) |
|---|---|
| Normal cümleler tekrar sanılıyor | `REPEAT_SIMILARITY=0.8` |
| Kesimde kelime kırpılıyor | `CUT_PAD_BEFORE=0.12`, `CUT_PAD_AFTER=0.12` |
| Sessizlikler çok sıkı | `SILENCE_KEEP_SEC=0.5` |
| Yeni bir komut ifadesi yakalanmıyor | `vedit/detector.py` içindeki listelere ASCII olarak ekleyin |
