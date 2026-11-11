# Dışa aktarım formatları

Tüm formatlar kaynak videoya referans verir; kesimler editörde düzenlenebilir bir timeline olarak açılır.
XML ve EDL dosyasını kaynak videoyla aynı klasörde tutun ya da editörde "Relink" ile bağlayın.

## Premiere Pro — XML (önerilen)
1. File > Import > `*_edit.xml`
2. Proje panelinde oluşan sekansı açın.
3. Medya çevrimdışı görünürse sağ tık > Link Media > kaynak videoyu seçin.

## DaVinci Resolve — XML
1. File > Import > Timeline > `*_edit.xml`
2. "Automatically import source clips into media pool" seçili olsun.
3. Klip bulunamazsa Media Pool'da sağ tık > Relink Selected Clips.

## DaVinci Resolve — EDL (alternatif)
1. Kaynak videoyu önce Media Pool'a ekleyin.
2. File > Import > Timeline > `*_edit.edl`
3. EDL tek kaynak varsayar (reel adı `AX`, klip adı yorum satırında).

## SRT
Kurgulanmış videonun zaman çizelgesine göre hizalıdır. Premiere: File > Import ve timeline'a sürükleyin. Resolve: Media Pool'a ekleyip altyazı izine bırakın.

## Kare hızı notu
29.97 / 23.976 / 59.94 fps için XML'de NTSC bayrağı otomatik ayarlanır. EDL non-drop-frame zaman kodu kullanır.
