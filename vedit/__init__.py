"""vedit — konuşma tabanlı otomatik video edit ve reels bölme çekirdeği.

Modüller:
    transcribe  : faster-whisper ile kelime zaman damgalı Türkçe transkripsiyon
    detector    : kesim komutu, yeniden çekim (retake), tekrar ve sessizlik tespiti
    reels       : uzun videodan birden çok reel'i ayırma (transkript + sahne + VLM)
    media       : ffmpeg yardımcıları (kesme, birleştirme, kare çıkarma, dikey render)
    exporters   : EDL / FCP7 XML (Premiere + DaVinci) / SRT / JSON
    claude_client: Claude Messages API istemcisi (metin + görsel)
"""

__version__ = "1.0.0"
