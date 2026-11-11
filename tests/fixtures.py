"""Testler için sentetik transkript üretici (Whisper gerektirmez)."""
from __future__ import annotations


def make_transcript(sentences: list[tuple[float, str]], word_dur: float = 0.32, gap: float = 0.08,
                    duration: float | None = None) -> dict:
    """[(başlangıç_sn, "cümle"), ...] → whisper formatında transkript."""
    segs = []
    for i, (start, text) in enumerate(sentences):
        t = start
        words = []
        for w in text.split():
            words.append({"start": round(t, 3), "end": round(t + word_dur, 3), "word": w, "prob": 0.95})
            t += word_dur + gap
        segs.append({"id": i, "start": words[0]["start"], "end": words[-1]["end"], "text": text, "words": words})
    return {"language": "tr", "duration": duration or (segs[-1]["end"] + 1.0), "segments": segs}


# Gerçekçi bir çekim: komutlu retake, komutsuz tekrar, yarım cümle, "burayı kes", kekeleme, uzun sessizlik
SHOOT = [
    (0.5, "Merhaba arkadaşlar bugün size yeni kahve makinemizi tanıtacağım."),
    (4.5, "Bu makine on beş bar basınçla çalışıyor ve süt köpürtücüsü var."),
    (9.0, "Ay pardon baştan alıyorum."),
    (11.0, "Bu makine on beş bar basınçla çalışıyor ve otomatik süt köpürtücüsü var."),
    (16.0, "Fiyatı da gayet uygun."),
    (19.0, "Kutusundan çıkan aksesuarlar şunlar."),
    (22.5, "Kutusundan çıkan aksesuarlar şunlar bir ölçü kaşığı ve bir temizleme fırçası."),
    (28.0, "Şimdi ilk kahvemizi hazırlayalım ya kamera açık mı."),
    (32.0, "Tamam şurayı kes."),
    (34.0, "Şimdi ilk kahvemizi hazırlayalım."),
    (37.0, "Gördüğünüz gibi gördüğünüz gibi köpük çok yoğun."),
    (45.0, "Beğendiyseniz takip etmeyi unutmayın görüşmek üzere."),
]
