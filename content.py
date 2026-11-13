"""Slayd matnini yaratish: Gemini AI (ixtiyoriy) yoki oflayn zaxira matn."""
import json
import logging
import os
import re
import time

import requests

log = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# Asosiy model topilmasa (404 — o'chirilgan/eskirgan), shu ro'yxatdan ketma-ket sinaladi.
FALLBACK_MODELS = ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash")


def _model_chain():
    return [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]


def ai_enabled():
    return bool(os.environ.get("GEMINI_API_KEY"))


# ----------------------------------------------------------------- AI qismi
PROMPT = """Sen o'zbek tilida talabalar uchun taqdimot matnini yozadigan yordamchisan.
Fan: {subject}
Mavzu: {topic}
Kerakli asosiy slaydlar soni: {n}

Faqat bitta JSON obyekt qaytar (izoh, markdown yoki ``` belgilarisiz). Tuzilishi:
{{
  "plan": ["reja bandi 1", "reja bandi 2", "reja bandi 3"],
  "slides": [
    {{
      "title": "qisqa sarlavha (6 so'zgacha)",
      "bullets": ["3-4 ta aniq, mazmunli band (har biri 22 so'zgacha)"],
      "notes": "taqdimotchi uchun 2-3 gaplik izoh",
      "image_query": "Wikimedia Commons uchun INGLIZCHA qidiruv so'rovi yoki bo'sh satr"
    }}
  ],
  "conclusion": ["2-3 ta xulosa bandi"]
}}

Qoidalar:
- "slides" ichida aynan {n} ta element bo'lsin; mantiqiy ketma-ketlikda (kirish -> asosiy qism -> misollar -> xulosaga yaqin).
- Har bir band MAVZUNING O'ZI haqida aniq ma'lumot bersin: nom, sana, asar, joy, voqea, ta'rif. Mavzuni nomlab, uni o'rganish haqida umumiy gap yozma.
- Quyidagi kabi umumiy/metamatnlarni HECH QACHON yozma: "ta'riflar ishonchli manbaga tayanishi kerak", "mavzuning dolzarbligini aniqlash kerak", "taqdimotda ... beriladi".
- Sana, raqam, asar nomi, ism va iqtibosni FAQAT aniq bilsang yoz. Ishonching komil bo'lmasa, o'sha ma'lumotni yozma yoki umumiyroq ayt. Hech narsani o'ylab topma.
- Mavzu tarixiy shaxs, asar yoki joy bo'lsa, ma'lumotlar haqiqiy va tekshirilgan faktlarga asoslansin.
- "image_query": faqat slaydning aniq mavzusi Vikipediyada nomi bilan mavjud bo'lsa (shaxs, bino, shahar, asar), o'sha nomni INGLIZCHA, Vikipediyadagi yozilishida, 2-4 so'z qilib yoz (masalan: Alisher Navoi monument). Mavhum slaydlar uchun va ishonching komil bo'lmasa "" qoldir. Mavzu bilan bog'liq bo'lmagan rasm chiqishidan ko'ra rasmsiz bo'lgani yaxshi.
- Matn o'zbek tilida (lotin yozuvi), akademik, lekin sodda bo'lsin.
"""


def _clip(s, n):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _ask_gemini(topic, subject, n):
    key = os.environ["GEMINI_API_KEY"]
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(topic=topic, subject=subject, n=n)}]}],
        "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
    }
    for model in _model_chain():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = requests.post(
            url, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body, timeout=90,
        )
        if r.status_code == 404:  # model topilmadi -> keyingisini sinaymiz
            log.warning("Gemini modeli topilmadi (%s), keyingisi sinaladi: %s", model, r.text[:200])
            continue
        if not r.ok:  # sababni logda ko'rish uchun javob matnini ham yozamiz
            raise RuntimeError(f"Gemini HTTP {r.status_code} ({model}): {r.text[:300]}")
        parts = r.json()["candidates"][0]["content"]["parts"]
        return _parse_json("".join(p.get("text", "") for p in parts))
    raise RuntimeError("Gemini: sinab ko'rilgan modellarning birortasi topilmadi")


def _validate(raw, n):
    slides = []
    for s in raw.get("slides", []):
        if not isinstance(s, dict):
            continue
        title = _clip(s.get("title"), 70)
        bullets = [_clip(b, 170) for b in s.get("bullets", []) if str(b).strip()][:5]
        if not title or not bullets:
            continue
        slides.append({
            "title": title,
            "bullets": bullets,
            "notes": _clip(s.get("notes"), 600),
            "image_query": _clip(s.get("image_query"), 60),
        })
    if len(slides) < max(1, n // 2):
        raise ValueError("AI javobi yetarli slayd bermadi")
    plan = [_clip(p, 80) for p in raw.get("plan", []) if str(p).strip()][:5]
    concl = [_clip(c, 200) for c in raw.get("conclusion", []) if str(c).strip()][:4]
    return {"plan": plan, "slides": slides[:n], "conclusion": concl}


# ------------------------------------------------------------ Zaxira matn
def _fallback_slides(topic, subject):
    t, f = topic, subject
    return [
        ("Mavzuga kirish", [
            f"{t} — {f} fanining o'rganiladigan masalalaridan biri.",
            "Mavzuning dolzarbligi va amaliy ahamiyatini aniqlash kerak.",
            "Taqdimotda asosiy tushunchalar, omillar va xulosalar ketma-ket beriladi.",
        ]),
        ("Asosiy tushunchalar", [
            f"{t} bo'yicha asosiy atamalar aniq ta'riflanishi lozim.",
            "Har bir tushuncha qisqa izoh va misol bilan beriladi.",
            "Ta'riflar ishonchli darslik yoki ilmiy manbaga tayanishi kerak.",
        ]),
        ("Maqsad va vazifalar", [
            f"Maqsad: {t} bo'yicha bilimlarni tizimlashtirish.",
            "Vazifa: tushunchalarni izohlash va omillarni aniqlash.",
            "Vazifa: amaliy misol keltirib, natijalarni umumlashtirish.",
        ]),
        ("Muhim omillar", [
            f"{t} ga ta'sir qiluvchi ichki va tashqi omillarni ajratish.",
            "Omillarning o'zaro bog'liqligini hisobga olish kerak.",
            "Har bir omilning ta'siri dalil bilan asoslanadi.",
        ]),
        ("Amaliy misol", [
            f"{t} ni real vaziyatda ko'rib chiqish mavzuni yaxshi tushuntiradi.",
            "Boshlang'ich shartlar, jarayon va natijani solishtiring.",
            "Raqamlar ishlatilsa, ularning manbasi ko'rsatilishi shart.",
        ]),
        ("Tahlil usullari", [
            "Taqqoslash, tasniflash va kuzatish eng ko'p qo'llaniladi.",
            "Ma'lumotlarni jadval yoki diagrammada ifodalash foydali.",
            "Usul mavzu va mavjud ma'lumotga mos tanlanadi.",
        ]),
        ("Afzallik va cheklovlar", [
            "Ijobiy jihatlar bilan birga cheklovlarni ham baholash kerak.",
            "Xulosalar dalillarga asoslanishi lozim.",
            "Ortiqcha umumlashtirishdan saqlanish zarur.",
        ]),
        ("Muammolar va yechimlar", [
            f"{t} bilan bog'liq asosiy muammolarni aniqlash.",
            "Har bir muammo uchun mumkin bo'lgan yechimni taqqoslash.",
            "Yechimning afzallik va xavflarini baholash.",
        ]),
        ("Natijalar", [
            f"Ko'rib chiqilgan fikrlar {t} ni tizimli tahlil qilish muhimligini ko'rsatadi.",
            "Natijalarni baholashda ma'lumot sifati va kontekst e'tiborga olinadi.",
        ]),
        ("Rivojlanish istiqbollari", [
            f"{t} bo'yicha keyingi tadqiqot yo'nalishlarini belgilash.",
            "Yangi ma'lumot va texnologiyalardan foydalanish imkoniyatlari.",
        ]),
    ]


def fallback_content(topic, subject, n):
    base = _fallback_slides(topic, subject)
    slides = []
    for i in range(n):
        title, bullets = base[i % len(base)]
        if i >= len(base):
            title = f"{title} (davomi)"
        slides.append({"title": title, "bullets": bullets, "notes": "", "image_query": ""})
    return {
        "plan": ["Kirish va asosiy tushunchalar", "Omillar va amaliy tahlil", "Natijalar va xulosa"],
        "slides": slides,
        "conclusion": [
            f"{topic} bo'yicha asosiy tushunchalarni aniqlash muhim.",
            "Tahlilda dalillar va ishonchli manbalarga tayaning.",
        ],
    }


class AIUnavailable(Exception):
    """AI matn bera olmadi va umumiy zaxira matn yuborish o'chirilgan."""


# Umumiy zaxira matn mavzu haqida hech qanday haqiqiy ma'lumot bermaydi, shuning uchun
# standart holatda yuborilmaydi. Kerak bo'lsa ALLOW_FALLBACK=1 qo'ying.
ALLOW_FALLBACK = os.environ.get("ALLOW_FALLBACK", "0") == "1"


def _ask_gemini_retry(topic, subject, n, attempts=3):
    """Vaqtinchalik xatolarda (kvota, tarmoq, noto'g'ri JSON) qayta urinadi."""
    last = None
    for k in range(attempts):
        try:
            return _validate(_ask_gemini(topic, subject, n), n)
        except Exception as e:
            last = e
            log.warning("Gemini urinish %s/%s muvaffaqiyatsiz: %s", k + 1, attempts, e)
            if k < attempts - 1:
                time.sleep(3 * (k + 1))
    raise last


def build_content(topic, subject, n):
    """(content, used_ai) qaytaradi. AI ishlamasa AIUnavailable (yoki ALLOW_FALLBACK=1 bo'lsa zaxira matn)."""
    if ai_enabled():
        try:
            content = _ask_gemini_retry(topic, subject, n)
            # AI kam slayd bergan bo'lsa, qolganini zaxira bilan to'ldiramiz
            if len(content["slides"]) < n:
                extra = fallback_content(topic, subject, n)["slides"]
                content["slides"] += extra[len(content["slides"]):n]
            fb = fallback_content(topic, subject, n)
            content["plan"] = content["plan"] or fb["plan"]
            content["conclusion"] = content["conclusion"] or fb["conclusion"]
            return content, True
        except Exception as e:  # tarmoq, kvota, noto'g'ri JSON va h.k.
            log.warning("AI matn yaratish muvaffaqiyatsiz: %s", e)
    else:
        log.warning("GEMINI_API_KEY o'rnatilmagan")
    if not ALLOW_FALLBACK:
        raise AIUnavailable("AI matn bera olmadi")
    return fallback_content(topic, subject, n), False
