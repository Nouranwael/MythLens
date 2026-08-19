"""Conservative clinical-safety screening for medical misinformation."""
from __future__ import annotations

from typing import Any, Dict, List


def _claim_text(claim: Any) -> str:
    if isinstance(claim, dict):
        return str(claim.get("normalized_claim") or claim.get("original_claim") or "").strip().lower()
    return str(claim or "").strip().lower()


def assess_risk(claim: Any, evidence: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    text = _claim_text(claim)

    emergency = [
        "بلع اللسان", "حط حاجة في بقه", "حط معلقة في بقه",
        "put something in the mouth during a seizure", "don't call an ambulance",
        "ما تروحش الطوارئ", "متروحش الطوارئ",
    ]
    abandonment = [
        "وقف الكيماوي", "سيب الكيماوي", "مش محتاج كيماوي", "stop chemotherapy",
        "وقف الانسولين", "مش محتاج انسولين", "stop insulin",
        "وقف العلاج", "سيب العلاج", "replace prescribed treatment",
    ]
    harmful = [
        "حط ثوم على الجرح", "ثوم على الجرح", "ضع ثوم على الجرح", "وضع الثوم على الجرح",
        "toothpaste on burn", "معجون اسنان على الحرق", "bleach", "كلور",
        "جرعة كبيرة", "double the dose", "ضاعف الجرعة",
    ]
    replacement = [
        "بديل للمضاد الحيوي", "بديل عن المضاد الحيوي", "بدل المضاد الحيوي",
        "بديل للمضادات الحيوية", "بديل عن المضادات الحيوية", "بدل المضادات الحيوية",
        "نستخدمه بدل المضاد الحيوي", "استخدمه بدل المضاد الحيوي",
        "replace antibiotics", "replace an antibiotic", "instead of antibiotics",
        "instead of an antibiotic", "alternative to antibiotics", "antibiotic alternative",
    ]

    if any(p in text for p in emergency):
        return {
            "risk_level": "CRITICAL",
            "safe_recommendation": "الادعاء ممكن يسبب ضرر فوري في حالة طارئة. اتبع إرشادات الإسعاف الموثوقة واطلب المساعدة الطبية العاجلة عند وجود أعراض خطيرة.",
            "detected_risks": ["unsafe_emergency_action"],
        }
    if any(p in text for p in abandonment):
        return {
            "risk_level": "HIGH",
            "safe_recommendation": "ما توقفش علاج موصوف أو تستبدله بناءً على الادعاء ده من غير الرجوع لطبيب مختص.",
            "detected_risks": ["treatment_delay_or_abandonment"],
        }
    if any(p in text for p in harmful):
        return {
            "risk_level": "HIGH",
            "safe_recommendation": "تجنب تجربة الوصفة على الجلد أو الجروح أو تغيير الجرعات بنفسك، واطلب نصيحة طبية مناسبة للحالة.",
            "detected_risks": ["direct_home_remedy_harm"],
        }
    if any(p in text for p in replacement):
        return {
            "risk_level": "HIGH",
            "safe_recommendation": "ما تستبدلش المضاد الحيوي أو أي دواء موصوف بوصفة منزلية أو مكمل من غير مراجعة الطبيب أو الصيدلي.",
            "detected_risks": ["medication_replacement", "treatment_delay_risk"],
        }
    if any(p in text for p in ["جرعة", "dose", "حامل", "pregnant", "حمل", "drug interaction", "تداخل دوائي"]):
        return {
            "risk_level": "MODERATE",
            "safe_recommendation": "الادعاء متعلق بموضوع ممكن يتأثر بالحالة الصحية أو الجرعة؛ الأفضل مراجعة طبيب أو صيدلي قبل التطبيق.",
            "detected_risks": ["needs_professional_guidance"],
        }
    return {
        "risk_level": "LOW",
        "safe_recommendation": "راجع المصادر الطبية الموثوقة قبل تطبيق أي نصيحة صحية، خصوصًا لو عندك مرض مزمن أو بتاخد أدوية.",
        "detected_risks": ["misinformation"],
    }
