# backend/src/rules.py
import re
from typing import Dict

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

AD_WORDS = [
    r"\bpromo\b", r"\bdiscount\b", r"\bdeal\b", r"\bbuy now\b",
    r"\bsubscribe\b", r"\buse code\b", r"\bcoupon\b", r"\bsale\b"
]

IRRELEVANT_HINTS = [
    r"\bunrelated\b", r"\bnot about\b", r"\boff-topic\b", r"\boff topic\b",
    r"\bthis post\b", r"\bthis review\b.*\bnot\b.*\b(place|location|restaurant|shop)\b"
]

NON_VISIT_HINTS = [
    r"\bnever been\b", r"\bnot been\b", r"\bhaven't been\b", r"\bhavent been\b",
    r"\bdidn'?t visit\b", r"\bnot visited\b", r"\blooks bad\b", r"\bheard it's\b"
]

def rule_flags(text: str) -> Dict[str, bool]:
    """Return policy flags: ad_spam, irrelevant, non_visit_rant."""
    t = text or ""
    ad_spam = bool(URL_RE.search(t)) or any(re.search(w, t, re.IGNORECASE) for w in AD_WORDS)
    irrelevant = any(re.search(w, t, re.IGNORECASE) for w in IRRELEVANT_HINTS)
    non_visit_rant = any(re.search(w, t, re.IGNORECASE) for w in NON_VISIT_HINTS)
    return {
        "ad_spam": ad_spam,
        "irrelevant": irrelevant,
        "non_visit_rant": non_visit_rant,
    }
