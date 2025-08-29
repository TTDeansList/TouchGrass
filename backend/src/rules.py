import re

AD = re.compile(r"(promo|discount|coupon|subscribe|buy my|use code|check my channel|follow me)", re.I)
IRREL = re.compile(r"(unrelated|off-topic|nothing to do with)", re.I)
NONVISIT = re.compile(r"(haven't been|never visited|didn't go|havent gone)", re.I)
CONTACT = re.compile(r"(\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|@|gmail\.com|outlook\.com)", re.I)

def rule_flags(text: str):
    return {
        "ad_spam": bool(AD.search(text) or CONTACT.search(text)),
        "irrelevant": bool(IRREL.search(text)),
        "non_visit_rant": bool(NONVISIT.search(text)),
    }
