import re, html

URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

def clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = s.strip().lower()
    s = URL_RE.sub(" <URL> ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s
