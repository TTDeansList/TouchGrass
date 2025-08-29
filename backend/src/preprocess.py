# backend/src/preprocess.py
import html
import re

URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
MULTISPACE_RE = re.compile(r'\s+')

def clean_text(s: str) -> str:
    """unescape HTML, lowercase, normalize urls/emojis, collapse spaces"""
    if s is None:
        return ""
    s = html.unescape(s)
    s = s.strip().lower()
    s = URL_RE.sub(" <URL> ", s)
    s = EMOJI_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip()
