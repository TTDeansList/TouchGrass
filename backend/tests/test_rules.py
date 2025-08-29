# tests/test_rules.py
from backend.src.rules import rule_flags

def test_ad_spam():
    txt = "Buy now! Use code SAVE20. Visit www.spam.com"
    flags = rule_flags(txt)
    assert flags["ad_spam"] is True

def test_non_visit():
    txt = "I haven't been here but it looks terrible"
    flags = rule_flags(txt)
    assert flags["non_visit_rant"] is True

def test_irrelevant():
    txt = "This post is unrelated to the cafe"
    flags = rule_flags(txt)
    assert flags["irrelevant"] is True
