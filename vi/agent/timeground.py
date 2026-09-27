"""Deterministic time grounding for questions (E-AGT-04). The model never resolves clock words;
this does, against the footage bounds, and the loop refuses or clarifies before any model call.
All times are absolute UTC milliseconds; display strings use the site timezone."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass
class Grounding:
    kind: str                      # ok | future | before_start | ambiguous | none
    t_start_ms: int | None = None
    t_end_ms: int | None = None
    message: str = ""
    matched: str = ""


NUMW = r"(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|couple of|few|half an?)"
REL = re.compile(r"\b(last|past|previous)\s+" + NUMW + r"?\s*(min|mins|minute|minutes|hour|hours|hr|hrs|sec|seconds)\b", re.I)
AGO = re.compile(r"\b" + NUMW + r"\s*(min|mins|minute|minutes|hour|hours|hr|hrs)\s+ago\b", re.I)
JUST_NOW = re.compile(r"\b(just now|right now|a moment ago|moments ago|just happened|currently|at the moment)\b", re.I)
WORDS_N = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "fifteen": 15, "twenty": 20, "thirty": 30, "couple of": 2, "few": 3, "half an": 0.5, "half a": 0.5}


def _n(tok: str | None) -> float:
    if not tok:
        return 1
    t = tok.strip().lower()
    return float(t) if t.isdigit() else WORDS_N.get(t, 1)
CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", re.I)
BETWEEN = re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[,.?]|$)", re.I)
AFTER = re.compile(r"\b(after|since|from)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", re.I)
BEFORE = re.compile(r"\b(before|until|till)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", re.I)
WORDS = {"now": 0, "right now": 0, "currently": 0, "at the moment": 0}
TOMORROW = re.compile(r"\b(tomorrow|next week|next hour|later today|tonight|in \d+ (minutes|hours))\b", re.I)


def _clock_to_ms(text: str, ref_day: datetime, tz: ZoneInfo) -> int | None:
    m = CLOCK.search(text)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "").lower()
    if ap == "pm" and h < 12: h += 12
    if ap == "am" and h == 12: h = 0
    if not ap and h > 23:
        return None
    local = ref_day.astimezone(tz).replace(hour=h, minute=mi, second=0, microsecond=0)
    return int(local.timestamp() * 1000)


def fmt(ms: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ms / 1000, tz).strftime("%H:%M:%S")


def ground(question: str, now_ms: int, start_ms: int, end_ms: int, tz_name: str = "UTC") -> Grounding:
    """now_ms = latest processed timestamp; [start_ms, end_ms] = footage bounds in the store."""
    tz = ZoneInfo(tz_name)
    q = question.strip()
    ref = datetime.fromtimestamp(now_ms / 1000, tz)
    if TOMORROW.search(q):
        return Grounding("future", message=f"That is after the latest footage I have, which ends at {fmt(end_ms, tz)}.", matched=TOMORROW.search(q).group(0))
    m = JUST_NOW.search(q)
    if m:
        return Grounding("ok", max(start_ms, now_ms - 120_000), now_ms, matched=m.group(0))
    m = REL.search(q)
    if m:
        n, unit = _n(m.group(2)), m.group(3).lower()
        span = int(n * (3600 if unit.startswith("h") else 1 if unit.startswith("s") else 60) * 1000)
        return Grounding("ok", max(start_ms, now_ms - span), now_ms, matched=m.group(0))
    m = AGO.search(q)
    if m:
        n, unit = _n(m.group(1)), m.group(2).lower()
        span = int(n * (3600 if unit.startswith("h") else 60) * 1000)
        t = now_ms - span
        if t < start_ms:
            return Grounding("before_start", message=f"That is before the footage starts at {fmt(start_ms, tz)}.", matched=m.group(0))
        return Grounding("ok", max(start_ms, t - 120_000), min(end_ms, t + 120_000), matched=m.group(0))
    m = BETWEEN.search(q)
    if m:
        a, b = _clock_to_ms(m.group(1), ref, tz), _clock_to_ms(m.group(2), ref, tz)
        if a is not None and b is not None:
            a, b = min(a, b), max(a, b)
            if a > end_ms:
                return Grounding("future", message=f"{fmt(a, tz)} is after the latest footage I have ({fmt(end_ms, tz)}).", matched=m.group(0))
            if b < start_ms:
                return Grounding("before_start", message=f"That window is before the footage starts at {fmt(start_ms, tz)}.", matched=m.group(0))
            return Grounding("ok", max(a, start_ms), min(b, end_ms), matched=m.group(0))
    m = AFTER.search(q)
    if m:
        a = _clock_to_ms(m.group(2), ref, tz)
        if a is not None:
            if a > end_ms:
                return Grounding("future", message=f"{fmt(a, tz)} is after the latest footage I have ({fmt(end_ms, tz)}).", matched=m.group(0))
            return Grounding("ok", max(a, start_ms), end_ms, matched=m.group(0))
    m = BEFORE.search(q)
    if m:
        b = _clock_to_ms(m.group(2), ref, tz)
        if b is not None:
            if b < start_ms:
                return Grounding("before_start", message=f"That is before the footage starts at {fmt(start_ms, tz)}.", matched=m.group(0))
            return Grounding("ok", start_ms, min(b, end_ms), matched=m.group(0))
    m = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)\b", q, re.I)
    if m:
        t = _clock_to_ms(m.group(1), ref, tz)
        if t is not None:
            if t > end_ms + 60_000:
                return Grounding("future", message=f"{fmt(t, tz)} is after the latest footage I have ({fmt(end_ms, tz)}).", matched=m.group(0))
            if t < start_ms - 60_000:
                return Grounding("before_start", message=f"{fmt(t, tz)} is before the footage starts at {fmt(start_ms, tz)}.", matched=m.group(0))
            return Grounding("ok", max(start_ms, t - 120_000), min(end_ms, t + 120_000), matched=m.group(0))
    return Grounding("none", start_ms, end_ms)
