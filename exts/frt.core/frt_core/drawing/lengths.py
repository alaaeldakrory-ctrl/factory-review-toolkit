"""Lengths as people write them on drawings, to centimetres: 40', 16'-10", 16' 10 1/2", 30'-10 3/4", 12.5 m, 3500 mm,
250 cm, 96 in, 8 ft, or a bare number (taken as cm).

    parse_length("16'-10\"")   -> 513.08
"""
import re

_FRAC = r"(\d+(?:\.\d+)?)(?:\s+(\d+)\s*/\s*(\d+))?|(\d+)\s*/\s*(\d+)"
_UNITS = {"mm": 0.1, "cm": 1.0, "m": 100.0, "in": 2.54, '"': 2.54, "inch": 2.54, "inches": 2.54,
          "ft": 30.48, "'": 30.48, "feet": 30.48, "foot": 30.48}


def _num(m, base):
    if m.group(base) is not None:
        v = float(m.group(base))
        if m.group(base + 1):
            v += float(m.group(base + 1)) / float(m.group(base + 2))
        return v
    if m.group(base + 3):
        return float(m.group(base + 3)) / float(m.group(base + 4))
    return 0.0


def parse_length(s):
    if isinstance(s, (int, float)):
        return float(s)
    t = s.strip().lower().replace("′", "'").replace("″", '"').replace("’", "'").replace("”", '"')
    t = t.replace("''", '"')
    # feet and inches: 16'-10", 16' 10 1/2", 16'
    m = re.fullmatch(rf"(?:({_FRAC})\s*(?:'|ft|feet|foot))?\s*-?\s*(?:({_FRAC})\s*(?:\"|in|inch|inches))?", t)
    if m and (m.group(1) or m.group(7)):
        ft = _num(m, 2) if m.group(1) else 0.0
        inch = _num(m, 8) if m.group(7) else 0.0
        return ft * 30.48 + inch * 2.54
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)?", t)
    if m:
        return float(m.group(1)) * _UNITS[m.group(2) or "cm"]
    raise ValueError(f"cannot read the length {s!r} (try 40', 16'-10\", 12.5 m, 3500 mm)")
