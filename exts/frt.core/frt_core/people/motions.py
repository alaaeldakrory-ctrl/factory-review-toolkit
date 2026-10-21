"""The manufacturing motion catalog (data/motions.toml): load, validate, query.

    lib = load(); lib["fasten.ratchet.tighten"].pattern; by_category(lib)["wiring"]
"""
import os
from dataclasses import dataclass, field

try:
    import tomllib
except ImportError:                                     # Python < 3.11 (older Kit)
    import tomli as tomllib

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "motions.toml")
HANDS = {"none", "one", "two", "either", "dom+sup", "team"}
POSTURES = {"standing", "kneeling", "squatting", "overhead", "seated", "lying", "on_stand"}
PATTERNS = {"reach_place", "operate", "inspect", "body", "walk", "carry", "team", "route_path"}
UAS = ("GP", "PL", "HT", "OP", "MC", "FL", "BM", "VC")


@dataclass
class Motion:
    id: str
    name: str
    category: str
    hands: str
    tools: list
    object: str
    postures: list
    pattern: str
    uas: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    risks: list = field(default_factory=list)

    @property
    def generator(self):
        """Frame generator name: 'cycle:ratchet' -> ('cycle', 'ratchet'), 'reach_place' -> ('reach_place', None)."""
        a, _, b = self.pattern.partition(":")
        return a, (b or None)


def validate(rows):
    """Problems in catalog rows (empty list = valid)."""
    out, seen = [], set()
    for r in rows:
        mid = r.get("id", "?")
        if mid in seen:
            out.append(f"{mid}: duplicate id")
        seen.add(mid)
        if r.get("hands") not in HANDS:
            out.append(f"{mid}: hands {r.get('hands')!r} not in {sorted(HANDS)}")
        bad = [p for p in r.get("postures", []) if p not in POSTURES]
        if bad or not r.get("postures"):
            out.append(f"{mid}: postures {bad or 'missing'}")
        kind = r.get("pattern", "").split(":")[0]
        if kind not in PATTERNS and kind != "cycle":
            out.append(f"{mid}: pattern {r.get('pattern')!r}")
        for u in r.get("uas", []):
            if not u.split("[")[0].split(" ")[0] in UAS:
                out.append(f"{mid}: uas block {u!r}")
    return out


def load(path=DATA, strict=True):
    with open(path, "rb") as f:
        data = tomllib.load(f)
    rows = data.get("motion", [])
    problems = validate(rows)
    if problems and strict:
        raise ValueError("motion catalog: " + "; ".join(problems[:10]))
    return {r["id"]: Motion(**{k: r.get(k) for k in Motion.__dataclass_fields__ if k in r}) for r in rows}


def by_category(lib):
    out = {}
    for m in lib.values():
        out.setdefault(m.category, []).append(m)
    return out


def cycles(lib):
    """Motion-cycle generators the catalog needs (what still has to be written, with how many motions use each)."""
    out = {}
    for m in lib.values():
        kind, name = m.generator
        key = f"cycle:{name}" if kind == "cycle" else kind
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
