"""Findings: the one output format of every check, so the issue list, reports and the assistant work the same for
all modules.

    f = Finding("validate.floating", "warning", "Table T3 floats 4.2 cm above the floor",
                prims=["/World/Items/T3"], location=(x, y, z), value=4.2, limit=1.0, unit="cm",
                rule="validate.floating_tolerance_cm")
    fl = FindingList(); fl.add(f); fl.save("findings.json")
"""
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

SEVERITIES = ("info", "warning", "critical")


@dataclass
class Finding:
    check: str                                  # module.check, e.g. "validate.floating", "layout.aisle_width"
    severity: str                               # info / warning / critical
    message: str                                # one plain sentence a person can act on
    prims: List[str] = field(default_factory=list)
    location: Optional[Tuple[float, float, float]] = None      # world point, cm
    value: Optional[float] = None
    limit: Optional[float] = None
    unit: str = ""
    rule: str = ""                              # rule-pack key that set the limit
    reference: str = ""                         # standard / source, e.g. "OSHA 1910.176(a)"
    view: Optional[dict] = None                 # {"eye": [x, y, z], "at": [x, y, z]} saved camera
    id: str = ""

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")


class FindingList:
    def __init__(self, items=None):
        self.items = list(items or [])

    def add(self, f):
        if not f.id:
            n = sum(1 for x in self.items if x.check == f.check) + 1
            f.id = f"{f.check}#{n}"
        if f.view is None and f.location is not None:
            x, y, z = f.location
            f.view = {"eye": [x + 450.0, y + 330.0, z + 450.0], "at": [x, y, z]}
        self.items.append(f)
        return f

    def extend(self, other):
        for f in (other.items if isinstance(other, FindingList) else other):
            self.add(f)
        return self

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def by_severity(self, sev):
        return [f for f in self.items if f.severity == sev]

    def counts(self):
        return {s: len(self.by_severity(s)) for s in SEVERITIES}

    def summary(self):
        c = self.counts()
        return f"{len(self.items)} findings: {c['critical']} critical, {c['warning']} warning, {c['info']} info"

    def to_dict(self):
        return {"summary": self.counts(), "findings": [asdict(f) for f in self.items]}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls([Finding(**{k: (tuple(v) if k == "location" and v else v) for k, v in f.items()}) for f in d["findings"]])
