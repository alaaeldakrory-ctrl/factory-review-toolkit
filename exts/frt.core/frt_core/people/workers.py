"""Worker profiles from anthropometric percentiles.

Body dimensions (cm, adults, standing, unclothed) are approximate percentile values from published anthropometric
tables (Pheasant & Haslegrave, Bodyspace, 3rd ed. - British adults 19-65; ANSUR II is similar). They set the
segment lengths of the canonical skeleton, so reach and posture are right for each body size: the 5th percentile
female is the one who over-reaches, the 95th percentile male the one who stoops at low work.

    w = WORKERS["W05F"]; w.dims["shoulder_height"]
"""
from dataclasses import dataclass, field

DIMS = {
    #                        5th F   50th M   95th M
    "stature":               (151.0, 174.0, 185.5),
    "eye_height":            (141.5, 163.0, 174.5),
    "shoulder_height":       (123.5, 142.5, 153.5),
    "elbow_height":          (93.5, 109.0, 118.0),
    "hip_height":            (79.0, 92.5, 100.0),      # greater trochanter
    "knuckle_height":        (66.5, 75.5, 83.5),
    "knee_height":           (45.5, 54.5, 59.0),
    "ankle_height":          (6.5, 7.5, 8.5),
    "shoulder_breadth":      (33.0, 40.0, 43.5),        # biacromial
    "hip_breadth":           (31.5, 36.0, 39.5),
    "upper_arm":             (30.0, 36.5, 39.5),        # shoulder - elbow
    "forearm":               (22.5, 26.5, 28.5),        # elbow - wrist
    "hand_length":           (16.5, 19.0, 20.5),
    "foot_length":           (22.0, 26.5, 28.5),
    "head_length":           (17.5, 19.5, 20.5),
    "mass_kg":               (47.0, 78.0, 98.0),
}


@dataclass
class Worker:
    id: str
    label: str
    percentile: int
    sex: str
    dims: dict
    handedness: str = "right"
    ppe: tuple = ("vest", "safety_glasses", "gloves")
    walk_speed: float = 120.0                 # cm/s, normal plant pace
    strength: str = "average"
    qualified: tuple = field(default_factory=tuple)

    @property
    def stature(self):
        return self.dims["stature"]


def _w(idx, wid, label, pct, sex, **kw):
    return Worker(wid, label, pct, sex, {k: v[idx] for k, v in DIMS.items()}, **kw)


WORKERS = {
    "W05F": _w(0, "W05F", "Short female (5th percentile)", 5, "F", walk_speed=112.0, strength="lower"),
    "W50M": _w(1, "W50M", "Average male (50th percentile)", 50, "M"),
    "W95M": _w(2, "W95M", "Tall male (95th percentile)", 95, "M", walk_speed=128.0, strength="higher"),
}


def custom(wid, stature, sex="M", **over):
    """A worker of any stature: the 50th percentile proportions scaled to it (then any dims overridden)."""
    base = WORKERS["W50M" if sex == "M" else "W05F"]
    k = stature / base.dims["stature"]
    dims = {n: (v * k if n != "mass_kg" else v * k ** 3) for n, v in base.dims.items()}
    dims.update(over)
    return Worker(wid, f"Custom {stature:.0f} cm", 0, sex, dims)
