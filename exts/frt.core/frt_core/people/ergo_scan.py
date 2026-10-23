"""Ergonomic scan of any animated scene: every operator (UsdSkel skeleton) found, RULA scored frame by frame from
the joint positions, and a short report (who, when, why, what to change).

Works on any rig whose joints follow a common naming (Unreal / frt canonical, Mixamo, Reallusion CC / NVIDIA
characters, SMPL; or a mapping you pass), any up axis and units. Scores come from frt_core.people.rula (RULA,
McAtamney & Corlett 1993). Angles from joint positions:
    trunk      pelvis -> chest against vertical (forward signed), side bend, twist (shoulder line vs hip line)
    neck       neck -> head against the trunk
    upper arm  shoulder -> elbow against the trunk's down axis (forward signed), abducted, shoulder raised
    lower arm  elbow flexion; hand across the midline / out to the side
    wrist      forearm vs hand (middle finger base) direction, relative to the rig's bind pose
    legs       knee bent past 60 deg = not supported / balanced
    walking    an operator walking empty-handed is not doing a task posture: capped at 2 (RULA rates tasks)
Loads / repeated work are not visible in a skeleton: pass them per operator (loads={op: kg}, repeated={op, ...}).

    from frt_core.people import ergo_scan
    res = ergo_scan.scan(stage, root="/World", step=2)
    ergo_scan.write_report(res, "C:/out/ergo")     # report .md/.json, frames.csv, dashboard.html, timeline / floor /
                                                   # score-panel PNGs, overlay.usda (markers to sublayer)

Caveat for every report: the scores describe the animation (the method as animated), not measured people. They
are good for comparing methods and finding bad steps.
"""
import csv
import json
import math
import os
import re

import numpy as np

from . import rula as R

BANDS = [(2, "acceptable"), (4, "investigate"), (6, "change soon"), (7, "change now")]
BAND_RGB = {"acceptable": (0.18, 0.67, 0.31), "investigate": (0.94, 0.78, 0.12), "change soon": (0.96, 0.47, 0.12),
            "change now": (0.84, 0.16, 0.16)}

# normalised joint names (lowercase, namespace and common prefixes removed, only letters / digits), by role.
# Order matters: the first name a skeleton has wins.
ALIASES = {
    "pelvis": ["pelvis", "hips", "hip"],
    "chest": ["spine03", "spine3", "spine2", "spine02", "chest", "upperchest", "spine1"],
    "neck": ["neck01", "necktwist01", "neck", "neck1"],
    "head": ["head"],
}
_SIDE = {
    "upperarm": ["upperarm{s}", "{s}upperarm", "{S}arm", "{S}upperarm", "upperarm{s}"],
    "forearm": ["lowerarm{s}", "{s}forearm", "{S}forearm", "{S}lowerarm", "{S}elbow", "forearm{s}"],
    "hand": ["hand{s}", "{s}hand", "{S}hand", "{S}wrist", "wrist{s}"],
    "mid": ["middle01{s}", "middle1{s}", "{s}mid1", "{S}handmiddle1", "middle{s}", "{s}middle1", "{S}middle1",
            "index01{s}", "{s}index1", "{S}handindex1", "{S}index1"],
    "thigh": ["thigh{s}", "{s}thigh", "{S}upleg", "{S}thigh", "{S}hip", "upperleg{s}", "{S}upperleg"],
    "calf": ["calf{s}", "{s}calf", "{S}leg", "{S}knee", "lowerleg{s}", "{S}lowerleg", "{S}shin"],
    "foot": ["foot{s}", "{s}foot", "{S}ankle", "{S}foot"],          # SMPL: the ankle, not the toe joint
}
for _role, _pats in _SIDE.items():
    for _s, _S in (("l", "left"), ("r", "right")):
        ALIASES[f"{_role}_{_s}"] = [p.format(s=_s, S=_S) for p in _pats]
REQUIRED = ["pelvis", "chest", "head", "upperarm_l", "upperarm_r", "forearm_l", "forearm_r", "hand_l", "hand_r",
            "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r"]
PART_LIMITS = {"trunk_s": (2, 3), "neck_s": (2, 3), "upper_arm_s": (2, 3), "legs_s": (1, 1)}   # green <=, yellow <=
GENERIC = {"body", "skelroot", "character", "skeleton", "root", "rig", "mesh", "geo", "model", "skel"}


def norm(name):
    n = name.split("/")[-1].split(":")[-1].lower()
    for pre in ("cc_base_", "mixamorig", "rl_", "bip01_", "bip01", "def_", "b_"):
        if n.startswith(pre):
            n = n[len(pre):]
    return re.sub(r"[^a-z0-9]", "", n)


def map_joints(joint_paths, override=None):
    """{role: joint index}. override: {role: joint name} for rigs with unusual names."""
    names = [norm(p) for p in joint_paths]
    out = {}
    for role, al in ALIASES.items():
        for a in al:
            if a in names:
                out[role] = names.index(a)
                break
    # SMPL: left_shoulder is the upper arm when there is a left_collar (Mixamo's LeftShoulder is the clavicle)
    for s, S in (("l", "left"), ("r", "right")):
        if f"{S}collar" in names and f"{S}shoulder" in names:
            out[f"upperarm_{s}"] = names.index(f"{S}shoulder")
    for role, jn in (override or {}).items():
        n = norm(jn)
        if n in names:
            out[role] = names.index(n)
    return out


def band(score):
    for hi, name in BANDS:
        if score <= hi:
            return name
    return BANDS[-1][1]


def _ang(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / na / nb)))))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


UP = np.array([0.0, 1.0, 0.0])


def frame_angles(J):
    """Body angles (deg) and flags from joint positions J (role -> Y-up cm)."""
    hip, ch, head = J["pelvis"], J["chest"], J["head"]
    neck = J.get("neck", ch + (head - ch) * 0.35)
    up_t = _unit(ch - hip)
    left = J["thigh_l"] - J["thigh_r"]
    left = _unit(np.array([left[0], 0.0, left[2]]))
    fwd = np.cross(left, UP)                                  # Y up, facing +Z -> left is +X
    a = {}
    a["trunk"] = _ang(up_t, UP) * (1 if np.dot(up_t, fwd) >= -0.05 else -1)
    a["trunk_side"] = abs(math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(up_t, left)))))))
    sh = J["upperarm_r"] - J["upperarm_l"]
    hp = J["thigh_r"] - J["thigh_l"]
    tw = abs(math.degrees(math.atan2(sh[0], sh[2]) - math.atan2(hp[0], hp[2])))
    a["trunk_twist"] = min(tw, 360 - tw)
    n = head - neck
    nf = _ang(n, up_t)
    a["neck"] = nf if np.dot(n - up_t * np.dot(n, up_t), fwd) >= 0 else -nf
    fwd_t = _unit(fwd - up_t * np.dot(fwd, up_t))
    left_t = _unit(np.cross(up_t, fwd_t))
    a["knee"] = max(180 - _ang(J[f"thigh_{s}"] - J[f"calf_{s}"], J[f"foot_{s}"] - J[f"calf_{s}"]) for s in "lr")
    for s in "lr":
        u = J[f"forearm_{s}"] - J[f"upperarm_{s}"]
        f = J[f"hand_{s}"] - J[f"forearm_{s}"]
        a[f"upper_arm_{s}"] = _ang(u, -up_t) * (1 if np.dot(u, fwd_t) >= 0 else -1)
        a[f"abducted_{s}"] = abs(float(np.dot(_unit(u), left_t))) > 0.62
        a[f"raised_{s}"] = J[f"hand_{s}"][1] > J[f"upperarm_{s}"][1] + 15.0
        a[f"elbow_{s}"] = 180 - _ang(J[f"upperarm_{s}"] - J[f"forearm_{s}"], J[f"hand_{s}"] - J[f"forearm_{s}"])
        side = float(np.dot(J[f"hand_{s}"] - hip, left_t))
        a[f"across_{s}"] = abs(side) > 45.0 or side * (1 if s == "l" else -1) < -8.0
        a[f"wrist_{s}"] = _ang(f, J[f"mid_{s}"] - J[f"hand_{s}"]) if f"mid_{s}" in J else 0.0
    return a


def score(a, rest, kg=0.0, repeated=False):
    """RULA for one frame (worst side); rest: the bind pose's neutral offsets (trunk / neck / wrists)."""
    trunk = max(0.0, abs(a["trunk"]) - rest["trunk"]) * (1 if a["trunk"] >= 0 else -1)
    neck = max(0.0, abs(a["neck"]) - rest["neck"]) * (1 if a["neck"] >= 0 else -1)
    if -8.0 < neck < 0.0:                                     # a few degrees back is neutral, not "in extension"
        neck = 0.0
    sides = {}
    for s in "lr":
        res = R.rula_from_angles(a[f"upper_arm_{s}"], a[f"elbow_{s}"], max(0.0, a[f"wrist_{s}"] - rest[f"wrist_{s}"]),
                                 neck, max(trunk, 0.0), mass_kg=kg, is_repeated=repeated,
                                 repeats_per_minute=10.0 if repeated else 0.0, shoulder_raised=a[f"raised_{s}"],
                                 upper_arm_abducted=a[f"abducted_{s}"], arm_across_midline=a[f"across_{s}"],
                                 trunk_twisted=a["trunk_twist"] > 15, trunk_side_bent=a["trunk_side"] > 12,
                                 legs_supported=a["knee"] < 60)
        sides[s] = res
    w = max(sides, key=lambda k: sides[k].grand_score)
    r = sides[w]
    return {"score": r.grand_score, "side": w, "trunk_s": r.trunk, "neck_s": r.neck, "legs_s": r.legs,
            "upper_arm_s": r.upper_arm, "lower_arm_s": r.lower_arm, "wrist_s": r.wrist,
            "trunk_deg": round(trunk, 1), "neck_deg": round(neck, 1), "knee_deg": round(a["knee"], 1),
            "upper_arm_deg": round(a[f"upper_arm_{w}"], 1), "elbow_deg": round(a[f"elbow_{w}"], 1),
            "trunk_twist_deg": round(a["trunk_twist"], 1), "trunk_side_deg": round(a["trunk_side"], 1)}


def driver(r):
    """The body part that pushes this frame's score up most, in plain words."""
    c = []
    if r["trunk_s"] >= 3:
        c.append((r["trunk_s"], f"trunk bent {r['trunk_deg']:.0f} deg"))
    if r["trunk_twist_deg"] > 15 or r["trunk_side_deg"] > 12:
        c.append((2.5, f"trunk twisted {r['trunk_twist_deg']:.0f} / side-bent {r['trunk_side_deg']:.0f} deg"))
    if r["neck_s"] >= 3:
        c.append((r["neck_s"], f"neck bent {r['neck_deg']:.0f} deg" if r["neck_deg"] >= 0 else
                  f"neck tipped back {-r['neck_deg']:.0f} deg"))
    if r["upper_arm_s"] >= 3:
        c.append((r["upper_arm_s"], f"{'left' if r['side'] == 'l' else 'right'} arm raised {r['upper_arm_deg']:.0f} deg"))
    if r["legs_s"] >= 2:
        c.append((2.6, f"deep knee bend {r['knee_deg']:.0f} deg (squat / kneel)"))
    if r["wrist_s"] >= 3:
        c.append((2.0, "wrist bent"))
    if not c:
        return "arm / wrist posture"
    return ", ".join(t for _, t in sorted(c, key=lambda x: -x[0])[:2])


FIX = {"trunk": "bring the work up / closer (lift table, tilt the part, step closer)",
       "neck": "raise or tilt the work / display so the head stays up",
       "arm": "lower the work or raise the operator (platform, step), bring the part closer",
       "knee": "raise the part (stand, turnover fixture) or give a creeper / stool",
       "twist": "turn the operator or the part instead of the body (rotate the fixture, move the bin in front)"}


def fix_hint(text):
    for k, key in (("trunk bent", "trunk"), ("twisted", "twist"), ("neck", "neck"), ("arm raised", "arm"),
                   ("knee", "knee")):
        if k in text:
            return FIX[key]
    return "check hand / wrist posture and tool grip"


def _to_cm_yup(stage):
    from pxr import UsdGeom
    k = (UsdGeom.GetStageMetersPerUnit(stage) or 0.01) * 100.0
    zup = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    if zup:
        return lambda p: np.array([p[0], p[2], -p[1]]) * k, k, True
    return lambda p: np.array([p[0], p[1], p[2]]) * k, k, False


def find_operators(stage, root="/World"):
    """[(name, SkelRoot prim, Skeleton prim)] for every skeleton under root (instance proxies included)."""
    from pxr import Usd, UsdSkel
    found = []
    start = stage.GetPrimAtPath(root) if root else stage.GetPseudoRoot()
    for p in Usd.PrimRange(start, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdSkel.Skeleton):
            continue
        sr = p
        while sr and not sr.IsA(UsdSkel.Root):
            sr = sr.GetParent()
        if not sr:
            continue
        name_prim = sr
        while name_prim.GetName().lower() in GENERIC and name_prim.GetParent() and name_prim.GetParent().GetPath() != start.GetPath():
            name_prim = name_prim.GetParent()
        found.append((name_prim, sr, p))
    # name each operator by the highest prim that holds only that one character (e.g. /World/Operator_07, not the
    # character asset's own root name); otherwise by the SkelRoot (or its parent when the SkelRoot name is generic)
    paths = [str(sr.GetPath()) for _, sr, _ in found]
    for k, (np_, sr, p) in enumerate(found):
        a = sr.GetParent()
        best = np_
        while a and a.GetPath() != start.GetPath() and str(a.GetPath()) != "/":
            pre = str(a.GetPath()) + "/"
            if sum(q.startswith(pre) or q == str(a.GetPath()) for q in paths) == 1:
                best = a
            a = a.GetParent()
        found[k] = (best, sr, p)
    names = [np_.GetName() for np_, _, _ in found]
    out, seen = [], set()
    for (np_, sr, p), name in zip(found, names):
        if names.count(name) > 1 and np_.GetParent():           # same name twice: say where (parent/name)
            name = f"{np_.GetParent().GetName()}/{name}"
        while name in seen:
            name = name + "_"
        seen.add(name)
        out.append((name, sr, p))
    return out


def scan(stage, root="/World", step=1, frames=None, loads=None, repeated=(), mapping=None, walk_cap=True,
         only=None, log=print):
    """Score every operator. frames: (start, end) in stage time codes (default: the stage's range).
    loads: {operator: kg} handled while working; repeated: operators doing repeated cycles (+1 muscle use);
    mapping: {role: joint name} (or {operator: {role: joint name}}) for unusual rigs."""
    from pxr import Usd, UsdGeom, UsdSkel
    conv, k, zup = _to_cm_yup(stage)
    t0, t1 = frames or (stage.GetStartTimeCode(), stage.GetEndTimeCode())
    if t1 <= t0:
        t1 = t0
    fps = stage.GetTimeCodesPerSecond() or 24.0
    times = list(np.arange(t0, t1 + 1e-6, step))
    loads, repeated = loads or {}, set(repeated or ())
    cache = UsdSkel.Cache()
    ops = []
    for name, sr, sk in find_operators(stage, root):
        if only and name not in only:
            continue
        cache.Populate(UsdSkel.Root(sr), Usd.TraverseInstanceProxies())
        q = cache.GetSkelQuery(UsdSkel.Skeleton(sk))
        joints = [str(j) for j in q.GetJointOrder()]
        ov = mapping.get(name, mapping) if mapping else None
        ov = ov if isinstance(ov, dict) and all(isinstance(v, str) for v in ov.values()) else None
        idx = map_joints(joints, ov)
        missing = [r for r in REQUIRED if r not in idx]
        op = {"name": name, "skelroot": str(sr.GetPath()), "skeleton": str(sk.GetPath()), "joints": len(joints),
              "animated": bool(q.GetAnimQuery()), "missing": missing, "rows": []}
        ops.append(op)
        if missing:
            log(f"{name}: skipped - joints not recognised: {missing} (pass mapping={{role: joint name}})")
            continue
        bind = q.GetJointWorldBindTransforms()
        Jb = {r: conv(np.array(bind[i].ExtractTranslation())) for r, i in idx.items()}
        try:
            ab = frame_angles(Jb)
            upright = abs(ab["trunk"]) < 20
        except Exception:
            ab, upright = None, False
        rest = {"trunk": abs(ab["trunk"]) if upright else 0.0, "neck": abs(ab["neck"]) if upright else 0.0,
                "wrist_l": ab["wrist_l"] if ab else 0.0, "wrist_r": ab["wrist_r"] if ab else 0.0}
        prev = []
        xc = UsdGeom.XformCache()
        for t in times:
            xc.SetTime(Usd.TimeCode(float(t)))
            W = q.ComputeJointWorldTransforms(xc)
            J = {r: conv(np.array(W[i].ExtractTranslation())) for r, i in idx.items()}
            a = frame_angles(J)
            row = score(a, rest, float(loads.get(name, 0.0)), name in repeated)
            prev.append(J["pelvis"])
            back = prev[-min(len(prev), max(2, int(round(0.25 * fps / step))))]
            dt = max(1e-6, (len(prev[-max(2, int(round(0.25 * fps / step))):]) - 1) * step / fps)
            v = np.hypot(*(J["pelvis"] - back)[[0, 2]]) / dt if len(prev) > 1 else 0.0
            row["walking"] = bool(v > 40.0)
            if walk_cap and row["walking"] and not loads.get(name):
                row["score"] = min(row["score"], 2)
            row["frame"] = float(t)
            row["t"] = round((t - t0) / fps, 3)
            row["pts"] = {r: [round(float(x), 4) for x in W[idx[r]].ExtractTranslation()]   # stage units / axes
                          for r in ("head", "chest", "upperarm_l", "upperarm_r", "calf_l", "calf_r", "pelvis")}
            row["head"] = row["pts"]["head"]
            row["floor_m"] = [round(float(J["pelvis"][0]) / 100.0, 3), round(float(J["pelvis"][2]) / 100.0, 3)]
            prev = prev[-32:]
            op["rows"].append(row)
        P = np.array([r["pts"]["pelvis"] + r["pts"]["head"] + r["pts"]["upperarm_r"] for r in op["rows"]])
        op["static"] = bool(len(P) > 1 and float(np.ptp(P, axis=0).max()) * k < 0.5)   # nothing moves in USD
        log(f"{name}: {len(op['rows'])} frames, max RULA {max(r['score'] for r in op['rows'])}"
            + ("  (STATIC: no animation stored in USD)" if op["static"] else ""))
    return {"stage": stage.GetRootLayer().identifier, "fps": fps, "step": step, "range": [t0, t1],
            "units_cm": k, "z_up": zup, "operators": ops}


def _smooth(v, r=5):
    if not v:
        return v
    pad = np.pad(np.array(v, float), r, mode="edge")
    return [int(np.median(pad[i:i + 2 * r + 1])) for i in range(len(v))]


def summarise(res, top=3):
    fps, step = res["fps"], res["step"]
    out = []
    for op in res["operators"]:
        rows = op["rows"]
        if not rows:
            out.append({"operator": op["name"], "skipped": op["missing"] or "no frames"})
            continue
        sm = _smooth([r["score"] for r in rows], r=max(1, int(round(0.25 * fps / step))))
        bands = {b: round(100.0 * sum(band(s) == b for s in sm) / len(sm), 1) for _, b in BANDS}
        # worst moments: peaks at least 2 s apart
        run, span = [1] * len(sm), [(0, 0)] * len(sm)          # the stretch at this level around each frame
        i = 0
        while i < len(sm):
            j = i
            while j + 1 < len(sm) and sm[j + 1] == sm[i]:
                j += 1
            for q in range(i, j + 1):
                run[q], span[q] = j - i + 1, (rows[i]["t"], rows[j]["t"])
            i = j + 1
        order = sorted(range(len(sm)), key=lambda i: (-sm[i], -run[i], -rows[i]["score"]))
        picks = []
        for i in order:
            if sm[i] < 5 and picks:
                break
            if all(rows[i]["t"] < span[j][0] - 2.0 or rows[i]["t"] > span[j][1] + 2.0 for j in picks):
                picks.append(i)
            if len(picks) >= top:
                break
        moments = []
        for i in picks:
            r = rows[i]
            j = i                                                  # how long it stays at that level
            while j + 1 < len(sm) and sm[j + 1] >= sm[i] - 0.5:
                j += 1
            h = i
            while h > 0 and sm[h - 1] >= sm[i] - 0.5:
                h -= 1
            why = driver(r)
            moments.append({"t": r["t"], "frame": r["frame"], "rula": sm[i], "band": band(sm[i]),
                            "lasts_s": round((rows[j]["t"] - rows[h]["t"]) + step / fps, 1), "why": why,
                            "fix": fix_hint(why), "angles": {k: r[k] for k in ("trunk_deg", "neck_deg", "upper_arm_deg",
                                                                               "elbow_deg", "knee_deg")}})
        drivers = {}
        for r, s in zip(rows, sm):
            if s >= 5:
                d = driver(r).split(",")[0].split(" ")[0:2]
                key = " ".join(d)
                drivers[key] = drivers.get(key, 0) + 1
        out.append({"operator": op["name"], "path": op["skelroot"], "frames": len(rows),
                    "seconds": round(len(rows) * step / fps, 1), "median": int(np.median(sm)),
                    "p90": int(np.percentile(sm, 90)), "max": max(sm), "band_pct": bands,
                    "walking_pct": round(100.0 * sum(r["walking"] for r in rows) / len(rows), 1),
                    "main_driver": max(drivers, key=drivers.get) if drivers else "-",
                    "worst": moments, "animated": op["animated"], "static": op.get("static", False)})
    return sorted(out, key=lambda s: (-(s.get("max") or 0), -(s.get("band_pct", {}).get("change now", 0)),
                                      -(s.get("p90") or 0)))


def _worst_frame(res, summ):
    for s_ in summ:
        if s_.get("worst"):
            return s_["worst"][0]["frame"]
    return res["range"][0]


def write_report(res, out_dir, title="Ergonomic scan (RULA)", overlay=True, dashboards=True):
    os.makedirs(out_dir, exist_ok=True)
    summ = summarise(res)
    with open(os.path.join(out_dir, "ergo_report.json"), "w") as f:
        json.dump({"stage": res["stage"], "fps": res["fps"], "step": res["step"], "range": res["range"],
                   "operators": summ}, f, indent=1)
    with open(os.path.join(out_dir, "ergo_frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["frame", "t", "score", "side", "trunk_deg", "neck_deg", "upper_arm_deg", "elbow_deg", "knee_deg",
                "trunk_twist_deg", "trunk_side_deg", "trunk_s", "neck_s", "upper_arm_s", "lower_arm_s", "wrist_s",
                "legs_s", "walking"]
        w.writerow(["operator"] + cols)
        for op in res["operators"]:
            for r in op["rows"]:
                w.writerow([op["name"]] + [r[c] for c in cols])
    L = [f"# {title}", "", f"Scene: `{res['stage']}`, frames {res['range'][0]:g}-{res['range'][1]:g} at "
         f"{res['fps']:g} fps, every {res['step']:g} frame(s). RULA 1-7: 1-2 acceptable, 3-4 investigate, 5-6 change "
         "soon, 7 change now. Walking empty-handed is not scored as a task (capped at 2).", "",
         "| operator | max | p90 | median | time at 5-6 | time at 7 | main driver | worst moment |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summ:
        if "skipped" in s:
            L.append(f"| {s['operator']} | - | - | - | - | - | skipped: {s['skipped']} | |")
            continue
        wm = s["worst"][0] if s["worst"] else None
        L.append(f"| {s['operator']} | **{s['max']}** | {s['p90']} | {s['median']} | {s['band_pct']['change soon']}% "
                 f"| {s['band_pct']['change now']}% | {s['main_driver']} | "
                 + (f"{wm['t']:.1f} s (frame {wm['frame']:g}): {wm['why']}" if wm else "") + " |")
    L += ["", "## Worst moments and what to change", ""]
    for s in summ:
        if "skipped" in s or not s["worst"] or s["max"] < 5:
            continue
        L.append(f"**{s['operator']}** (max {s['max']}, {s['band_pct']['change soon'] + s['band_pct']['change now']:.0f}% "
                 f"of the time at 5+)")
        for m in s["worst"]:
            L.append(f"- {m['t']:.1f} s, frame {m['frame']:g}, for {m['lasts_s']} s: RULA {m['rula']} ({m['band']}): "
                     f"{m['why']}. Fix: {m['fix']}.")
        L.append("")
    st_ = [s["operator"] for s in summ if s.get("static")]
    if st_:
        L += [f"**Not moving in the USD file** (animation not stored, e.g. driven by Animation Graph / behaviour "
              f"scripts; record or bake it first, these rows only score one pose): {', '.join(st_)}.", ""]
    ok = [s["operator"] for s in summ if "skipped" not in s and s["max"] <= 4]
    if ok:
        L += [f"No step above 4: {', '.join(ok)}.", ""]
    L += ["Caveat: scores describe the animation (the method as animated), not measured people. Loads and repeated "
          "work are only included where they were given. Use for comparing methods and finding bad steps."]
    with open(os.path.join(out_dir, "ergo_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    if dashboards:
        from . import ergo_dash as DB
        for name, fn in (("ergo_dashboard.html", lambda p: DB.dashboard_html(res, p, title)),
                         ("ergo_timeline.png", lambda p: DB.timeline_png(res, p)),
                         ("ergo_floor.png", lambda p: DB.floor_png(res, p)),
                         ("ergo_panel_worst.png", lambda p: DB.hud_png(res, _worst_frame(res, summ), p))):
            try:
                fn(os.path.join(out_dir, name))
            except Exception as e:                               # PIL missing etc.: the rest of the report stands
                print(f"{name} skipped: {e}")
    if overlay:
        _overlay(res, os.path.join(out_dir, "ergo_overlay.usda"))
    return summ


def _overlay(res, path):
    """A layer to sublayer into the scene: one marker sphere above each operator's head, coloured by the RULA band
    over time (stage units / axes)."""
    from pxr import Gf, Tf, Usd, UsdGeom, Vt
    st = Usd.Stage.CreateNew(path) if not os.path.exists(path) else None
    if st is None:
        os.remove(path)
        st = Usd.Stage.CreateNew(path)
    k = res["units_cm"]
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z if res["z_up"] else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(st, k / 100.0)
    st.SetTimeCodesPerSecond(res["fps"])
    st.SetStartTimeCode(res["range"][0])
    st.SetEndTimeCode(res["range"][1])
    base = "/World/ErgoScan"
    UsdGeom.Scope.Define(st, "/World") if not st.GetPrimAtPath("/World") else None
    st.OverridePrim("/World")
    UsdGeom.Xform.Define(st, base)
    up = Gf.Vec3d(0, 0, 1) if res["z_up"] else Gf.Vec3d(0, 1, 0)
    r = 6.0 / k                                               # 6 cm sphere, 28 cm above the head joint
    for op in res["operators"]:
        if not op["rows"]:
            continue
        sp = UsdGeom.Sphere.Define(st, f"{base}/{Tf.MakeValidIdentifier(op['name'])}")
        sp.CreateRadiusAttr(r)
        tr = sp.AddTranslateOp()
        col = sp.CreateDisplayColorAttr()
        sm = _smooth([x["score"] for x in op["rows"]], r=max(1, int(round(0.25 * res["fps"] / res["step"]))))
        last = None
        for row, s in zip(op["rows"], sm):
            tc = Usd.TimeCode(row["frame"])
            tr.Set(Gf.Vec3d(*row["head"]) + up * (28.0 / k), tc)
            b = band(s)
            if b != last:
                col.Set(Vt.Vec3fArray([Gf.Vec3f(*BAND_RGB[b])]), tc)
                last = b
        # body-part dots (back, shoulders, knees) coloured by their part score, as in the assembly reviews
        nm = Tf.MakeValidIdentifier(op["name"])
        for key, joints in (("trunk_s", ("chest",)), ("upper_arm_s", ("upperarm_l", "upperarm_r")),
                            ("legs_s", ("calf_l", "calf_r"))):
            for jn in joints:
                dot = UsdGeom.Sphere.Define(st, f"{base}/{nm}_parts/{jn}")
                dot.CreateRadiusAttr(4.0 / k)
                tr2, c2, lastc = dot.AddTranslateOp(), dot.CreateDisplayColorAttr(), None
                for row in op["rows"]:
                    tc = Usd.TimeCode(row["frame"])
                    tr2.Set(Gf.Vec3d(*row["pts"][jn]), tc)
                    g, y = PART_LIMITS[key]
                    c = (0.18, 0.67, 0.31) if row[key] <= g else (0.94, 0.78, 0.12) if row[key] <= y else (0.84, 0.16, 0.16)
                    if c != lastc:
                        c2.Set(Vt.Vec3fArray([Gf.Vec3f(*c)]), tc)
                        lastc = c
    st.GetRootLayer().Save()
