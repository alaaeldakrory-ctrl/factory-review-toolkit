"""Rules on a baked process: ergonomics (RULA per frame from the solved joint angles), reach, overhead work,
efficiency (walking, waiting, idle hands) and traffic. Findings are plain dicts (severity, worker, step, what, why,
fix) for the report and the Kit window.

RULA mapping from the solver's angles: upper arm = shoulder elevation, lower arm = elbow flexion, wrist = deviation
from the forearm, neck = neck flexion (+ twist), trunk = trunk flexion (+ side bend / twist), legs = 2 when a knee
is bent past 60 deg (squatting / lunging, not supported). Muscle use +1 for repeated cycles (ratchet, hammer, ...).
"""
from collections import defaultdict

from . import rula as R
from .motion_gen import FPS

CYCLES = ("ratchet", "hammer", "drive", "torque", "push")


def rula_frame(a, label="", side="r"):
    ua = R.score_upper_arm(a.get(f"shoulder_{side}", 0.0))
    la = R.score_lower_arm(a.get(f"elbow_{side}", 90.0))
    wr = R.score_wrist(a.get(f"wrist_{side}", 0.0) * 0.5)          # deviation is split flexion / side bend
    wa = R.table_a(ua, la, wr, 1)
    muscle = 1 if any(c in label for c in CYCLES) else 0
    neck = R.score_neck(a.get("neck_flexion", 0.0), twisted=abs(a.get("neck_twist", 0.0)) > 20)
    trunk = R.score_trunk(a.get("trunk_flexion", 0.0), twisted=abs(a.get("trunk_twist", 0.0)) > 10,
                          side_bent=abs(a.get("trunk_side", 0.0)) > 10)
    legs = R.score_legs(max(a.get("knee_l", 0.0), a.get("knee_r", 0.0)) < 60)
    nb = R.table_b(neck, trunk, legs)
    return R.table_c(wa + muscle, nb + muscle), {"upper_arm": ua, "lower_arm": la, "wrist": wr, "neck": neck,
                                                  "trunk": trunk, "legs": legs}


def niosh(H, V1, V2, mass_kg, freq_per_min=1.0, coupling="fair", A=0.0):
    """Revised NIOSH lifting equation (metric, Waters et al. 1993) at the lift origin: recommended weight limit and
    lifting index LI = load / RWL (LI > 1 raised risk, > 3 high). Frequency multiplier for <= 1 h, V < 75 cm."""
    H = max(25.0, H)
    HM = 0.0 if H > 63 else 25.0 / H
    VM = max(0.0, 1 - 0.003 * abs(V1 - 75.0)) if V1 <= 175 else 0.0
    D = abs(V2 - V1)
    DM = 1.0 if D < 25 else min(1.0, 0.82 + 4.5 / D)
    AM = max(0.0, 1 - 0.0032 * A)
    table = {0.2: 1.0, 0.5: 0.97, 1: 0.94, 2: 0.91, 3: 0.88, 4: 0.84, 5: 0.80, 6: 0.75, 8: 0.60, 10: 0.45}
    FM = table[min(table, key=lambda k: abs(k - freq_per_min))]
    CM = {"good": 1.0, "fair": 0.95 if V1 < 75 else 1.0, "poor": 0.90}[coupling]
    rwl = 23.0 * HM * VM * DM * AM * FM * CM
    return {"RWL": round(rwl, 1), "LI": round(mass_kg / rwl, 2) if rwl > 0 else 99.0,
            "HM": round(HM, 2), "VM": round(VM, 2), "DM": round(DM, 2)}


def evaluate(baked, plan, planner):
    """baked: worker -> bake() report. Returns (findings, summary)."""
    findings, summary = [], {}
    for wid, rep in baked.items():
        labels = rep["labels"]
        scores = []
        per_step = defaultdict(list)
        for i, (a, lab) in enumerate(zip(rep["angles"], labels)):
            work = ":" in lab and not lab.endswith((":walk", ":wait", ":give way")) and lab != "idle"
            if not work:
                scores.append(None)
                continue
            s, parts = rula_frame(a, lab)
            scores.append(s)
            per_step[lab.split(":")[0]].append((s, parts, a, i))
        for step, rows in per_step.items():
            hi = [r for r in rows if r[0] >= 5]
            t_hi = len(hi) / FPS
            worst = max(rows, key=lambda r: r[0])
            if worst[0] >= 5 and t_hi >= 1.0:
                parts = worst[1]
                bad = [k.replace("_", " ") for k, v in parts.items() if v >= 3]
                findings.append({"severity": "high" if worst[0] >= 6 else "medium", "worker": wid, "step": step,
                                 "what": f"RULA {worst[0]} for {t_hi:.1f} s", "why": "posture: " + (", ".join(bad) or "combined"),
                                 "fix": _fix(parts, worst[2]), "frame": worst[3]})
            over = [r for r in rows if r[2].get("shoulder_r", 0) > 90 or r[2].get("shoulder_l", 0) > 90]
            if len(over) / FPS >= 2.0:
                findings.append({"severity": "medium", "worker": wid, "step": step,
                                 "what": f"hands above shoulder for {len(over) / FPS:.1f} s", "why": "overhead work",
                                 "fix": "lower the work (stand, lift table) or raise the worker (step stand)"})
            neck = [r for r in rows if r[2].get("neck_flexion", 0) > 40]
            if len(neck) / FPS >= 3.0:
                findings.append({"severity": "low", "worker": wid, "step": step,
                                 "what": f"neck bent over 40 deg for {len(neck) / FPS:.1f} s", "why": "work too low "
                                 "for this worker's eye height", "fix": "raise or tilt the work towards the worker"})
        reach = [i for i, (r, lab) in enumerate(zip(rep.get("reach", []), labels))
                 if r > 0.5 and ":" in lab and not lab.endswith((":walk", ":carry", ":wait", ":give way"))]
        walk_f = sum(1 for lab in labels if lab.endswith(":walk"))
        wait_f = sum(1 for lab in labels if lab.endswith((":wait", ":give way")) or lab == "idle")
        tot = max(1, len(labels))
        valid = [s for s in scores if s is not None]
        summary[wid] = {"frames": tot, "seconds": round(tot / FPS, 1), "walking_s": round(walk_f / FPS, 1),
                        "waiting_s": round(wait_f / FPS, 1), "rula_max": max(valid) if valid else None,
                        "rula_mean": round(sum(valid) / len(valid), 2) if valid else None,
                        "rula_5plus_s": round(sum(1 for s in valid if s >= 5) / FPS, 1)}
        if walk_f / tot > 0.25:
            findings.append({"severity": "medium", "worker": wid, "step": "", "what": f"walking {100 * walk_f / tot:.0f} % "
                             "of the cycle", "why": "parts / tools far from the work", "fix": "bring the parts to the "
                             "point of use (flow rack lane at the station)"})
        if wait_f / tot > 0.20:
            findings.append({"severity": "low", "worker": wid, "step": "", "what": f"waiting {100 * wait_f / tot:.0f} % "
                             "of the cycle", "why": "unbalanced work between workers", "fix": "rebalance steps"})
        if reach:
            findings.append({"severity": "high", "worker": wid, "step": labels[reach[0]].split(":")[0],
                             "what": "target out of comfortable reach", "why": "the solver could not reach it without "
                             "large discomfort", "fix": "move the work closer or let the worker step in"})
    for sid, L in getattr(planner, "lifts", {}).items():          # NIOSH for every lift with a known mass
        r = niosh(L["H1"], L["V1"], L["V2"], L["mass_kg"])
        summary.setdefault("_lifts", {})[sid] = {**r, "mass_kg": L["mass_kg"], "H_cm": round(L["H1"]),
                                                 "V_cm": round(L["V1"])}
        if r["LI"] > 1.0:
            low, far = r["VM"] < 0.85, r["HM"] < 0.7
            why = ", ".join(x for x, c in (("lift starts low", low), ("load held far from the body", far)) if c)
            findings.append({"severity": "high" if r["LI"] > 2.0 else "medium", "worker": "", "step": sid,
                             "what": f"lifting index {r['LI']} ({L['mass_kg']:.0f} kg, limit {r['RWL']} kg here)",
                             "why": "NIOSH: " + (why or "lifting equation"),
                             "fix": "raise the pick height (pallet on a stand / lift table)" if low else
                                    "bring the load closer / use a lift assist"})
    for n in plan.notes:
        if n.startswith("traffic:") and "conflict" in n:
            findings.append({"severity": "high", "worker": "", "step": "", "what": n, "why": "paths cross",
                             "fix": "separate walkways or re-sequence"})
    return findings, summary


def _fix(parts, a):
    out = []
    if parts["trunk"] >= 3:
        if a.get("trunk_flexion", 0) > 20:
            out.append("raise the work (trunk bent %.0f deg)" % a.get("trunk_flexion", 0))
        else:
            out.append("face the work squarely (trunk twisted %.0f / side bent %.0f deg)"
                       % (abs(a.get("trunk_twist", 0)), abs(a.get("trunk_side", 0))))
    if parts["upper_arm"] >= 3:
        out.append("bring the work closer / lower (shoulder %.0f deg)" % max(a.get("shoulder_r", 0), a.get("shoulder_l", 0)))
    if parts["neck"] >= 3:
        if a.get("neck_flexion", 0) < 0:
            out.append("lower the work or bring it nearer (head tipped back %.0f deg)" % -a.get("neck_flexion", 0))
        else:
            out.append("tilt or raise the work (neck %.0f deg)" % a.get("neck_flexion", 0))
    if parts["legs"] >= 2:
        out.append("avoid squatting: work stand or lift table")
    if parts["wrist"] >= 3:
        out.append("change the tool angle / handle (wrist bent)")
    return "; ".join(out) or "review the workstation layout"
