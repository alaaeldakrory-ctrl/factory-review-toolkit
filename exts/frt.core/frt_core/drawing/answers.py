"""Answer keys: what a drawing really contains, confirmed by a person, so every change to the reader (or a profile) can
be scored against all drawings already reviewed.

    save_key(reading)                  # after review: <drawing>.answers.json
    score(reading)                     # {type: {"found", "missed", "extra", "wrong"}} and an overall F1

Objects are matched by type and position (centre within max(60 cm, 30 % of the size)); stations by name, with their
tasks. Unknowns and ignored objects are not part of a key.
"""
import json
import math
import os

SKIP = ("unknown", "ignore")


def key_path(path):
    return path + ".answers.json"


def _entries(reading):
    out = []
    for o, c in reading.items:
        if c.type in SKIP:
            continue
        out.append({"type": c.type, "center": [round(o.center[0], 1), round(o.center[1], 1)],
                    "size": [round(max(o.size), 1), round(min(o.size), 1)], "params": c.params})
    sts = [{"name": s.name, "tasks": s.tasks, "box": list(s.box)} for s in reading.stations]
    return out, sts


def save_key(reading, path=None, corrections=None):
    path = path or key_path(reading.drawing.source)
    items, sts = _entries(reading)
    for fix in corrections or []:        # {"center": [x, y], "type": "..."}; type "none" removes an entry
        near = min(items, key=lambda e: math.dist(e["center"], fix["center"]), default=None)
        if near is not None and math.dist(near["center"], fix["center"]) < 100:
            if fix["type"] == "none":
                items.remove(near)
            else:
                near["type"] = fix["type"]
        elif fix["type"] != "none":
            items.append({"type": fix["type"], "center": fix["center"], "size": fix.get("size", [0, 0]), "params": {}})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"drawing": os.path.basename(reading.drawing.source), "items": items, "stations": sts}, fh, indent=1)
    return path


def load_key(path):
    with open(path if path.endswith(".json") else key_path(path), encoding="utf-8") as fh:
        return json.load(fh)


def score(reading, key=None):
    key = key or load_key(reading.drawing.source)
    got, sts = _entries(reading)
    want = key["items"]
    per = {}
    used = set()
    for w in want:
        t = per.setdefault(w["type"], {"found": 0, "missed": 0, "extra": 0, "wrong": 0})
        tol = max(60.0, 0.3 * max(w["size"]))
        best, bd = None, tol
        for i, g in enumerate(got):
            if i in used:
                continue
            d = math.dist(g["center"], w["center"])
            if d < bd:
                best, bd = i, d
        if best is None:
            t["missed"] += 1
        elif got[best]["type"] == w["type"]:
            t["found"] += 1
            used.add(best)
        else:
            t["wrong"] += 1
            used.add(best)
    for i, g in enumerate(got):
        if i not in used:
            per.setdefault(g["type"], {"found": 0, "missed": 0, "extra": 0, "wrong": 0})["extra"] += 1
    ks = {s["name"]: s for s in key.get("stations", [])}
    st = {"found": 0, "missed": 0, "extra": 0, "tasks_ok": 0}
    for s in sts:
        if s["name"] in ks:
            st["found"] += 1
            st["tasks_ok"] += int(s["tasks"] == ks[s["name"]]["tasks"])
        else:
            st["extra"] += 1
    st["missed"] = len(set(ks) - {s["name"] for s in sts})
    tp = sum(v["found"] for v in per.values())
    fp = sum(v["extra"] + v["wrong"] for v in per.values())
    fn = sum(v["missed"] + v["wrong"] for v in per.values())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"types": per, "stations": st, "f1": round(f1, 3), "items": len(want)}
