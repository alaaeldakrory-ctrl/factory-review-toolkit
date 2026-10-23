"""Score the original and the re-solved scene with the same Ergo Scan settings (RULA + the anatomy judge), on the
working frames only (walking and idle standing are the same in both), and write a side-by-side table.

    python tools/compare_report.py out/people/compare [--ergo=C:/Users/alaae/Dev/ergo-scan] [--step=2]
    -> <dir>/compare.md, compare.json, worst_moments.json (frames for side-by-side renders)
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")


def stats(op, frames):
    from ergo_scan.scan import _smooth, band
    rows = [r for r in op["rows"] if r["frame"] in frames]
    if not rows:
        return {}
    sm = _smooth([r["score"] for r in rows], r=3)
    imp = [r for r in rows if r.get("implausible")]
    why = {}
    for r in imp:
        for k in r["implausible"]:
            k2 = k[:-2] if k.endswith(("_l", "_r")) else k
            why[k2] = why.get(k2, 0) + 1
    return {"frames": len(rows), "rula_max": max(sm), "rula_p90": int(np.percentile(sm, 90)),
            "rula_median": int(np.median(sm)), "pct_5plus": round(100.0 * sum(s >= 5 for s in sm) / len(sm), 1),
            "pct_7": round(100.0 * sum(s >= 7 for s in sm) / len(sm), 1),
            "implausible_pct": round(100.0 * len(imp) / len(rows), 1),
            "implausible_top": dict(sorted(why.items(), key=lambda kv: -kv[1])[:5]),
            "trunk_p90": round(float(np.percentile([abs(r["trunk_deg"]) for r in rows], 90)), 1),
            "arm_p90": round(float(np.percentile([r["upper_arm_deg"] for r in rows], 90)), 1),
            "knee_p90": round(float(np.percentile([r["knee_deg"] for r in rows], 90)), 1),
            "band": band(max(sm))}


def main():
    d = os.path.abspath(sys.argv[1])
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[2:] if a.startswith("--") and "=" in a}
    sys.path.insert(0, os.path.join(opt.get("ergo", r"C:/Users/alaae/Dev/ergo-scan"), "exts", "ergo.scan"))
    from pxr import Usd

    from ergo_scan import scan as E
    summ = json.load(open(os.path.join(d, "summary.json")))
    step = float(opt.get("step", 2))
    res = {}
    for tag, path in (("before", summ["scene"]), ("after", os.path.join(d, "after.usda"))):
        st = Usd.Stage.Open(path)
        res[tag] = {o["name"]: o for o in E.scan(st, "/World", step=step, frames=tuple(summ["frames"]),
                                                 log=lambda *a: None)["operators"]}
    out, L, worst = {}, [], []
    L += ["# Apples to apples: original animation vs re-solved by our solver", "",
          f"Scene `{summ['scene']}`. The same characters, frames, hand contact points, feet and gaze. Only the body posture "
          "in the working frames differs. Scored by Ergo Scan with the same settings: RULA, plus the anatomy judge "
          "(joint positions only, so any rig is judged the same way). Working frames only: walking and idle standing "
          "are identical in both.", "",
          "| worker | version | RULA max | p90 | median | time at 5+ | time at 7 | physically impossible frames | main reasons | back p90 | arm p90 | knee p90 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if any("working" not in w for w in summ["workers"].values()):       # older summary: classify the frames again
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exts", "frt.core"))
        import resolve_compare as RC
        from frt_core.people.character import Character
        from frt_core.people.solver import PostureSolver
        st = Usd.Stage.Open(summ["scene"])
        times = list(np.arange(summ["frames"][0], summ["frames"][1] + 1e-6, 1.0))
        for name, skel_path, asset in RC.find_workers(st):
            ch = Character.load(asset)
            S = PostureSolver(ch.skeleton, ch.worker)
            recs = ch.read(st, st.GetPrimAtPath(skel_path), times)
            _, kinds, _, _ = RC.build_frames(ch, S, recs, summ["fps"])
            summ["workers"][name]["working"] = [t for t, k in zip(times, kinds) if k == "solve"]
    for name, w in summ["workers"].items():
        frames = set(round(f) for f in w["working"])
        frames = {float(f) for f in frames}
        out[name] = {}
        for tag in ("before", "after"):
            op = res[tag].get(name)
            s_ = stats(op, frames) if op else {}
            out[name][tag] = s_
            if s_:
                L.append(f"| {name} | {tag} | {s_['rula_max']} | {s_['rula_p90']} | {s_['rula_median']} | {s_['pct_5plus']}% | "
                         f"{s_['pct_7']}% | {s_['implausible_pct']}% | {', '.join(s_['implausible_top']) or '-'} | "
                         f"{s_['trunk_p90']}° | {s_['arm_p90']}° | {s_['knee_p90']}° |")
        # the original's worst moments (working frames): RULA 7 or impossible, at least 4 s apart
        op = res["before"][name]
        cand = sorted([r for r in op["rows"] if r["frame"] in frames],
                      key=lambda r: (-r["score"], -len(r.get("implausible") or {})))
        picks = []
        for r in cand:
            if all(abs(r["frame"] - p) > 4 * summ["fps"] for p in picks):
                picks.append(r["frame"])
            if len(picks) >= 4:
                break
        worst.append({"worker": name, "frames": picks})
    L += ["", "Caveats:",
          "- Re-solved with the feet where the original put them. The solver could not step closer, so problem 2 "
          "(standing in the wrong place) is only partly addressed.",
          "- The hands' contact points are copied. Where the original put a hand somewhere a person cannot reach "
          "comfortably, the re-solve is the best posture for that point, and the score is a real workplace finding.",
          "- The hand orientation is chosen by the solver, so a held part may be gripped differently."]
    with open(os.path.join(d, "compare.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(d, "compare.json"), "w"), indent=1)
    json.dump(worst, open(os.path.join(d, "worst_moments.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
