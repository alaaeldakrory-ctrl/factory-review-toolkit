"""Ergonomic scan (RULA) of every animated operator in a USD scene -> a short report (frt_core/people/ergo_scan.py).

    python tools/ergo_scan.py C:/scenes/line.usd --out=C:/scenes/line_ergo [--root=/World] [--step=2]
        [--range=0,1800] [--loads=Operator_03:12,Operator_07:5] [--repeated=Operator_02,Operator_09]
        [--mapping=mapping.json] [--only=Operator_01,Operator_02] [--list]
        [--frames-dir=C:/renders/line]      # rendered movie frames (number in the file name): adds the score panel

--list only prints the operators found and how their joints were recognised.
Writes ergo_report.md / ergo_report.json / ergo_frames.csv / ergo_timeline.png / ergo_overlay.usda (sublayer it into
the scene to see a marker above each operator coloured by risk) and prints the summary as JSON.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))


def main():
    from pxr import Usd, UsdSkel

    from frt_core.people import ergo_scan as E
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {}
    for a in sys.argv[1:]:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opt[k] = v
    if not args:
        print(__doc__)
        sys.exit(1)
    st = Usd.Stage.Open(args[0])
    root = opt.get("root", "/World")
    if not st.GetPrimAtPath(root):
        root = str(st.GetDefaultPrim().GetPath()) if st.GetDefaultPrim() else "/"
    mapping = json.load(open(opt["mapping"])) if opt.get("mapping") else None
    if "list" in opt:
        rows = []
        cache = UsdSkel.Cache()
        for name, sr, sk in E.find_operators(st, root):
            cache.Populate(UsdSkel.Root(sr), Usd.TraverseInstanceProxies())
            q = cache.GetSkelQuery(UsdSkel.Skeleton(sk))
            joints = [str(j) for j in q.GetJointOrder()]
            idx = E.map_joints(joints, mapping.get(name, mapping) if isinstance(mapping, dict) else None)
            rows.append({"operator": name, "skelroot": str(sr.GetPath()), "joints": len(joints),
                         "animated": bool(q.GetAnimQuery()),
                         "recognised": {r: joints[i].split("/")[-1] for r, i in sorted(idx.items())},
                         "missing": [r for r in E.REQUIRED if r not in idx]})
        print(json.dumps(rows, indent=1))
        return
    rng = tuple(float(v) for v in opt["range"].split(",")) if opt.get("range") else None
    loads = {k: float(v) for k, v in (p.split(":") for p in opt["loads"].split(","))} if opt.get("loads") else None
    rep = opt["repeated"].split(",") if opt.get("repeated") else ()
    only = opt["only"].split(",") if opt.get("only") else None
    res = E.scan(st, root, step=float(opt.get("step", 2)), frames=rng, loads=loads, repeated=rep, mapping=mapping,
                 only=only)
    out = opt.get("out") or os.path.splitext(args[0])[0] + "_ergo"
    summ = E.write_report(res, out)
    if opt.get("frames-dir"):
        from frt_core.people import ergo_dash as DB
        n = DB.composite_frames(res, opt["frames-dir"], os.path.join(out, "frames_with_panel"))
        print(f"{n} frames with the score panel -> {os.path.join(out, 'frames_with_panel')}")
    print(json.dumps({"out": out, "operators": [{k: s.get(k) for k in ("operator", "max", "p90", "median",
                                                                        "main_driver", "skipped")} for s in summ]},
                     indent=1))


if __name__ == "__main__":
    main()
