"""Factory Review Toolkit command line (no Kit needed).

    python tools/frt.py build    samples/recipes/demo_plant.toml      build a scene from a recipe (+ validate)
    python tools/frt.py drawing  plant.dxf|plant.pdf [legend.toml]     read a drawing and print what was recognised
    python tools/frt.py diff     old.dxf|old.plant.json new.dxf|.pdf     what changed between two drawing revisions
    python tools/frt.py layout   plant.dxf|.pdf|scene.plant.json       layout checks: aisles, exits, stations
    python tools/frt.py validate scene.usd [--auto-tag]                 check a scene, write <scene>.findings.json
    python tools/frt.py package  scene.usd out_dir                      portable folder with every dependency
    python tools/frt.py catalog                                         list catalog items and their parameters
    python tools/frt.py preview  scene.usd out.png [overview|plan]      quick offline picture
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
sys.path.insert(0, HERE)


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "build":
        from frt_core import recipe
        recipe.build(args[0])
    elif cmd == "drawing":
        from frt_core import drawing
        p = drawing.read_plant(args[0], args[1] if len(args) > 1 else None)
        print(f"{p.width / 100:.2f} x {p.depth / 100:.2f} m; grid x {[v / 100 for v in p.grid_x]} m, y {[v / 100 for v in p.grid_y]} m")
        for n in p.notes:
            print("  -", n)
        for d in p.doors:
            print(f"  door {d.kind:7s} {d.wall} wall at {d.pos / 100:6.2f} m, {d.width / 100:.2f} m wide")
        print(f"  {len(p.aisles)} aisles, {len(p.walkways)} walkways")
        for e in p.equipment:
            print(f"  {e.id:5s} {e.catalog:17s} at ({e.x / 100:6.2f}, {e.y / 100:6.2f}) m  {e.length:.0f} x {e.depth:.0f} cm  rot {e.rot:.0f}")
        for s in p.stations:
            print(f"  {s.name:10s} {(s.x1 - s.x0) / 100:.1f} x {(s.y1 - s.y0) / 100:.1f} m at ({s.x0 / 100:.2f}, {s.y0 / 100:.2f}) m, "
                  f"name {'inside' if s.label_inside else 'beside'}")
        for u in p.unknown:
            print("  ? unrecognised:", u)
    elif cmd == "calibrate":
        from frt_core.drawing.read import save_settings
        p0, p1, dist = args[1].split(";")
        cal = {"a": [float(v) for v in p0.split(",")], "b": [float(v) for v in p1.split(",")], "distance": dist}
        print("saved", save_settings(args[0], calibration=cal, profile=args[2] if len(args) > 2 else None))
    elif cmd in ("read", "teach"):
        from frt_core.drawing.read import read, save_settings
        a = list(args)

        def opt(name):
            if name in a:
                i = a.index(name)
                v = a[i + 1]
                del a[i:i + 2]
                return v
        prof, view, crop = opt("--profile"), opt("--view"), opt("--crop")
        unknown = "--unknown" in a
        a = [x for x in a if x != "--unknown"]
        if prof:
            save_settings(a[0], profile=prof)
        r = read(a[0], prof)
        if cmd == "teach":
            from frt_core.drawing.classify import teach_symbol
            oid, typ = int(a[1]), a[2]
            params = dict(kv.split("=", 1) for kv in a[3:])
            o = next(o for o in r.objects if o.id == oid)
            ex = f"{os.path.basename(a[0])} #{oid} at ({o.center[0] / 100:.1f}, {o.center[1] / 100:.1f}) m"
            e = teach_symbol(r.profile, o, typ, ex, params or None)
            same = [x.id for x in r.objects if x.id != oid and __import__("frt_core.drawing.classify", fromlist=["x"])._symbol_match(x, e) is not None]
            print(f"taught: #{oid} is a {typ} ({o.layer}, {max(o.size) / 100:.2f} x {min(o.size) / 100:.2f} m) -> "
                  f"{r.profile['path']}; {len(same)} more like it in this drawing: {same[:20]}")
            return 0
        print(" ; ".join(r.drawing.notes))
        print("profile:", " > ".join(r.profile["names"]))
        print("found:", ", ".join(f"{n} {t.replace('_', ' ')}" for t, n in r.counts().items()))
        for st in r.stations:
            print(f"  {st.name:10s} tasks: {'; '.join(st.tasks) or '-'}   (area from {st.box_from})")
        if unknown:
            unk = sorted(((o, c) for o, c in r.items if c.type == "unknown" and max(o.size) >= 40),
                         key=lambda oc: -oc[0].size[0] * oc[0].size[1])
            print(f"unknown objects over 40 cm: {len(unk)}")
            for o, c in unk[:60]:
                print(f"  #{o.id:<5d} {o.layer[:24]:24s} {max(o.size) / 100:6.2f} x {min(o.size) / 100:5.2f} m  "
                      f"{o.n:5d} strokes  at ({o.center[0] / 100:7.1f}, {o.center[1] / 100:7.1f})  {'; '.join(o.texts + o.near_texts)[:40]}")
        if view:
            import read_view
            read_view.render(r, view, tuple(map(float, crop.split(","))) if crop else None, 40)
            print("->", view)
    elif cmd in ("confirm", "score"):
        from frt_core.drawing import answers
        from frt_core.drawing.read import read
        if cmd == "confirm":
            print("answer key ->", answers.save_key(read(args[0])))
            return 0
        tot = []
        for path in args:
            sc = answers.score(read(path))
            tot.append(sc["f1"])
            print(f"{os.path.basename(path)}: F1 {sc['f1']:.3f} over {sc['items']} items; stations {sc['stations']}")
            for t, v in sorted(sc["types"].items()):
                if v["missed"] or v["extra"] or v["wrong"]:
                    print(f"   {t:18s} found {v['found']:3d}  missed {v['missed']:3d}  wrong {v['wrong']:3d}  extra {v['extra']:3d}")
        if len(tot) > 1:
            print(f"mean F1 {sum(tot) / len(tot):.3f}")
    elif cmd == "diff":
        from frt_core import drawing, revision
        old, new = [a if a.endswith(".json") else drawing.read_plant(a) for a in args[:2]]
        ch = revision.diff(old, new)
        print(f"{len(ch)} changes")
        for c in ch:
            print("  ", c)
    elif cmd == "layout":
        from frt_core import drawing, layout
        p = args[0] if args[0].endswith(".json") else drawing.read_plant(args[0])
        fl = layout.check(p)
        print(fl.summary())
        for f in fl:
            print(f"  [{f.severity}] {f.message}")
    elif cmd == "validate":
        from pxr import Usd
        from frt_core import tags, validate
        st = Usd.Stage.Open(args[0])
        if "--auto-tag" in args:
            print("auto-tagged", len(tags.auto_tag(st)))
        fl = validate.validate(st)
        out = os.path.splitext(args[0])[0] + ".findings.json"
        fl.save(out)
        print(fl.summary(), "->", out)
        for f in fl:
            print(f"  [{f.severity}] {f.message}")
    elif cmd == "package":
        from frt_core import package
        rep = package.package(args[0], args[1])
        print(f"{rep['files']} files, {rep['bytes'] / 1e6:.1f} MB, {rep['rewritten']} paths rewritten -> {rep['scene']}")
        for m in rep["missing"]:
            print("  missing in the source:", m)
        left = package.check(rep["scene"])
        print("  package check:", "all dependencies resolve" if not left else f"{len(left)} unresolved: {left[:5]}")
    elif cmd == "catalog":
        from frt_core import catalog
        for k, it in catalog.CATALOG.items():
            print(f"{k:18s} {it.title}  [{it.source}, role {it.role}]")
            for n, p in it.params.items():
                rng = f" {p.lo}..{p.hi} {p.unit}" if p.lo is not None else (f" one of {p.choices}" if p.choices else "")
                print(f"    {n:12s} = {p.default!r:16}{rng}  {p.doc}")
    elif cmd == "preview":
        import preview_scene
        preview_scene.main(args[0], args[1], args[2] if len(args) > 2 else "overview")
    else:
        print("unknown command", cmd)
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:   # noqa: BLE001  known, explained failures: one line, no traceback
        from frt_core.usdio import SwapFailed
        if isinstance(e, (SwapFailed, FileNotFoundError, KeyError, ValueError)):
            print("error:", e)
            sys.exit(1)
        raise
