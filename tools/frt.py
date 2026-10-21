"""Factory Review Toolkit command line (no Kit needed).

    python tools/frt.py build    samples/recipes/demo_plant.toml      build a scene from a recipe (+ validate)
    python tools/frt.py drawing  plant.dxf|plant.pdf [legend.toml]     read a drawing and print what was recognised
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
        for u in p.unknown:
            print("  ? unrecognised:", u)
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
        print(f"{rep['files']} files, {rep['bytes'] / 1e6:.1f} MB -> {rep['scene']}")
        if rep["missing"]:
            print("  missing before packaging:", rep["missing"])
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
    sys.exit(main(sys.argv[1:]))
