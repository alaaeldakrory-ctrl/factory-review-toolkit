"""A crew of real characters (NVIDIA / Reallusion CC assets) doing library jobs side by side: each job is solved on
that character's own proportions (character.py), checked, and baked onto its rig. The result is one scene plus
render shots, and a scan with the independent ergonomics tool as a cross-check.

    python tools/crew_demo.py [--n=15] [--workers=11] [--cols=5] [--out=out/people/crew]
    -> <out>/crew.usd, shots.json, check.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")

CC = os.environ.get("FRT_CHARACTERS", r"C:/Nvidia_Assets/Characters_NVD/Assets/Characters/Reallusion")
CHARACTERS = [CC + "/ActorCore/Uniform_M_0001/uniform_m_0001.usd", CC + "/ActorCore/Uniform_F_0001/uniform_f_0001.usd",
              CC + "/Worker/Worker.usd", CC + "/Debra/Debra.usd", CC + "/ActorCore/Party_M_0001/party-m-0001.usd",
              CC + "/ActorCore/Business_F_0002/business-f-0002.usd"]
JOBS = ["ratchet", "connector", "hammer", "box", "overhead", "kneel_drive", "lug_torque", "battery", "inspect",
        "guide_hoist", "control_box", "scan", "torque", "cable_tie", "wheel", "two_wrench", "decal", "grease",
        "screwdriver", "circlip"]
DX, DZ = 330.0, 380.0


def run(task):
    i, jid, asset = task
    from frt_core.people import api
    from frt_core.people.character import load
    t0 = time.time()
    try:
        ch = load(asset)
        res = api.solve_job(jid, worker=ch)
        s = res.summary()
    except Exception as e:
        import traceback
        return {"i": i, "job": jid, "asset": asset, "error": f"{e}\n{traceback.format_exc()[-1200:]}"}
    s.update(i=i, asset=asset, solve_s=round(time.time() - t0, 1))
    return s


def main():
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    n = int(opt.get("n", 15))
    cols = int(opt.get("cols", 5))
    out = os.path.abspath(opt.get("out", os.path.join(ROOT, "out", "people", "crew")))
    os.makedirs(out, exist_ok=True)
    tasks = [(i, JOBS[i % len(JOBS)], CHARACTERS[i % len(CHARACTERS)]) for i in range(n)]
    t0 = time.time()
    with mp.Pool(int(opt.get("workers", 11))) as pool:
        rows = []
        for r in pool.imap_unordered(run, tasks):
            rows.append(r)
            print(f"{len(rows):2d}/{n} {r['job']:12s} {os.path.basename(r['asset']):22s} "
                  f"{'ERROR ' + r['error'][:200] if 'error' in r else ('ok' if r['ok'] else 'FAIL ' + str(r['counts']))}"
                  f" {r.get('solve_s', '')} s", flush=True)
    rows.sort(key=lambda r: r["i"])
    with open(os.path.join(out, "check.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"solved in {time.time() - t0:.0f} s; {sum(bool(r.get('ok')) for r in rows)}/{n} clean")
    # the scene: every result comes from the cache now
    from pxr import Gf, UsdGeom

    from frt_core import units as U
    from frt_core.catalog.details import sign
    from frt_core.geom import Shapes
    from frt_core.materials import MaterialLibrary
    from frt_core.people import api
    from frt_core.people.character import load
    from frt_core.people.motion_gen import FPS
    from frt_core.studio import dress
    from frt_core.usdio import safe_stage
    path = os.path.join(out, "crew.usd")
    shots, length = [], 0
    with safe_stage(path) as st:
        U.setup_stage(st, fps=FPS)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        for r in rows:
            if "error" in r:
                continue
            i = r["i"]
            ch = load(r["asset"])
            res = api.solve_job(r["job"], worker=ch)
            x, z = ((i % cols) - (cols - 1) / 2) * DX, (i // cols) * DZ
            base = f"/World/Asset/S{i:02d}_{r['job']}"
            UsdGeom.Xform.Define(st, base).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, z))
            end = api.bake_into(st, base, res, lib, fixture=True)
            s = Shapes()
            sign(s, (68.0, 150.0, 70.0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 34.0,
                 [res.job.title.upper(), os.path.splitext(os.path.basename(r["asset"]))[0]], header=f"{i + 1:02d}",
                 header_mat="powder_blue")
            s.build(st, f"{base}/Sign", lib.refs(list(s.groups)))
            length = max(length, end + 1)
            fx, fy, fz = res.job.focus
            # from the front-left, over the work, at the face and hands
            shots.append({"seq": f"{i + 1:02d}_{r['job']}", "eye": [x + fx - 140.0, 165.0, z + fz + 200.0],
                          "at": [x + fx * 0.5, max(80.0, fy) + 5.0, z + fz * 0.4], "mm": 30, "frames": [0, end, 1]})
            print(f"baked {i + 1:02d} {r['job']} on {ch.id}")
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    rws = (n + cols - 1) // cols
    shots.insert(0, {"name": "overview.png", "eye": [-700.0, 520.0, (rws - 1) * DZ + 900.0],
                     "at": [0.0, 80.0, (rws - 1) * DZ / 2], "mm": 24, "frame": 60, "settle": 220})
    with open(os.path.join(out, "shots.json"), "w") as f:
        json.dump(shots, f, indent=1)
    print("->", path, length, "frames")


if __name__ == "__main__":
    main()
