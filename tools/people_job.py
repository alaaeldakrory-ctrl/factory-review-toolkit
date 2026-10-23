"""Command line for the people API (frt_core/people/api.py): list the job library, solve + check one job, write it
as a USD to reference into any scene. Prints JSON, so scripts and AI agents can read the result.

    python tools/people_job.py --list
    python tools/people_job.py ratchet --worker=W05F --dy=10 --out=C:/scenes/ops/ratchet.usd
    python tools/people_job.py ratchet --work-point=300,100,200 --facing=90 --out=...   # placed by its work point
    python tools/people_job.py ratchet ... --scene=C:/scenes/cell.usd --prim=/World/People/Op1 --at=3,2,0 --yaw=90
                                                                         --start=120     # and referenced into a scene

Options: --dx/--dy/--dz (cm, work moved left / up / away), --yaw-work / --tilt (deg, work turned / tipped back),
--no-step (keep the authored standing spot), --no-fixture (leave out the bench / panel: your scene has its own).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))


def main():
    from frt_core.people import api
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {}
    for a in sys.argv[1:]:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opt[k] = v
    if "list" in opt or not args:
        print(json.dumps(api.list_jobs(), indent=1))
        return
    jid = args[0]
    d = [float(opt.get(k, 0.0)) for k in ("dx", "dy", "dz")]
    placed = None
    if "work-point" in opt:
        wp = [float(v) for v in opt["work-point"].split(",")]
        at, dy = api.station_for(jid, wp, float(opt.get("facing", 0.0)))
        d[1] += dy
        placed = {"station_at": at, "dy_cm": round(dy, 1)}
    work = {"d": tuple(d), "yaw": float(opt.get("yaw-work", 0.0)), "tilt": float(opt.get("tilt", 0.0))}
    res = api.solve_job(jid, worker=opt.get("worker") or None, work=work, step="no-step" not in opt)
    out = dict(res.summary())
    if placed:
        out.update(placed)
    if "out" in opt:
        api.write_job_usd(res, opt["out"], fixture="no-fixture" not in opt)
        out["usd"] = opt["out"]
        if "scene" in opt:
            from pxr import Usd
            st = Usd.Stage.Open(opt["scene"])
            at = [float(v) for v in opt.get("at", "0,0,0").split(",")]
            out["scene"] = api.add_to_scene(st, opt.get("prim", f"/World/People/{jid}"), opt["out"], at=at,
                                            yaw=float(opt.get("yaw", 0.0)), start_frame=float(opt.get("start", 0)))
            st.Save()
    print(json.dumps(out, indent=1))
    sys.exit(0 if res.ok else 2)


if __name__ == "__main__":
    main()
