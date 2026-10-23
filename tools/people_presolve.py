"""Solve every library job on every body at its authored placement into the API cache (data/solved), so scripts
and agents get those results instantly (the offline bundle ships them).

    python tools/people_presolve.py [--workers=11] [--only=ratchet,torque] [--bodies=W05F,W50M,W95M]
    -> data/solved/*.pkl + data/solved/index.json (job, body, ok, stepped, counts)
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))


def run(task):
    jid, wid = task
    from frt_core.people import api
    t0 = time.time()
    try:
        s = api.solve_job(jid, worker=wid).summary()
    except Exception as e:
        s = {"job": jid, "worker": wid, "ok": False, "error": str(e)}
    s["solve_s"] = round(time.time() - t0, 1)
    return s


def main():
    from frt_core.people import api
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    only = opt["only"].split(",") if "only" in opt else None
    bodies = opt.get("bodies", "W05F,W50M,W95M").split(",")
    tasks = [(j["id"], b) for j in api.list_jobs() if not only or j["id"] in only for b in bodies]
    os.makedirs(api.CACHE, exist_ok=True)
    ip = os.path.join(api.CACHE, "index.json")
    index = json.load(open(ip)) if os.path.isfile(ip) else {}
    with mp.Pool(int(opt.get("workers", 11))) as pool:
        for i, s in enumerate(pool.imap_unordered(run, tasks)):
            index[f"{s['job']}/{s['worker']}"] = {k: s.get(k) for k in ("ok", "stepped_cm", "counts", "findings",
                                                                       "seconds", "solve_s", "error")}
            with open(ip, "w") as f:
                json.dump(index, f, indent=1)
            print(f"{i + 1:3d}/{len(tasks)} {s['job']:14s} {s['worker']} {'ok' if s['ok'] else 'FAIL'}"
                  f" {s.get('counts', s.get('error'))} {s['solve_s']} s", flush=True)
    n = sum(bool(v["ok"]) for v in index.values())
    print(f"{n}/{len(index)} clean")


if __name__ == "__main__":
    main()
