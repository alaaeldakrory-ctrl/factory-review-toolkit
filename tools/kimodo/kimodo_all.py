"""Kimodo across the whole job library: keyframes for every job, one Kimodo process generating them all (GPU),
final passes in parallel (CPU), and a pass-rate table against the rule-based versions.

    python tools/kimodo/kimodo_all.py [--workers=10] [--only=a,b] -> out/people/kimodo/all.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "exts", "frt.core"))


def _fin(jid):
    import kimodo_job as KJ
    try:
        return KJ.finalize(jid)
    except Exception as e:
        return {"job": jid, "error": f"{type(e).__name__}: {e}"}


def main():
    import kimodo_job as KJ
    from frt_core.people.jobs import JOBS
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--")}
    ids = [j.id for j in JOBS if "only" not in opt or j.id in opt["only"].split(",")]
    t0 = time.time()
    pairs = [KJ.prepare(j) for j in ids]
    if "nogen" in opt:                                    # re-run the final passes on existing Kimodo output
        so, se = "", ""
    else:
        so, se = KJ.generate(pairs)
    print("generated", so.count("ok ("), "of", len(ids), f"in {time.time() - t0:.0f} s")
    for ln in so.splitlines():
        if ln.startswith("failed"):
            print(ln)
    with mp.Pool(int(opt.get("workers", 10))) as pool:
        res = list(pool.imap_unordered(_fin, ids))
    out = os.path.join(HERE, "..", "..", "out", "people", "kimodo", f"all_{os.environ.get('KIMODO_PIN', 'work')}.json")
    json.dump(res, open(out, "w"), indent=1)
    ok_k = [r["job"] for r in res if r.get("kimodo", {}).get("ok")]
    print(f"\nKimodo-based: {len(ok_k)}/{len(res)} pass  ({time.time() - t0:.0f} s)")
    for r in sorted(res, key=lambda r: r["job"]):
        if not r.get("kimodo", {}).get("ok"):
            print(f"  {r['job']:14s} {r.get('kimodo', {}).get('counts') or r.get('error')}")


if __name__ == "__main__":
    main()
