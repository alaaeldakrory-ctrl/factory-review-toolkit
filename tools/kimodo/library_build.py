"""Motion-library clips from the job library, on the website's character, with Kimodo's body motion:

    1. each job solved on the character's own skeleton (rule-based: exact hands, tools, stance)   - CPU pool
    2. its work poses -> Kimodo keyframes, Kimodo generates the whole body (one GPU process)
    3. final pass: Kimodo's body, our hands; checks                                              - CPU pool
    4. the clip: GLB (character + animation), BVH, preview .bin, tags and checks (tools/clip_export.py)

    python tools/kimodo/library_build.py <site>/library [--only=ratchet,box] [--workers=10] [--character=<glb>]
"""
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "tools"), os.path.join(ROOT, "exts", "frt.core")]
SITE = os.environ.get("FRT_SITE", os.path.join(ROOT, "..", "frt-site"))
CHAR = os.path.join(SITE, "assets", "worker.glb")
OUT = os.path.join(ROOT, "out", "people", "library")
_CH = {}


def character(path=CHAR):
    if path not in _CH:
        from frt_core.people.mixamo import load
        _CH[path] = load(path)
    return _CH[path]


def src(jid, ch=None):
    """The job solved (rule-based) on the character, cached in out/people/library/<job>/rules.pkl."""
    ch = ch or character()
    out = os.path.join(OUT, jid)
    p = os.path.join(out, "rules.pkl")
    if os.path.isfile(p):
        d = pickle.load(open(p, "rb"))
    else:
        from frt_core.people import api
        r = api.solve_job(jid, worker=ch, step=False, cache=False)     # (its cache does not know code changes)
        d = {"frames": r.frames, "sols": r.sols}
        os.makedirs(out, exist_ok=True)
        pickle.dump(d, open(p, "wb"), protocol=4)
    return {"frames": d["frames"], "sols": d["sols"], "sk": ch.skeleton, "worker": ch.worker, "out": out}


def _rules(jid):
    t = time.time()
    try:
        src(jid)
        return jid, None, time.time() - t
    except Exception as e:
        import traceback
        return jid, f"{type(e).__name__}: {e} {traceback.format_exc()[-600:]}", time.time() - t


def _final(jid):
    import kimodo_job as KJ
    try:
        return KJ.finalize(jid, src(jid))
    except Exception as e:
        return {"job": jid, "error": f"{type(e).__name__}: {e}"}


def main():
    from frt_core.people.jobs import JOBS
    import kimodo_job as KJ
    opt = {a.split("=")[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    out = os.path.abspath([a for a in sys.argv[1:] if not a.startswith("--")][0])
    ids = [j.id for j in JOBS if "only" not in opt or j.id in opt["only"].split(",")]
    nw = int(opt.get("workers", 10))
    t0 = time.time()
    with mp.Pool(nw) as pool:                                          # 1. rule-based on the character
        for jid, err, dt in pool.imap_unordered(_rules, ids):
            print(f"[{time.time() - t0:5.0f} s] rules {jid}: {err or 'ok'} ({dt:.0f} s)", flush=True)
    ids = [j for j in ids if os.path.isfile(os.path.join(OUT, j, "rules.pkl"))]
    if "export" in sys.argv[1:] or "--export" in sys.argv:            # (only step 4, on what is there)
        export_all(ids, out)
        return
    if "nogen" not in opt:                                             # 2. Kimodo
        pairs = [KJ.prepare(j, src=src(j)) for j in ids]
        so, se = KJ.generate(pairs)
        print(f"[{time.time() - t0:5.0f} s] Kimodo generated {so.count('ok (')} of {len(ids)}", flush=True)
        if so.count("ok (") < len(ids):
            print(se[-2000:])
    with mp.Pool(nw) as pool:                                          # 3. final pass
        res = list(pool.imap_unordered(_final, ids))
    json.dump(res, open(os.path.join(OUT, "final.json"), "w"), indent=1)
    for r in sorted(res, key=lambda r: r["job"]):
        print(f"  {r['job']:14s} {r.get('kimodo', {}).get('counts') or r.get('error') or 'clean'}")
    export_all(ids, out)
    print(f"[{time.time() - t0:5.0f} s] done")


def export_all(ids, out):
    """4. clips: GLB / BVH / preview + the job folder the library preview plays (fixture, tool) + the on-character
    checks (tools/clip_audit.py) in each clip's entry."""
    from clip_audit import audit
    from clip_export import export_job_clip
    from web_export import export_job
    ch = character()
    for jid in ids:
        p = os.path.join(OUT, jid, "kimodo.pkl")
        if not os.path.isfile(p):
            continue
        d = pickle.load(open(p, "rb"))
        e = export_job_clip(jid, d["frames"], d["sols"], ch.skeleton, out)
        if e is None:
            continue
        export_job(jid, d["frames"], d["sols"], ch, os.path.join(out, "jobs", jid))
        e["scene"] = f"jobs/{jid}"
        e["audit"] = audit(d["frames"], d["sols"], ch.skeleton)
        json.dump(e, open(os.path.join(out, e["id"] + ".json"), "w"))
        print(f"  clip {jid}: {e['audit']['counts'] or 'clean'}", flush=True)
    from clip_export import rebuild_index
    from web_export import Retarget
    rebuild_index(out, Retarget(ch.skeleton))


if __name__ == "__main__":
    main()
