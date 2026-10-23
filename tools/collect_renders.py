"""Collect per-job frame sequences from several render folders into one set numbered in library order, taking each
job from the first folder (newest first) that has it - so a partial re-render replaces only the jobs it covers.

    python tools/collect_renders.py out/people/jobs/final out/people/jobs/render_fix out/people/jobs/render_awp out/people/jobs/render
    python tools/make_mp4s.py out/people/jobs/final --fps=24 --grid=10
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))


def main():
    from frt_core.people.jobs import JOBS
    out, sources = sys.argv[1], sys.argv[2:]
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    index = {}
    for src in sources:
        for d in sorted(os.listdir(src)) if os.path.isdir(src) else []:
            p = os.path.join(src, d)
            if os.path.isdir(p) and "_" in d and d.split("_", 1)[0].isdigit():
                jid = d.split("_", 1)[1]
                index.setdefault(jid, p)                        # the first source listed wins
    missing = []
    for i, j in enumerate(JOBS):
        if j.id not in index:
            missing.append(j.id)
            continue
        shutil.copytree(index[j.id], os.path.join(out, f"{i + 1:02d}_{j.id}"))
        print(f"{i + 1:02d} {j.id:14s} <- {os.path.relpath(index[j.id], os.path.dirname(out))}")
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
