"""Hand lab: every grip drawn as a stick hand from three sides (top, front, thumb side), so finger and thumb poses
can be judged without a render. Also prints the thumb tip's distance to the index tip / middle phalanx.

    python tools/hand_lab.py [W50M] [r] -> out/people/hand_lab.png
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
import numpy as np   # noqa: E402
from PIL import Image, ImageDraw   # noqa: E402

from frt_core.people.kinematics import Pose, fk   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import PostureSolver   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402

GRIPS = ["relaxed", "open", "flat", "point", "pinch", "tripod", "lateral", "power:1.5", "power:2.4", "hook:1.5",
         "trigger:1.7:0.0", "trigger:1.7:1.0"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def main():
    wid = sys.argv[1] if len(sys.argv) > 1 else "W50M"
    s = sys.argv[2] if len(sys.argv) > 2 else "r"
    w = WORKERS[wid]
    sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    cell = 190
    views = [("top (palm down)", 0, 2), ("front", 0, 1), ("thumb side", 2, 1)]
    img = Image.new("RGB", (cell * len(views), cell * len(GRIPS)), (250, 250, 248))
    dr = ImageDraw.Draw(img)
    h = sk.world[f"hand_{s}"]
    for gi, g in enumerate(GRIPS):
        pose = Pose()
        S._grip(pose, s, g)
        R, P = fk(sk, pose)
        chains = []
        for f in FINGERS:
            pts = [P[f"hand_{s}"]] + [P[f"{f}_0{i}_{s}"] for i in (1, 2, 3)]
            # tip from the last bone's direction and rest length
            Rl = R[f"{f}_03_{s}"]
            pts.append(P[f"{f}_03_{s}"] + Rl @ (sk.tips[f"{f}_tip_{s}"] - sk.world[f"{f}_03_{s}"]))
            chains.append((f, [p - h for p in pts]))
        th = dict(chains)["thumb"][-1]
        ix = dict(chains)["index"]
        print(f"{g:16s} thumb tip -> index tip {np.linalg.norm(th - ix[-1]):5.1f} cm, -> index mid {np.linalg.norm(th - (ix[2] + ix[3]) / 2):5.1f} cm,"
              f" thumb tip {np.round(th, 1)}")
        for vi, (name, a, b) in enumerate(views):
            ox, oy = vi * cell, gi * cell
            dr.rectangle([ox, oy, ox + cell - 1, oy + cell - 1], outline=(200, 200, 200))
            dr.text((ox + 4, oy + 3), f"{g}  {name}", fill=(0, 0, 0))

            def to(p):
                sgn = -1 if a == 0 and s == "r" else 1
                return (ox + cell / 2 + sgn * p[a] * 6.5 + (-55 if a == 0 else 0) * (1 if s == "r" else -1) * -1,
                        oy + cell / 2 - p[b] * 6.5 * (1 if b == 1 else -1))
            for f, pts in chains:
                col = (220, 60, 40) if f == "thumb" else (40, 90, 200) if f == "index" else (80, 80, 80)
                q = [to(p) for p in pts]
                dr.line(q, fill=col, width=4)
                for c in q:
                    dr.ellipse([c[0] - 2.5, c[1] - 2.5, c[0] + 2.5, c[1] + 2.5], fill=col)
    out = os.path.join(HERE, "..", "out", "people", "hand_lab.png")
    img.save(out)
    print("->", out)


if __name__ == "__main__":
    main()
