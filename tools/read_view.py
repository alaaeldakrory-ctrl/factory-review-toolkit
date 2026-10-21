"""Recognition view from the command line (see frt_core.drawing.view).

    python tools/read_view.py layout.pdf out.png [--profile lift_team] [--calib "x0,y0;x1,y1;40'"]
        [--crop x0,y0,x1,y1]   (drawing cm)   [--px-per-m 12]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from frt_core.drawing.view import render   # noqa: E402,F401


def main(argv):
    from frt_core.drawing.read import read
    a = list(argv)

    def opt(name, default=None):
        if name in a:
            i = a.index(name)
            v = a[i + 1]
            del a[i:i + 2]
            return v
        return default
    prof = opt("--profile", "base")
    calib = opt("--calib")
    crop = opt("--crop")
    ppm = float(opt("--px-per-m", 12))
    cal = None
    if calib:
        p0, p1, dist = calib.split(";")
        cal = (tuple(map(float, p0.split(","))), tuple(map(float, p1.split(","))), dist)
    r = read(a[0], prof, calibration=cal)
    render(r, a[1], tuple(map(float, crop.split(","))) if crop else None, ppm)
    print("->", a[1], r.counts(), f"{len(r.stations)} stations")


if __name__ == "__main__":
    main(sys.argv[1:])
