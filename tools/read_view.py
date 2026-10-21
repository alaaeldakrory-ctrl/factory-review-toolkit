"""Recognition view: the drawing in grey with every object the reader found boxed in its type's colour, unknown
objects in red, stations in purple with their tasks. For reviewing what was understood before building anything.

    python tools/read_view.py layout.pdf out.png --profile lift_team --calib "1627,802.3;1821.5,802.3;40'"
        [--crop x0,y0,x1,y1]   (drawing cm)   [--px-per-m 12]
"""
import colorsys
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from PIL import Image, ImageDraw, ImageFont   # noqa: E402

FIXED = {"unknown": (225, 40, 40), "station": (150, 60, 200), "staged_kit": (0, 150, 220), "product_unit": (30, 170, 60),
         "column": (230, 140, 0), "wall": (60, 60, 60), "operator": (0, 120, 90), "area": (140, 140, 140)}


def colour(t):
    if t in FIXED:
        return FIXED[t]
    h = int(hashlib.md5(t.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.8)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(reading, out, crop=None, px_per_m=12.0, show_unknown=True, min_unknown_cm=25):
    dr = reading.drawing
    if crop is None:
        xs = [v for s in dr.segs for v in (s.a[0], s.b[0])] + [q[0] for p in dr.polys for q in p.pts]
        ys = [v for s in dr.segs for v in (s.a[1], s.b[1])] + [q[1] for p in dr.polys for q in p.pts]
        crop = (min(xs), min(ys), max(xs), max(ys))
    x0, y0, x1, y1 = crop
    k = px_per_m / 100.0
    W, H = int((x1 - x0) * k) + 1, int((y1 - y0) * k) + 1
    scale_down = 1.0
    if W * H > 60e6:                                   # keep the image reasonable
        scale_down = (60e6 / (W * H)) ** 0.5
        k *= scale_down
        W, H = int((x1 - x0) * k) + 1, int((y1 - y0) * k) + 1
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    P = lambda p: ((p[0] - x0) * k, (y1 - p[1]) * k)

    def inside(bb):
        return bb[2] >= x0 and bb[0] <= x1 and bb[3] >= y0 and bb[1] <= y1
    for s in dr.segs:
        if inside((min(s.a[0], s.b[0]), min(s.a[1], s.b[1]), max(s.a[0], s.b[0]), max(s.a[1], s.b[1]))):
            d.line([P(s.a), P(s.b)], fill=(200, 200, 200), width=1)
    for p in dr.polys:
        if inside(p.bbox()):
            d.line([P(q) for q in p.pts] + [P(p.pts[0])], fill=(185, 185, 185), width=1)
    f_small, f_big = _font(12), _font(15)
    for o, c in reading.items:
        if not inside(o.bbox):
            continue
        if c.type == "unknown" and (not show_unknown or max(o.size) < min_unknown_cm):
            continue
        if c.type in ("ignore",):
            continue
        col = colour(c.type)
        a, b = P((o.bbox[0], o.bbox[3])), P((o.bbox[2], o.bbox[1]))
        wdt = 1 if c.type in ("unknown", "area", "column") else 2
        d.rectangle([a, b], outline=col, width=wdt)
        if c.type == "unknown" and max(o.size) >= 40:
            d.text((a[0] + 1, a[1] + 1), f"#{o.id}", fill=(200, 60, 60), font=f_small)
        if c.type not in ("column", "unknown"):
            label = c.type.replace("_", " ")
            if c.params:
                label += " " + " ".join(str(v) for v in c.params.values())
            d.text((a[0] + 2, a[1] + 1), label, fill=col, font=f_small)
    for st in reading.stations:
        a, b = P((st.box[0], st.box[3])), P((st.box[2], st.box[1]))
        d.rectangle([a, b], outline=FIXED["station"], width=3)
        txt = st.name + ("\n" + "\n".join("· " + t for t in st.tasks[:4]) if st.tasks else "")
        d.multiline_text((a[0] + 3, b[1] + 3), txt, fill=FIXED["station"], font=f_big, spacing=2)
    # legend
    counts = reading.counts()
    lines = [f"{t.replace('_', ' ')}: {n}" for t, n in counts.items()] + [f"stations: {len(reading.stations)}"]
    lw = max(d.textlength(t, font=f_big) for t in lines) + 30
    d.rectangle([8, 8, 8 + lw, 14 + 20 * len(lines)], fill=(255, 255, 255), outline=(120, 120, 120))
    for i, t in enumerate(lines):
        name = t.split(":")[0].replace(" ", "_")
        d.rectangle([14, 15 + 20 * i, 26, 27 + 20 * i], fill=colour("station" if name == "stations" else name))
        d.text((32, 12 + 20 * i), t, fill=(30, 30, 30), font=f_big)
    img.save(out)
    return out


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
