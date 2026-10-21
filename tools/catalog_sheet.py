"""Render every catalog item (default parameters) to a contact sheet, offline.

    python tools/catalog_sheet.py out.png [item ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
sys.path.insert(0, HERE)
from PIL import Image, ImageDraw   # noqa: E402
from pxr import Usd   # noqa: E402
from frt_core import catalog as CAT, units as U   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
import preview as PV   # noqa: E402


def main(out, keys=None):
    keys = keys or list(CAT.CATALOG)
    ims = []
    for k in keys:
        st = Usd.Stage.CreateInMemory()
        U.setup_stage(st)
        lib = MaterialLibrary(st)
        prim, p = CAT.place(st, "/World", k, "Item", lib=lib)
        L, D = CAT.CATALOG[k].footprint(p)
        r = max(L, D, 120.0)
        eye = (r * 1.05, r * 0.85 + 60, r * 1.35)
        tmp = out + f".{k}.png"
        PV.render(PV.collect(st), tmp, eye, (0, 55, 0), size=(420, 320), fov=40)
        im = Image.open(tmp)
        ImageDraw.Draw(im).text((8, 6), f"{k}  {L:.0f} x {D:.0f} cm", fill=(20, 20, 20))
        ims.append(im)
        os.remove(tmp)
    cols = 4
    sheet = Image.new("RGB", (420 * cols, 320 * ((len(ims) + cols - 1) // cols)), (255, 255, 255))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * 420, (i // cols) * 320))
    sheet.save(out)
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or None)
