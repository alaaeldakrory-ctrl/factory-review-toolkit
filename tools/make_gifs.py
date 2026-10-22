"""Frame sequences (tools/kit_render.py "seq" shots) -> GIFs: one per sequence and a looping mosaic of all of them.

    python tools/make_gifs.py out/people/jobs/render [--aspect=1.5] [--width=480] [--mosaic=6] [--fps=12]
"""
import os
import sys

from PIL import Image, ImageDraw


def load(seq_dir, aspect, width):
    fr = []
    for n in sorted(os.listdir(seq_dir)):
        if not n.endswith(".png"):
            continue
        im = Image.open(os.path.join(seq_dir, n)).convert("RGB")
        w, h = im.size
        cw = min(w, int(h * aspect))
        im = im.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h))
        fr.append(im.resize((width, int(width / aspect)), Image.LANCZOS))
    return fr


def save(frames, path, fps):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


def main():
    root = sys.argv[1]
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[2:] if a.startswith("--")}
    aspect, width = float(opt.get("aspect", 1.5)), int(opt.get("width", 480))
    cols, fps = int(opt.get("mosaic", 6)), float(opt.get("fps", 12))
    out = os.path.join(root, "gif")
    os.makedirs(out, exist_ok=True)
    seqs = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and d != "gif":
            fr = load(p, aspect, width)
            if fr:
                seqs[d] = fr
                save(fr, os.path.join(out, d + ".gif"), fps)
                print(d, len(fr))
    if len(seqs) > 1:
        tw = int(opt.get("tile", 240))
        th = int(tw / aspect)
        rows = (len(seqs) + cols - 1) // cols
        n = max(len(f) for f in seqs.values())
        mos = []
        for k in range(n):
            canvas = Image.new("RGB", (cols * tw, rows * th), (24, 24, 26))
            dr = ImageDraw.Draw(canvas)
            for i, (name, fr) in enumerate(seqs.items()):
                im = fr[k % len(fr)].resize((tw, th), Image.LANCZOS)
                x, y = (i % cols) * tw, (i // cols) * th
                canvas.paste(im, (x, y))
                dr.text((x + 5, y + 4), name.replace("_", " "), fill=(240, 240, 240))
            mos.append(canvas)
        save(mos, os.path.join(out, "mosaic.gif"), fps)
        print("mosaic", n, "frames")


if __name__ == "__main__":
    main()
