"""Frame sequences (tools/kit_render.py "seq" shots) -> H.264 MP4s (one per sequence), plus a grid MP4 of them all.

    python tools/make_mp4s.py out/people/jobs/render [--fps=12] [--grid=6] [--width=1280]
Uses the ffmpeg binary that ships with imageio-ffmpeg (no separate install).
"""
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(frames_dir, out, fps, width=None, pattern="f%03d.png"):
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2" if not width else f"scale={width}:-2"
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames_dir, pattern),
           "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)


def main():
    root = sys.argv[1]
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[2:] if a.startswith("--")}
    fps, cols = float(opt.get("fps", 12)), int(opt.get("grid", 6))
    width = int(opt["width"]) if "width" in opt else None
    out = os.path.join(root, "mp4")
    os.makedirs(out, exist_ok=True)
    seqs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and d not in ("gif", "mp4")
                  and any(f.endswith(".png") for f in os.listdir(os.path.join(root, d))))
    for d in seqs:
        encode(os.path.join(root, d), os.path.join(out, d + ".mp4"), fps, width)
        print(d + ".mp4")
    if len(seqs) > 1:                                     # grid of all sequences, looping the shorter ones
        lists = {d: sorted(f for f in os.listdir(os.path.join(root, d)) if f.endswith(".png")) for d in seqs}
        n = max(len(v) for v in lists.values())
        w0, h0 = Image.open(os.path.join(root, seqs[0], lists[seqs[0]][0])).size
        aspect = 1.5
        tw = 320
        th = int(tw / aspect) // 2 * 2
        rows = (len(seqs) + cols - 1) // cols
        with tempfile.TemporaryDirectory() as tmp:
            for k in range(n):
                canvas = Image.new("RGB", (cols * tw, rows * th), (24, 24, 26))
                dr = ImageDraw.Draw(canvas)
                for i, d in enumerate(seqs):
                    im = Image.open(os.path.join(root, d, lists[d][k % len(lists[d])])).convert("RGB")
                    w, h = im.size
                    cw = min(w, int(h * aspect))
                    im = im.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h)).resize((tw, th), Image.LANCZOS)
                    x, y = (i % cols) * tw, (i // cols) * th
                    canvas.paste(im, (x, y))
                    dr.text((x + 6, y + 4), d.replace("_", " "), fill=(240, 240, 240))
                canvas.save(os.path.join(tmp, f"g{k:04d}.png"))
            encode(tmp, os.path.join(out, "all_grid.mp4"), fps, None, pattern="g%04d.png")
        print("all_grid.mp4")


if __name__ == "__main__":
    main()
