"""Contact sheets from tools/kit/task_review.py shots: one row per moment (step, frame, phase), one column per view.

    python tools/review_sheet.py <shots folder> [--width=1800]
-> <shots folder>/sheet_<task>.jpg (several pages if long: sheet_<task>_2.jpg ...)
"""
import json
import os
import sys

from PIL import Image, ImageDraw


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a.split("=")[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    root = args[0]
    W = int(opt.get("width", 1800))
    shots = [s for s in json.load(open(os.path.join(root, "shots.json"))) if "png" in s and os.path.isfile(s["png"])]
    tasks = sorted({s["task"] for s in shots})
    for task in tasks:
        sh = [s for s in shots if s["task"] == task]
        views = list(dict.fromkeys(s["view"] for s in sh))
        rows = list(dict.fromkeys((s["step"], s["frame"], s["phase"]) for s in sh))
        cw = W // len(views)
        im0 = Image.open(sh[0]["png"])
        ch = int(cw * im0.height / im0.width)
        per = max(1, 9000 // (ch + 24))
        for page in range(0, len(rows), per):
            rr = rows[page:page + per]
            sheet = Image.new("RGB", (W, len(rr) * (ch + 24)), (24, 24, 24))
            d = ImageDraw.Draw(sheet)
            for i, (step, f, ph) in enumerate(rr):
                y = i * (ch + 24)
                d.text((6, y + 4), f"{step}   frame {f}   {ph}", fill=(230, 230, 230))
                for j, v in enumerate(views):
                    s = next((s for s in sh if (s["step"], s["frame"], s["view"]) == (step, f, v)), None)
                    if s:
                        sheet.paste(Image.open(s["png"]).convert("RGB").resize((cw, ch)), (j * cw, y + 24))
                        d.text((j * cw + 6, y + 28), v, fill=(255, 220, 120))
            n = page // per + 1
            fn = os.path.join(root, f"sheet_{task}{'' if n == 1 else f'_{n}'}.jpg")
            sheet.save(fn, quality=85)
            print(fn)


if __name__ == "__main__":
    main()
