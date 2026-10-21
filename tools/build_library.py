"""Portable asset library: every generated catalog item as a self-contained folder - the asset itself (to reference
into any scene), a studio scene (preview / turntable), a thumbnail and a turntable GIF - plus a browsable index page
and a manifest. Copy the library folder to any computer; nothing points outside it.

    python tools/build_library.py build  [out/library] [--only=key,key]   # USD assets + studio scenes + Kit job
    (Kit renders thumbnails and turntable frames: C:/OmniverseData/kit_jobs/frt_library_render.py)
    python tools/build_library.py index  [out/library]      # turntable GIFs, library.json, index.html

Items that need NVIDIA asset packs (forklift, crate, ...) are listed as not portable instead of being included.
"""
import html
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import Sdf, Usd, UsdGeom   # noqa: E402

from frt_core import catalog as CAT   # noqa: E402
from frt_core import units as U   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.studio import build_studio   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402

JOBS = r"C:\OmniverseData\kit_jobs"
TURN = 240                       # turntable length (frames at 24 fps)
GIF_STEP = 5                     # every 5th frame in the GIF
RIGGED = {"bridge_crane": "bridge travels, hoist traverses, hook lifts (frt_core.rig.crane)",
          "lift_table": "deck lifts, scissor and cylinder follow, carries what stands on it (frt_core.rig.lift_table)",
          "flow_rack": "each lane's load is its own Xform (Load/L1_B1_N1) for animation"}


def foreign_assets(path):
    """Asset paths in a saved USD that point outside its folder (absolute or ../)."""
    st = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    here = os.path.normcase(os.path.dirname(os.path.abspath(path)))
    bad = set()
    for prim in st.Traverse():
        for a in prim.GetAttributes():
            if a.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            v = a.Get()
            if v is None or not v.path or v.path.endswith(".mdl") and not os.path.isabs(v.path) and "/" not in v.path:
                continue                                    # core MDL modules (OmniPBR.mdl) ship with every renderer
            p = v.path
            if os.path.isabs(p) or p.startswith(".."):
                bad.add(p)
        for arc in prim.GetPrimStack():
            for ref in arc.referenceList.prependedItems:
                if ref.assetPath and (os.path.isabs(ref.assetPath) or ref.assetPath.startswith("..")):
                    bad.add(ref.assetPath)
            for pl in arc.payloadList.prependedItems:
                if pl.assetPath and (os.path.isabs(pl.assetPath) or pl.assetPath.startswith("..")):
                    bad.add(pl.assetPath)
    del st
    return sorted(bad)


def build_asset(out, key):
    """The item alone, default prim = the item (reference it anywhere); materials inside it."""
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        lib = MaterialLibrary(st, f"/{key}/Looks")
        prim, p = CAT.place(st, "", key, key, (0, 0, 0), 0.0, {}, lib, source="library")
        st.SetDefaultPrim(prim)
        Usd.ModelAPI(prim).SetKind("component")
    return p


def build(root, only=None):
    """Build every portable item (or only the given keys; the others keep their entries in library.json)."""
    os.makedirs(root, exist_ok=True)
    items, skipped = [], []
    old, prev_skipped = {}, []
    if only and os.path.isfile(os.path.join(root, "library.json")):
        with open(os.path.join(root, "library.json"), encoding="utf-8") as f:
            prev = json.load(f)
        old = {i["key"]: i for i in prev["items"]}
        prev_skipped = prev.get("not_included", [])
    for key, it in sorted(CAT.CATALOG.items()):
        if only and key not in only:
            if key in old:
                items.append(old[key])
            continue
        if it.source == "pack":
            skipped.append({"key": key, "title": it.title, "why": "built from NVIDIA asset packs (not portable)"})
            continue
        t0 = time.time()
        d = os.path.join(root, key)
        asset = os.path.join(d, f"{key}.usd")                      # asset + studio share one textures/ folder
        try:
            p = build_asset(asset, key)
        except Exception as e:                                   # one broken item never stops the library
            skipped.append({"key": key, "title": it.title, "why": f"build failed: {e}"})
            continue
        bad = foreign_assets(asset)
        if bad:
            skipped.append({"key": key, "title": it.title, "why": "refers outside the library: " + ", ".join(bad[:3])})
            continue
        studio = os.path.join(d, f"{key}_studio.usd")
        build_studio(studio, key, {}, turntable=TURN)
        items.append({"key": key, "title": it.title, "role": it.role,
                      "params": {k: {"default": v.default, "doc": v.doc, "lo": v.lo, "hi": v.hi, "unit": v.unit}
                                 for k, v in it.params.items()},
                      "asset": f"{key}/{key}.usd", "studio": f"{key}/{key}_studio.usd",
                      "rig": RIGGED.get(key, "")})
        print(f"{key}: {time.time() - t0:.1f} s")
    if only and old is not None:                                    # keep what was not rebuilt
        done = {i["key"] for i in items} | {x["key"] for x in skipped}
        skipped += [x for x in prev_skipped if x["key"] not in done]
    with open(os.path.join(root, "library.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 1, "items": items, "not_included": skipped}, f, indent=1, default=str)
    write_job(root, [i["key"] for i in items])
    for s in skipped:
        print(f"  not included: {s['key']} - {s['why']}")
    return items, skipped


def write_job(root, keys):
    root = os.path.abspath(root)
    job = f'''# Render the asset library: per item a square path-traced thumbnail and real-time turntable frames.
import os

import carb
import omni.kit.app
import omni.timeline as TL
import omni.usd
from pxr import UsdGeom

ROOT = r"{root}"
KEYS = {keys!r}
TURN, STEP = {TURN}, {GIF_STEP}


async def frames(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def run(log):
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    ctx = omni.usd.get_context()
    s = carb.settings.get_settings()
    for k in ("/app/viewport/grid/enabled",):                # session only - never persistent settings
        s.set(k, False)
    vp = get_active_viewport()
    old_res = vp.resolution
    try:
        vp.resolution = (1024, 1024)
    except Exception as e:
        log(f"square resolution not set: {{e}}")

    async def grab(path, frame, settle):
        TL.get_timeline_interface().set_current_time(frame / 24.0)
        await frames(settle)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cap = capture_viewport_to_file(vp, path)
        if hasattr(cap, "wait_for_result"):
            await cap.wait_for_result(completion_frames=20)
        return os.path.exists(path)
    try:
        for key in KEYS:
            d = os.path.join(ROOT, key)
            studio = os.path.join(d, key + "_studio.usd")
            thumb = os.path.join(d, "thumb_full.png")
            done = os.path.isfile(thumb) or os.path.isfile(os.path.join(d, "thumb.png"))
            newest = max(os.path.getmtime(p_) for p_ in (thumb, os.path.join(d, "thumb.png")) if os.path.isfile(p_)) if done else 0
            if done and newest > os.path.getmtime(studio):
                log(f"{{key}}: up to date")
                continue
            await ctx.new_stage_async()
            await frames(10)
            await ctx.open_stage_async(studio)
            await frames(300)
            vp.camera_path = "/World/Studio/Hero"
            # the viewport keeps the window's shape: widen the lens to it so the centre square holds the
            # studio's square framing (the index crops that square)
            w, h = vp.resolution
            cam = UsdGeom.Camera(ctx.get_stage().GetPrimAtPath("/World/Studio/Hero"))
            cam.GetVerticalApertureAttr().Set(24.0)
            cam.GetHorizontalApertureAttr().Set(24.0 * w / max(h, 1))
            s.set("/rtx/rendermode", "PathTracing")
            s.set("/rtx/pathtracing/spp", 4)
            s.set("/rtx/pathtracing/totalSpp", 192)
            s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
            ok = await grab(os.path.join(d, "thumb_full.png"), 30, 160)
            s.set("/rtx/rendermode", "RaytracedLighting")
            n = 0
            for k, f in enumerate(range(0, TURN, STEP)):
                n += await grab(os.path.join(d, "turn", f"f{{k:03d}}.png"), f, 8)
            log(f"{{key}}: thumb {{ok}}, turntable {{n}}")
    finally:
        s.set("/rtx/rendermode", "RaytracedLighting")
        vp.camera_path = "/OmniverseKit_Persp"
        try:
            vp.resolution = old_res
        except Exception:
            pass
        TL.get_timeline_interface().set_current_time(0.0)
        await ctx.new_stage_async()
    log("done")
'''
    if os.path.isdir(JOBS):
        with open(os.path.join(JOBS, "frt_library_render.py"), "w", encoding="utf-8") as f:
            f.write(job)
        print("render job queued in Kit:", os.path.join(JOBS, "frt_library_render.py"))


# ---------------------------------------------------------------------------------------------- index
def square(im):
    """Centre square of a capture (the studio frames the item in it)."""
    w, h = im.size
    k = min(w, h)
    return im.crop(((w - k) // 2, (h - k) // 2, (w - k) // 2 + k, (h - k) // 2 + k))


def make_gif(d):
    from PIL import Image
    fs = sorted(f for f in os.listdir(os.path.join(d, "turn")) if f.endswith(".png")) if os.path.isdir(
        os.path.join(d, "turn")) else []
    if not fs:
        return None
    ims = [square(Image.open(os.path.join(d, "turn", f)).convert("RGB")).resize((480, 480), Image.LANCZOS) for f in fs]
    pal = [i.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for i in ims]
    out = os.path.join(d, "turntable.gif")
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=GIF_STEP * 1000 // 24, loop=0, optimize=True)
    import shutil
    shutil.rmtree(os.path.join(d, "turn"), ignore_errors=True)        # the GIF replaces the raw frames
    return out


def _size(path):
    total = 0
    for dp, _, fs in os.walk(path):
        total += sum(os.path.getsize(os.path.join(dp, f)) for f in fs)
    return total


CSS = """
:root { --bg:#141518; --card:#1d1f23; --line:#2c2f35; --text:#e8e8e6; --dim:#9a9ca1; --accent:#e3a700; }
:root[data-theme="light"] { --bg:#f3f3f1; --card:#ffffff; --line:#dcdcd8; --text:#1d1f23; --dim:#5f6166; --accent:#a87b00; }
@media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) { --bg:#f3f3f1; --card:#ffffff; --line:#dcdcd8;
  --text:#1d1f23; --dim:#5f6166; --accent:#a87b00; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
header { padding:28px 16px 8px; max-width:1240px; margin:0 auto; }
h1 { margin:0 0 4px; font-size:24px; letter-spacing:.2px; }
header p { margin:0; color:var(--dim); }
input { margin:16px 0 4px; width:100%; max-width:420px; padding:9px 12px; border-radius:8px; border:1px solid var(--line);
  background:var(--card); color:var(--text); font-size:15px; }
main { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:16px; padding:16px;
  max-width:1240px; margin:0 auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; display:flex;
  flex-direction:column; }
.card img { width:100%; aspect-ratio:1/1; object-fit:cover; background:#202226; display:block; }
.card .body { padding:12px 14px 14px; }
.card h2 { margin:0; font-size:16px; }
.key { font-family:ui-monospace, Consolas, monospace; color:var(--accent); font-size:13px; }
.tag { display:inline-block; margin:6px 6px 0 0; padding:2px 8px; border-radius:99px; border:1px solid var(--line);
  color:var(--dim); font-size:12px; }
.rig { color:var(--dim); font-size:13px; margin-top:6px; }
details { margin-top:10px; }
summary { cursor:pointer; color:var(--dim); font-size:13px; }
table { width:100%; border-collapse:collapse; margin-top:6px; font-size:12.5px; }
td { border-top:1px solid var(--line); padding:4px 4px; vertical-align:top; }
td:first-child { font-family:ui-monospace, Consolas, monospace; white-space:nowrap; }
.links { margin-top:10px; font-size:13px; }
.links a { color:var(--accent); margin-right:12px; }
.skipped { max-width:1240px; margin:8px auto 40px; padding:0 16px; color:var(--dim); font-size:13px; }
"""


def index(root):
    with open(os.path.join(root, "library.json"), encoding="utf-8") as f:
        lib = json.load(f)
    cards = []
    for it in lib["items"]:
        d = os.path.join(root, it["key"])
        gif = make_gif(d) or (os.path.join(d, "turntable.gif") if os.path.isfile(os.path.join(d, "turntable.gif")) else None)
        if os.path.isfile(os.path.join(d, "thumb_full.png")):
            from PIL import Image
            square(Image.open(os.path.join(d, "thumb_full.png")).convert("RGB")).save(os.path.join(d, "thumb.png"))
            os.remove(os.path.join(d, "thumb_full.png"))
        it["thumb"] = f"{it['key']}/thumb.png" if os.path.isfile(os.path.join(d, "thumb.png")) else ""
        it["turntable"] = f"{it['key']}/turntable.gif" if gif else ""
        it["size_mb"] = round(_size(d) / 1e6, 1)
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v['default']))}"
                       f"{html.escape(' ' + v['unit']) if v.get('unit') else ''}</td><td>{html.escape(v['doc'])}</td></tr>"
                       for k, v in it["params"].items())
        img = it["thumb"] or it["turntable"]
        hover = f' onmouseover="this.src=\'{it["turntable"]}\'" onmouseout="this.src=\'{it["thumb"]}\'"' \
            if it["thumb"] and it["turntable"] else ""
        cards.append(f'''<article class="card" data-k="{html.escape(it['key'] + ' ' + it['title'] + ' ' + it['role']).lower()}">
<img src="{img}" alt="{html.escape(it['title'])}" loading="lazy"{hover}>
<div class="body"><h2>{html.escape(it['title'])}</h2><div class="key">{it['key']}</div>
<span class="tag">{html.escape(it['role'])}</span><span class="tag">{len(it['params'])} parameters</span>
<span class="tag">{it['size_mb']} MB</span>{'<span class="tag">animated</span>' if it['rig'] else ''}
{f'<div class="rig">{html.escape(it["rig"])}</div>' if it['rig'] else ''}
<details><summary>Parameters</summary><table>{rows}</table></details>
<div class="links"><a href="{it['asset']}">asset .usd</a><a href="{it['studio']}">studio scene</a>
{f'<a href="{it["turntable"]}">turntable</a>' if it['turntable'] else ''}</div></div></article>''')
    skipped = "".join(f"<li><b>{html.escape(s['key'])}</b> - {html.escape(s['why'])}</li>" for s in lib["not_included"])
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Factory Asset Library</title>
<style>{CSS}</style></head><body>
<header><h1>Factory Asset Library</h1>
<p>{len(lib['items'])} parametric assets from the Factory Review Toolkit. Each folder is self-contained: reference
<code>&lt;key&gt;/&lt;key&gt;.usd</code> into any scene; textures and materials travel with it. Hover a card to turn it.</p>
<input id="q" placeholder="Filter (crane, rack, table...)" oninput="f(this.value)"></header>
<main id="grid">{''.join(cards)}</main>
<div class="skipped">{f'Not included (need NVIDIA asset packs or failed): <ul>{skipped}</ul>' if skipped else ''}</div>
<script>function f(q){{q=q.toLowerCase();document.querySelectorAll('.card').forEach(c=>c.style.display=
c.dataset.k.includes(q)?'':'none')}}</script></body></html>'''
    with open(os.path.join(root, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    with open(os.path.join(root, "library.json"), "w", encoding="utf-8") as f:
        json.dump(lib, f, indent=1, default=str)
    print("->", os.path.join(root, "index.html"), f"({len(lib['items'])} items, {_size(root) / 1e6:.0f} MB)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    args = [a for a in sys.argv[2:] if not a.startswith("--only=")]
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--only=")), None)
    root = args[0] if args else os.path.join(HERE, "..", "out", "library")
    if cmd == "build":
        build(root, only)
    else:
        index(root)
