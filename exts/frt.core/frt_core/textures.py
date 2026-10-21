"""Generated surface textures: tileable albedo variation, roughness and normal maps per finish class, at real scale
(the meshes carry UVs in metres, so a map covers `tile` metres). Generic presets are uniform colour; real finishes
are not - powder coat has orange peel and mottling, hot-dip galvanising has spangle crystals, steel is brushed,
cast aluminium is pitted, plastic is grained, rubber collects dust, concrete has aggregate and stains, and every
painted surface has smudges and wiped, glossier patches.

    maps = texture_set("powder", roughness=0.5)   # {"albedo": path, "roughness": path, "normal": path, "mean": m, "tile": t}

Maps are generated once (numpy, seeded, so every computer makes identical files) into a cache, ~/.frt/textures
(or FRT_TEXTURES). Materials refer to them relative to the scene ("./textures/powder/powder_r050_albedo.png");
when a scene is saved, localize() copies the maps it uses next to it, so a scene folder is self-contained and
opens on any computer. Albedo maps are grey multipliers: the material tints them with the finish colour / mean.
"""
import json
import os
import re
import shutil
import struct
import zlib

import numpy as np

VERSION = 1
N = 1024                                       # map resolution (tileable)
ROOT = os.environ.get("FRT_TEXTURES", os.path.join(os.path.expanduser("~"), ".frt", "textures", f"v{VERSION}"))

# finish class -> tile size (m) and seed
KINDS = {
    "powder": (0.5, 11), "satin": (1.0, 12), "galv": (0.6, 13), "steel": (0.4, 14), "polished": (0.4, 15),
    "cast": (0.3, 16), "plastic": (0.3, 17), "rubber": (0.4, 18), "concrete": (2.0, 19), "cardboard": (0.5, 20),
}


# ---------------------------------------------------------------------------------------------- noise
def _freqs(n):
    f = np.fft.fftfreq(n) * n                  # cycles per tile
    return np.meshgrid(f, f, indexing="xy")


def spectral(seed, lo, hi, beta=1.0, aniso=1.0, n=N):
    """Tileable band-limited noise (lo..hi cycles per tile, 1/f^beta), zero mean, unit std. aniso > 1 stretches
    the features along x (brushed / fibre streaks)."""
    rng = np.random.default_rng(seed)
    fx, fy = _freqs(n)
    f = np.sqrt((fx / aniso) ** 2 + (fy * 1.0) ** 2)
    amp = np.where((f >= lo) & (f <= hi), 1.0 / np.maximum(f, 1e-6) ** (beta / 2 + 0.5), 0.0)
    w = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    out = np.real(np.fft.ifft2(w * amp))
    s = out.std() or 1.0
    return (out - out.mean()) / s


def specks(seed, density, size, n=N):
    """Sparse soft specks (dust, aggregate): values 0..1."""
    rng = np.random.default_rng(seed)
    base = (rng.random((n, n)) < density).astype(float)
    fx, fy = _freqs(n)
    g = np.exp(-(fx ** 2 + fy ** 2) * (size / n) ** 2 * 20.0)       # gaussian blur in frequency space
    out = np.real(np.fft.ifft2(np.fft.fft2(base) * g))
    return np.clip(out / (out.max() or 1.0), 0, 1)


def cells(seed, count, n=N, res=256):
    """Tileable Voronoi cell ids -> random value per cell (0..1), plus distance to the cell border (0..1)."""
    rng = np.random.default_rng(seed)
    pts = rng.random((count, 2))
    val = rng.random(count)
    g = (np.arange(res) + 0.5) / res
    X, Y = np.meshgrid(g, g, indexing="xy")
    best = np.full((res, res), 9.0)
    second = np.full((res, res), 9.0)
    idx = np.zeros((res, res), int)
    for k, (px, py) in enumerate(pts):
        dx = np.abs(X - px)
        dy = np.abs(Y - py)
        d = np.hypot(np.minimum(dx, 1 - dx), np.minimum(dy, 1 - dy))      # torus distance: tileable
        closer = d < best
        second = np.where(closer, best, np.minimum(second, d))
        idx = np.where(closer, k, idx)
        best = np.where(closer, d, best)
    rep = n // res
    v = _blur(np.kron(val[idx], np.ones((rep, rep))), 3.0)
    edge = _blur(np.kron(np.clip((second - best) * count ** 0.5 * 2.0, 0, 1), np.ones((rep, rep))), 3.0)
    return v, edge


def _blur(a, px):
    fx, fy = _freqs(a.shape[0])
    g = np.exp(-(fx ** 2 + fy ** 2) * (px / a.shape[0]) ** 2 * 20.0)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))


# ---------------------------------------------------------------------------------------------- finishes
def _maps(kind, rough):
    """(albedo multiplier, roughness, height, normal strength) for a finish class."""
    tile, seed = KINDS[kind]
    s = seed * 1000
    smudge = spectral(s + 1, 2, 14, 1.2)
    wipes = np.clip(spectral(s + 2, 1, 5, 1.5, aniso=3.0) - 1.1, 0, None)    # glossier wiped / handled patches
    if kind in ("powder", "satin"):
        mottle = spectral(s + 3, 1, 8, 1.5)
        fine = spectral(s + 4, 30, 120, 0.5)
        dust = specks(s + 5, 0.0006 if kind == "powder" else 0.002, 2.0)
        alb = 1 + 0.035 * mottle + 0.012 * fine + 0.10 * dust - (0.03 * smudge.clip(0) if kind == "satin" else 0)
        rgh = rough + 0.06 * smudge - 0.12 * wipes + 0.15 * dust
        peel_hi = 420 if kind == "powder" else 220                         # orange peel: ~1-2 mm blobs
        h = spectral(s + 6, peel_hi * 0.35, peel_hi, 0.8)
        return alb, rgh, h, 2.2 if kind == "powder" else 1.2
    if kind == "galv":
        v, edge = cells(s + 7, 140)                                        # spangle crystals ~5 cm
        grain = spectral(s + 8, 40, 200, 0.6, aniso=1.0)
        alb = 1 + 0.10 * (v - 0.5) + 0.02 * grain - 0.04 * (1 - edge) ** 6 + 0.03 * smudge
        rgh = rough + 0.14 * (v - 0.5) + 0.05 * smudge - 0.08 * wipes
        h = 0.6 * edge + 0.15 * grain
        return alb, rgh, h, 1.0
    if kind in ("steel", "polished"):
        brushed = spectral(s + 9, 40, 600, 0.4, aniso=25.0)                 # streaks along U
        k = 0.4 if kind == "polished" else 1.0
        alb = 1 + 0.03 * k * brushed + 0.03 * smudge
        rgh = rough + 0.05 * k * brushed + 0.06 * smudge - 0.08 * wipes
        h = brushed
        return alb, rgh, h, 0.8 * k
    if kind == "cast":
        pits = spectral(s + 10, 60, 300, 0.7)
        alb = 1 + 0.05 * pits + 0.04 * smudge
        rgh = rough + 0.08 * pits + 0.05 * smudge
        return alb, rgh, pits, 3.0
    if kind == "plastic":
        grain = spectral(s + 11, 150, 450, 0.5)
        alb = 1 + 0.012 * grain + 0.02 * smudge
        rgh = rough + 0.03 * grain + 0.05 * smudge - 0.08 * wipes
        return alb, rgh, grain, 1.6
    if kind == "rubber":
        dust = specks(s + 12, 0.003, 2.5)
        tex = spectral(s + 13, 60, 300, 0.6)
        alb = 1 + 0.25 * dust + 0.06 * smudge.clip(0) + 0.03 * tex
        rgh = rough + 0.06 * smudge + 0.1 * dust
        return alb, rgh, tex, 1.5
    if kind == "concrete":
        agg = specks(s + 14, 0.01, 3.0)
        stain = spectral(s + 15, 1, 6, 1.6)
        tex = spectral(s + 16, 40, 400, 0.6)
        alb = 1 + 0.12 * (agg - 0.2) + 0.08 * stain + 0.04 * tex
        rgh = rough + 0.06 * stain + 0.05 * tex
        return alb, rgh, tex + 0.8 * agg, 2.0
    if kind == "cardboard":
        fib = spectral(s + 17, 20, 300, 0.6, aniso=8.0)
        alb = 1 + 0.05 * fib + 0.04 * smudge
        rgh = rough + 0.05 * fib
        return alb, rgh, fib, 1.0
    raise KeyError(kind)


def _normal(h, strength):
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    k = strength * 0.08
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(h)
    L = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / L, ny / L, nz / L], -1) * 0.5 + 0.5


def write_png(path, arr):
    """Minimal PNG writer (8-bit grey or RGB) so this runs where PIL is not installed (Kit)."""
    a = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    ch = 1 if a.ndim == 2 else a.shape[2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0 if ch == 1 else 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(png)
    os.replace(tmp, path)


def texture_set(kind, roughness):
    """Paths of the generated maps for a finish class at a base roughness (cached)."""
    r = round(float(roughness) * 20) / 20                       # 0.05 steps: few files, shared by many looks
    tile = KINDS[kind][0]
    d = os.path.join(ROOT, kind)
    stem = os.path.join(d, f"{kind}_r{int(r * 100):03d}")
    meta = stem + ".json"
    if os.path.isfile(meta):
        with open(meta) as f:
            m = json.load(f)
        m["rel"] = {k: rel(m[k]) for k in ("albedo", "roughness", "normal")}
        return m
    os.makedirs(d, exist_ok=True)
    alb, rgh, h, ns = _maps(kind, r)
    alb = np.clip(0.82 * alb, 0.02, 1.0)                          # a grey multiplier with headroom
    srgb = np.where(alb <= 0.0031308, alb * 12.92, 1.055 * alb ** (1 / 2.4) - 0.055)
    shared = os.path.join(d, kind)                               # albedo + normal do not depend on roughness
    out = {"albedo": shared + "_albedo.png", "roughness": stem + "_rough.png", "normal": shared + "_normal.png",
           "mean": float(alb.mean()), "tile": tile}
    out["rel"] = {k: rel(out[k]) for k in ("albedo", "roughness", "normal")}
    if not os.path.isfile(out["albedo"]):
        write_png(out["albedo"], srgb)
    write_png(out["roughness"], np.clip(rgh, 0.02, 1.0))
    if not os.path.isfile(out["normal"]):
        write_png(out["normal"], _normal(h, ns))
    with open(meta, "w") as f:
        json.dump(out, f)
    return out


def rel(path):
    """The scene-relative name of a cached map: ./textures/<kind>/<file>."""
    return "./textures/" + os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)


REL = re.compile(r"\./textures/([a-z_]+)/([a-z_0-9]+)\.(png|hdr)$")


def used(scene):
    """(kind, file stem) of every generated map a saved scene refers to."""
    from pxr import Sdf, Usd
    st = Usd.Stage.Open(scene, Usd.Stage.LoadNone)
    out = set()
    for prim in st.Traverse():                     # materials, and lights (a dome's environment map)
        for a in prim.GetAttributes():
            if a.GetTypeName() == Sdf.ValueTypeNames.Asset:
                v = a.Get()
                m = REL.match(v.path) if v is not None else None
                if m:
                    out.add((m.group(1), m.group(2) + "." + m.group(3)))
    return out


def localize(scene):
    """Copy the generated maps a saved scene refers to into <scene dir>/textures (regenerating any missing from
    the cache). Returns the number of files copied."""
    n = 0
    here = os.path.dirname(os.path.abspath(scene))
    for kind, name in used(scene):
        src = os.path.join(ROOT, kind, name)
        if not os.path.isfile(src) and kind == "wear":
            wear_maps()
        elif not os.path.isfile(src) and kind == "studio":
            studio_hdri()
        elif not os.path.isfile(src) and kind in KINDS:
            m = re.match(r"[a-z]+_r(\d+)_", name)
            texture_set(kind, int(m.group(1)) / 100.0 if m else 0.5)
        dst = os.path.join(os.path.dirname(os.path.abspath(scene)), "textures", kind, name)
        if os.path.isfile(src) and (not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src)):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            n += 1
    mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frt_worn.mdl")
    if _uses_worn(scene) and os.path.isfile(mdl):                       # the wear material travels too
        dst = os.path.join(here, "materials", "frt_worn.mdl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(mdl, dst)
        n += 1
    return n


def _uses_worn(scene):
    from pxr import Usd
    st = Usd.Stage.Open(scene, Usd.Stage.LoadNone)
    for prim in st.Traverse():
        if prim.GetTypeName() == "Shader":
            a = prim.GetAttribute("info:mdl:sourceAsset")
            v = a.Get() if a else None
            if v is not None and v.path.endswith("frt_worn.mdl"):
                return True
    return False


# ---------------------------------------------------------------------------------------------- wear masks
def _scratches(seed, count, n=N):
    """Thin random scratches (mostly short, a few long, slightly curved), 0..1."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n))
    for _ in range(count):
        x, y = rng.random(2) * n
        ang = rng.random() * np.pi
        L = n * (0.02 + 0.12 * rng.random() ** 3)
        bend = (rng.random() - 0.5) * 0.6
        k = int(L)
        t = np.linspace(0, 1, max(k, 2))
        th = ang + bend * t
        xs = (x + np.cumsum(np.cos(th)) * L / len(t)).astype(int) % n
        ys = (y + np.cumsum(np.sin(th)) * L / len(t)).astype(int) % n
        np.add.at(a, (ys, xs), 0.5 + 0.5 * rng.random())
    a = _blur(a, 0.8)
    return np.clip(a / (np.percentile(a[a > 0], 99) if (a > 0).any() else 1.0), 0, 1)


def wear_maps():
    """Grey masks the wear material samples: chip breakup noise, scratches, stains (oil spots, drips, blotches)."""
    d = os.path.join(ROOT, "wear")
    out = {k: os.path.join(d, f"wear_{k}.png") for k in ("chip", "scratch", "stain")}
    if all(os.path.isfile(p) for p in out.values()):
        out["rel"] = {k: rel(p) for k, p in out.items() if k != "rel"}
        return out
    os.makedirs(d, exist_ok=True)
    s = 77000
    chip = 0.5 + 0.5 * np.tanh(0.9 * (0.55 * spectral(s + 1, 4, 40, 1.2) + 0.45 * spectral(s + 2, 40, 260, 0.6)))
    write_png(out["chip"], chip)
    write_png(out["scratch"], _scratches(s + 3, 220))
    spots = specks(s + 4, 0.00025, 14.0)
    blot = np.clip(spectral(s + 5, 2, 10, 1.5) - 0.6, 0, None)
    drips = np.clip(spectral(s + 6, 2, 30, 1.0, aniso=0.15) - 1.4, 0, None)
    stain = np.clip(1.6 * spots + 0.6 * blot + 0.8 * drips, 0, 1) * (0.85 + 0.15 * spectral(s + 7, 4, 30, 1.0).clip(-1, 1))
    write_png(out["stain"], np.clip(_blur(stain, 1.5), 0, 1))
    out["rel"] = {k: rel(p) for k, p in out.items() if k != "rel"}
    return out


# ---------------------------------------------------------------------------------------------- studio light
def write_hdr(path, rgb):
    """Radiance .hdr (RGBE, flat scanlines) of a float (H, W, 3) image."""
    v = np.maximum(rgb.max(-1), 1e-32)
    m, e = np.frexp(v)
    scale = np.where(v > 1e-30, m * 256.0 / v, 0.0)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    out[..., 3] = np.where(v > 1e-30, e + 128, 0).astype(np.uint8)
    h, w = rgb.shape[:2]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode())
        f.write(out.tobytes())
    os.replace(tmp, path)


def studio_hdri(w=2048):
    """Lat-long studio environment: dark graded cyclorama with a large key softbox, a fill, an overhead panel and a
    thin rim strip - soft, long highlights on paint and chrome like product renders. Returns its scene-relative
    name and cache path."""
    d = os.path.join(ROOT, "studio")
    path = os.path.join(d, "studio_softbox.hdr")
    if not os.path.isfile(path):
        os.makedirs(d, exist_ok=True)
        h = w // 2
        lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi            # azimuth
        lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi             # elevation
        LON, LAT = np.meshgrid(lon, lat)
        img = np.zeros((h, w, 3))
        g = np.clip((LAT + 0.2) / 1.6, 0, 1)
        img += (0.018 + 0.03 * np.exp(-((LAT - 0.05) / 0.25) ** 2) + 0.01 * g)[..., None] * np.array([1.0, 1.0, 1.03])

        def box(az, el, wa, we, k, tint=(1.0, 1.0, 1.0)):
            da = np.angle(np.exp(1j * (LON - np.radians(az))))
            u = np.abs(da) / np.radians(wa / 2)
            v = np.abs(LAT - np.radians(el)) / np.radians(we / 2)
            m = np.clip(1.0 - np.maximum(u, v), 0, None)
            return (np.clip(m * 6.0, 0, 1) ** 2)[..., None] * k * np.array(tint)
        img += box(40, 32, 34, 24, 14.0, (1.0, 0.98, 0.95))           # key
        img += box(-65, 18, 44, 28, 3.5, (0.95, 0.98, 1.0))           # fill
        img += box(175, 48, 70, 7, 9.0)                               # rim strip
        img += box(0, 82, 60, 18, 5.0)                                # overhead panel
        write_hdr(path, img.astype(np.float32))
    return {"path": path, "rel": rel(path)}
