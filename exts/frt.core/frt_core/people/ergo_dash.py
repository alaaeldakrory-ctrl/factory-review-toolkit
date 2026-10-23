"""Dashboards for an ergonomic scan (ergo_scan.scan): the same set as the rail / rack assembly reviews, sized for
many operators.

    hud_png(res, frame, path)            dark score panel at one frame: a card per operator (score badge, band,
                                         angles, body figure coloured by body part, rolling 20 s chart); grid layout
    composite_frames(res, in_dir, out_dir)   the panel next to every rendered movie frame (frame number in the name)
    timeline_png(res, path)              every operator's RULA over time, coloured by band
    floor_png(res, path)                 plan view: where the risky work (RULA 5-7) happens, operators labelled
    dashboard_html(res, path)            one self-contained HTML file (no internet): KPIs, ranking, per-operator
                                         timeline, body-part breakdown, angles, worst moments, floor map
Viewport markers (sphere above the head + body-part dots) are in ergo_scan's overlay layer.
"""
import json
import math
import os
import re

import numpy as np

from .ergo_scan import BAND_RGB, band, summarise, _smooth

BG, PANEL, FG, DIM, GRID = (24, 26, 30), (34, 37, 43), (235, 236, 240), (150, 154, 162), (60, 64, 72)
PART_LIMITS = {"trunk_s": (2, 3), "neck_s": (2, 3), "upper_arm_s": (2, 3), "legs_s": (1, 1)}   # green <=, yellow <=
GREEN, YELLOW, RED = (46, 170, 80), (240, 200, 30), (215, 40, 40)


def part_rgb(key, s):
    g, y = PART_LIMITS[key]
    return GREEN if s <= g else YELLOW if s <= y else RED


def band_rgb(score):
    return tuple(int(255 * v) for v in BAND_RGB[band(score)])


def _font(size, bold=False):
    from PIL import ImageFont
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf"),
                 ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _ops(res):
    return [o for o in res["operators"] if o["rows"]]


def _smoothed(res, op):
    if "_sm" not in op:
        op["_sm"] = _smooth([r["score"] for r in op["rows"]], r=max(1, int(round(0.25 * res["fps"] / res["step"]))))
    return op["_sm"]


def _row_at(op, frame):
    fr = [r["frame"] for r in op["rows"]]
    i = int(np.searchsorted(fr, frame, side="right")) - 1
    return max(0, min(i, len(fr) - 1))


def _figure(d, x, y, h, r):
    """Small stick figure, body parts coloured by their RULA part score."""
    c_neck, c_trunk = part_rgb("neck_s", r["neck_s"]), part_rgb("trunk_s", r["trunk_s"])
    c_arm, c_leg = part_rgb("upper_arm_s", r["upper_arm_s"]), part_rgb("legs_s", r["legs_s"])
    s = h / 100.0
    P = lambda a, b: (x + a * s, y + b * s)                       # noqa: E731
    w = max(2, int(3 * s))
    d.line([P(0, 18), P(0, 54)], fill=c_trunk, width=w)            # trunk
    d.line([P(-16, 22), P(16, 22)], fill=c_trunk, width=w)
    d.line([P(-16, 22), P(-22, 46)], fill=c_arm, width=w)          # arms
    d.line([P(16, 22), P(22, 46)], fill=c_arm, width=w)
    d.line([P(-9, 56), P(-11, 96)], fill=c_leg, width=w)           # legs
    d.line([P(9, 56), P(11, 96)], fill=c_leg, width=w)
    rr = 7 * s
    d.ellipse([x - rr, y + 9 * s - rr, x + rr, y + 9 * s + rr], outline=c_neck, width=w)


def hud_png(res, frame, path=None, cols=None, card=(440, 210), title="Ergonomics (RULA)", subtitle=None):
    """The score panel at `frame` (stage time code): one card per operator, worst first. Returns the PIL image."""
    from PIL import Image, ImageDraw
    ops = _ops(res)
    rows_now = []
    for o in ops:
        i = _row_at(o, frame)
        rows_now.append((_smoothed(res, o)[i], o, i))
    rows_now.sort(key=lambda t: -t[0])
    n = len(rows_now)
    cols = cols or (1 if n <= 4 else 2 if n <= 8 else 3)
    rws = math.ceil(n / cols)
    cw, ch = card
    W, H = 24 + cols * (cw + 14), 92 + rws * (ch + 14)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((22, 16), title, fill=FG, font=_font(34, True))
    t = (frame - res["range"][0]) / res["fps"]
    d.text((24, 58), subtitle or f"t = {t:5.1f} s   frame {frame:g}   {n} operators", fill=DIM, font=_font(18))
    span = int(round(20.0 * res["fps"] / res["step"]))            # rolling 20 s
    for k, (sc, o, i) in enumerate(rows_now):
        cx, cy = 18 + (k % cols) * (cw + 14), 92 + (k // cols) * (ch + 14)
        d.rounded_rectangle([cx, cy, cx + cw, cy + ch], 12, fill=PANEL, outline=GRID, width=2)
        r = o["rows"][i]
        d.text((cx + 14, cy + 8), o["name"][:30], fill=FG, font=_font(18, True))
        col = band_rgb(sc)
        d.rounded_rectangle([cx + 14, cy + 36, cx + 74, cy + 96], 10, fill=col)
        f = _font(40, True)
        tw = d.textlength(str(sc), font=f)
        d.text((cx + 44 - tw / 2, cy + 40), str(sc), fill=(255, 255, 255) if sc >= 5 else (20, 20, 20), font=f)
        d.text((cx + 14, cy + 100), band(sc).upper() + ("  (walking)" if r["walking"] else ""), fill=col,
               font=_font(15, True))
        info = [f"back {r['trunk_deg']:.0f}°   neck {r['neck_deg']:.0f}°",
                f"arm {r['upper_arm_deg']:.0f}°   elbow {r['elbow_deg']:.0f}°   knee {r['knee_deg']:.0f}°"]
        for j, line in enumerate(info):
            d.text((cx + 86, cy + 42 + j * 20), line, fill=DIM, font=_font(14))
        _figure(d, cx + cw - 36, cy + 10, 88, r)
        gx0, gy0, gx1, gy1 = cx + 14, cy + 124, cx + cw - 14, cy + ch - 14
        d.rectangle([gx0, gy0, gx1, gy1], fill=(18, 19, 22))
        sm = _smoothed(res, o)
        lo = max(0, i - span)
        seg = sm[lo:i + 1]
        if seg:
            bw = (gx1 - gx0) / max(1, span)
            for j, v in enumerate(seg):
                x0 = gx1 - (len(seg) - j) * bw
                d.rectangle([x0, gy1 - (gy1 - gy0) * v / 7.0, x0 + bw + 0.5, gy1], fill=band_rgb(v))
        d.text((gx0 + 2, gy1 - 14), "last 20 s", fill=DIM, font=_font(11))
    if path:
        img.save(path)
    return img


def composite_frames(res, in_dir, out_dir, panel_w=None, pattern=r"(\d+)\.(png|jpg)$"):
    """Put the score panel next to each rendered frame in in_dir (the frame number is read from the file name)."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    for fn in sorted(os.listdir(in_dir)):
        m = re.search(pattern, fn)
        if not m:
            continue
        frame = float(int(m.group(1)))
        im = Image.open(os.path.join(in_dir, fn)).convert("RGB")
        hud = hud_png(res, frame)
        pw = panel_w or int(im.height * hud.width / hud.height)
        hud = hud.resize((pw, im.height))
        out = Image.new("RGB", (im.width + pw, im.height), BG)
        out.paste(im, (0, 0))
        out.paste(hud, (im.width, 0))
        out.save(os.path.join(out_dir, fn))
        done += 1
    return done


def timeline_png(res, path, width=1800, row=46):
    from PIL import Image, ImageDraw
    summ = {s["operator"]: s for s in summarise(res)}
    ops = sorted(_ops(res), key=lambda o: (-(summ[o["name"]].get("max") or 0), -(summ[o["name"]].get("p90") or 0)))
    lab = 240
    H = 90 + row * len(ops) + 40
    img = Image.new("RGB", (width, H), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 16), "RULA over time, every operator (worst first)", fill=FG, font=_font(28, True))
    n = max(len(o["rows"]) for o in ops)
    secs = n * res["step"] / res["fps"]
    x0, x1 = lab, width - 20
    for k, o in enumerate(ops):
        y = 70 + k * row
        d.text((20, y + 10), o["name"][:26], fill=FG, font=_font(17, True))
        s = summ[o["name"]]
        d.text((lab - 54, y + 10), f"{s['max']}", fill=band_rgb(s["max"]), font=_font(18, True))
        sm = _smoothed(res, o)
        for i, v in enumerate(sm):
            a = x0 + (x1 - x0) * i / n
            b = x0 + (x1 - x0) * (i + 1) / n
            hgt = (row - 10) * v / 7.0
            d.rectangle([a, y + row - 6 - hgt, max(a, b - 0.5), y + row - 6], fill=band_rgb(v))
        d.line([x0, y + row - 5, x1, y + row - 5], fill=GRID)
    tick = max(1, int(round(secs / 12.0)))
    for t in range(0, int(secs) + 1, tick):
        x = x0 + (x1 - x0) * t / secs if secs else x0
        d.text((x - 8, H - 32), f"{t}s", fill=DIM, font=_font(14))
    lx = width - 560
    for j, (name, rgb) in enumerate(BAND_RGB.items()):
        c = tuple(int(255 * v) for v in rgb)
        d.rectangle([lx + j * 135, 26, lx + j * 135 + 16, 42], fill=c)
        d.text((lx + j * 135 + 22, 24), name, fill=DIM, font=_font(15))
    img.save(path)


def _plan(res, row):
    x, z = row["floor_m"]
    return (x, -z) if res["z_up"] else (x, z)                     # stage plan axes (Z-up: x / y; Y-up: x / z)


def floor_png(res, path, px_per_m=90, sigma_m=0.3, margin_m=1.5):
    """Plan view: where RULA 5-7 happens (time-weighted, 7 counts double), operators labelled at their usual spot."""
    from PIL import Image, ImageDraw
    ops = _ops(res)
    pts = [(_plan(res, r), s) for o in ops for r, s in zip(o["rows"], _smoothed(res, o))]
    xs = [p[0][0] for p in pts]
    ys = [p[0][1] for p in pts]
    mnx, mxx, mny, mxy = min(xs) - margin_m, max(xs) + margin_m + 2.0, min(ys) - margin_m, max(ys) + margin_m   # labels: room on the right
    W, H = int((mxx - mnx) * px_per_m), int((mxy - mny) * px_per_m)
    scale = min(1.0, 1600.0 / max(W, H))
    px = px_per_m * scale
    W, H = int((mxx - mnx) * px), int((mxy - mny) * px) + 60
    cell = 4
    gw, gh = W // cell + 1, H // cell + 1
    G = np.zeros((gh, gw))
    sig = sigma_m * px / cell
    rr = int(3 * sig) + 1
    ky = np.arange(-rr, rr + 1)
    K = np.exp(-(ky[None, :] ** 2 + ky[:, None] ** 2) / (2 * sig * sig))
    for (x, y), s in pts:
        if s < 5:
            continue
        cx = int((x - mnx) * px / cell)
        cy = int((mxy - y) * px / cell) + 60 // cell
        y0, y1, x0, x1 = max(0, cy - rr), min(gh, cy + rr + 1), max(0, cx - rr), min(gw, cx + rr + 1)
        G[y0:y1, x0:x1] += K[y0 - cy + rr:y1 - cy + rr, x0 - cx + rr:x1 - cx + rr] * (2.0 if s >= 7 else 1.0)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    step = 1.0
    gx = math.ceil(mnx)
    while gx < mxx:
        X = (gx - mnx) * px
        d.line([X, 60, X, H], fill=(40, 43, 50))
        gx += step
    gy = math.ceil(mny)
    while gy < mxy:
        Y = (mxy - gy) * px + 60
        d.line([0, Y, W, Y], fill=(40, 43, 50))
        gy += step
    if G.max() > 0:
        g = G / G.max()
        heat = Image.new("RGBA", (gw, gh))
        hp = heat.load()
        for j in range(gh):
            for i in range(gw):
                v = g[j, i]
                if v < 0.04:
                    continue
                c = (int(240 - 30 * v), int(200 - 160 * v), int(30 + 10 * v), int(255 * min(1.0, 0.35 + v)))
                hp[i, j] = c
        heat = heat.resize((gw * cell, gh * cell))
        img.paste(heat, (0, 0), heat)
        d = ImageDraw.Draw(img)
    summ = {s["operator"]: s for s in summarise(res)}
    for o in ops:
        P = np.array([_plan(res, r) for r in o["rows"]])
        x, y = np.median(P[:, 0]), np.median(P[:, 1])
        X, Y = (x - mnx) * px, (mxy - y) * px + 60
        c = band_rgb(summ[o["name"]]["max"])
        d.ellipse([X - 6, Y - 6, X + 6, Y + 6], fill=c, outline=(255, 255, 255))
        d.text((X + 9, Y - 9), f"{o['name']} ({summ[o['name']]['max']})", fill=FG, font=_font(15))
    d.text((14, 14), "Where the risky work happens (time at RULA 5-7, all operators)", fill=FG, font=_font(24, True))
    d.text((W - 110, H - 24), "1 m grid", fill=DIM, font=_font(14))
    img.save(path)


def dashboard_html(res, path, title="Ergonomic scan (RULA)", max_points=900):
    """One HTML file with everything embedded (opens offline in any browser)."""
    summ = summarise(res)
    data = {"title": title, "stage": res["stage"], "fps": res["fps"], "step": res["step"], "range": res["range"],
            "zup": res["z_up"], "summary": summ, "ops": {}}
    for o in _ops(res):
        sm = _smoothed(res, o)
        k = max(1, int(math.ceil(len(sm) / max_points)))
        idx = list(range(0, len(sm), k))
        rows = o["rows"]
        data["ops"][o["name"]] = {
            "t": [rows[i]["t"] for i in idx], "frame": [rows[i]["frame"] for i in idx],
            "s": [max(sm[i:i + k]) for i in idx],
            "trunk": [rows[i]["trunk_deg"] for i in idx], "neck": [rows[i]["neck_deg"] for i in idx],
            "arm": [rows[i]["upper_arm_deg"] for i in idx], "knee": [rows[i]["knee_deg"] for i in idx],
            "xy": [[round(v, 2) for v in _plan(res, rows[i])] for i in idx],
            "parts": {p: [round(100.0 * sum(part_rgb(p, r[p]) == c for r in rows) / len(rows), 1) for c in (YELLOW, RED)]
                      for p in PART_LIMITS},
            "walk": [1 if rows[i]["walking"] else 0 for i in idx]}
    html = _HTML.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))).replace("%TITLE%", title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>%TITLE%</title>
<style>
:root{--bg:#181a1e;--panel:#22252b;--line:#3c4048;--fg:#ebecf0;--dim:#9a9ea6;--g:#2eaa50;--y:#f0c81e;--o:#f5781e;--r:#d72828}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 "Segoe UI",system-ui,sans-serif}
header{padding:18px 22px 6px}h1{margin:0;font-size:24px}.sub{color:var(--dim);font-size:12px;word-break:break-all}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;padding:10px 22px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px}.kpi b{font-size:26px;display:block}
.kpi span{color:var(--dim);font-size:12px}main{display:grid;grid-template-columns:minmax(320px,1fr) minmax(360px,1.4fr);gap:12px;padding:6px 22px 22px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px;overflow:hidden}
.card h2{font-size:15px;margin:0 0 8px}table{width:100%;border-collapse:collapse}td,th{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left;font-size:13px}
th{color:var(--dim);font-weight:600}tr.op{cursor:pointer}tr.op:hover,tr.sel{background:#2c3038}
.badge{display:inline-block;min-width:26px;text-align:center;border-radius:6px;padding:1px 6px;font-weight:700;color:#111}
.bar{height:10px;border-radius:3px;display:flex;overflow:hidden;min-width:90px}.bar i{display:block;height:100%}
svg{width:100%;display:block}.mom{border-left:4px solid var(--o);padding:4px 10px;margin:6px 0;background:#1d2025;border-radius:4px}
.mom small{color:var(--dim)}.legend span{margin-right:12px;color:var(--dim);font-size:12px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
.parts{display:grid;grid-template-columns:90px 1fr 46px;gap:6px;align-items:center;font-size:13px}.pbar{height:10px;background:#1d2025;border-radius:3px;display:flex;overflow:hidden}.pbar i{display:block;height:100%}
.note{color:var(--dim);font-size:12px}
</style></head><body>
<header><h1 id="ttl"></h1><div class="sub" id="sub"></div></header>
<div class="kpis" id="kpis"></div>
<main><div>
<div class="card"><h2>Operators (worst first) - click one</h2><table id="tbl"></table>
<div class="legend" style="margin-top:8px"><span><i style="background:var(--g)"></i>1-2 acceptable</span><span><i style="background:var(--y)"></i>3-4 investigate</span><span><i style="background:var(--o)"></i>5-6 change soon</span><span><i style="background:var(--r)"></i>7 change now</span></div></div>
<div class="card"><h2>Where the risky work happens (RULA 5-7)</h2><svg id="floor"></svg><div class="note" id="floornote"></div></div>
</div><div>
<div class="card"><h2 id="opname"></h2><svg id="tl" viewBox="0 0 800 170"></svg></div>
<div class="card"><h2>Body angles over time</h2><svg id="ang" viewBox="0 0 800 190"></svg>
<div class="legend"><span><i style="background:#6fb7ff"></i>back</span><span><i style="background:#c38cff"></i>neck</span><span><i style="background:#ffb86b"></i>upper arm</span><span><i style="background:#7ee0c0"></i>knee</span></div></div>
<div class="card"><h2>Which body part is at risk (% of time: yellow = moderate, red = high)</h2><div class="parts" id="parts"></div></div>
<div class="card"><h2>Worst moments and what to change</h2><div id="moms"></div></div>
</div></main>
<div class="note" style="padding:0 22px 22px">Scores describe the animation (the method as animated), not measured people. Loads and repeated work are included only where they were given. Use for comparing methods and finding bad steps.</div>
<script>
const D=/*DATA*/null;
const C=s=>s<=2?'#2eaa50':s<=4?'#f0c81e':s<=6?'#f5781e':'#d72828';
const $=id=>document.getElementById(id);
const S=D.summary.filter(s=>!s.skipped), SK=D.summary.filter(s=>s.skipped);
$('ttl').textContent=D.title;
$('sub').textContent=`${D.stage}  |  frames ${D.range[0]}-${D.range[1]} @ ${D.fps} fps, every ${D.step}`;
const at5=S.filter(s=>s.max>=5&&s.max<7).length, at7=S.filter(s=>s.max>=7).length;
const tot=S.reduce((a,s)=>a+s.frames,0), hi=S.reduce((a,s)=>a+s.frames*(s.band_pct['change soon']+s.band_pct['change now'])/100,0);
const k=[[S.length+(SK.length?` (+${SK.length} skipped)`:''),'operators scored'],[at7,'reach RULA 7 (change now)'],[at5,'peak at 5-6 (change soon)'],
 [(tot?100*hi/tot:0).toFixed(0)+'%','of all working time at 5+'],[S.length?S[0].operator:'-','worst operator']];
$('kpis').innerHTML=k.map(([b,s])=>`<div class="kpi"><b>${b}</b><span>${s}</span></div>`).join('');
const bar=p=>`<div class="bar">${[['acceptable','#2eaa50'],['investigate','#f0c81e'],['change soon','#f5781e'],['change now','#d72828']].map(([n,c])=>`<i style="width:${p[n]}%;background:${c}"></i>`).join('')}</div>`;
const ST=D.summary.filter(s=>s.static).map(s=>s.operator);
$('tbl').innerHTML='<tr><th>operator</th><th>max</th><th>p90</th><th>time in each band</th><th>main driver</th></tr>'+
 S.map(s=>`<tr class="op" data-op="${s.operator}"><td>${s.operator}</td><td><span class="badge" style="background:${C(s.max)}">${s.max}</span></td><td>${s.p90}</td><td>${bar(s.band_pct)}</td><td>${s.main_driver}</td></tr>`).join('')+
 SK.map(s=>`<tr><td>${s.operator}</td><td colspan="4" class="note">skipped: ${s.skipped}</td></tr>`).join('');
if(ST.length)document.querySelector('.kpis').insertAdjacentHTML('afterend',`<div class="note" style="padding:0 22px;color:#f0c81e">Not moving in the USD file (animation not stored, e.g. Animation Graph / behaviour scripts - record or bake first): ${ST.join(', ')}</div>`);
function tl(name){const o=D.ops[name],n=o.s.length,T=o.t[n-1]||1;let h='';
 for(let i=0;i<n;i++){const x=40+740*o.t[i]/T,w=Math.max(1,740/n+0.6),v=o.s[i],y=150-130*v/7;h+=`<rect x="${x}" y="${y}" width="${w}" height="${150-y}" fill="${C(v)}"/>`;}
 for(let v=1;v<=7;v+=2)h+=`<text x="30" y="${154-130*v/7}" fill="#9a9ea6" font-size="11" text-anchor="end">${v}</text>`;
 for(let i=0;i<=6;i++)h+=`<text x="${40+740*i/6}" y="166" fill="#9a9ea6" font-size="11" text-anchor="middle">${(T*i/6).toFixed(0)}s</text>`;
 $('tl').innerHTML=h;}
function ang(name){const o=D.ops[name],n=o.t.length,T=o.t[n-1]||1;let h='';const mx=160;
 for(const v of [0,45,90,135])h+=`<line x1="40" x2="780" y1="${170-150*v/mx}" y2="${170-150*v/mx}" stroke="#3c4048"/><text x="34" y="${174-150*v/mx}" fill="#9a9ea6" font-size="11" text-anchor="end">${v}°</text>`;
 for(const [k,c] of [['trunk','#6fb7ff'],['neck','#c38cff'],['arm','#ffb86b'],['knee','#7ee0c0']]){
  h+=`<polyline fill="none" stroke="${c}" stroke-width="1.6" points="${o[k].map((v,i)=>`${(40+740*o.t[i]/T).toFixed(1)},${(170-150*Math.max(-10,Math.min(mx,Math.abs(v)))/mx).toFixed(1)}`).join(' ')}"/>`;}
 $('ang').innerHTML=h;}
function parts(name){const p=D.ops[name].parts,L={trunk_s:'back',neck_s:'neck',upper_arm_s:'upper arm',legs_s:'legs / knees'};
 $('parts').innerHTML=Object.keys(L).map(k=>`<div>${L[k]}</div><div class="pbar"><i style="width:${p[k][1]}%;background:var(--r)"></i><i style="width:${p[k][0]}%;background:var(--y)"></i></div><div>${(p[k][0]+p[k][1]).toFixed(0)}%</div>`).join('');}
function moms(name){const s=S.find(x=>x.operator===name);
 $('moms').innerHTML=(s.worst||[]).filter(m=>m.rula>=3).map(m=>`<div class="mom" style="border-color:${C(m.rula)}"><span class="badge" style="background:${C(m.rula)}">${m.rula}</span> <b>${m.t.toFixed(1)} s</b> <small>frame ${m.frame}, lasts ${m.lasts_s} s</small><br>${m.why}<br><small>Fix: ${m.fix}</small></div>`).join('')||'<div class="note">Nothing above 2.</div>';}
function floor(sel){const all=[];for(const n in D.ops){const o=D.ops[n];o.xy.forEach((p,i)=>all.push([p,o.s[i],n]));}
 if(!all.length)return;let xs=all.map(a=>a[0][0]),ys=all.map(a=>a[0][1]);let x0=Math.min(...xs)-1,x1=Math.max(...xs)+1,y0=Math.min(...ys)-1,y1=Math.max(...ys)+1;
 const W=600,sc=W/(x1-x0),H=Math.max(200,(y1-y0)*sc);$('floor').setAttribute('viewBox',`0 0 ${W} ${H}`);let h='';
 for(let g=Math.ceil(x0);g<x1;g++)h+=`<line x1="${(g-x0)*sc}" x2="${(g-x0)*sc}" y1="0" y2="${H}" stroke="#2a2d34"/>`;
 for(let g=Math.ceil(y0);g<y1;g++)h+=`<line y1="${(y1-g)*sc}" y2="${(y1-g)*sc}" x1="0" x2="${W}" stroke="#2a2d34"/>`;
 for(const [p,s,n] of all){if(s<5)continue;h+=`<circle cx="${(p[0]-x0)*sc}" cy="${(y1-p[1])*sc}" r="${0.25*sc}" fill="${C(s)}" opacity="${n===sel?0.22:0.07}"/>`;}
 for(const n in D.ops){const o=D.ops[n],xs=o.xy.map(p=>p[0]).sort((a,b)=>a-b),ys=o.xy.map(p=>p[1]).sort((a,b)=>a-b),m=xs.length>>1;
  const s=S.find(x=>x.operator===n);const X=(xs[m]-x0)*sc,Y=(y1-ys[m])*sc;
  h+=`<circle cx="${X}" cy="${Y}" r="${n===sel?7:5}" fill="${C(s.max)}" stroke="#fff" stroke-width="${n===sel?2:1}"/><text x="${X+9}" y="${Y+4}" fill="${n===sel?'#fff':'#9a9ea6'}" font-size="12">${n}</text>`;}
 $('floor').innerHTML=h;$('floornote').textContent=`Plan view (${D.zup?'x / y':'x / z'}), 1 m grid. Glow = time at RULA 5-7; dot = the operator's usual spot, coloured by their worst score.`;}
function pick(name){document.querySelectorAll('tr.op').forEach(r=>r.classList.toggle('sel',r.dataset.op===name));
 const s=S.find(x=>x.operator===name);$('opname').innerHTML=`${name} - RULA over time <span class="badge" style="background:${C(s.max)}">max ${s.max}</span> <span class="note">median ${s.median}, p90 ${s.p90}, walking ${s.walking_pct}% of the time</span>`;
 tl(name);ang(name);parts(name);moms(name);floor(name);}
document.querySelectorAll('tr.op').forEach(r=>r.onclick=()=>pick(r.dataset.op));
if(S.length)pick(S[0].operator);
</script></body></html>"""
