"""The engine behind the website's "Try the engine" page: a small HTTP API that takes an edited sample job, solves it
with task_run.py and serves the result files the web viewer plays.

    python tools/engine_server.py [--port=8765]

    GET  /api/health                      {"ok": true, "queue": n}
    GET  /api/samples                     the sample jobs a request can start from
    POST /api/solve                       {"base": "wheel_mounting", "assets": {id: [x, z]}, "spots": {id: [x, z, yaw]},
                                           "steps": [indices into the base job's steps]}   -> {"id": ...}
    GET  /api/job/<id>                    {"state": queued|running|done|error, "msg", "progress", "ahead"}
    GET  /api/job/<id>/<job.json|motion.bin|scene.glb>

A request only moves what is already in a sample (assets and spots, within 10 m of where they were) and picks and
orders the sample's own steps: nothing else of the task file can be set from outside. One solve at a time; each
sample keeps its step cache, so re-solving after a small edit only redoes the steps that changed.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "out", "web_jobs")
SAMPLES = {
    "wheel_mounting": ("docs/people/demo/wheel_mounting.toml", "Automotive: wheel mounting"),
    "control_cabinet": ("docs/people/demo/control_cabinet.toml", "Electrical: control cabinet wiring"),
    "overhead_bracket": ("docs/people/demo/overhead_bracket.toml", "Aerospace: overhead bracket"),
    "valve_block": ("docs/people/demo/valve_block.toml", "Hydraulics: valve block into a housing"),
    "base_panel": ("docs/people/demo/base_panel.toml", "Appliances: base service panel"),
    "battery_module": ("docs/people/demo/battery_module.toml", "Energy storage: battery module"),
}
MAX_MOVE = 1000.0                                                    # cm an asset or spot may move from its place
FILES = {"job.json": "application/json", "motion.bin": "application/octet-stream", "scene.glb": "model/gltf-binary"}

jobs = {}                                                            # id -> {"state", "msg", "progress", "t"}
todo = queue.Queue()
lock = threading.Lock()


# ---------------------------------------------------------------------------------------------- task files
def base_data(base):
    """The sample's task with its scene merged in (spots and assets inline)."""
    path = os.path.join(ROOT, SAMPLES[base][0])
    with open(path, "rb") as f:
        data = tomllib.load(f)
    sc = data["task"].pop("scene", None)
    if sc:
        with open(os.path.join(os.path.dirname(path), sc), "rb") as f:
            s = tomllib.load(f)
        data["spot"] = s.get("spot", []) + data.get("spot", [])
        data["asset"] = s.get("asset", []) + data.get("asset", [])
    return data


def edited(base, req):
    """The base task with the request's moves and step choice applied (only those)."""
    data = base_data(base)
    moves = req.get("assets") or {}
    for a in data.get("asset", []):
        m = moves.get(a["id"])
        if m is not None and "at" in a:
            x, z = (float(v) for v in m[:2])
            if abs(x - a["at"][0]) > MAX_MOVE or abs(z - a["at"][2]) > MAX_MOVE:
                raise ValueError(f"{a['id']} moved too far")
            a["at"] = [round(x, 1), a["at"][1], round(z, 1)]
    spots = req.get("spots") or {}
    for s in data.get("spot", []):
        m = spots.get(s["id"])
        if m is not None:
            x, z = (float(v) for v in m[:2])
            if abs(x - s["at"][0]) > MAX_MOVE or abs(z - s["at"][1]) > MAX_MOVE:
                raise ValueError(f"spot {s['id']} moved too far")
            s["at"] = [round(x, 1), round(z, 1)]
            if len(m) > 2:
                s["yaw"] = round(float(m[2]), 1)
    steps = data.get("step", [])
    pick = req.get("steps")
    if pick is not None:
        if not pick or len(pick) > 20 or any(not isinstance(i, int) or not 0 <= i < len(steps) for i in pick):
            raise ValueError("steps: indices of the sample's steps")
        data["step"] = [steps[i] for i in pick]
    return data


def _val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    raise TypeError(type(v))


def _table(name, d, out, array=False):
    out.append(f"[[{name}]]" if array else f"[{name}]")
    sub = []
    for k, v in d.items():
        if isinstance(v, dict):
            sub.append((k, v))
        else:
            out.append(f"{k} = {_val(v)}")
    for k, v in sub:
        _table(f"{name}.{k}", v, out)
    out.append("")


def dump_toml(data):
    out = []
    for k, v in data.items():
        if isinstance(v, dict):
            _table(k, v, out)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for item in v:
                _table(k, item, out, array=True)
        else:
            out.insert(0, f"{k} = {_val(v)}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- the solver queue
def worker():
    while True:
        jid, base, data = todo.get()
        d = os.path.join(WORK, jid)
        cache_dir = os.path.join(WORK, f"cache_{base}")              # the sample's step cache, carried between runs
        os.makedirs(cache_dir, exist_ok=True)
        out = os.path.join(d, "out")
        os.makedirs(out, exist_ok=True)
        src = os.path.join(cache_dir, "step_cache.pkl")
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(out, "step_cache.pkl"))
        task = os.path.join(d, "task.toml")
        with open(task, "w", encoding="utf-8") as f:
            f.write(dump_toml(data))
        jobs[jid].update(state="running", msg="starting the engine", t=time.time())
        p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "tools", "task_run.py"), task, f"--out={out}",
                              f"--web={os.path.join(d, 'web')}"], cwd=ROOT, stdout=open(os.path.join(d, "log.txt"), "w"),
                             stderr=subprocess.STDOUT, env={**os.environ, "PXR_USDC_EMIT_DEPRECATION_WARNINGS": "0"})
        st = os.path.join(out, "status.json")
        while p.poll() is None:
            time.sleep(0.5)
            try:
                s = json.load(open(st))
                jobs[jid].update(msg=s.get("msg", ""), progress=s.get("progress", 0))
            except Exception:
                pass
        ok = p.returncode == 0 and os.path.isfile(os.path.join(d, "web", "job.json"))
        if ok and os.path.isfile(os.path.join(out, "step_cache.pkl")):
            shutil.copy(os.path.join(out, "step_cache.pkl"), src)
        err = None
        if not ok:
            try:
                err = json.load(open(st)).get("error")
            except Exception:
                err = "the engine stopped - see log.txt"
        jobs[jid].update(state="done" if ok else "error", msg="done" if ok else (err or "failed"), progress=1.0,
                         seconds=round(time.time() - jobs[jid]["t"], 1))
        todo.task_done()


# ---------------------------------------------------------------------------------------------- http
class Api(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")   # (the hosted site calling a local engine)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        parts = self.path.split("?")[0].strip("/").split("/")
        if parts[:2] == ["api", "health"]:
            return self._send(200, {"ok": True, "queue": todo.qsize()})
        if parts[:2] == ["api", "samples"]:
            return self._send(200, [{"id": k, "title": v[1]} for k, v in SAMPLES.items()])
        if len(parts) >= 3 and parts[:2] == ["api", "job"] and parts[2] in jobs:
            jid = parts[2]
            if len(parts) == 3:
                j = dict(jobs[jid])
                j["ahead"] = sum(1 for k, v in jobs.items() if v["state"] == "queued" and v["n"] < j["n"])
                return self._send(200, j)
            if len(parts) == 4 and parts[3] in FILES and jobs[jid]["state"] == "done":
                with open(os.path.join(WORK, jid, "web", parts[3]), "rb") as f:
                    return self._send(200, f.read(), FILES[parts[3]])
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") != "/api/solve":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 20000:
                raise ValueError("request too big")
            req = json.loads(self.rfile.read(n))
            base = req.get("base")
            if base not in SAMPLES:
                raise ValueError("base: one of " + ", ".join(SAMPLES))
            if todo.qsize() >= 8:
                return self._send(503, {"error": "the engine is busy - try again in a minute"})
            data = edited(base, req)
        except Exception as e:
            return self._send(400, {"error": str(e)})
        jid = uuid.uuid4().hex[:12]
        with lock:
            jobs[jid] = {"state": "queued", "msg": "waiting for the engine", "progress": 0.0, "t": time.time(),
                         "n": len(jobs), "base": base}
        todo.put((jid, base, data))
        self._send(200, {"id": jid})

    def log_message(self, fmt, *a):
        pass


def main():
    port = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--port=")), 8765))
    os.makedirs(WORK, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    print(f"engine on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Api).serve_forever()


if __name__ == "__main__":
    main()
