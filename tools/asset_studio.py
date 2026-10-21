"""Studio scene for one catalog item (product shots, library thumbnails, turntables).

    python tools/asset_studio.py lift_table out/studio/lift_table.usd --param cabinet=true --param castors=red
           [--age 1.0] [--turntable 240]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from frt_core.studio import build_studio   # noqa: E402


def _value(v):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("item")
    ap.add_argument("out")
    ap.add_argument("--param", action="append", default=[], help="name=value")
    ap.add_argument("--age", type=float, default=1.0)
    ap.add_argument("--turntable", type=int, default=0)
    a = ap.parse_args(argv)
    params = dict(kv.split("=", 1) for kv in a.param)
    print("->", build_studio(a.out, a.item, {k: _value(v) for k, v in params.items()}, a.age, a.turntable))


if __name__ == "__main__":
    main()
