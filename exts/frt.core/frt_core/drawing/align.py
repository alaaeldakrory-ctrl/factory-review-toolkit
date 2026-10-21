"""Alignment: put a drawing into plant coordinates.

two_point(dr, (a, b), (A, B)) maps drawing points a, b onto plant points A, B (rotation + translation, optionally
uniform scale) — the "click two column centres" workflow. to_origin(dr, corner) moves a corner (e.g. the building
outline's south-west corner found by the interpreter) to (0, 0).
"""
import math


def similarity(a, b, A, B, allow_scale=False):
    va = (b[0] - a[0], b[1] - a[1])
    vA = (B[0] - A[0], B[1] - A[1])
    rot = math.atan2(vA[1], vA[0]) - math.atan2(va[1], va[0])
    s = (math.hypot(*vA) / math.hypot(*va)) if allow_scale else 1.0
    c, n = math.cos(rot) * s, math.sin(rot) * s

    def f(p):
        x, y = p[0] - a[0], p[1] - a[1]
        return (A[0] + c * x - n * y, A[1] + n * x + c * y)
    return f, math.degrees(rot), s


def two_point(dr, src, dst, allow_scale=False):
    f, rot, s = similarity(src[0], src[1], dst[0], dst[1], allow_scale)
    dr.transform(f, rot, s)
    dr.notes.append(f"aligned by two points: rotated {rot:.2f} deg" + (f", scaled x{s:.4f}" if allow_scale else ""))
    return dr


def to_origin(dr, corner):
    cx, cy = corner
    dr.transform(lambda p: (p[0] - cx, p[1] - cy))
    dr.notes.append(f"moved ({cx:.0f}, {cy:.0f}) cm to the origin")
    return dr
