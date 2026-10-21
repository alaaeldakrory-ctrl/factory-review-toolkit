"""Floor paths for walking: A* on a clearance grid built from obstacle footprints (benches, racks, machines),
then string-pulled to straight segments with gentle corners.

    nav = NavGrid.from_stage(stage, ["/World/Asset"], skip=["/World/Asset/W50M"])
    pts = nav.path((x0, z0), (x1, z1))
"""
import heapq
import math

import numpy as np
from pxr import Usd, UsdGeom

CELL = 10.0          # cm
CLEAR = 28.0         # body half-width + margin


class NavGrid:
    def __init__(self, rects, bounds=None, clear=CLEAR):
        """rects: obstacle footprints (x0, z0, x1, z1) in world cm."""
        self.rects = [tuple(map(float, r)) for r in rects]
        xs = [v for r in self.rects for v in (r[0], r[2])] or [0.0]
        zs = [v for r in self.rects for v in (r[1], r[3])] or [0.0]
        self.x0, self.z0 = (bounds[0], bounds[1]) if bounds else (min(xs) - 500, min(zs) - 500)
        x1, z1 = (bounds[2], bounds[3]) if bounds else (max(xs) + 500, max(zs) + 500)
        self.nx, self.nz = int((x1 - self.x0) / CELL) + 1, int((z1 - self.z0) / CELL) + 1
        self.block = np.zeros((self.nx, self.nz), bool)
        for (a, b, c, d) in self.rects:
            i0, i1 = self._ix(a - clear), self._ix(c + clear)
            j0, j1 = self._iz(b - clear), self._iz(d + clear)
            self.block[max(i0, 0):min(i1 + 1, self.nx), max(j0, 0):min(j1 + 1, self.nz)] = True

    @classmethod
    def from_stage(cls, stage, roots, skip=(), min_height=20.0, time=Usd.TimeCode.Default()):
        """Obstacles = world bounding boxes of the children of `roots` that stand at least min_height tall."""
        bc = UsdGeom.BBoxCache(time, ["default", "render"])
        rects = []
        for r in roots:
            for c in stage.GetPrimAtPath(r).GetChildren():
                if any(str(c.GetPath()).startswith(s) for s in skip):
                    continue
                if c.GetAttribute("frt:worker") or c.GetName() in ("Targets",):
                    continue
                box = bc.ComputeWorldBound(c).ComputeAlignedRange()
                if box.IsEmpty() or box.GetMax()[1] - box.GetMin()[1] < min_height:
                    continue
                mn, mx = box.GetMin(), box.GetMax()
                rects.append((mn[0], mn[2], mx[0], mx[2]))
        return cls(rects)

    def _ix(self, x):
        return int(math.floor((x - self.x0) / CELL))

    def _iz(self, z):
        return int(math.floor((z - self.z0) / CELL))

    def _xz(self, i, j):
        return (self.x0 + (i + 0.5) * CELL, self.z0 + (j + 0.5) * CELL)

    def free(self, x, z):
        i, j = self._ix(x), self._iz(z)
        return 0 <= i < self.nx and 0 <= j < self.nz and not self.block[i, j]

    def _line_free(self, a, b):
        d = math.dist(a, b)
        n = max(2, int(d / (CELL * 0.5)))
        return all(self.free(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n + 1))

    def _nearest_free(self, p):
        i0, j0 = self._ix(p[0]), self._iz(p[1])
        for r in range(0, 40):
            for i in range(i0 - r, i0 + r + 1):
                for j in (j0 - r, j0 + r) if abs(i - i0) != r else range(j0 - r, j0 + r + 1):
                    if 0 <= i < self.nx and 0 <= j < self.nz and not self.block[i, j]:
                        return i, j
        return i0, j0

    def path(self, a, b):
        """Shortest clear path from a to b (world xz). The end points may sit inside a clearance zone (a worker's
        spot at a bench): they connect to the nearest free cell."""
        a, b = tuple(map(float, a)), tuple(map(float, b))
        if self._line_free(a, b):
            return [a, b]
        s, g = self._nearest_free(a), self._nearest_free(b)
        openq = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414),
                 (-1, -1, 1.414)]
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for di, dj, w in moves:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < self.nx and 0 <= nb[1] < self.nz) or self.block[nb]:
                    continue
                c = cost[cur] + w
                if c < cost.get(nb, 1e18):
                    cost[nb] = c
                    came[nb] = cur
                    heapq.heappush(openq, (c + math.dist(nb, g), nb))
        if g not in came:
            return [a, b]                                       # no route: walk straight (the rules will flag it)
        cells = []
        n = g
        while n is not None:
            cells.append(self._xz(*n))
            n = came[n]
        pts = [a] + cells[::-1][1:-1] + [b]
        out = [pts[0]]                                           # string pulling
        k = 0
        while k < len(pts) - 1:
            j = len(pts) - 1
            while j > k + 1 and not self._line_free(pts[k], pts[j]):
                j -= 1
            out.append(pts[j])
            k = j
        return out
