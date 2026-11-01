"""Procedural violin-shaped mid-surface meshes (top, back, ribs).

This exists for two reasons: (1) so the whole pipeline can be exercised and
validated without CAD data, and (2) so f-hole shape/position can be treated as
*search parameters* (the mesh is regenerated for every candidate).  When you
have real CAD, use ``luthier.mesh.load_parts`` instead and keep everything
else.

Coordinates: x across the body (treble side +x, bass side -x), y along the
body from the bottom button (y=0) to the neck (y=length), z up out of the top
plate.  Back plate edge at z=0, top plate edge at z=rib_height.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Tuple

import numpy as np
import shapely
from scipy.interpolate import splev, splprep
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely import affinity

from .mesh import SurfaceMesh

# Half-outline control points (x, y) in mm, right-hand (treble) side, bottom -> top.
_OUTLINE_MM = np.array([
    (0, 0), (30, 2), (62, 12), (86, 33), (100, 62), (103, 92), (98, 120), (84, 148),
    (66, 168), (57, 183), (58, 198), (68, 216), (79, 238), (84, 262), (82, 286),
    (72, 311), (52, 335), (26, 351), (0, 355),
], float)


@dataclass(frozen=True)
class FHole:
    """Parametric f-hole (the treble-side hole; the bass-side hole is its mirror image)."""
    x: float = 0.034       # centreline offset from the body axis [m]
    y: float = 0.1625      # centre height along the body [m]
    length: float = 0.075  # end-to-end [m]
    width: float = 0.0035  # stem width [m]
    tilt: float = 0.10     # outward lean of the upper end (dx/dy)
    s_amp: float = 0.002   # S-curve amplitude [m]
    eye_top: float = 0.0048
    eye_low: float = 0.0058


@dataclass(frozen=True)
class PlateGeometry:
    length: float = 0.355
    rib_height: float = 0.030
    arch_top: float = 0.0150
    arch_back: float = 0.0150
    arch_exponent: float = 2.2
    mesh_size: float = 0.009
    fhole: FHole = field(default_factory=FHole)


# --------------------------------------------------------------------------- #
def outline_polygon(length: float = 0.355) -> Polygon:
    half = _OUTLINE_MM * 1e-3
    half = half * np.array([1.0, length / 0.355])
    right = half
    left = half[-2:0:-1] * np.array([-1.0, 1.0])
    loop = np.vstack([right, left, right[:1]])
    tck, _ = splprep(loop.T, s=0, per=True)
    u = np.linspace(0, 1, 800, endpoint=False)
    x, y = splev(u, tck)
    return Polygon(np.column_stack([x, y]))


def fhole_polygon(f: FHole, side: int = +1) -> Polygon:
    s = np.linspace(0, 1, 40)
    cx = f.x + f.tilt * (s - 0.5) * f.length + f.s_amp * np.sin(2 * np.pi * s)
    cy = f.y + (s - 0.5) * f.length
    line = LineString(np.column_stack([cx, cy]))
    shape = unary_union([
        line.buffer(f.width / 2, quad_segs=4),
        Point(cx[-1], cy[-1]).buffer(f.eye_top, quad_segs=8),
        Point(cx[0], cy[0]).buffer(f.eye_low, quad_segs=8),
    ])
    if side < 0:
        shape = affinity.scale(shape, xfact=-1, yfact=1, origin=(0, 0))
    return shape


def _resample(ring: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(8, int(round(s[-1] / spacing)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, ring[:, 0]), np.interp(t, s, ring[:, 1])])


def _smooth_profile(d: np.ndarray, dmax: float, p: float) -> np.ndarray:
    t = np.clip(d / dmax, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** p


@dataclass
class Plates:
    outline: Polygon
    holes: Tuple[Polygon, Polygon]
    top: SurfaceMesh
    back: SurfaceMesh
    ribs: SurfaceMesh
    boundary_xy: np.ndarray
    dmax: float
    geom: PlateGeometry

    def dist_to_edge(self, xy: np.ndarray) -> np.ndarray:
        return shapely.distance(self.outline.exterior, shapely.points(np.asarray(xy)[:, :2]))

    def top_z(self, xy):
        d = self.dist_to_edge(xy)
        return self.geom.rib_height + self.geom.arch_top * _smooth_profile(d, self.dmax, self.geom.arch_exponent)

    def back_z(self, xy):
        d = self.dist_to_edge(xy)
        return -self.geom.arch_back * _smooth_profile(d, self.dmax, self.geom.arch_exponent)


def _planar_mesh(domain, boundary_xy, hole_rings, h):
    """Delaunay mesh of a polygon-with-holes using given boundary points."""
    inner = domain.buffer(-0.55 * h)
    pts = [boundary_xy] + [hr for hr in hole_rings]
    minx, miny, maxx, maxy = domain.bounds
    dy = h * np.sqrt(3) / 2
    ys = np.arange(miny, maxy + dy, dy)
    lattice = []
    for j, y in enumerate(ys):
        xs = np.arange(minx - h, maxx + h, h) + (0.5 * h if j % 2 else 0.0)
        lattice.append(np.column_stack([xs, np.full_like(xs, y)]))
    lattice = np.vstack(lattice)
    keep = shapely.contains(inner, shapely.points(lattice))
    pts.append(lattice[keep])
    P = np.vstack(pts)
    tri = Delaunay(P).simplices
    cen = P[tri].mean(1)
    inside = shapely.contains(domain, shapely.points(cen))
    tri = tri[inside]
    a = P[tri]
    def cross2(a):
        u, v = a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    tri = tri[0.5 * np.abs(cross2(a)) > 0.02 * h * h]
    # orient counter-clockwise (normal +z)
    cr = cross2(P[tri])
    flip = cr < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return P, tri


@lru_cache(maxsize=32)
def build_plates(geom: PlateGeometry) -> Plates:
    h = geom.mesh_size
    outline = outline_polygon(geom.length)
    holes = (fhole_polygon(geom.fhole, +1), fhole_polygon(geom.fhole, -1))
    bnd = _resample(np.asarray(outline.exterior.coords)[:-1], h)
    bnd = np.vstack([bnd])
    domain = Polygon(outline.exterior.coords, [hl.exterior.coords for hl in holes])
    hole_rings = [_resample(np.asarray(hl.exterior.coords)[:-1], 0.45 * h) for hl in holes]
    P, tri = _planar_mesh(domain, bnd, hole_rings, h)

    # unused vertices (e.g. lattice points swallowed by holes) are dropped
    used = np.unique(tri)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, tri = P[used], remap[tri]

    dist_all = shapely.distance(outline.exterior, shapely.points(P))
    dmax = float(dist_all.max())
    pl = Plates(outline, holes, None, None, None, bnd, dmax, geom)
    ztop = geom.rib_height + geom.arch_top * _smooth_profile(dist_all, dmax, geom.arch_exponent)
    zback = -geom.arch_back * _smooth_profile(dist_all, dmax, geom.arch_exponent)
    top = SurfaceMesh(np.column_stack([P, ztop]), tri)
    back = SurfaceMesh(np.column_stack([P, zback]), tri[:, [0, 2, 1]])  # normal points down (-z)

    # ribs: ring of vertical quads between z=0 and z=rib_height
    nz = max(2, int(round(geom.rib_height / h)))
    zs = np.linspace(0.0, geom.rib_height, nz + 1)
    nb = len(bnd)
    V = np.vstack([np.column_stack([bnd, np.full(nb, z)]) for z in zs])
    F = []
    for k in range(nz):
        for i in range(nb):
            a, b = k * nb + i, k * nb + (i + 1) % nb
            c, d = (k + 1) * nb + (i + 1) % nb, (k + 1) * nb + i
            F += [(a, b, c), (a, c, d)]
    ribs = SurfaceMesh(V, np.array(F))
    pl.top, pl.back, pl.ribs = top, back, ribs
    return pl


def outline_area(geom: PlateGeometry = PlateGeometry()) -> float:
    return outline_polygon(geom.length).area


def plot_plan(plates: Plates, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    fig, ax = plt.subplots(figsize=(5, 8))
    t = plates.top
    ax.triplot(mtri.Triangulation(t.vertices[:, 0], t.vertices[:, 1], t.faces), lw=0.3, color="0.3")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=130)
    plt.close(fig)
