"""Tetrahedral air-volume meshes and acoustic openings.

The 3-D wind-instrument model works on a mesh of the *air* inside the
instrument (bore + tone-hole chimneys + window).  Wherever that air meets the
outside world there is an **opening** (foot end, tone-hole outer face, window).
Openings are recognised *geometrically* after meshing -- a boundary triangle
belongs to an opening if it lies in the opening's plane and inside its outline
-- so the same machinery works for meshes built here (gmsh/OpenCASCADE) and for
air-volume meshes you import from CAD.

A finger closing a hole simply turns that opening into rigid wall, so every
fingering shares one mesh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ellipk


@dataclass(frozen=True)
class OpeningSpec:
    """Planar opening where the air meets the outside.

    ``center`` / ``normal`` (outward) define the plane; ``axis1`` is the in-plane
    direction of semi-axis ``half_extents[0]`` (the other semi-axis is
    ``half_extents[1]``).  ``shape`` is "ellipse" or "rect".

    ``end_corr`` is the radiation end correction in units of the equivalent-circle
    radius: 0.6133 = unflanged pipe, 0.8216 = flanged (Levine-Schwinger, ka << 1).
    A tone hole in a cylindrical wall sits between the two (default 0.7).
    """
    name: str
    center: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    axis1: Tuple[float, float, float]
    half_extents: Tuple[float, float]
    shape: str = "ellipse"
    end_corr: float = 0.7
    kind: str = "hole"


@dataclass
class Opening:
    spec: OpeningSpec
    faces: np.ndarray          # indices into AirMesh.tris
    area: float                # *true* (analytic) aperture area [m^2]
    a_eq: float                # equivalent-circle radius sqrt(S/pi)
    shape_factor: float        # end-correction scale for non-circular outlines (1 for a circle)
    mesh_area: float = 0.0     # area of the meshed faces (slightly below `area` for curved outlines)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def robin_scale(self) -> float:
        """Correction so the discrete radiating area equals the true aperture area."""
        return self.area / self.mesh_area if self.mesh_area > 0 else 1.0

    @property
    def end_length(self) -> float:
        """Acoustic end-correction length l [m] of this opening."""
        return self.spec.end_corr * self.a_eq * self.shape_factor


def ellipse_shape_factor(area: float, semi_major: float) -> float:
    """Ratio of the end correction of an elliptical aperture to that of the equal-area circle,
    from Rayleigh conductances (circle: 2a, ellipse: pi*a_maj/K(e))."""
    a_eq = np.sqrt(area / np.pi)
    a = max(semi_major, a_eq)
    b = area / (np.pi * a)
    m = max(0.0, 1.0 - (b / a) ** 2)
    return float(2 * a_eq * ellipk(m) / (np.pi * a))


@dataclass
class AirMesh:
    nodes: np.ndarray                  # (N,3) m
    tets: np.ndarray                   # (M,4)
    tris: np.ndarray                   # (T,3) boundary triangles, outward oriented
    tri_tet: np.ndarray                # (T,) adjacent tet of each boundary triangle
    openings: Dict[str, Opening] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_tets(cls, nodes: np.ndarray, tets: np.ndarray, specs: Sequence[OpeningSpec] = (), tol: float = 2e-4) -> "AirMesh":
        nodes = np.asarray(nodes, float)
        tets = np.asarray(tets, int)
        # fix orientation (positive volume)
        P = nodes[tets]
        vol = np.einsum("mi,mi->m", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])
        flip = vol < 0
        tets = tets.copy()
        tets[flip] = tets[flip][:, [0, 2, 1, 3]]
        # boundary = faces that appear once
        local = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])  # face opposite vertex i, outward for positive tets
        faces = tets[:, local].reshape(-1, 3)
        owner = np.repeat(np.arange(len(tets)), 4)
        key = np.sort(faces, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        bnd = cnt[inv] == 1
        tris, tri_tet = faces[bnd], owner[bnd]
        mesh = cls(nodes, tets, tris, tri_tet)
        mesh.find_openings(specs, tol)
        return mesh

    # ------------------------------------------------------------------ #
    def tri_geometry(self):
        P = self.nodes[self.tris]
        cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        A = 0.5 * np.linalg.norm(cr, axis=1)
        return A, cr / (2 * A[:, None]), P.mean(1)

    def find_openings(self, specs: Sequence[OpeningSpec], tol: float = 2e-4) -> None:
        A, nrm, cen = self.tri_geometry()
        self.openings = {}
        for s in specs:
            n = np.asarray(s.normal, float)
            n /= np.linalg.norm(n)
            u = np.asarray(s.axis1, float)
            u = u - u.dot(n) * n
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            d = cen - np.asarray(s.center)
            h, x, y = d @ n, d @ u, d @ v
            ra, rb = s.half_extents
            inside = (np.abs(h) < tol) & (nrm @ n > 0.99)
            if s.shape == "ellipse":
                inside &= (x / ra) ** 2 + (y / rb) ** 2 <= 1.0 + 0.06
            else:
                inside &= (np.abs(x) <= ra + tol) & (np.abs(y) <= rb + tol)
            idx = np.where(inside)[0]
            if len(idx) == 0:
                raise ValueError(f"opening {s.name!r}: no boundary faces found in its plane (check center/normal/size)")
            S_mesh = float(A[idx].sum())
            S = float(np.pi * ra * rb) if s.shape == "ellipse" else float(4 * ra * rb)
            if abs(S_mesh / S - 1) > 0.25:
                raise ValueError(f"opening {s.name!r}: meshed area {S_mesh:.3e} differs from the nominal {S:.3e} by >25% "
                                 "(refine the mesh, or the opening outline does not match the mesh)")
            self.openings[s.name] = Opening(s, idx, S, float(np.sqrt(S / np.pi)),
                                            ellipse_shape_factor(S, max(ra, rb)) if s.shape == "ellipse" else
                                            ellipse_shape_factor(S, 0.5 * np.hypot(ra, rb)), S_mesh)

    # ------------------------------------------------------------------ #
    @property
    def volume(self) -> float:
        P = self.nodes[self.tets]
        return float(np.sum(np.abs(np.einsum("mi,mi->m", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])) / 6))

    def wall_faces(self, open_names: Sequence[str]) -> np.ndarray:
        """Boundary faces that are rigid wall for the given set of *open* openings."""
        is_open = np.zeros(len(self.tris), bool)
        for n in open_names:
            is_open[self.openings[n].faces] = True
        return np.where(~is_open)[0]


# --------------------------------------------------------------------------- #
def mesh_from_gmsh(specs: Sequence[OpeningSpec] = ()) -> AirMesh:
    """Extract the current gmsh model's tetrahedra into an :class:`AirMesh`."""
    import gmsh

    tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags = tags[order]
    X = np.asarray(coords).reshape(-1, 3)[order]
    remap = {int(t): i for i, t in enumerate(tags)}
    types, _, enodes = gmsh.model.mesh.getElements(3)
    tets = np.concatenate([np.asarray(en).reshape(-1, 4) for t, en in zip(types, enodes) if t == 4])
    lut = np.full(int(tags.max()) + 1, -1)
    lut[tags.astype(int)] = np.arange(len(tags))
    tets = lut[tets]
    used = np.unique(tets)
    re = -np.ones(len(X), int)
    re[used] = np.arange(len(used))
    return AirMesh.from_tets(X[used], re[tets], specs)
