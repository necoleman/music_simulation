"""Finite-element core: flat-facet orthotropic shells, 3-D beams, links, masses.

Why shells?  Violin plates are 2.5-4 mm thick over a ~350 mm body.  Solid
tetrahedra lock badly in such thin parts and need prohibitive element counts;
a shell model on the *mid-surface* mesh captures arching, thickness grading and
orthotropy at a few thousand elements, which is what makes searching over
design parameters affordable.

Element formulations
--------------------
* membrane : constant-strain triangle (CST), orthotropic plane stress
* bending  : Mindlin-Reissner with *discrete shear gap* shear strains (DSG3,
             Bletzinger et al. 2000) -- locking-free, thickness-robust
* drilling : small penalty tying the nodal drilling rotation to the element's
             mean in-plane rotation (Hughes-Brezzi style), so that all six
             nodal dofs are well-posed at folds such as plate/rib junctions
* beams    : 3-D Euler-Bernoulli with rigid end offsets (bass bar sits *on*
             the plate, not in its mid-plane)
* links    : pin-ended truss with optional lateral friction (sound post)
* mass     : lumped (diagonal) translational + rotary inertia

Units are SI.  Node dofs are ordered (ux, uy, uz, rx, ry, rz).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .materials import Orthotropic

ArrayLike = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]
SHEAR_CORRECTION = 5.0 / 6.0
DRILL_ALPHA = 0.1


# --------------------------------------------------------------------------- #
# Beam sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Section:
    """Cross-section: area A, second moments Iy (bending that deflects local z),
    Iz (deflects local y) and torsion constant J."""

    A: float
    Iy: float
    Iz: float
    J: float

    @classmethod
    def rect(cls, width: float, thickness: float) -> "Section":
        """``width`` along local y, ``thickness`` along local z (the 'up' direction)."""
        b, t = width, thickness
        a, c = max(b, t), min(b, t)
        J = a * c ** 3 * (1 / 3 - 0.21 * (c / a) * (1 - (c / a) ** 4 / 12))
        return cls(b * t, b * t ** 3 / 12, t * b ** 3 / 12, J)

    @classmethod
    def circle(cls, diameter: float) -> "Section":
        r = diameter / 2
        I = np.pi * r ** 4 / 4
        return cls(np.pi * r ** 2, I, I, 2 * I)


@dataclass(frozen=True)
class End:
    """Attachment of a beam/link end to the model.

    The end sits at ``sum_i w_i x_i + offset`` and moves rigidly with the
    weighted nodal motion.  Use several nodes + barycentric weights to attach
    *anywhere* on a shell (so design parameters such as post position vary
    smoothly instead of snapping to mesh nodes).
    """

    nodes: tuple
    weights: tuple
    offset: tuple = (0.0, 0.0, 0.0)


def as_end(x) -> End:
    return x if isinstance(x, End) else End((int(x),), (1.0,))


# --------------------------------------------------------------------------- #
# Shell element kernels (vectorised over elements)
# --------------------------------------------------------------------------- #
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


def shell_frames(P: np.ndarray):
    """Element frames and local 2-D coordinates.  P: (M,3,3) node coordinates."""
    e1 = _unit(P[:, 1] - P[:, 0])
    n = _unit(np.cross(e1, P[:, 2] - P[:, 0]))
    e2 = np.cross(n, e1)
    R = np.stack([e1, e2, n], axis=1)  # rows are local axes in global coords
    loc = np.einsum("mij,mkj->mki", R, P - P[:, :1])
    return R, loc


def _rotated_stiffness(mat: Orthotropic, phi: np.ndarray):
    """Plane-stress (3x3) and shear (2x2) stiffness rotated into element axes."""
    c, s = np.cos(phi), np.sin(phi)
    one, zero = np.ones_like(c), np.zeros_like(c)
    Te = np.stack(  # engineering-strain transform: local -> material
        [
            np.stack([c * c, s * s, c * s], -1),
            np.stack([s * s, c * c, -c * s], -1),
            np.stack([-2 * c * s, 2 * c * s, c * c - s * s], -1),
        ],
        -2,
    )
    Ts = np.stack(  # stress transform: local -> material
        [
            np.stack([c * c, s * s, 2 * c * s], -1),
            np.stack([s * s, c * c, -2 * c * s], -1),
            np.stack([-c * s, c * s, c * c - s * s], -1),
        ],
        -2,
    )
    Q = np.linalg.inv(Ts) @ mat.plane_stress() @ Te
    Rm = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    G = Rm @ mat.shear() @ np.swapaxes(Rm, -1, -2)
    return Q, G


def shell_matrices(P, h, mat: Orthotropic, grain):
    """Global 18x18 stiffness and lumped masses for each triangle.

    Parameters
    ----------
    P     : (M,3,3) node coordinates
    h     : (M,) thickness
    mat   : material (single material per call)
    grain : (M,3) grain direction (any length; projected onto the element)
    """
    M = P.shape[0]
    R, loc = shell_frames(P)
    x, y = loc[..., 0], loc[..., 1]
    A = 0.5 * ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / (2 * A[:, None])
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / (2 * A[:, None])

    gx = np.einsum("mi,mi->m", grain, R[:, 0])
    gy = np.einsum("mi,mi->m", grain, R[:, 1])
    phi = np.where(np.hypot(gx, gy) > 1e-9, np.arctan2(gy, gx), 0.0)
    Q, Gs = _rotated_stiffness(mat, phi)

    # --- membrane (u,v) -----------------------------------------------------
    Bm = np.zeros((M, 3, 18))
    for i in range(3):
        Bm[:, 0, 6 * i] = b[:, i]
        Bm[:, 1, 6 * i + 1] = c[:, i]
        Bm[:, 2, 6 * i] = c[:, i]
        Bm[:, 2, 6 * i + 1] = b[:, i]
    # --- bending (w, bx = ry, by = -rx) --------------------------------------
    Bb = np.zeros((M, 3, 18))
    for i in range(3):
        Bb[:, 0, 6 * i + 4] = b[:, i]
        Bb[:, 1, 6 * i + 3] = -c[:, i]
        Bb[:, 2, 6 * i + 4] = c[:, i]
        Bb[:, 2, 6 * i + 3] = -b[:, i]
    # --- shear via discrete shear gap -----------------------------------------
    gap = np.zeros((M, 2, 18))
    for k, j in enumerate((1, 2)):
        dx, dy = x[:, j] - x[:, 0], y[:, j] - y[:, 0]
        gap[:, k, 6 * j + 2] += 1.0
        gap[:, k, 2] -= 1.0
        for i in (0, j):
            gap[:, k, 6 * i + 4] += 0.5 * dx  # bx = ry
            gap[:, k, 6 * i + 3] -= 0.5 * dy  # by = -rx
    Bs = np.zeros((M, 2, 18))
    Bs[:, 0] = b[:, 1, None] * gap[:, 0] + b[:, 2, None] * gap[:, 1]
    Bs[:, 1] = c[:, 1, None] * gap[:, 0] + c[:, 2, None] * gap[:, 1]

    hh = h[:, None, None]
    AA = A[:, None, None]
    tr = np.swapaxes
    K = AA * hh * (tr(Bm, 1, 2) @ Q @ Bm)
    K += AA * hh ** 3 / 12.0 * (tr(Bb, 1, 2) @ Q @ Bb)
    K += AA * SHEAR_CORRECTION * hh * (tr(Bs, 1, 2) @ Gs @ Bs)

    # --- drilling penalty -----------------------------------------------------
    t = np.zeros((M, 18))
    for j in range(3):
        t[:, 6 * j + 1] += 0.5 * b[:, j]
        t[:, 6 * j] -= 0.5 * c[:, j]
    kd = DRILL_ALPHA * mat.G_LC * h * A / 3.0
    for i in range(3):
        ci = -t.copy()
        ci[:, 6 * i + 5] += 1.0
        K += kd[:, None, None] * (ci[:, :, None] * ci[:, None, :])

    # --- local -> global --------------------------------------------------------
    T = np.zeros((M, 18, 18))
    for i in range(6):
        T[:, 3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = R
    Kg = tr(T, 1, 2) @ K @ T

    m_node = mat.rho * h * A / 3.0
    I_node = mat.rho * h ** 3 * A / 36.0 + 1e-3 * m_node * A  # tiny floor keeps drilling dofs high-frequency
    return Kg, m_node, I_node, A


# --------------------------------------------------------------------------- #
# Beam / link kernels
# --------------------------------------------------------------------------- #
def beam_local(E, G, sec: Section, L):
    A, Iy, Iz, J = sec.A, sec.Iy, sec.Iz, sec.J
    K = np.zeros((12, 12))
    a = E * A / L
    K[np.ix_([0, 6], [0, 6])] = a * np.array([[1, -1], [-1, 1]])
    t = G * J / L
    K[np.ix_([3, 9], [3, 9])] = t * np.array([[1, -1], [-1, 1]])
    k = E * Iz / L ** 3
    K[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = k * np.array(
        [[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
         [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
    k = E * Iy / L ** 3
    K[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = k * np.array(
        [[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L * L, 6 * L, 2 * L * L],
         [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L * L, 6 * L, 4 * L * L]])
    return K


def skew(o):
    return np.array([[0, -o[2], o[1]], [o[2], 0, -o[0]], [-o[1], o[0], 0]])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class _Part:
    name: str
    tris: np.ndarray  # (m,3) global node ids
    thickness: np.ndarray  # (m,)
    material: Orthotropic
    grain: np.ndarray  # (m,3)


class Model:
    """Container that accumulates geometry and produces sparse K, M."""

    def __init__(self, merge_tol: float = 2e-4):
        self.tol = merge_tol
        self.nodes = np.zeros((0, 3))
        self.parts: dict[str, _Part] = {}
        self._beams: list[dict] = []
        self._links: list[dict] = []
        self._masses: list[tuple[int, float, float]] = []
        self._terms: list[tuple] = []
        self._parts_cache: dict = {}
        self._extra_mass: list[float] = []

    # ------------------------------------------------------------------ nodes
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def add_free_node(self, xyz) -> int:
        self.nodes = np.vstack([self.nodes, np.asarray(xyz, float)[None]])
        return self.n_nodes - 1

    def _merge_points(self, pts: np.ndarray) -> np.ndarray:
        n0 = len(self.nodes)
        allp = np.vstack([self.nodes, pts]) if n0 else pts
        pairs = cKDTree(allp).query_pairs(self.tol, output_type="ndarray")
        N = len(allp)
        g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(N, N)) if len(pairs) else sp.coo_matrix((N, N))
        _, lab = connected_components(g, directed=False)
        first = np.full(lab.max() + 1, N, dtype=int)
        np.minimum.at(first, lab, np.arange(N))
        rep = first[lab]  # representative index (smallest) of each point
        new_rep = np.unique(rep[n0:][rep[n0:] >= n0])
        newid = {int(r): n0 + k for k, r in enumerate(new_rep)}
        mapped = np.array([r if r < n0 else newid[int(r)] for r in rep[n0:]], dtype=int)
        if len(new_rep):
            self.nodes = np.vstack([self.nodes, allp[new_rep]])
        return mapped

    def nearest_node(self, xyz, among: Optional[Sequence[int]] = None) -> int:
        idx = np.arange(self.n_nodes) if among is None else np.asarray(among)
        d = np.linalg.norm(self.nodes[idx] - np.asarray(xyz), axis=1)
        return int(idx[np.argmin(d)])

    def part_nodes(self, name: str) -> np.ndarray:
        return np.unique(self.parts[name].tris)

    # ----------------------------------------------------------------- shells
    def add_shell_part(self, name, vertices, faces, thickness: ArrayLike, material: Orthotropic,
                       grain: Union[Sequence[float], Callable] = (0.0, 1.0, 0.0)) -> None:
        """Add a triangulated mid-surface.

        ``thickness`` : float, per-face array, or callable(face_centroids)->array.
        ``grain``     : 3-vector, or callable(centroids, normals)->(m,3) (use this for
                        ribs, whose grain follows the outline).
        """
        vertices = np.asarray(vertices, float)
        faces = np.asarray(faces, int)
        ids = self._merge_points(vertices)
        tris = ids[faces]
        cen = vertices[faces].mean(1)
        if callable(thickness):
            h = np.asarray(thickness(cen), float)
        else:
            h = np.broadcast_to(np.asarray(thickness, float), (len(faces),)).copy()
        if callable(grain):
            nrm = _unit(np.cross(vertices[faces][:, 1] - vertices[faces][:, 0], vertices[faces][:, 2] - vertices[faces][:, 0]))
            g = np.asarray(grain(cen, nrm), float)
        else:
            g = np.broadcast_to(np.asarray(grain, float), (len(faces), 3)).copy()
        if name in self.parts:
            raise ValueError(f"part {name!r} already exists")
        self.parts[name] = _Part(name, tris, h, material, g)

    # ------------------------------------------------------- discrete members
    def add_beam(self, end1, end2, section: Section, material: Orthotropic, *, up=(0, 0, 1)) -> None:
        """Euler-Bernoulli beam between two ends (node ids or :class:`End`).
        ``up`` fixes the section's local z (thickness) direction."""
        self._beams.append(dict(e1=as_end(end1), e2=as_end(end2), sec=section, mat=material, up=np.asarray(up, float)))

    def add_link(self, end1, end2, EA_over_L: float, mass: float = 0.0) -> None:
        """Pin-ended axial truss (translations only).

        Deliberately has *no* lateral stiffness: a constant lateral spring is not
        invariant under rigid rotation and would give the free body spurious
        rotational resonances.
        """
        self._links.append(dict(e1=as_end(end1), e2=as_end(end2), k=EA_over_L, kl=0.0, m=mass))

    def add_point_mass(self, where, mass: float, rotary: float = 0.0) -> None:
        e = as_end(where)
        for n, w in zip(e.nodes, e.weights):
            self._masses.append((n, mass * w, rotary * w))

    def end_position(self, end) -> np.ndarray:
        e = as_end(end)
        return np.einsum("i,ij->j", np.asarray(e.weights), self.nodes[list(e.nodes)]) + np.asarray(e.offset)

    # ------------------------------------------------------------- locate
    def locate(self, part: str, xy) -> tuple:
        """Find the triangle of ``part`` under plan position ``xy``.

        Returns ``(nodes(3), weights(3), unit_normal, point)``.  Points outside the
        part (inside an f-hole, say) are clamped to the nearest triangle.
        """
        pt = self._parts_cache.get(part)
        if pt is None:
            tris = self.parts[part].tris
            cen = self.nodes[tris].mean(1)
            rad = np.linalg.norm(self.nodes[tris] - cen[:, None], axis=2).max()
            pt = (cKDTree(cen[:, :2]), cen, rad)
            self._parts_cache[part] = pt
        tree, cen, rad = pt
        tris = self.parts[part].tris
        xy = np.asarray(xy, float)[:2]
        cand = tree.query_ball_point(xy, 1.5 * rad)
        if not cand:
            cand = [int(tree.query(xy)[1])]
        best, bw, bmin = None, None, -1e30
        for k in cand:
            a, b, c = self.nodes[tris[k]][:, :2]
            T = np.array([[a[0] - c[0], b[0] - c[0]], [a[1] - c[1], b[1] - c[1]]])
            try:
                l = np.linalg.solve(T, xy - c)
            except np.linalg.LinAlgError:
                continue
            w = np.array([l[0], l[1], 1 - l.sum()])
            if w.min() > bmin:
                best, bw, bmin = k, w, w.min()
        w = np.clip(bw, 0, None)
        w /= w.sum()
        P = self.nodes[tris[best]]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        n /= np.linalg.norm(n)
        return tuple(int(i) for i in tris[best]), tuple(w), n, w @ P

    def node_normals(self, part: str) -> dict:
        """Area-weighted vertex normals of a part (dict node -> unit vector)."""
        acc: dict = {}
        tris = self.parts[part].tris
        P = self.nodes[tris]
        fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        for t, n in zip(tris, fn):
            for i in t:
                acc[int(i)] = acc.get(int(i), 0) + n
        return {k: v / np.linalg.norm(v) for k, v in acc.items()}

    # ----------------------------------------------------------- extra dofs
    def add_extra_dof(self, mass: float) -> tuple:
        self._extra_mass.append(mass)
        return ("x", len(self._extra_mass) - 1)

    def add_cavity(self, dofs, b, volume: float, bulk_modulus: float, air_mass: float) -> dict:
        """Lumped (uniform-pressure) air cavity with a Helmholtz plug.

        Mixed formulation so K stays sparse:  extra dofs ``p`` (cavity pressure,
        massless) and ``q`` (volume of air displaced through the openings, mass
        ``air_mass`` = rho_air / conductance).  ``b`` holds the outward-area
        weights of the wall dofs, i.e. a unit outward wall displacement changes
        the cavity volume by ``b . u``.
            M u'' + K u - b p = 0 ;  air_mass q'' + p = 0 ;  (V/kappa) p + b.u - q = 0
        """
        p = self.add_extra_dof(0.0)
        q = self.add_extra_dof(air_mass)
        for d, bv in zip(dofs, b):
            self.add_stiffness_entry(int(d), p, -float(bv))
        self.add_stiffness_entry(p, p, -volume / bulk_modulus)
        self.add_stiffness_entry(q, p, 1.0)
        self.cavity = dict(p=p, q=q, air_mass=air_mass, volume=volume)
        return self.cavity

    def add_stiffness_entry(self, i, j, value: float) -> None:
        """Add ``value`` to K[i,j] and K[j,i] (i, j: dof ints or ("x", k) extra dofs)."""
        self._terms.append((i, j, float(value)))

    # --------------------------------------------------------------- assemble
    @property
    def n_dof(self) -> int:
        return 6 * self.n_nodes + len(self._extra_mass)

    def dof(self, node: int, comp: int) -> int:
        return 6 * node + comp

    def _resolve(self, d) -> int:
        return 6 * self.n_nodes + d[1] if isinstance(d, tuple) else int(d)

    def assemble(self):
        N = self.n_dof
        rows, cols, vals = [], [], []
        mdiag = np.zeros(N)
        ar = np.arange(6)

        for part in self.parts.values():
            P = self.nodes[part.tris]
            Kg, m, I, A = shell_matrices(P, part.thickness, part.material, part.grain)
            if np.any(A <= 0):
                raise ValueError(f"part {part.name!r} has degenerate triangles")
            idx = (6 * part.tris[:, :, None] + ar).reshape(len(part.tris), 18)
            rows.append(np.repeat(idx, 18, axis=1).ravel())
            cols.append(np.tile(idx, (1, 18)).ravel())
            vals.append(Kg.ravel())
            for i in range(3):
                for k in range(3):
                    np.add.at(mdiag, 6 * part.tris[:, i] + k, m)
                    np.add.at(mdiag, 6 * part.tris[:, i] + 3 + k, I)

        for bm in self._beams:
            e1, e2 = bm["e1"], bm["e2"]
            p1, p2 = self.end_position(e1), self.end_position(e2)
            L = np.linalg.norm(p2 - p1)
            ex = (p2 - p1) / L
            ez = bm["up"] - bm["up"].dot(ex) * ex
            ez /= np.linalg.norm(ez)
            ey = np.cross(ez, ex)
            mat, sec = bm["mat"], bm["sec"]
            Kl = beam_local(mat.E_L, mat.G_Lz, sec, L)
            Tb = np.kron(np.eye(4), np.vstack([ex, ey, ez]))
            Kb = Tb.T @ Kl @ Tb
            nodes_all = list(e1.nodes) + list(e2.nodes)
            Tn = np.zeros((12, 6 * len(nodes_all)))
            col = 0
            for side, e in enumerate((e1, e2)):
                for n, w in zip(e.nodes, e.weights):
                    blk = w * np.eye(6)
                    blk[0:3, 3:6] = -w * skew(np.asarray(e.offset))
                    Tn[6 * side : 6 * side + 6, col : col + 6] = blk
                    col += 6
            Ke = Tn.T @ Kb @ Tn
            idx = np.concatenate([6 * n + ar for n in nodes_all])
            rows.append(np.repeat(idx, len(idx))); cols.append(np.tile(idx, len(idx))); vals.append(Ke.ravel())
            mb = mat.rho * sec.A * L / 2
            Ib = mat.rho * sec.A * L ** 3 / 24 + mat.rho * L * (sec.Iy + sec.Iz) / 2
            for e in (e1, e2):
                for n, w in zip(e.nodes, e.weights):
                    mdiag[6 * n : 6 * n + 3] += mb * w
                    mdiag[6 * n + 3 : 6 * n + 6] += Ib * w

        for lk in self._links:
            e1, e2 = lk["e1"], lk["e2"]
            d = self.end_position(e2) - self.end_position(e1)
            ex = d / np.linalg.norm(d)
            Kt = lk["k"] * np.outer(ex, ex) + lk["kl"] * (np.eye(3) - np.outer(ex, ex))
            Ke6 = np.block([[Kt, -Kt], [-Kt, Kt]])
            nodes_all = list(e1.nodes) + list(e2.nodes)
            Tn = np.zeros((6, 6 * len(nodes_all)))
            col = 0
            for side, e in enumerate((e1, e2)):
                for n, w in zip(e.nodes, e.weights):
                    Tn[3 * side : 3 * side + 3, col : col + 3] = w * np.eye(3)
                    Tn[3 * side : 3 * side + 3, col + 3 : col + 6] = -w * skew(np.asarray(e.offset))  # u + theta x o
                    col += 6
            Ke = Tn.T @ Ke6 @ Tn
            idx = np.concatenate([6 * n + ar for n in nodes_all])
            rows.append(np.repeat(idx, len(idx))); cols.append(np.tile(idx, len(idx))); vals.append(Ke.ravel())
            for e in (e1, e2):
                for n, w in zip(e.nodes, e.weights):
                    mdiag[6 * n : 6 * n + 3] += 0.5 * lk["m"] * w

        for node, m, I in self._masses:
            mdiag[6 * node : 6 * node + 3] += m
            mdiag[6 * node + 3 : 6 * node + 6] += I
        for j, m in enumerate(self._extra_mass):
            mdiag[6 * self.n_nodes + j] += m

        for i, j, v in self._terms:
            i, j = self._resolve(i), self._resolve(j)
            rows.append(np.array([i])); cols.append(np.array([j])); vals.append(np.array([v]))
            if i != j:
                rows.append(np.array([j])); cols.append(np.array([i])); vals.append(np.array([v]))

        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsc()
        K = 0.5 * (K + K.T)
        # nodes that carry no element would give singular M/K rows -> pin them
        M = sp.diags(mdiag).tocsc()
        return K, M


# --------------------------------------------------------------------------- #
# Modal analysis
# --------------------------------------------------------------------------- #
@dataclass
class Modes:
    model: Model
    freqs: np.ndarray  # Hz, ascending (includes rigid-body modes ~0 Hz when free)
    shapes: np.ndarray  # (n_dof, n_modes), mass-normalised
    zeta: np.ndarray  # modal damping ratios

    def flexible(self, fmin: float = 5.0) -> np.ndarray:
        return np.where(self.freqs > fmin)[0]

    def admittance(self, out_dof: int, in_dof: int, freqs: np.ndarray, fmin: float = 5.0) -> np.ndarray:
        """Velocity-per-force frequency response (mobility), complex, e^{+j w t} convention."""
        sel = self.flexible(fmin)
        w = 2 * np.pi * np.asarray(freqs)[:, None]
        wn = 2 * np.pi * self.freqs[sel][None]
        z = self.zeta[sel][None]
        num = self.shapes[out_dof, sel][None] * self.shapes[in_dof, sel][None]
        return np.sum(1j * w * num / (wn ** 2 - w ** 2 + 2j * z * wn * w), axis=1)

    def admittance_vec(self, out_vec: np.ndarray, in_vec: np.ndarray, freqs: np.ndarray, fmin: float = 5.0) -> np.ndarray:
        """Mobility between arbitrary dof-space vectors (e.g. a force direction at a node)."""
        sel = self.flexible(fmin)
        w = 2 * np.pi * np.asarray(freqs)[:, None]
        wn = 2 * np.pi * self.freqs[sel][None]
        z = self.zeta[sel][None]
        num = (out_vec @ self.shapes[:, sel])[None] * (in_vec @ self.shapes[:, sel])[None]
        return np.sum(1j * w * num / (wn ** 2 - w ** 2 + 2j * z * wn * w), axis=1)

    def node_vector(self, node: int, direction) -> np.ndarray:
        v = np.zeros(self.shapes.shape[0])
        v[6 * node : 6 * node + 3] = np.asarray(direction, float) / np.linalg.norm(direction)
        return v


def damping_from(zeta: Union[float, Callable[[np.ndarray], np.ndarray]], freqs: np.ndarray) -> np.ndarray:
    return np.asarray(zeta(freqs), float) if callable(zeta) else np.full(len(freqs), float(zeta))


def rigid_body_basis(model: Model, M: sp.spmatrix, ii: np.ndarray) -> np.ndarray:
    """M-orthonormal basis of the six rigid-body motions (restricted to dofs ``ii``)."""
    N = model.n_dof
    nn = model.n_nodes
    X = model.nodes
    R = np.zeros((N, 6))
    for k in range(3):
        R[k : 6 * nn : 6, k] = 1.0
    for k in range(3):  # rotation about axis k: translation = e_k x r, rotation dof = e_k
        e = np.zeros(3)
        e[k] = 1.0
        u = np.cross(e, X)
        R[0 : 6 * nn : 6, 3 + k] = u[:, 0]
        R[1 : 6 * nn : 6, 3 + k] = u[:, 1]
        R[2 : 6 * nn : 6, 3 + k] = u[:, 2]
        R[3 + k : 6 * nn : 6, 3 + k] = 1.0
    R = R[ii]
    Mf = M[ii][:, ii]
    Q = []
    for j in range(6):  # Gram-Schmidt in the M inner product
        v = R[:, j].copy()
        for q in Q:
            v -= q * (q @ (Mf @ v))
        v /= np.sqrt(v @ (Mf @ v))
        Q.append(v)
    return np.column_stack(Q)


def modal_analysis(model: Model, n_modes: int = 40, *, fixed: Optional[Sequence[int]] = None,
                   zeta: Union[float, Callable] = 0.015, sigma_hz: float = -20.0) -> Modes:
    """Lowest ``n_modes`` *flexible* eigenpairs of K phi = w^2 M phi.

    Shift-invert Lanczos with sparse LU.  For an unconstrained model the six
    rigid-body modes are removed analytically (ARPACK is unreliable on a
    6-fold zero eigenvalue) and returned as the first six zero-frequency modes.
    """
    K, M = model.assemble()
    N = K.shape[0]
    free = np.ones(N, bool)
    if fixed is not None:
        free[np.asarray(fixed, int)] = False
    live = (np.abs(K).sum(0).A1 > 0) | (M.diagonal() > 0)
    free &= live
    ii = np.where(free)[0]
    Kf, Mf = K[ii][:, ii], M[ii][:, ii]
    sigma = -((2 * np.pi * abs(sigma_hz)) ** 2)
    k = min(n_modes, len(ii) - 8)

    rigid = None
    kwargs = {}
    if fixed is None:
        rigid = rigid_body_basis(model, M, ii)
        lu = spla.splu((Kf - sigma * Mf).tocsc())
        MR = Mf @ rigid

        def op(x):
            z = lu.solve(np.asarray(x, float))
            return z - rigid @ (MR.T @ z) if z.ndim == 1 else z - rigid @ (MR.T @ z)

        kwargs["OPinv"] = spla.LinearOperator(Kf.shape, matvec=op, matmat=op, dtype=float)

    vals, vecs = spla.eigsh(Kf, k=k, M=Mf, sigma=sigma, which="LM", **kwargs)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    f = np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi)
    if rigid is not None:
        f = np.concatenate([np.zeros(6), f])
        vecs = np.column_stack([rigid, vecs])
    full = np.zeros((N, len(f)))
    full[ii] = vecs
    return Modes(model, f, full, damping_from(zeta, f))
