"""Violin assembly: plates + ribs + bass bar + sound post + bridge + air cavity.

``ViolinSpec`` is a frozen dataclass: use ``dataclasses.replace`` (or the helper
``spec.with_(...)``) to make variants -- that is what the search layer does.

Modelling choices (and what they leave out) are listed in docs/DESIGN.md.  The
short version: structural shell model of the wooden body in vacuo, plus a
lumped Helmholtz air cavity.  Not (yet) modelled: radiation/air loading of the
plates, corner blocks & linings, varnish, neck/fingerboard/tailpiece, the
bridge's detailed kidney cut-outs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Optional, Tuple

import numpy as np
import shapely
from scipy.special import ellipk

from .fem import End, Model, Modes, Section, modal_analysis
from .materials import C_AIR, MAPLE, RHO_AIR, SPRUCE, Orthotropic
from .violin_geometry import FHole, PlateGeometry, Plates, build_plates


@dataclass(frozen=True)
class BassBar:
    enabled: bool = True
    x: float = -0.0195       # centre-line offset from body axis (negative = bass side)
    y: float = 0.1625        # centre of the bar along the body
    length: float = 0.270
    width: float = 0.006
    height: float = 0.0135   # at mid-length; tapers elliptically to the ends
    material: Orthotropic = SPRUCE


@dataclass(frozen=True)
class SoundPost:
    enabled: bool = True
    x: float = 0.0225        # treble side
    y: float = 0.1585        # just behind the treble bridge foot
    diameter: float = 0.006
    material: Orthotropic = SPRUCE


@dataclass(frozen=True)
class Bridge:
    y: float = 0.1625
    foot_half_spacing: float = 0.0205
    leg_height: float = 0.021      # feet -> top bar
    crown_height: float = 0.010    # top bar -> string notches
    leg_width: float = 0.0185      # along the body
    leg_thickness: float = 0.0030  # "waist" thickness, in the bending direction
    top_mass: float = 0.0007       # kg lumped at the string notches (rest of bridge mass comes from beams)
    material: Orthotropic = MAPLE


@dataclass(frozen=True)
class ViolinSpec:
    plates: PlateGeometry = field(default_factory=PlateGeometry)
    top_thickness: Tuple[float, float] = (0.0029, 0.0025)    # (centre, edge)
    back_thickness: Tuple[float, float] = (0.0044, 0.0026)
    rib_thickness: float = 0.0011
    graduation_width: float = 0.060  # distance from edge over which thickness grades to centre value
    top_material: Orthotropic = SPRUCE
    back_material: Orthotropic = MAPLE
    rib_material: Orthotropic = MAPLE
    bass_bar: BassBar = field(default_factory=BassBar)
    post: SoundPost = field(default_factory=SoundPost)
    bridge: Bridge = field(default_factory=Bridge)
    cavity: bool = True
    zeta: float = 0.015  # modal damping ratio (typical violin Q ~ 25-60)

    def with_(self, **kw) -> "ViolinSpec":
        """Nested update: ``spec.with_(post__x=0.025, plates__fhole__length=0.08)``."""
        out = self
        groups: dict = {}
        for k, v in kw.items():
            head, _, rest = k.partition("__")
            if rest:
                groups.setdefault(head, {})[rest] = v
            else:
                out = replace(out, **{head: v})
        for head, sub in groups.items():
            out = replace(out, **{head: _nested_replace(getattr(out, head), sub)})
        return out


def _nested_replace(obj, kw):
    flat, nest = {}, {}
    for k, v in kw.items():
        head, _, rest = k.partition("__")
        if rest:
            nest.setdefault(head, {})[rest] = v
        else:
            flat[head] = v
    for head, sub in nest.items():
        flat[head] = _nested_replace(getattr(obj, head), sub)
    return replace(obj, **flat)


def _graded(plates: Plates, centre: float, edge: float, width: float):
    def f(cen):
        d = plates.dist_to_edge(cen[:, :2])
        t = np.clip(d / width, 0, 1)
        return edge + (centre - edge) * (3 * t ** 2 - 2 * t ** 3)
    return f


def _rib_grain(cen, nrm):
    tang = np.cross(np.array([0.0, 0.0, 1.0]), nrm)
    return tang


def hole_conductance(poly, plate_thickness: float) -> float:
    """Rayleigh conductance [m] of one thin-plate opening, via the equivalent ellipse.

    Circle: C = 2a.  Ellipse (semi-axes a>=b): C = pi a / K(e).  Finite plate
    thickness adds an acoustic mass in series: 1/C_eff = 1/C + t/S.
    """
    S = poly.area
    rect = poly.minimum_rotated_rectangle
    xy = np.asarray(rect.exterior.coords)
    e = sorted([np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1])])
    a = e[1] / 2
    b = S / (np.pi * a)
    m = max(0.0, 1.0 - (b / a) ** 2)
    C = np.pi * a / ellipk(m)
    return 1.0 / (1.0 / C + plate_thickness / S)


class Violin:
    """A built, ready-to-analyse violin model."""

    def __init__(self, spec: ViolinSpec):
        self.spec = spec
        self.plates = build_plates(spec.plates)
        self.model = Model()
        self._build()
        self._modes: dict = {}

    # ------------------------------------------------------------------ build
    def _build(self):
        sp, pl, m = self.spec, self.plates, self.model
        self.t_top = _graded(pl, *sp.top_thickness, sp.graduation_width)
        self.t_back = _graded(pl, *sp.back_thickness, sp.graduation_width)
        m.add_shell_part("top", pl.top.vertices, pl.top.faces, self.t_top, sp.top_material, grain=(0, 1, 0))
        m.add_shell_part("back", pl.back.vertices, pl.back.faces, self.t_back, sp.back_material, grain=(0, 1, 0))
        m.add_shell_part("ribs", pl.ribs.vertices, pl.ribs.faces, sp.rib_thickness, sp.rib_material, grain=_rib_grain)
        if sp.bass_bar.enabled:
            self._add_bass_bar()
        if sp.post.enabled:
            self._add_post()
        self._add_bridge()
        if sp.cavity:
            self._add_cavity()

    def _top_point(self, x, y):
        nodes, w, n, p = self.model.locate("top", (x, y))
        h = float(self.t_top(np.array([[p[0], p[1], p[2]]]))[0])
        return nodes, w, n, p, h

    def _add_bass_bar(self):
        bb, m = self.spec.bass_bar, self.model
        ds = 0.8 * self.spec.plates.mesh_size
        n = max(6, int(round(bb.length / ds)))
        ys = bb.y + np.linspace(-bb.length / 2, bb.length / 2, n + 1)
        s = np.linspace(0, 1, n + 1)
        hb = np.maximum(0.0015, bb.height * np.sqrt(np.clip(1 - (2 * s - 1) ** 2, 0, 1)))
        ends, norms = [], []
        for y, hk in zip(ys, hb):
            nodes, w, nrm, p, ht = self._top_point(bb.x, y)
            ends.append(End(nodes, w, tuple(-nrm * (ht / 2 + hk / 2))))
            norms.append(nrm)
        for k in range(n):
            sec = Section.rect(bb.width, 0.5 * (hb[k] + hb[k + 1]))
            m.add_beam(ends[k], ends[k + 1], sec, bb.material, up=-(norms[k] + norms[k + 1]))

    def _add_post(self):
        ps, m = self.spec.post, self.model
        nodes, w, n, p, ht = self._top_point(ps.x, ps.y)
        top_end = End(nodes, w, tuple(-n * ht / 2))
        bn, bw, bnrm, bp = m.locate("back", (ps.x, ps.y))
        hb = float(self.t_back(np.array([bp]))[0])
        back_end = End(bn, bw, tuple(-bnrm * hb / 2))
        L = np.linalg.norm(m.end_position(top_end) - m.end_position(back_end))
        A = np.pi * ps.diameter ** 2 / 4
        m.add_link(top_end, back_end, ps.material.E_L * A / L, mass=ps.material.rho * A * L)
        self.post_length = L

    def _add_bridge(self):
        br, m = self.spec.bridge, self.model
        feet, W = [], []
        for sx in (-1.0, +1.0):
            nodes, w, n, p, ht = self._top_point(sx * br.foot_half_spacing, br.y)
            feet.append(End(nodes, w, tuple(n * ht / 2)))
        pf = [m.end_position(f) for f in feet]
        for f in pf:
            W.append(m.add_free_node(f + np.array([0, 0, br.leg_height])))
        Wm = m.add_free_node(0.5 * (m.nodes[W[0]] + m.nodes[W[1]]))
        T = m.add_free_node(m.nodes[Wm] + np.array([0, 0, br.crown_height]))
        leg = Section.rect(br.leg_width, br.leg_thickness)
        bar = Section.rect(0.012, 0.006)
        stiff = Section.rect(0.010, 0.008)
        for f, w in zip(feet, W):
            m.add_beam(f, w, leg, br.material, up=(1, 0, 0))
        m.add_beam(W[0], Wm, bar, br.material, up=(0, 0, 1))
        m.add_beam(Wm, W[1], bar, br.material, up=(0, 0, 1))
        m.add_beam(Wm, T, stiff, br.material, up=(1, 0, 0))
        m.add_point_mass(T, br.top_mass, rotary=br.top_mass * 1e-5)
        self.bridge_top = T
        self.bridge_feet = feet
        self.bridge_nodes = dict(bass_waist=W[0], treble_waist=W[1], waist_mid=Wm, top=T)

    def _add_cavity(self):
        sp, m, pl = self.spec, self.model, self.plates
        parts = ("top", "back", "ribs")
        allnodes = np.unique(np.concatenate([m.parts[p].tris.ravel() for p in parts]))
        c0 = m.nodes[allnodes].mean(0)
        b = {}
        V = 0.0
        for name in parts:
            tris = m.parts[name].tris
            P = m.nodes[tris]
            cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            A = 0.5 * np.linalg.norm(cr, axis=1)
            nrm = cr / np.linalg.norm(cr, axis=1, keepdims=True)
            cen = P.mean(1)
            out = np.einsum("ij,ij->i", nrm, cen - c0) < 0
            nrm[out] *= -1
            V += np.sum(A * np.einsum("ij,ij->i", cen - c0, nrm)) / 3.0
            for i in range(3):
                for k in range(3):
                    for node, val in zip(tris[:, i], A / 3.0 * nrm[:, k]):
                        key = 6 * int(node) + k
                        b[key] = b.get(key, 0.0) + val
        # the f-hole openings are missing from the closed surface: add their "lids" to the volume
        # The lid's outward-area weight is spread over the mesh nodes on the hole rim, so
        # the weights sum to zero over the closed surface (rigid translation => no volume
        # change) and the plug dof q is measured relative to the moving plate.
        C_total = 0.0
        top_nodes = m.part_nodes("top")
        top_xy = m.nodes[top_nodes][:, :2]
        for hole in pl.holes:
            c = hole.centroid
            z = float(pl.top_z(np.array([[c.x, c.y]]))[0])
            V += hole.area * (z - c0[2]) / 3.0
            t = float(self.t_top(np.array([[c.x, c.y, z]]))[0])
            C_total += hole_conductance(hole, t)
            rim = top_nodes[shapely.distance(hole.exterior, shapely.points(top_xy)) < 1.0e-3]
            if len(rim) < 3:
                raise RuntimeError("f-hole rim not found in the top-plate mesh")
            # weights: uniform, plus the minimum-norm linear correction that makes the
            # first moment equal the hole's area centroid (keeps rigid rotations exactly
            # volume-neutral, so the free body has six true zero-energy modes)
            r = m.nodes[rim][:, :2]
            rbar = r.mean(0)
            S = (r - rbar).T @ (r - rbar)
            lam = np.linalg.solve(S, hole.area * (np.array([c.x, c.y]) - rbar))
            wts = hole.area / len(rim) + (r - rbar) @ lam
            for node, wj in zip(rim, wts):
                key = 6 * int(node) + 2  # lid normal is +z (top plate faces up)
                b[key] = b.get(key, 0.0) + float(wj)
        dofs = np.array(list(b.keys()))
        vals = np.array([b[k] for k in dofs])
        kappa = RHO_AIR * C_AIR ** 2
        info = m.add_cavity(dofs, vals, V, kappa, RHO_AIR / C_total)
        self.cavity_volume = V
        self.hole_conductance = C_total
        self.helmholtz_rigid_hz = C_AIR / (2 * np.pi) * np.sqrt(C_total / V)

    # ----------------------------------------------------------------- analysis
    def modes(self, n_modes: int = 40, zeta: Optional[float] = None) -> Modes:
        key = (n_modes, zeta)
        if key not in self._modes:
            self._modes[key] = modal_analysis(self.model, n_modes, zeta=self.spec.zeta if zeta is None else zeta)
        return self._modes[key]

    def bridge_admittance(self, freqs, direction=(1.0, 0.0, 0.0), n_modes: int = 40) -> np.ndarray:
        """Mobility (velocity / force) at the string notches along ``direction``.

        direction=(1,0,0) is the bowing direction (across the body); (0,0,1) is vertical.
        """
        md = self.modes(n_modes)
        v = md.node_vector(self.bridge_top, direction)
        return md.admittance_vec(v, v, np.asarray(freqs, float))

    def mode_table(self, n_modes: int = 30):
        """List of (index, frequency_hz, air_fraction, bridge_x_participation)."""
        md = self.modes(n_modes)
        rows = []
        q = self.model.cavity["q"][1] + 6 * self.model.n_nodes if self.spec.cavity else None
        for i in md.flexible():
            af = float(md.shapes[q, i] ** 2 * self.model.cavity["air_mass"]) if q is not None else 0.0
            bx = float(md.shapes[6 * self.bridge_top, i])
            rows.append((int(i), float(md.freqs[i]), af, bx))
        return rows

    def total_mass(self) -> float:
        _, M = self.model.assemble()
        d = M.diagonal()[: 6 * self.model.n_nodes]
        return float(sum(d[k::6].sum() for k in range(3)) / 3.0)
