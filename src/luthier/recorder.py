"""Recorder (fipple flute) built as a full 3-D air-volume model.

Geometry (all metres).  The bore axis is z, z=0 at the closed upstream end (block
face / upstream edge of the window), z increasing toward the foot.  "Front" (the
finger side, window side) is +x; angle 0 deg is the front, 180 deg the back (thumb).

* ``bore``  : (z, radius) points, piecewise linear, revolved about z -> any taper/flare
* ``wall``  : outer wall thickness = chimney height of the tone holes
* ``holes`` : :class:`ToneHole` -- position, outer size, oval aspect, undercut, angle
* ``window``: rectangular cut-out in the front wall

What the 3-D model *includes* that a 1-D transfer-matrix model must parametrise:
chimney height and shape, undercutting, the actual volume of closed holes, bore
taper, and the real 3-D field around the window and holes.  What it does *not*
model (yet): the jet / edge-tone excitation (the playing pitch is the air-column
resonance with the window as an open end; use ``Recorder.jet_cents`` for an empirical
offset), the windway, wall vibration, and radiation beyond the end-correction
impedance at each opening.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .acoustics import AcousticModes, Air, build_operators, solve_modes
from .airmesh import AirMesh, OpeningSpec, mesh_from_gmsh
from .specutil import update


@dataclass(frozen=True)
class ToneHole:
    name: str
    z: float                    # axial position of the hole centre [m]
    diameter: float             # outer (finger-side) diameter, tangential direction [m]
    axial_ratio: float = 1.0    # axial diameter / tangential diameter (1 = round, <1 = narrow slot)
    undercut: float = 0.0       # radius added at the bore end, tapering to 0 at the outer face [m]
    angle_deg: float = 0.0      # position round the bore: 0 = front, 180 = back (thumb hole)
    end_corr: float = 0.7       # radiation end correction / equivalent radius (0.6133 unflanged .. 0.8216 flanged)


@dataclass(frozen=True)
class Window:
    length: float = 0.0065      # axial length of the opening (labium edge at z0 + length)
    width: float = 0.0085       # tangential width
    z0: float = 0.0


@dataclass(frozen=True)
class RecorderSpec:
    bore: Tuple[Tuple[float, float], ...]
    holes: Tuple[ToneHole, ...]
    window: Window = field(default_factory=Window)
    wall: float = 0.004
    mesh_size: float = 0.004        # global tetra size [m]; refined automatically near small features
    mesh_min: float = 0.0008
    curvature_points: int = 16       # mesh segments per full circle on curved outlines (16: ~2 cents vs 24, 3x cheaper)
    air_temp_c: float = 26.0        # breath-warmed air in a played instrument is warmer than the room
    window_end_corr: float = 0.6133
    foot_end_corr: float = 0.6133

    def with_(self, **kw) -> "RecorderSpec":
        """Nested update by "__" path; numeric parts index tuples: ``spec.with_(holes__2__z=0.15)``."""
        return update(self, kw)

    def bore_radius(self, z: float) -> float:
        zs, rs = zip(*self.bore)
        return float(np.interp(z, zs, rs))

    @property
    def length(self) -> float:
        return self.bore[-1][0]


# --------------------------------------------------------------------------- #
def _hole_frames(spec: RecorderSpec, h: ToneHole):
    phi = np.deg2rad(h.angle_deg)
    n = np.array([np.cos(phi), np.sin(phi), 0.0])
    t = np.array([-np.sin(phi), np.cos(phi), 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    r = spec.bore_radius(h.z)
    return n, t, ez, r


@lru_cache(maxsize=16)
def build_air_mesh(spec: RecorderSpec) -> AirMesh:
    """Mesh the air in the bore, chimneys and window; identify every opening."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("recorder")
        occ = gmsh.model.occ

        # --- bore: revolve the (z, r) profile -----------------------------------
        pts = [occ.addPoint(0, 0, spec.bore[0][0])]
        pts += [occ.addPoint(r, 0, z) for z, r in spec.bore]
        pts += [occ.addPoint(0, 0, spec.bore[-1][0])]
        lines = [occ.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)] + [occ.addLine(pts[-1], pts[0])]
        face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        rev = occ.revolve([(2, face)], 0, 0, 0, 0, 0, 1, 2 * np.pi)
        bore = [dt for dt in rev if dt[0] == 3]
        tools = []

        # --- tone-hole chimneys: loft inner ellipse -> outer ellipse ------------------
        specs: List[OpeningSpec] = []
        for h in spec.holes:
            n, t, ez, r = _hole_frames(spec, h)
            a_t = h.diameter / 2
            a_z = a_t * h.axial_ratio
            rho_in = r - 0.003             # start inside the bore so the loft overlaps it
            rho_out = r + spec.wall
            grow = (rho_out - rho_in) / spec.wall            # undercut taper extended to rho_in
            walls = []
            for rho, u in ((rho_in, h.undercut * grow), (rho_out, 0.0)):
                A_z, A_t = a_z + u, a_t + u
                c = rho * n + np.array([0, 0, h.z])
                if A_z >= A_t:
                    cur = occ.addEllipse(*c, A_z, A_t, zAxis=list(n), xAxis=list(ez))
                else:
                    cur = occ.addEllipse(*c, A_t, A_z, zAxis=list(n), xAxis=list(t))
                walls.append(occ.addWire([cur]))
            loft = occ.addThruSections(walls, makeSolid=True, makeRuled=True)
            tools += [dt for dt in loft if dt[0] == 3]
            cap = rho_out * n + np.array([0, 0, h.z])
            specs.append(OpeningSpec(h.name, tuple(cap), tuple(n), tuple(ez), (a_z, a_t), "ellipse", h.end_corr, "hole"))

        # --- window: box through the front wall --------------------------------------------
        w = spec.window
        zc = w.z0 + w.length / 2
        r_w = spec.bore_radius(zc)
        x0, x1 = 0.3 * r_w, r_w + spec.wall
        box = occ.addBox(x0, -w.width / 2, w.z0, x1 - x0, w.width, w.length)
        tools.append((3, box))
        specs.append(OpeningSpec("window", (x1, 0.0, zc), (1, 0, 0), (0, 0, 1), (w.length / 2, w.width / 2),
                                 "rect", spec.window_end_corr, "window"))
        specs.append(OpeningSpec("foot", (0, 0, spec.length), (0, 0, 1), (1, 0, 0),
                                 (spec.bore[-1][1],) * 2, "ellipse", spec.foot_end_corr, "foot"))

        occ.fuse(bore, tools)
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", spec.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", spec.mesh_min)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", spec.curvature_points)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)
        return mesh_from_gmsh(specs)
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------- #
# Notes and fingerings
# --------------------------------------------------------------------------- #
_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_hz(name: str, A4: float = 440.0) -> float:
    """Equal-tempered frequency of a note name such as ``C5``, ``F#5``, ``Bb5`` (A4 = ``A4`` Hz)."""
    letter, rest = name[0], name[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    n = _SEMI[letter] + acc + 12 * (int(rest) + 1)       # MIDI number
    return A4 * 2.0 ** ((n - 69) / 12.0)


# Baroque-style fingerings for an 8-hole recorder with single holes: holes T(humb), 1..7.
# Value = the holes that are OPEN (all others are covered).  Double holes 6/7 are modelled as single holes,
# so F and F# are approximations of the real cross-fingerings.
BAROQUE_SOPRANO = {
    "C5": (),
    "D5": ("7",),
    "E5": ("6", "7"),
    "F5": ("5",),
    "F#5": ("4", "7"),
    "G5": ("4", "5", "6", "7"),
    "A5": ("3", "4", "5", "6", "7"),
    "B5": ("2", "3", "4", "5", "6", "7"),
    "C6": ("1", "3", "4", "5", "6", "7"),
    "D6": ("T", "1", "3", "4", "5", "6", "7"),
}


@dataclass
class NoteReport:
    note: str
    target_hz: float
    f1: float
    cents: float                 # pitch error of the fundamental (positive = sharp)
    Q1: float
    ratios: np.ndarray           # f_n / f_1 for the next modes
    stretch_cents: float         # 2nd mode vs an exact octave above f1 (0 = perfectly harmonic)
    zpeak: float                 # window input-impedance peak of mode 1 [Pa s/m^3] (relative loudness/ease indicator)
    open_holes: tuple
    f_res: float = float("nan")  # lowest air-column resonance (== f1)
    gain: float = float("nan")   # jet loop gain |L| of the sounding candidate (>= 1: speaks)
    register: int = 0            # 0 = fundamental register, 1 = octave, 2 = twelfth ...  (-1: none found)
    speaks: bool = True          # False when the jet model finds no sustained first-register oscillation

    def row(self) -> str:
        r = " ".join(f"{x:5.3f}" for x in self.ratios)
        g = "" if np.isnan(self.gain) else f"  jet gain {self.gain:4.2f}{'' if self.speaks else ' (WEAK)'}"
        return (f"{self.note:>4}  target {self.target_hz:7.2f}  model {self.f1:7.2f} Hz  {self.cents:+6.1f} c{g}   "
                f"Q {self.Q1:5.0f}  f2/f1..: {r}   octave stretch {self.stretch_cents:+6.1f} c   open: {','.join(self.open_holes) or '-'}")


class Recorder:
    """A built recorder: one air mesh, any number of fingerings."""

    def __init__(self, spec: RecorderSpec, jet_cents: float = 0.0):
        self.spec = spec
        self.mesh = build_air_mesh(spec)
        self.ops = build_operators(self.mesh)
        self.air = Air.at(spec.air_temp_c)
        self.jet_cents = jet_cents
        w = self.mesh.openings["window"]
        self._window_nodes = np.unique(self.mesh.tris[w.faces])

    @property
    def hole_names(self) -> List[str]:
        return [h.name for h in self.spec.holes]

    def modes(self, open_holes: Iterable[str] = (), n_modes: int = 5, f_guess: float = 100.0) -> AcousticModes:
        """Resonances for a fingering: the window and foot are open, ``open_holes`` are uncovered, the rest covered."""
        open_holes = tuple(open_holes)
        unknown = set(open_holes) - set(self.hole_names)
        if unknown:
            raise KeyError(f"unknown holes {sorted(unknown)}; recorder has {self.hole_names}")
        return solve_modes(self.ops, ("window", "foot") + open_holes, n_modes, air=self.air, f_guess=f_guess)

    def analyse(self, fingerings: Dict[str, Sequence[str]] = BAROQUE_SOPRANO, A4: float = 440.0,
                n_modes: int = 4, jet=None) -> List[NoteReport]:
        """Per-note report.  Without ``jet`` the pitch is the air-column resonance; with a :class:`luthier.jet.Jet`
        it is the *playing* pitch of the first register under that blowing pressure."""
        out = []
        for note, open_h in fingerings.items():
            md = self.modes(open_h, max(n_modes, 5) if jet else n_modes)
            f = md.freqs
            tgt = note_hz(note, A4)
            wn = 2 * np.pi * f[0]
            zpk = float(self.air.rho * self.air.c ** 2 * np.mean(md.p[self._window_nodes, 0] ** 2) * md.Q[0] / wn)
            stretch = 1200 * np.log2(f[1] / (2 * f[0])) if len(f) > 1 else float("nan")
            f_play, gain, reg, speaks = float(f[0]), float("nan"), 0, True
            if jet is not None:
                from .jet import play
                w = self.spec.window
                cands = play(md, jet.for_pitch(tgt), w.length, w.width, float(np.mean([r for _, r in self.spec.bore[:2]])))
                first = [c for c in cands if c.mode == 0]
                if first:
                    c0 = max(first, key=lambda c: c.gain)
                    f_play, gain = c0.f, c0.gain
                    speaks = c0.gain >= 1.0 and (cands[0].mode == 0)
                    reg = 0 if speaks else (cands[0].mode if cands[0].gain >= 1 else 0)
                else:
                    speaks, reg = False, -1
            out.append(NoteReport(note, tgt, float(f_play), 1200 * np.log2(f_play / tgt) + self.jet_cents, float(md.Q[0]),
                                  f[1:] / f[0], float(stretch), zpk, tuple(open_h), float(f[0]), gain, reg, speaks))
        return out


# --------------------------------------------------------------------------- #
# Intonation tuning of hole positions
# --------------------------------------------------------------------------- #
# Which hole "governs" each note: the upstream-most open hole of the plain fingering.
GOVERNING_HOLE = {"D5": "7", "E5": "6", "F5": "5", "G5": "4", "A5": "3", "B5": "2", "C6": "1", "D6": "T"}


def tune_hole_positions(spec: RecorderSpec, A4: float = 440.0, iterations: int = 4, damping: float = 0.8,
                        jet_cents: float = 0.0, notes: Sequence[str] = tuple(GOVERNING_HOLE),
                        on_iter=None, jet=None) -> Tuple[RecorderSpec, List[List[NoteReport]]]:
    """Fixed-point intonation correction: move each governing hole by the amount that would cancel its
    note's pitch error under a 1-D sensitivity estimate (df/f = -dz/L_eff, L_eff = c/2f), re-solve in full 3-D,
    repeat.  Holes interact, so a few iterations are needed; this is a starting point for a real search, not
    a replacement for one.  Returns the final spec and the per-iteration reports.
    """
    history = []
    fing = {n: BAROQUE_SOPRANO[n] for n in BAROQUE_SOPRANO}
    for it in range(iterations + 1):
        rec = Recorder(spec, jet_cents)
        rep = rec.analyse(fing, A4, jet=jet)
        history.append(rep)
        if on_iter:
            on_iter(it, spec, rep)
        if it == iterations:
            break
        by_note = {r.note: r for r in rep}
        updates = {}
        c = rec.air.c
        for i, h in enumerate(spec.holes):
            for note in notes:
                if GOVERNING_HOLE.get(note) == h.name and note in by_note:
                    r = by_note[note]
                    L = c / (2 * r.target_hz)
                    dz = damping * L * (np.log(2) / 1200.0) * r.cents      # sharp -> move the hole toward the foot
                    updates[f"holes__{i}__z"] = h.z + dz
        spec = spec.with_(**updates)
        zs = [h.z for h in spec.holes]
        if any(b - a < 0.012 for a, b in zip(sorted(zs)[:-1], sorted(zs)[1:])):
            raise ValueError("tuning pushed two holes closer than 12 mm; widen the design or change the hole diameters")
    return spec, history


# --------------------------------------------------------------------------- #
# Playing pitch (air-jet model)
# --------------------------------------------------------------------------- #
def _play(self, open_holes: Iterable[str] = (), jet=None, n_modes: int = 5):
    """Playing candidates for a fingering: ``[PlayCandidate]`` strongest first (see :mod:`luthier.jet`)."""
    from .jet import Jet, play
    jet = jet or Jet()
    md = self.modes(open_holes, n_modes)
    r0 = float(np.mean([r for _, r in self.spec.bore[:2]]))
    w = self.spec.window
    return md, play(md, jet, w.length, w.width, r0)


Recorder.play = _play     # type: ignore[attr-defined]


def _blow_range(self, open_holes: Iterable[str], pressures: Sequence[float], jet=None):
    """Sweep blowing pressure for one fingering: [(P, f_play, gain, mode of the strongest candidate)]."""
    from dataclasses import replace
    from .jet import Jet, play
    jet = jet or Jet()
    md = self.modes(open_holes, 5)
    w = self.spec.window
    r0 = float(np.mean([r for _, r in self.spec.bore[:2]]))
    out = []
    for P in pressures:
        c = play(md, replace(jet, blowing_pressure=float(P), pressure_ref_hz=None), w.length, w.width, r0)
        out.append((float(P), c[0].f, c[0].gain, c[0].mode) if c else (float(P), float("nan"), 0.0, -1))
    return out


Recorder.blow_range = _blow_range     # type: ignore[attr-defined]
