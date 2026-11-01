"""String <-> body coupling and wolf-tone analysis.

Physics
-------
A string of length L, tension T, linear density mu (wave impedance
``Zs = sqrt(T mu)``) fixed at one end and attached to the bridge at the other has
driving-point mobility (velocity per force) at the bridge

        Ys(w) = (j / Zs) tan(k L),         k = w / (c (1 + j eta / 2)).

The body has mobility ``Yb(w)`` at the bridge notch (from the FE modal model).
Joining them, force balance and velocity continuity give the free-vibration
condition  ``Ys(w) + Yb(w) = 0``, i.e.

        g(w) = sin(kL) - j Zs Yb(w) cos(kL) = 0 .

Its complex roots w = w_r + j w_i are the coupled string/body modes (frequency
w_r/2pi, decay rate w_i).  Far from a body resonance there is one root close to
the string's own harmonic.  Near a body resonance there are two hybrid roots; if
their frequency splitting exceeds their difference in damping the two modes are
~50/50 mixtures, the bow cannot lock onto either, and the note beats: a *wolf*.

``wolf_scan`` evaluates this at every semitone of every string and reports, per
note, the root pair, their splitting (the beat rate), and a 0..1 *mixing* score.

Caveats: a single polarisation (default: across the body) and a single
bridge-notch force point are used; the string is lossless except for a uniform
loss factor; and the body model is the in-vacuo shell model plus lumped cavity,
so absolute wolf frequencies inherit that model's frequency errors.  Use the
tool for *relative* comparisons between designs, then calibrate against a
measured admittance (``admittance_from_measurement``) when you have one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .fem import Modes


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class String:
    name: str
    f0: float           # open-string pitch [Hz]
    mu: float           # linear density [kg/m]
    length: float = 0.325  # open vibrating length, nut to bridge [m]
    Q: float = 800.0    # intrinsic string quality factor (1/eta)

    @property
    def tension(self) -> float:
        return self.mu * (2 * self.length * self.f0) ** 2

    @property
    def Zs(self) -> float:
        return float(np.sqrt(self.tension * self.mu))

    @property
    def wave_speed(self) -> float:
        return float(np.sqrt(self.tension / self.mu))

    def note(self, semitones: int) -> Tuple[float, float]:
        """(pitch, stopped length) when fingered ``semitones`` above the open string."""
        r = 2.0 ** (semitones / 12.0)
        return self.f0 * r, self.length / r


# typical modern violin strings (steel E, synthetic-core D/A, wound G); tensions come out 50-75 N
VIOLIN_STRINGS: Dict[str, String] = {
    "G": String("G", 196.00, 3.2e-3),
    "D": String("D", 293.66, 1.7e-3),
    "A": String("A", 440.00, 0.9e-3),
    "E": String("E", 659.26, 0.42e-3),
}


# --------------------------------------------------------------------------- #
class BodyMobility:
    """Modal mobility  Yb(w) = sum_n  j w a_n / (wn^2 - w^2 + 2 j z_n wn w)  (complex w allowed)."""

    def __init__(self, a, wn, zeta):
        self.a, self.wn, self.z = np.asarray(a, float), np.asarray(wn, float), np.asarray(zeta, float)

    def __call__(self, w):
        w = np.asarray(w, complex)[..., None]
        return np.sum(1j * w * self.a / (self.wn ** 2 - w ** 2 + 2j * self.z * self.wn * w), axis=-1)

    def pole_free_parts(self, w_lo: float, w_hi: float):
        """Split Yb = Nn/Dn + Yfar where Dn is the product of the denominators of modes
        inside [w_lo, w_hi]: Nn, Dn, Yfar have no poles in that band, so
        ``sin(kL) Dn - j Zs cos(kL) (Nn + Dn Yfar)`` is entire there and Newton behaves."""
        near = (self.wn >= w_lo) & (self.wn <= w_hi)
        an, wn, zn = self.a[near], self.wn[near], self.z[near]
        far = BodyMobility(self.a[~near], self.wn[~near], self.z[~near])

        def den(w):  # (..., n_near)
            w = np.asarray(w, complex)[..., None]
            return wn ** 2 - w ** 2 + 2j * zn * wn * w

        def Dn(w):
            return np.prod(den(w), axis=-1) if len(an) else np.ones_like(np.asarray(w, complex))

        def Nn(w):
            d = den(w)
            wc = np.asarray(w, complex)
            out = np.zeros_like(wc)
            for i in range(len(an)):
                others = np.prod(np.delete(d, i, axis=-1), axis=-1) if len(an) > 1 else 1.0
                out = out + 1j * wc * an[i] * others
            return out

        return Nn, Dn, far


def body_mobility(modes: Modes, out_vec: np.ndarray, in_vec: Optional[np.ndarray] = None,
                  fmin: float = 5.0) -> BodyMobility:
    """Yb(omega) at a point/direction from the FE modes (valid for complex omega)."""
    in_vec = out_vec if in_vec is None else in_vec
    sel = modes.flexible(fmin)
    a = (out_vec @ modes.shapes[:, sel]) * (in_vec @ modes.shapes[:, sel])
    return BodyMobility(a, 2 * np.pi * modes.freqs[sel], modes.zeta[sel])


def admittance_from_measurement(freqs_hz: np.ndarray, mobility: np.ndarray, n_poles: int = 30):
    """Placeholder hook for coupling a *measured* bridge mobility instead of the FE one.

    Fit a rational (modal) model to ``mobility`` (vector fitting) and return a
    callable Yb(omega).  Not implemented yet -- see docs/DESIGN.md roadmap.
    """
    raise NotImplementedError("measured-admittance import is on the roadmap")


# --------------------------------------------------------------------------- #
def _g(w, Yb, string: String, L: float):
    """Characteristic function g(w) = sin(kL) - j Zs Yb(w) cos(kL)."""
    eta = 1.0 / string.Q
    k = w / (string.wave_speed * (1 + 0.5j * eta))
    return np.sin(k * L) - 1j * string.Zs * Yb(w) * np.cos(k * L)


def _g_entire(w, parts, string: String, L: float):
    Nn, Dn, far = parts
    eta = 1.0 / string.Q
    k = w / (string.wave_speed * (1 + 0.5j * eta))
    D = Dn(w)
    return np.sin(k * L) * D - 1j * string.Zs * np.cos(k * L) * (Nn(w) + D * far(w))


def coupled_roots(Yb, string: String, L: float, f_guess: float, *, window: float = 0.08,
                  n_start: int = 81, imag_starts=(0.002, 0.008, 0.02, 0.04, 0.08)) -> np.ndarray:
    """All damped roots of ``g`` within +/- ``window`` (fractional) of ``f_guess``.

    Vectorised damped Newton from a dense grid of complex starting points
    (heavily damped hybrid roots have small basins of attraction, so a sparse
    set of starts misses them).  Roots are returned sorted by frequency.
    """
    w0 = 2 * np.pi * f_guess
    if isinstance(Yb, BodyMobility):  # pole-free formulation (robust near body resonances)
        parts = Yb.pole_free_parts(w0 * (1 - 3 * window), w0 * (1 + 3 * window))
        f = lambda z: _g_entire(z, parts, string, L)
    else:
        f = lambda z: _g(z, Yb, string, L)
    re = w0 * (1 + np.linspace(-1.5 * window, 1.5 * window, n_start))
    Z = (re[:, None] + 1j * w0 * np.asarray(imag_starts)[None, :]).ravel()
    dz = np.ones_like(Z)
    for _ in range(80):
        h = 1e-6 * np.abs(Z)
        fz = f(Z)
        dz = fz / ((f(Z + h) - fz) / h)
        scale = np.minimum(1.0, 0.08 * np.abs(Z) / np.maximum(np.abs(dz), 1e-300))
        Z = Z - dz * scale
    ok = np.isfinite(Z) & (np.abs(dz) < 1e-9 * np.abs(Z)) & (Z.imag > 0) & (np.abs(Z.real / w0 - 1) < window)
    roots: List[complex] = []
    for z in sorted(Z[ok], key=lambda r: r.real):
        if not roots or abs(z - roots[-1]) > 1e-6 * abs(z):
            roots.append(z)
    return np.array(roots)


# --------------------------------------------------------------------------- #
@dataclass
class NoteResult:
    string: str
    semitone: int
    f_note: float
    roots_hz: np.ndarray       # frequencies of the damped roots near the note
    q: np.ndarray              # their quality factors
    body_mode_hz: float        # nearest uncoupled body resonance
    split_hz: float            # frequency separation of the string-like / body-like roots (beat rate)
    mixing: float              # 0..1; 1 => roots are 50/50 string/body hybrids, 0 => decoupled
    coupling: float            # Zs |Yb(f_note)|, dimensionless body/string coupling strength
    is_wolf: bool

    def row(self) -> str:
        r = ", ".join(f"{f:7.2f}(Q{q:5.0f})" for f, q in zip(self.roots_hz, self.q))
        return (f"{self.string}{self.semitone:+3d}  {self.f_note:7.2f} Hz  body {self.body_mode_hz:7.2f}  "
                f"coupling {self.coupling:5.3f}  split {self.split_hz:5.2f} Hz  mixing {self.mixing:4.2f}  "
                f"{'WOLF' if self.is_wolf else '    '}  [{r}]")


def analyse_note(Yb: BodyMobility, string: String, semitone: int, *, beat_range=(0.4, 25.0),
                 mixing_threshold=0.5, window: float = 0.06) -> NoteResult:
    """Coupled-mode analysis of one fingered note (string fundamental vs nearest body mode).

    Wolf criterion.  Let w_s, w_b be the uncoupled complex frequencies of the
    string fundamental and of the nearest body mode, and w_1, w_2 the two
    coupled roots nearest to them.  In a two-oscillator picture the body
    participation in root i is  p_i = |w_i - w_s| / (|w_i - w_s| + |w_i - w_b|)
    (p_1 + p_2 = 1), so  mixing = 1 - |p_1 - p_2|  is 0 for decoupled and 1 for
    fully hybridised modes.  A note is flagged when mixing >= ``mixing_threshold``
    and the roots are separated by a beat rate within ``beat_range`` [Hz].
    """
    f_note, L = string.note(semitone)
    roots = coupled_roots(Yb, string, L, f_note, window=window)
    f = roots.real / (2 * np.pi) if len(roots) else np.array([])
    q = roots.real / (2 * roots.imag) if len(roots) else np.array([])
    coupling = float(string.Zs * abs(Yb(2 * np.pi * f_note)))

    eta = 1.0 / string.Q
    w_s = np.pi * string.wave_speed / L * (1 + 0.5j * eta)
    poles = Yb.wn * (np.sqrt(1 - Yb.z ** 2) + 1j * Yb.z)
    ib = int(np.argmin(np.abs(poles - w_s)))
    w_b = poles[ib]
    split, mixing = 0.0, 0.0
    if len(roots) >= 2 and abs(w_b.real / w_s.real - 1) < 3 * window:
        i1 = int(np.argmin(np.abs(roots - w_s)))
        d_b = np.abs(roots - w_b)
        d_b[i1] = np.inf
        i2 = int(np.argmin(d_b))
        r1, r2 = roots[i1], roots[i2]
        split = abs(r1.real - r2.real) / (2 * np.pi)
        p = lambda r: abs(r - w_s) / (abs(r - w_s) + abs(r - w_b))
        mixing = float(max(0.0, 1.0 - abs(p(r1) - p(r2))))
    is_wolf = bool(beat_range[0] <= split <= beat_range[1] and mixing >= mixing_threshold)
    return NoteResult(string.name, semitone, f_note, f, q, float(poles[ib].real / (2 * np.pi)), split, mixing, coupling, is_wolf)


def wolf_scan(modes: Modes, bridge_node: int, strings: Optional[Dict[str, String]] = None,
              semitones: Iterable[int] = range(0, 15), direction=(1.0, 0.0, 0.0), **kw) -> List[NoteResult]:
    """Scan every semitone of every string for wolf-prone notes."""
    strings = VIOLIN_STRINGS if strings is None else strings
    v = modes.node_vector(bridge_node, direction)
    Yb = body_mobility(modes, v)
    out: List[NoteResult] = []
    for s in strings.values():
        for k in semitones:
            out.append(analyse_note(Yb, s, k, **kw))
    return out


def wolf_summary(results: Sequence[NoteResult]) -> dict:
    wolves = [r for r in results if r.is_wolf]
    return dict(
        n_wolf=len(wolves),
        worst_mixing=max((r.mixing for r in results), default=0.0),
        wolf_notes=[(r.string, r.semitone, round(r.f_note, 1), round(r.split_hz, 2)) for r in wolves],
        max_coupling=max((r.coupling for r in results), default=0.0),
    )


# --------------------------------------------------------------------------- #
# Overtones and an indicative rendering of a note
# --------------------------------------------------------------------------- #
@dataclass
class Partial:
    n: int                 # harmonic number of the uncoupled string
    f_ideal: float         # n * f_note (rigid-termination string)
    f: float               # body-coupled frequency [Hz]
    q: float               # quality factor of the coupled root (decay = pi f / Q nepers/s... see render_pluck)
    shift_cents: float     # deviation from the ideal harmonic caused by the body


def note_partials(modes: Modes, bridge_node: int, string: String, semitone: int = 0, n_harm: int = 12,
                  direction=(1.0, 0.0, 0.0)) -> List[Partial]:
    """Body-coupled partial frequencies and decays of one note.

    For each harmonic the *string-like* coupled root is selected (the root nearest the
    uncoupled complex harmonic).  This is what a luthier hears as 'the body pulls
    this overtone flat/sharp and damps it'.
    """
    Yb = body_mobility(modes, modes.node_vector(bridge_node, direction))
    f_note, L = string.note(semitone)
    eta = 1.0 / string.Q
    out: List[Partial] = []
    for n in range(1, n_harm + 1):
        w_s = n * np.pi * string.wave_speed / L * (1 + 0.5j * eta)
        roots = coupled_roots(Yb, string, L, n * f_note, window=0.03)
        if len(roots) == 0:
            continue
        r = roots[np.argmin(np.abs(roots - w_s))]
        f = r.real / (2 * np.pi)
        out.append(Partial(n, n * f_note, f, float(r.real / (2 * r.imag)), 1200 * np.log2(f / (n * f_note))))
    return out


def render_pluck(partials: Sequence[Partial], duration: float = 2.0, sr: int = 44100, pluck_pos: float = 0.12) -> np.ndarray:
    """Additive rendering of a plucked note from coupled partials (INDICATIVE ONLY).

    Amplitudes follow the ideal plucked string, sin(n pi beta)/n^2; decays come from each
    partial's coupled Q.  There is no radiation/directivity model yet, so treat the result
    as a way to *hear pitch, beating and overtone damping*, not the final timbre.
    """
    t = np.arange(int(duration * sr)) / sr
    y = np.zeros_like(t)
    for p in partials:
        amp = abs(np.sin(p.n * np.pi * pluck_pos)) / p.n ** 2
        y += amp * np.exp(-np.pi * p.f / p.q * t) * np.sin(2 * np.pi * p.f * t)
    return y / max(1e-12, np.abs(y).max())
