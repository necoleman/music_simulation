"""Air-jet (fipple) excitation: from air-column resonances to *playing* pitches.

The 3-D model (acoustics.py) gives the window input impedance Z(w) of the air column.  Here that
impedance is closed into a feedback loop with the air jet of the windway (Fletcher 1976, Coltman 1976,
Verge/Fabre et al. 1990s):

  window pressure p  --(window air plug, inertance)-->  transverse velocity v at the flue exit
  v  --(jet: delay W/c_ph and spatial growth e^{mu W}, integration 1/jw)-->  jet displacement eta at the labium
  eta  --(jet cut by the labium)-->  volume flow Q_j = gain * b * U * eta injected into the pipe
  Q_j  --(3-D air column)-->  p = Z(w) Q_j

Closing the loop gives

        L(w) = - Z(w) * gain * b * U * exp((mu - j w / c_ph) W) / (rho * l_w * w^2)  = 1 .

* the real part of the oscillation condition (arg L = 0) fixes the playing frequency: it sits where the
  pipe's impedance phase compensates the jet delay -- so it is a bit above a resonance for a fast jet
  and below for a slow one, and rises with blowing pressure, as on a real recorder;
* |L| >= 1 is the start-up (threshold) condition; among the candidates the strongest wins, which is how
  the register is chosen (hard blowing -> upper register).

What is physical: the structure, the delay W/c_ph with c_ph ~ 0.4 U, the dimensional scalings.
What is empirical: ``gain`` (jet/labium coupling, ~1), ``window_inertance`` (effective length of the
air plug in the window, default = window length + a bore-radius end term), the amplification rate ``mu``
(default = Rayleigh/Bickley-jet estimate).  Calibrate those against one real instrument.  Nonlinear
saturation (amplitude, loudness, harmonic content) is *not* modelled, so this gives pitch, register and
ease of speaking, not timbre.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .acoustics import AcousticModes


@dataclass(frozen=True)
class Jet:
    flue_height: float = 0.0010          # h, slit height [m]
    blowing_pressure: float = 400.0      # Pa above ambient, in the windway
    c_ratio: float = 0.40                # disturbance phase speed / jet centre speed
    gain: float = 1.0                    # labium coupling (empirical, order 1)
    window_inertance: Optional[float] = None   # effective plug length l_w [m]; None -> window length + 0.5 * bore radius
    growth: float = 0.30                 # mu*h at its optimum kh ~ 0.4 (Bickley jet), scaled below that
    pressure_ref_hz: Optional[float] = None    # if set, blowing_pressure applies at this pitch and scales as f^2 for other notes
                                               # (players blow harder for higher notes); None = same pressure for every note

    def for_pitch(self, f_hz: float) -> "Jet":
        if self.pressure_ref_hz is None:
            return self
        from dataclasses import replace
        return replace(self, blowing_pressure=self.blowing_pressure * (f_hz / self.pressure_ref_hz) ** 2)

    def speed(self, rho: float) -> float:
        return float(np.sqrt(2.0 * self.blowing_pressure / rho))


@dataclass
class PlayCandidate:
    f: float              # Hz
    gain: float           # |L| at that frequency (>= 1: sustained oscillation possible)
    mode: int             # index of the nearest air-column resonance
    f_res: float          # that resonance, Hz
    cents_from_res: float # playing frequency relative to the resonance
    delay_ratio: float    # jet delay / period


def window_admittance_weights(modes: AcousticModes, window_name: str = "window") -> np.ndarray:
    """Area-weighted average of every mode shape over the window face (the effective drive/response point)."""
    mesh = modes.ops.mesh
    Bw = modes.ops.B[window_name]
    w = np.asarray(Bw.sum(axis=1)).ravel()
    return (w @ modes.p) / w.sum()          # (n_modes,)


def window_impedance(modes: AcousticModes, freqs: np.ndarray, window_name: str = "window") -> np.ndarray:
    """Z = p/Q seen by a volume-flow source spread over the window [Pa s/m^3] (modal sum + residual)."""
    phi = window_admittance_weights(modes, window_name)
    w = 2 * np.pi * np.asarray(freqs, float)[:, None]
    wn = 2 * np.pi * modes.freqs[None]
    z = 1.0 / (2.0 * modes.Q[None])
    num = 1j * w * modes.air.rho * modes.air.c ** 2 * phi[None] ** 2
    return np.sum(num / (wn ** 2 - w ** 2 + 2j * z * wn * w), axis=1)


def loop_gain(modes: AcousticModes, jet: Jet, window_len: float, window_width: float, bore_radius: float,
              freqs: np.ndarray, window_name: str = "window"):
    air = modes.air
    U = jet.speed(air.rho)
    cph = jet.c_ratio * U
    w = 2 * np.pi * np.asarray(freqs, float)
    lw = jet.window_inertance if jet.window_inertance is not None else window_len + 0.5 * bore_radius
    h = jet.flue_height
    k = w / cph
    mu = (jet.growth / h) * np.minimum(1.0, k * h / 0.4)
    # growth over the window length, limited (the real jet saturates: cap at e^3)
    amp = np.exp(np.minimum(mu * window_len, 3.0))
    delay = window_len / cph
    L = -window_impedance(modes, freqs, window_name) * jet.gain * window_width * U * amp * np.exp(-1j * w * delay) \
        / (air.rho * lw * w ** 2)
    return L, delay


def play(modes: AcousticModes, jet: Jet, window_len: float, window_width: float, bore_radius: float,
         n_scan: int = 4000, window_name: str = "window") -> List[PlayCandidate]:
    """Frequencies where arg L = 0 (phase condition) with their loop gain, strongest first."""
    f_lo, f_hi = 0.6 * modes.freqs[0], 1.25 * modes.freqs[min(len(modes.freqs), 3) - 1]
    f = np.linspace(f_lo, f_hi, n_scan)
    L, delay = loop_gain(modes, jet, window_len, window_width, bore_radius, f, window_name)
    ph = np.unwrap(np.angle(L))
    out: List[PlayCandidate] = []
    # phase crosses a multiple of 2 pi going down (increasing f makes the delay phase more negative)
    m = np.floor(ph / (2 * np.pi))
    for i in np.where(m[:-1] != m[1:])[0]:
        # linear interpolation of the crossing
        target = 2 * np.pi * max(m[i], m[i + 1])
        t = (target - ph[i]) / (ph[i + 1] - ph[i])
        fc = f[i] + t * (f[i + 1] - f[i])
        gc = abs(L[i] + t * (L[i + 1] - L[i]))
        j = int(np.argmin(np.abs(modes.freqs - fc)))
        out.append(PlayCandidate(float(fc), float(gc), j, float(modes.freqs[j]),
                                 float(1200 * np.log2(fc / modes.freqs[j])), float(delay * fc)))
    out.sort(key=lambda c: -c.gain)
    return out
