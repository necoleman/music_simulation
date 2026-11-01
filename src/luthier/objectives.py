"""Ready-made objectives for violin searches.

A ``ViolinObjective`` turns a parameter dict (keys are ``ViolinSpec.with_``
paths, e.g. ``post__x`` or ``plates__fhole__length``) into a score, so any
``ParameterSpace`` over the spec can be searched directly:

    obj = ViolinObjective(base=ViolinSpec(), a0_target=275.0, wolf_weight=2.0)
    result = search(obj, space, method="lhs", n=40)

Score = sum of weighted terms (lower is better):
  * (cents error of the air resonance A0 from its target / 100)^2
  * for each additional named mode target, same form
  * wolf_weight * (number of wolf-flagged notes + sum of mixing over all scanned notes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np

from .strings import VIOLIN_STRINGS, String, wolf_scan
from .violin import Violin, ViolinSpec

if TYPE_CHECKING:
    from .recorder import RecorderSpec


def cents(f: float, target: float) -> float:
    return 1200.0 * np.log2(f / target)


def air_mode_hz(v: Violin, n_modes: int = 30) -> Optional[float]:
    """Frequency of the lowest mode that is mostly air-plug motion (the A0 'Helmholtz' mode)."""
    for _, f, air, _ in v.mode_table(n_modes):
        if air > 0.5:
            return f
    return None


def nearest_mode_hz(v: Violin, f_target: float, n_modes: int = 40) -> float:
    fs = np.array([r[1] for r in v.mode_table(n_modes)])
    return float(fs[np.argmin(np.abs(fs - f_target))])


@dataclass
class ViolinObjective:
    base: ViolinSpec = field(default_factory=ViolinSpec)
    a0_target: Optional[float] = 275.0
    a0_weight: float = 1.0
    mode_targets: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # name -> (Hz, weight); matched to nearest mode
    wolf_weight: float = 1.0
    strings: Optional[Dict[str, String]] = None
    semitones: Sequence[int] = tuple(range(0, 15))
    n_modes: int = 36

    def __call__(self, params: dict):
        spec = self.base.with_(**params)
        v = Violin(spec)
        md = v.modes(self.n_modes)
        metrics: dict = {}
        score = 0.0
        a0 = air_mode_hz(v, self.n_modes)
        metrics["A0_hz"] = a0
        if self.a0_target is not None and self.a0_weight:
            if a0 is None:
                raise RuntimeError("no air mode found")
            score += self.a0_weight * (cents(a0, self.a0_target) / 100.0) ** 2
        for name, (ft, w) in self.mode_targets.items():
            f = nearest_mode_hz(v, ft, self.n_modes)
            metrics[f"{name}_hz"] = f
            score += w * (cents(f, ft) / 100.0) ** 2
        if self.wolf_weight:
            res = wolf_scan(md, v.bridge_top, self.strings or VIOLIN_STRINGS, self.semitones)
            n_wolf = sum(r.is_wolf for r in res)
            mix = float(sum(r.mixing for r in res))
            metrics.update(n_wolf=n_wolf, mixing_sum=mix, worst_mixing=max(r.mixing for r in res),
                           wolf_notes=[(r.string, r.semitone) for r in res if r.is_wolf])
            score += self.wolf_weight * (n_wolf + mix)
        metrics["mass_g"] = 1e3 * v.total_mass()
        return score, metrics


@dataclass
class RecorderObjective:
    """Score a recorder design (``RecorderSpec.with_`` paths such as ``holes__2__z`` or ``bore__3__1``).

    score = sum over notes of (cents error / 100)^2
            + overtone_weight * mean((stretch of the 2nd mode vs a true octave, clipped to 150 c) / 100)^2   [plain-fingered notes]
            - q_weight * mean(log10 Q)           (a mild preference for strong, focused resonances)
    """
    base: "RecorderSpec" = None          # required
    fingerings: Optional[Dict[str, Sequence[str]]] = None
    A4: float = 440.0
    jet_cents: float = 0.0
    jet: object = None                   # luthier.jet.Jet -> score *playing* pitches; WEAK notes cost 1 each
    overtone_weight: float = 0.3
    q_weight: float = 0.0

    def __call__(self, params: dict):
        from .recorder import BAROQUE_SOPRANO, Recorder
        fing = self.fingerings or BAROQUE_SOPRANO
        rec = Recorder(self.base.with_(**params), self.jet_cents)
        rep = rec.analyse(fing, self.A4, jet=self.jet)
        score = sum((r.cents / 100.0) ** 2 for r in rep) + sum(1.0 for r in rep if not r.speaks)
        plain = [r for r in rep if abs(r.stretch_cents) < 400]
        if plain and self.overtone_weight:
            score += self.overtone_weight * float(np.mean([(min(abs(r.stretch_cents), 150.0) / 100.0) ** 2 for r in plain]))
        if self.q_weight:
            score -= self.q_weight * float(np.mean([np.log10(r.Q1) for r in rep]))
        metrics = {"rms_cents": float(np.sqrt(np.mean([r.cents ** 2 for r in rep]))),
                   "worst_cents": float(max(abs(r.cents) for r in rep)),
                   **{f"{r.note}_cents": r.cents for r in rep}}
        return score, metrics
