"""Material models.

All quantities are SI (Pa, kg/m^3, m).  Wood is modelled as an orthotropic
plate material: stiffness *along the grain* (L) and *across the grain in the
plane of the plate* (C), plus the two through-thickness shear moduli that
matter for Mindlin plate theory.

The preset numbers are typical handbook values (Wood Handbook, FPL-GTR-190;
Haines; Caldersmith) and vary by 20-30 % between boards.  Treat them as
starting points and replace them with measured values (see
``Orthotropic.from_measured``) whenever you can: tap-tone / Chladni
measurements of the actual board are by far the most valuable calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Orthotropic:
    """Orthotropic (or isotropic) material for thin plates and shells.

    Attributes
    ----------
    rho   : density [kg/m^3]
    E_L   : Young's modulus along the grain [Pa]
    E_C   : Young's modulus across the grain, in the plate plane [Pa]
    G_LC  : in-plane shear modulus [Pa]
    nu_LC : Poisson ratio (strain across grain caused by stress along grain)
    G_Lz  : through-thickness shear modulus, plane containing the grain [Pa]
    G_Cz  : through-thickness shear modulus, plane across the grain [Pa]
    """

    name: str
    rho: float
    E_L: float
    E_C: float
    G_LC: float
    nu_LC: float
    G_Lz: float
    G_Cz: float

    # ------------------------------------------------------------------ #
    @classmethod
    def isotropic(cls, name: str, rho: float, E: float, nu: float) -> "Orthotropic":
        G = E / (2.0 * (1.0 + nu))
        return cls(name, rho, E, E, G, nu, G, G)

    def scaled(self, *, rho=1.0, E_L=1.0, E_C=1.0, G=1.0) -> "Orthotropic":
        """Return a copy with multiplicative tweaks (handy in searches)."""
        return replace(
            self,
            rho=self.rho * rho,
            E_L=self.E_L * E_L,
            E_C=self.E_C * E_C,
            G_LC=self.G_LC * G,
            G_Lz=self.G_Lz * G,
            G_Cz=self.G_Cz * G,
        )

    # ------------------------------------------------------------------ #
    @property
    def nu_CL(self) -> float:
        return self.nu_LC * self.E_C / self.E_L

    def plane_stress(self) -> np.ndarray:
        """3x3 plane-stress stiffness in material axes (L, C): [s_L, s_C, t_LC]."""
        nu_cl = self.nu_CL
        d = 1.0 - self.nu_LC * nu_cl
        return np.array(
            [
                [self.E_L / d, self.nu_LC * self.E_C / d, 0.0],
                [self.nu_LC * self.E_C / d, self.E_C / d, 0.0],
                [0.0, 0.0, self.G_LC],
            ]
        )

    def shear(self) -> np.ndarray:
        """2x2 through-thickness shear stiffness in material axes (L, C)."""
        return np.diag([self.G_Lz, self.G_Cz])

    @property
    def speed_of_sound_L(self) -> float:
        return float(np.sqrt(self.E_L / self.rho))

    @classmethod
    def from_measured(cls, base: "Orthotropic", *, rho=None, f_long=None, f_cross=None,
                      length=None, width=None, thickness=None) -> "Orthotropic":
        """Tap-tone calibration of one board.

        Given the free-free first bending frequencies of a rectangular strip cut
        along (``f_long``) and across (``f_cross``) the grain, back out E_L and
        E_C using the Euler-Bernoulli free-free beam formula
        ``f1 = (4.730^2 / 2 pi) sqrt(E I / (rho A L^4))``.
        Only ``length`` and ``thickness`` are needed (the same strip geometry
        is assumed for both measurements); ``width`` is accepted for symmetry.
        """
        rho = base.rho if rho is None else rho
        out = base
        out = replace(out, rho=rho)
        coef = (4.730040745 ** 2 / (2 * np.pi))
        if f_long is not None:
            E = (f_long / coef) ** 2 * 12.0 * rho * length ** 4 / thickness ** 2
            out = replace(out, E_L=E)
        if f_cross is not None:
            E = (f_cross / coef) ** 2 * 12.0 * rho * length ** 4 / thickness ** 2
            out = replace(out, E_C=E)
        return out


# --------------------------------------------------------------------------- #
# Presets (plates are quarter-sawn: face = L-R plane, thickness = tangential)
# --------------------------------------------------------------------------- #
SPRUCE = Orthotropic("spruce", 420.0, 11.6e9, 0.90e9, 0.72e9, 0.37, 0.71e9, 0.04e9)
MAPLE = Orthotropic("maple", 620.0, 10.0e9, 2.00e9, 1.20e9, 0.42, 1.10e9, 0.40e9)
EBONY = Orthotropic("ebony", 1150.0, 17.0e9, 2.5e9, 1.5e9, 0.40, 1.5e9, 0.6e9)
STEEL = Orthotropic.isotropic("steel", 7850.0, 200e9, 0.30)
BRASS = Orthotropic.isotropic("brass", 8500.0, 100e9, 0.34)
ALUMINIUM = Orthotropic.isotropic("aluminium", 2700.0, 69e9, 0.33)
BIRCH_PLYWOOD = Orthotropic("birch ply", 680.0, 12.0e9, 6.0e9, 0.8e9, 0.30, 0.6e9, 0.5e9)

PRESETS = {m.name: m for m in (SPRUCE, MAPLE, EBONY, STEEL, BRASS, ALUMINIUM, BIRCH_PLYWOOD)}

# Air at 20 C
RHO_AIR = 1.204
C_AIR = 343.2
