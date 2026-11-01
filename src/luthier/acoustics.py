"""3-D linear acoustics of an air volume: Helmholtz FEM + modal analysis.

Governing equation (harmonic, e^{+j w t}):   laplace p + k^2 p = 0 in the air.

Boundaries
----------
* rigid walls            dp/dn = 0 (a closed finger hole is just more wall)
* open ends / tone holes radiate.  The reactive part of the radiation impedance
  is the inertia of the end correction, Z = j w rho l, which gives the
  *frequency-independent* Robin condition  dp/dn = -p / l  and keeps the problem
  a real symmetric generalised eigenproblem

        (K + sum_open B_i / l_i) p = k^2 M p .

* losses are then added per mode by perturbation, from the mode's own fields:
    - radiation:   P = 1/2 Re(1/Z) |p|^2 integrated over each open face, with the
                   Levine-Schwinger resistance (ka)^2/4 .. (ka)^2/2 per rho c
    - viscous:     P = (delta_v / 4 rho w) |grad_t p|^2 integrated over the walls
    - thermal:     P = (gamma-1) w delta_t |p|^2 / (4 rho c^2) over the walls
  and the amplitude decay rate is alpha = P_total / (2 E),  E = int p^2/(2 rho c^2) dV,
  so Q = w / (2 alpha).  (Checked against Kirchhoff's tube attenuation in the tests.)

The 1-D "transfer matrix" picture is *not* assumed anywhere: bore taper, chimney
height, undercut, hole shape and the volume of closed holes all enter through the
real geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .airmesh import AirMesh


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Air:
    c: float
    rho: float
    nu: float       # kinematic viscosity [m^2/s]
    chi: float      # thermal diffusivity [m^2/s]
    gamma: float = 1.4

    @classmethod
    def at(cls, temp_c: float = 20.0) -> "Air":
        """Dry air at 101.325 kPa.  Breath-warmed air in a played recorder is ~ 25-30 C."""
        T = temp_c + 273.15
        r = (T / 293.15)
        return cls(c=343.2 * np.sqrt(r), rho=1.204 / r, nu=1.516e-5 * r ** 1.76, chi=2.14e-5 * r ** 1.76)


# --------------------------------------------------------------------------- #
@dataclass
class Operators:
    mesh: AirMesh
    K: sp.csc_matrix
    M: sp.csc_matrix
    B: Dict[str, sp.csc_matrix]     # boundary mass of each opening
    B_wall_all: sp.csc_matrix       # boundary mass of *all* boundary faces (restrict per fingering)
    grad: sp.csr_matrix             # (3M, N) gradient operator
    tri_mass_blocks: tuple          # cached triangle data


def _tri_mass(mesh: AirMesh, face_idx: np.ndarray) -> sp.csc_matrix:
    P = mesh.nodes[mesh.tris[face_idx]]
    A = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    Be = (A / 12.0)[:, None, None] * (np.ones((3, 3)) + np.eye(3))
    idx = mesh.tris[face_idx]
    r = np.repeat(idx, 3, axis=1).ravel()
    c = np.tile(idx, (1, 3)).ravel()
    N = len(mesh.nodes)
    return sp.coo_matrix((Be.ravel(), (r, c)), shape=(N, N)).tocsc()


def build_operators(mesh: AirMesh) -> Operators:
    X, T = mesh.nodes, mesh.tets
    N, Mn = len(X), len(T)
    P = X[T]
    J = P[:, 1:] - P[:, :1]
    detJ = np.linalg.det(J)
    V = np.abs(detJ) / 6.0
    Jinv = np.linalg.inv(J)
    gradN = np.concatenate([-Jinv.sum(2)[:, :, None], Jinv], axis=2)  # (M,3,4)
    Ke = V[:, None, None] * np.einsum("mdi,mdj->mij", gradN, gradN)
    Me = (V / 20.0)[:, None, None] * (np.ones((4, 4)) + np.eye(4))
    r = np.repeat(T, 4, axis=1).ravel()
    c = np.tile(T, (1, 4)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (r, c)), shape=(N, N)).tocsc()
    M = sp.coo_matrix((Me.ravel(), (r, c)), shape=(N, N)).tocsc()
    rows = (3 * np.arange(Mn)[:, None, None] + np.arange(3)[None, :, None]) * np.ones((1, 1, 4), int)
    cols = T[:, None, :] * np.ones((1, 3, 1), int)
    grad = sp.coo_matrix((gradN.ravel(), (rows.ravel(), cols.ravel())), shape=(3 * Mn, N)).tocsr()
    B = {name: _tri_mass(mesh, op.faces) for name, op in mesh.openings.items()}
    return Operators(mesh, K, M, B, _tri_mass(mesh, np.arange(len(mesh.tris))), grad, ())


# --------------------------------------------------------------------------- #
@dataclass
class AcousticModes:
    freqs: np.ndarray          # Hz
    Q: np.ndarray              # quality factors
    p: np.ndarray              # (N, n) pressure shapes, M-normalised (p^T M p = 1)
    loss: Dict[str, np.ndarray]  # per-mode power-loss shares: 'radiation', 'viscous', 'thermal' (each as 1/Q contribution)
    open_names: tuple
    ops: Operators
    air: Air

    def impedance(self, freqs: np.ndarray, node: int) -> np.ndarray:
        """Point input impedance p/q [Pa s/m^3] at ``node`` for a volume-velocity source there
        (modal sum, lightly damped modes)."""
        w = 2 * np.pi * np.asarray(freqs)[:, None]
        wn = 2 * np.pi * self.freqs[None]
        z = (1.0 / (2.0 * self.Q))[None]
        num = 1j * w * self.air.rho * self.air.c ** 2 * (self.p[node][None]) ** 2
        return np.sum(num / (wn ** 2 - w ** 2 + 2j * z * wn * w), axis=1)

    def pressure_at(self, mode: int, xyz) -> float:
        i = int(np.argmin(np.linalg.norm(self.ops.mesh.nodes - np.asarray(xyz), axis=1)))
        return float(self.p[i, mode])


def solve_modes(ops: Operators, open_names: Sequence[str], n_modes: int = 8, *, air: Air = Air.at(20.0),
                f_guess: float = 40.0, radiation_loss: bool = True, wall_loss: bool = True) -> AcousticModes:
    """Lowest ``n_modes`` modes above ``f_guess`` Hz for a given set of open openings (all others rigid)."""
    mesh = ops.mesh
    open_names = tuple(open_names)
    A = ops.K.copy()
    for n in open_names:
        A = A + (mesh.openings[n].robin_scale / mesh.openings[n].end_length) * ops.B[n]
    sigma = (2 * np.pi * f_guess / air.c) ** 2
    if not open_names:
        sigma = -abs(sigma) - 1.0
    k2, vec = spla.eigsh(A.tocsc(), k=n_modes, M=ops.M, sigma=sigma, which="LM")
    order = np.argsort(k2)
    k2, vec = k2[order], vec[:, order]
    keep = k2 > 1e-6
    k2, vec = k2[keep], vec[:, keep]
    k = np.sqrt(k2)
    w = k * air.c
    f = w / (2 * np.pi)
    # M-normalise
    vec = vec / np.sqrt(np.einsum("ij,ij->j", vec, ops.M @ vec))[None]

    E = 0.5 * 1.0 / (air.rho * air.c ** 2)        # p^T M p = 1  ->  E = int p^2 / (2 rho c^2)
    n = len(f)
    P_rad, P_visc, P_th = np.zeros(n), np.zeros(n), np.zeros(n)

    if radiation_loss:
        for name in open_names:
            op = mesh.openings[name]
            ell = op.end_length
            frac = np.clip((op.spec.end_corr - 0.6133) / (0.8216 - 0.6133), 0.0, 1.0)   # 0 unflanged .. 1 flanged
            for i in range(n):
                ka = k[i] * op.a_eq
                R = ka ** 2 * (0.25 + 0.25 * frac)
                X = k[i] * ell
                reY = R / (R ** 2 + X ** 2) / (air.rho * air.c)
                P_rad[i] += 0.5 * reY * op.robin_scale * vec[:, i] @ (ops.B[name] @ vec[:, i])
    if wall_loss:
        wall = mesh.wall_faces(open_names)
        Bw = _tri_mass(mesh, wall)
        Pn = mesh.nodes[mesh.tris[wall]]
        nrm = np.cross(Pn[:, 1] - Pn[:, 0], Pn[:, 2] - Pn[:, 0])
        Aw = 0.5 * np.linalg.norm(nrm, axis=1)
        nrm /= 2 * Aw[:, None]
        tet = mesh.tri_tet[wall]
        for i in range(n):
            wi = w[i]
            dv = np.sqrt(2 * air.nu / wi)
            dt = np.sqrt(2 * air.chi / wi)
            g = (ops.grad @ vec[:, i]).reshape(-1, 3)[tet]
            gt = g - np.einsum("ij,ij->i", g, nrm)[:, None] * nrm
            P_visc[i] = np.sum(Aw * np.einsum("ij,ij->i", gt, gt)) * dv / (4 * air.rho * wi)
            P_th[i] = (air.gamma - 1) * wi * dt / (4 * air.rho * air.c ** 2) * vec[:, i] @ (Bw @ vec[:, i])
    alpha = (P_rad + P_visc + P_th) / (2 * E)
    Q = w / (2 * np.maximum(alpha, 1e-30))
    loss = {"radiation": 2 * (P_rad / (2 * E)) / w, "viscous": 2 * (P_visc / (2 * E)) / w, "thermal": 2 * (P_th / (2 * E)) / w}
    return AcousticModes(f, Q, vec, loss, open_names, ops, air)
