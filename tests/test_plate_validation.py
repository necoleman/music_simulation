"""Validate the shell element against closed-form plate solutions."""
import numpy as np
import pytest

from luthier import Model, modal_analysis
from luthier.materials import ALUMINIUM, SPRUCE, Orthotropic
from luthier.mesh import rect_mesh


def _plate(a, b, h, mat, n, grain=(1, 0, 0)):
    m = Model()
    s = rect_mesh(a, b, n, max(2, int(round(n * b / a))))
    m.add_shell_part("p", s.vertices, s.faces, h, mat, grain=grain)
    return m, s


def _edge_nodes(model, a, b, tol=1e-9):
    X = model.nodes
    return np.where((X[:, 0] < tol) | (X[:, 0] > a - tol) | (X[:, 1] < tol) | (X[:, 1] > b - tol))[0]


def _simply_supported_fixed(model, a, b):
    fixed = []
    for n in _edge_nodes(model, a, b):
        fixed += [6 * n, 6 * n + 1, 6 * n + 2]  # u, v, w = 0 ; rotations free
    return fixed


@pytest.mark.parametrize("mode", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_isotropic_simply_supported_plate(mode):
    a = b = 0.3
    h = 0.003
    m, _ = _plate(a, b, h, ALUMINIUM, 24)
    modes = modal_analysis(m, 12, fixed=_simply_supported_fixed(m, a, b))
    D = ALUMINIUM.E_L * h ** 3 / (12 * (1 - ALUMINIUM.nu_LC ** 2))
    mm, nn = mode
    f_exact = np.pi / 2 * (mm ** 2 / a ** 2 + nn ** 2 / b ** 2) * np.sqrt(D / (ALUMINIUM.rho * h))
    assert np.min(np.abs(modes.freqs - f_exact) / f_exact) < 0.03


def test_orthotropic_simply_supported_plate_grain_rotation_invariant():
    """Spruce plate, SS edges.  Exact: w^2 = pi^4/(rho h) [D11 (m/a)^4 + 2(D12+2D66)(mn/ab)^2 + D22 (n/b)^4]."""
    a, b, h = 0.36, 0.20, 0.003
    mat = SPRUCE
    Q = mat.plane_stress() * h ** 3 / 12
    D11, D22, D12, D66 = Q[0, 0], Q[1, 1], Q[0, 1], Q[2, 2]

    def exact(m_, n_):
        w2 = np.pi ** 4 / (mat.rho * h) * (D11 * (m_ / a) ** 4 + 2 * (D12 + 2 * D66) * (m_ * n_ / (a * b)) ** 2 + D22 * (n_ / b) ** 4)
        return np.sqrt(w2) / (2 * np.pi)

    m, _ = _plate(a, b, h, mat, 36, grain=(1, 0, 0))
    modes = modal_analysis(m, 10, fixed=_simply_supported_fixed(m, a, b))
    for mn in [(1, 1), (2, 1), (3, 1), (1, 2)]:
        fe = exact(*mn)
        assert np.min(np.abs(modes.freqs - fe) / fe) < 0.04, (mn, fe, modes.freqs[:8])


def test_free_square_plate_leissa():
    """Free square plate, nu=0.3: first flexible frequency parameters 13.47, 19.60, 24.27 (Leissa)."""
    mat = Orthotropic.isotropic("x", 2700.0, 69e9, 0.3)
    a, h = 0.25, 0.003
    m, _ = _plate(a, a, h, mat, 24)
    modes = modal_analysis(m, 14)
    D = mat.E_L * h ** 3 / (12 * (1 - 0.09))
    lam = 2 * np.pi * modes.freqs[modes.flexible(5.0)] * a ** 2 * np.sqrt(mat.rho * h / D)
    for target, got in zip([13.47, 19.60, 24.27], lam[:3]):
        assert abs(got - target) / target < 0.04, lam[:5]


def test_rigid_body_modes_present():
    mat = ALUMINIUM
    m, _ = _plate(0.2, 0.2, 0.002, mat, 10)
    modes = modal_analysis(m, 10)
    assert np.sum(modes.freqs < 2.0) == 6


def test_mesh_convergence_is_monotone_and_small():
    errs = []
    a = b = 0.3
    h = 0.003
    D = ALUMINIUM.E_L * h ** 3 / (12 * (1 - ALUMINIUM.nu_LC ** 2))
    f_exact = np.pi / 2 * (2 / a ** 2) * np.sqrt(D / (ALUMINIUM.rho * h))
    for n in (8, 16, 32):
        m, _ = _plate(a, b, h, ALUMINIUM, n)
        modes = modal_analysis(m, 6, fixed=_simply_supported_fixed(m, a, b))
        errs.append(abs(modes.freqs[0] - f_exact) / f_exact)
    assert errs[2] < errs[0]
    assert errs[2] < 0.01
