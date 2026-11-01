"""Beam element vs closed-form solutions."""
import numpy as np

from luthier import Model, Section, modal_analysis
from luthier.materials import STEEL, Orthotropic
from luthier.mesh import rect_mesh


def _cantilever(up, axis, n=20):
    m = Model()
    L = 0.5
    ids = [m.add_free_node(np.array(axis, float) * L * i / n) for i in range(n + 1)]
    sec = Section.rect(0.02, 0.004)
    for i in range(n):
        m.add_beam(ids[i], ids[i + 1], sec, STEEL, up=up)
    return modal_analysis(m, 6, fixed=[6 * ids[0] + k for k in range(6)], sigma_hz=-1).freqs


def test_cantilever_both_bending_planes():
    b, t, L = 0.02, 0.004, 0.5
    A = b * t
    f_soft = (1.8751 ** 2 / (2 * np.pi)) * np.sqrt(STEEL.E_L * (b * t ** 3 / 12) / (STEEL.rho * A * L ** 4))
    f_stiff = (1.8751 ** 2 / (2 * np.pi)) * np.sqrt(STEEL.E_L * (t * b ** 3 / 12) / (STEEL.rho * A * L ** 4))
    for up, axis in (((0, 0, 1), (1, 0, 0)), ((0, 1, 0), (1, 0, 0)), ((1, 0, 0), (0, 0, 1)), ((1, 1, 0), (0, 1, 0))):
        f = _cantilever(up, axis)
        assert abs(f[0] - f_soft) / f_soft < 0.01
        assert abs(f[1] - f_stiff) / f_stiff < 0.01


def test_eccentric_beam_on_plate_matches_composite_section():
    """A bar glued on a plate strip (rigid offset) must reproduce the composite-beam EI
    about the shifted neutral axis."""
    mat = Orthotropic.isotropic("nu0", 2700.0, 70e9, 0.0)
    L, b, h = 0.6, 0.03, 0.003
    bw, bh = 0.01, 0.012
    n = 60
    s = rect_mesh(L, b, n, 2)
    m = Model()
    m.add_shell_part("strip", s.vertices, s.faces, h, mat, grain=(1, 0, 0))
    # beam along the strip's centre line, on top (+z) of the plate
    xs = np.linspace(0, L, n + 1)
    off = (0, 0, h / 2 + bh / 2)
    for k in range(n):
        e1 = m.locate("strip", (xs[k], b / 2))
        e2 = m.locate("strip", (xs[k + 1], b / 2))
        from luthier.fem import End
        m.add_beam(End(e1[0], e1[1], off), End(e2[0], e2[1], off), Section.rect(bw, bh), mat, up=(0, 0, 1))
    f = m.__class__  # noqa
    modes = modal_analysis(m, 12)
    f_fe = modes.freqs[modes.flexible(5.0)][0]
    Ap, Ab = b * h, bw * bh
    zc = h / 2 + bh / 2
    zna = Ab * zc / (Ap + Ab)
    I = b * h ** 3 / 12 + Ap * zna ** 2 + bw * bh ** 3 / 12 + Ab * (zc - zna) ** 2
    rhoA = mat.rho * (Ap + Ab)
    f_ex = (4.730040745 ** 2 / (2 * np.pi)) * np.sqrt(mat.E_L * I / (rhoA * L ** 4))
    assert abs(f_fe - f_ex) / f_ex < 0.03, (f_fe, f_ex)


def test_from_measured_recovers_modulus_from_tap_tone():
    """Free-free strip tap tone -> E, round trip through the Euler-Bernoulli formula."""
    from luthier.materials import SPRUCE
    L, t = 0.30, 0.004
    coef = 4.730040745 ** 2 / (2 * np.pi)
    f_long = coef * np.sqrt(SPRUCE.E_L * t ** 2 / 12 / (SPRUCE.rho * L ** 4))
    f_cross = coef * np.sqrt(SPRUCE.E_C * t ** 2 / 12 / (SPRUCE.rho * L ** 4))
    cal = Orthotropic.from_measured(SPRUCE.scaled(E_L=0.5, E_C=2.0), f_long=f_long, f_cross=f_cross, length=L, thickness=t)
    assert abs(cal.E_L / SPRUCE.E_L - 1) < 1e-9 and abs(cal.E_C / SPRUCE.E_C - 1) < 1e-9
