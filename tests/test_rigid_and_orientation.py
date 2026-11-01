"""Frame invariance: results must not depend on how the model is oriented in space."""
import numpy as np

from luthier import Model, modal_analysis
from luthier.materials import SPRUCE
from luthier.mesh import rect_mesh


def _rot(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def test_rotated_orthotropic_plate_same_frequencies():
    s = rect_mesh(0.36, 0.2, 18, 10)
    base = Model()
    base.add_shell_part("p", s.vertices, s.faces, 0.003, SPRUCE, grain=(1, 0, 0))
    f0 = modal_analysis(base, 14).freqs
    R = _rot(0.7, -1.1, 2.0)
    rot = Model()
    rot.add_shell_part("p", s.vertices @ R.T, s.faces, 0.003, SPRUCE, grain=R @ np.array([1.0, 0, 0]))
    f1 = modal_analysis(rot, 14).freqs
    assert np.allclose(f0, f1, rtol=2e-4, atol=1e-3)


def test_grain_direction_matters():
    s = rect_mesh(0.36, 0.2, 18, 10)
    out = []
    for g in ((1, 0, 0), (0, 1, 0)):
        m = Model()
        m.add_shell_part("p", s.vertices, s.faces, 0.003, SPRUCE, grain=g)
        md = modal_analysis(m, 8)
        out.append(md.freqs[md.flexible()][0])
    assert abs(out[0] - out[1]) / out[0] > 0.1


def test_violin_has_six_zero_energy_rigid_modes():
    from luthier.violin import Violin, ViolinSpec
    v = Violin(ViolinSpec())
    md = v.modes(20)
    K, _ = v.model.assemble()
    R = md.shapes[:, :6]
    scale = abs(K).max()
    for j in range(6):
        assert abs(R[:, j] @ (K @ R[:, j])) < 1e-8 * scale
    assert np.all(md.freqs[:6] == 0.0) and md.freqs[6] > 50
