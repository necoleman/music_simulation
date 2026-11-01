"""Air-cavity coupling against the rigid-wall Helmholtz formula, and CAD (OBJ) round trip."""
import numpy as np

from luthier import Model, modal_analysis
from luthier.materials import MAPLE, SPRUCE
from luthier.mesh import load_parts, save_obj
from luthier.objectives import air_mode_hz
from luthier.violin import Violin, ViolinSpec
from luthier.violin_geometry import PlateGeometry, build_plates


def test_rigid_wall_helmholtz():
    stiff = lambda m: m.scaled(E_L=1e3, E_C=1e3, G=1e3)
    spec = ViolinSpec(top_material=stiff(SPRUCE), back_material=stiff(MAPLE), rib_material=stiff(MAPLE))
    v = Violin(spec)
    f_fe = air_mode_hz(v, 12)
    assert abs(f_fe - v.helmholtz_rigid_hz) / v.helmholtz_rigid_hz < 0.03, (f_fe, v.helmholtz_rigid_hz)


def test_flexible_walls_lower_the_air_mode():
    v = Violin(ViolinSpec())
    assert air_mode_hz(v, 12) < v.helmholtz_rigid_hz


def test_enlarging_fholes_raises_air_mode():
    base = air_mode_hz(Violin(ViolinSpec()), 12)
    big = air_mode_hz(Violin(ViolinSpec().with_(plates__fhole__length=0.085, plates__fhole__width=0.0042)), 12)
    assert big > base


def test_obj_round_trip(tmp_path):
    pl = build_plates(PlateGeometry())
    parts = {"top": pl.top, "back": pl.back, "ribs": pl.ribs}
    path = tmp_path / "body.obj"
    save_obj(parts, path, units="mm")
    loaded = load_parts(path, units="mm")
    assert set(loaded) == {"top", "back", "ribs"}
    for k in parts:
        assert loaded[k].faces.shape == parts[k].faces.shape
        assert np.allclose(np.sort(loaded[k].vertices, axis=0), np.sort(parts[k].vertices, axis=0), atol=1e-6)

    def freqs(src):
        m = Model()
        for (name, p), mat in zip(src.items(), (SPRUCE, MAPLE, MAPLE)):
            m.add_shell_part(name, p.vertices, p.faces, 0.003, mat, grain=(0, 1, 0))
        md = modal_analysis(m, 12)
        return md.freqs[md.flexible()][:6]

    f_ref = freqs(parts)
    f_obj = freqs({k: loaded[k] for k in parts})
    assert np.allclose(f_ref, f_obj, rtol=1e-3)


def test_obj_parser_handles_quads_negative_indices_and_slashes(tmp_path):
    path = tmp_path / "q.obj"
    path.write_text("g plate\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1/1 2/2/1 3/3/1 4/4/1\ng other\nf -4//1 -3//1 -2//1\n")
    parts = load_parts(path, units="m")
    assert parts["plate"].faces.shape == (2, 3)  # quad -> 2 triangles
    assert parts["other"].faces.shape == (1, 3)
    assert abs(parts["plate"].areas().sum() - 1.0) < 1e-12
