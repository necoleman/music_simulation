"""3-D acoustic kernel + recorder: validation against closed-form acoustics."""
import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from luthier.acoustics import Air, build_operators, solve_modes
from luthier.recorder import (BAROQUE_SOPRANO, Recorder, RecorderSpec, ToneHole, build_air_mesh, note_hz)

BORE = ((0.0, 0.0063), (0.10, 0.0055), (0.20, 0.0047), (0.308, 0.0040))
Z = {"T": 0.087, "1": 0.108, "2": 0.128, "3": 0.150, "4": 0.182, "5": 0.208, "6": 0.234, "7": 0.262}


def make_spec(**kw):
    holes = tuple(ToneHole(n, z, 0.0045, angle_deg=180 if n == "T" else 0) for n, z in Z.items())
    return RecorderSpec(BORE, holes, **kw)


@pytest.fixture(scope="module")
def rec():
    return Recorder(make_spec())


def test_note_hz():
    assert note_hz("A4") == pytest.approx(440.0)
    assert note_hz("C5") == pytest.approx(523.251, abs=1e-2)
    assert note_hz("Bb4") == pytest.approx(note_hz("A#4"))


def test_openings_detected_and_areas(rec):
    ops = rec.mesh.openings
    assert {"window", "foot", *Z} <= set(ops)
    for n in Z:
        assert ops[n].area == pytest.approx(np.pi * 0.00225 ** 2, rel=1e-9)
        assert 0.85 < ops[n].mesh_area / ops[n].area <= 1.0     # polygonal outline slightly smaller
    assert rec.mesh.volume > 0


def test_closing_all_holes_changes_only_bcs(rec):
    # same mesh, more open holes -> sharper, monotonic along the standard scale
    f = [rec.modes(h, 1).freqs[0] for h in [(), ("7",), ("6", "7"), ("4", "5", "6", "7")]]
    assert f == sorted(f)


def test_open_closed_pipe_pitch_and_wall_Q():
    """Straight pipe, open at the foot only: f ~ c/(2(L+2*0.6133a)) style check via the same Robin model,
    and Q within a Kirchhoff-theory band."""
    import gmsh as g
    from luthier.airmesh import AirMesh, OpeningSpec, mesh_from_gmsh
    a, L = 0.01, 0.30
    g.initialize()
    try:
        g.option.setNumber("General.Terminal", 0)
        g.model.add("p")
        g.model.occ.addCylinder(0, 0, 0, 0, 0, L, a)
        g.model.occ.synchronize()
        g.option.setNumber("Mesh.MeshSizeMax", 0.004)
        g.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        g.model.mesh.generate(3)
        specs = [OpeningSpec("end", (0, 0, L), (0, 0, 1), (1, 0, 0), (a, a), end_corr=0.6133),
                 OpeningSpec("end0", (0, 0, 0), (0, 0, -1), (1, 0, 0), (a, a), end_corr=0.6133)]
        mesh = mesh_from_gmsh(specs)
    finally:
        g.finalize()
    air = Air.at(20.0)
    ops = build_operators(mesh)
    md = solve_modes(ops, ("end",), 3, air=air, f_guess=100.0)       # closed-open: f_n = (2n-1) c / 4 L_eff
    Leff = L + 0.6133 * a
    expect = air.c / (4 * Leff)
    assert md.freqs[0] == pytest.approx(expect, rel=0.01)
    assert md.freqs[1] / md.freqs[0] == pytest.approx(3.0, rel=0.01)
    assert md.Q[0] > 20


def test_baroque_fingerings_monotonic_and_harmonic(rec):
    rep = rec.analyse()
    f1 = [r.f1 for r in rep]
    assert f1 == sorted(f1)
    # plain closed bore: second mode ~ octave
    assert abs(rep[0].stretch_cents) < 40
    assert all(r.Q1 > 5 for r in rep)


def test_hole_position_moves_pitch_the_right_way():
    base = make_spec()
    r0 = Recorder(base).modes(("7",), 1).freqs[0]
    r1 = Recorder(base.with_(holes__7__z=0.272)).modes(("7",), 1).freqs[0]    # hole 7 moved toward the foot
    assert r1 < r0


def test_unknown_hole_rejected(rec):
    with pytest.raises(KeyError):
        rec.modes(("nope",))


def test_mesh_convergence_cents():
    s16, s20 = make_spec(), make_spec(curvature_points=20, mesh_size=0.0035, mesh_min=0.0007)
    f16 = Recorder(s16).modes(("4", "5", "6", "7"), 1).freqs[0]
    f20 = Recorder(s20).modes(("4", "5", "6", "7"), 1).freqs[0]
    assert abs(1200 * np.log2(f16 / f20)) < 8.0


# ---------------------------------------------------------------- jet / playing pitch
def test_jet_pitch_rises_with_pressure_and_overblows(rec):
    pytest.importorskip("luthier.jet")
    f = []
    for P in (150.0, 220.0, 300.0):
        _, c = rec.play((), __import__("luthier.jet", fromlist=["Jet"]).Jet(blowing_pressure=P))
        assert c[0].mode == 0 and c[0].gain >= 1.0
        f.append(c[0].f)
    assert f == sorted(f) and f[-1] / f[0] > 1.005          # sharpens with blowing pressure
    f_res = rec.modes((), 1).freqs[0]
    assert abs(1200 * np.log2(f[0] / f_res)) < 60           # stays near the air-column resonance
    from luthier.jet import Jet
    rows = rec.blow_range((), [1200.0], Jet())
    assert rows[0][3] >= 1 and rows[0][1] > 1.8 * f_res      # hard blowing -> upper register


def test_jet_loop_gain_scales_with_coupling(rec):
    from luthier.jet import Jet
    _, c1 = rec.play((), Jet(blowing_pressure=200.0, gain=1.0))
    _, c2 = rec.play((), Jet(blowing_pressure=200.0, gain=0.4))
    assert c2[0].gain < c1[0].gain if c2 else True


def test_playing_report_flags_weak_notes(rec):
    from luthier.jet import Jet
    rep = rec.analyse(jet=Jet(blowing_pressure=300.0))                       # same pressure for every note
    assert rep[0].speaks and not rep[-1].speaks                              # D6 does not speak at 300 Pa
    rep2 = rec.analyse(jet=Jet(blowing_pressure=300.0, pressure_ref_hz=523.25))
    assert all(r.speaks for r in rep2)
