"""String-body coupling vs closed-form two-oscillator theory."""
import numpy as np

from luthier.strings import VIOLIN_STRINGS, BodyMobility, analyse_note, coupled_roots


def test_uncoupled_string_roots_are_harmonic():
    s = VIOLIN_STRINGS["G"]
    null = BodyMobility([0.0], [2 * np.pi * 1000], [0.02])
    r = coupled_roots(null, s, s.length, s.f0)
    assert len(r) == 1
    assert abs(r[0].real / (2 * np.pi) - s.f0) < 1e-3
    assert abs(r[0].real / (2 * r[0].imag) - s.Q) / s.Q < 1e-3


def test_roots_match_two_oscillator_theory():
    s = VIOLIN_STRINGS["A"]
    L = s.length
    rng = np.random.default_rng(1)
    for _ in range(80):
        fb, M, zb = rng.uniform(380, 500), rng.uniform(0.01, 0.15), rng.uniform(0.01, 0.04)
        wb = 2 * np.pi * fb
        r = coupled_roots(BodyMobility([1 / M], [wb], [zb]), s, L, 440.0, window=0.08)
        ws = 2 * np.pi * 440.0
        b = (wb - ws) + 1j * zb * wb
        g2 = s.tension / (2 * L * M)
        d = np.array([(b + np.sqrt(b * b + 4 * g2)) / 2, (b - np.sqrt(b * b + 4 * g2)) / 2])
        th = ws + d
        th = th[(abs(th.real / ws - 1) < 0.06) & (th.imag > 0)]
        for x in th:
            assert len(r) and np.min(abs(r - x)) < 0.01 * abs(x), (fb, M, zb)


def test_mixing_limits():
    s = VIOLIN_STRINGS["A"]
    heavy = analyse_note(BodyMobility([1 / 20.0], [2 * np.pi * 440.5], [0.02]), s, 0)
    light = analyse_note(BodyMobility([1 / 0.05], [2 * np.pi * 440.5], [0.02]), s, 0)
    assert heavy.mixing < 0.05 and not heavy.is_wolf
    assert light.mixing > 0.8 and light.is_wolf


def test_string_tension_is_physical():
    for st in VIOLIN_STRINGS.values():
        assert 40 < st.tension < 90


def test_partials_of_uncoupled_string_are_harmonic_and_body_shifts_them():
    from luthier.strings import note_partials, render_pluck
    from luthier.violin import Violin, ViolinSpec
    v = Violin(ViolinSpec())
    md = v.modes(30)
    parts = note_partials(md, v.bridge_top, VIOLIN_STRINGS["A"], 0, n_harm=6)
    assert [p.n for p in parts] == [1, 2, 3, 4, 5, 6]
    assert all(abs(p.shift_cents) < 60 for p in parts)          # body perturbs, does not wreck, the series
    assert any(abs(p.shift_cents) > 0.5 for p in parts)          # ...but does perturb it
    y = render_pluck(parts, duration=0.5)
    assert np.isfinite(y).all() and abs(abs(y).max() - 1) < 1e-9
