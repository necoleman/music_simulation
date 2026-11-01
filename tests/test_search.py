import numpy as np

from luthier.search import Param, ParameterSpace, search


def test_decode_and_kinds():
    sp = ParameterSpace(Param("a", 0, 10), Param("n", 1, 5, kind="int"), Param("c", kind="choice", choices=("x", "y", "z")),
                        Param("l", 1e-3, 1e3, log=True))
    d = sp.decode([0.5, 1.0, 0.99, 0.5])
    assert d["a"] == 5 and d["n"] == 5 and d["c"] == "z" and abs(d["l"] - 1.0) < 1e-9


def test_methods_find_quadratic_minimum():
    sp = ParameterSpace(Param("x", -2, 2), Param("y", -2, 2))
    f = lambda p: (p["x"] - 0.7) ** 2 + (p["y"] + 0.4) ** 2
    for method, n in (("lhs", 80), ("sobol", 64), ("grid", 81), ("random", 80)):
        r = search(f, sp, method=method, n=n, seed=1)
        assert r.best.score < 0.2, method
    r = search(f, sp, method="de", n=300, seed=1)
    assert r.best.score < 1e-3


def test_failures_are_recorded_not_fatal():
    sp = ParameterSpace(Param("x", 0, 1))

    def f(p):
        if p["x"] > 0.5:
            raise ValueError("bad geometry")
        return p["x"], {"note": 1}

    r = search(f, sp, method="grid", n=10)
    assert any(t.error for t in r.trials) and r.best.score == 0.0
