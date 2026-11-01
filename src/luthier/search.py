"""Configuration-space search.

Define a space of named parameters, an objective that maps a parameter dict to
a score (lower is better) plus any diagnostic metrics, and run a search:

    space = ParameterSpace(Param("post__x", 0.018, 0.030),
                           Param("post__y", 0.150, 0.168))
    result = search(objective, space, method="lhs", n=40, seed=1)
    print(result.summary())

Methods: ``grid``, ``random``, ``lhs`` (Latin hypercube), ``sobol``, and ``de``
(scipy differential evolution; serial).  Every trial is recorded -- including
failures (invalid geometry etc., stored with ``score = inf``) -- and can be
dumped to JSON/CSV, so a long search is never a black box.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

Objective = Callable[[Dict[str, Any]], Union[float, Tuple[float, Dict[str, Any]], Dict[str, Any]]]


@dataclass(frozen=True)
class Param:
    """One searchable dimension.  ``name`` may be a ``ViolinSpec.with_`` path like ``post__x``."""
    name: str
    low: float = 0.0
    high: float = 1.0
    kind: str = "float"          # "float" | "int" | "choice"
    choices: Tuple = ()
    log: bool = False

    def decode(self, u: float):
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "choice":
            return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return int(round(v)) if self.kind == "int" else v


class ParameterSpace:
    def __init__(self, *params: Param):
        self.params = list(params)
        if len({p.name for p in params}) != len(params):
            raise ValueError("duplicate parameter names")

    def __len__(self):
        return len(self.params)

    def decode(self, u: Sequence[float]) -> Dict[str, Any]:
        return {p.name: p.decode(x) for p, x in zip(self.params, u)}

    def unit_samples(self, method: str, n: int, seed: int = 0) -> np.ndarray:
        d = len(self)
        if method == "random":
            return np.random.default_rng(seed).random((n, d))
        if method == "lhs":
            return qmc.LatinHypercube(d=d, seed=seed).random(n)
        if method == "sobol":
            return qmc.Sobol(d=d, seed=seed).random(n)
        if method == "grid":
            k = max(2, int(round(n ** (1.0 / d))))
            axes = []
            for p in self.params:
                m = len(p.choices) if p.kind == "choice" else k
                axes.append((np.arange(m) + 0.5) / m if p.kind == "choice" else np.linspace(0, 1, m))
            return np.array(list(itertools.product(*axes)))
        raise ValueError(f"unknown method {method!r}")


@dataclass
class Trial:
    params: Dict[str, Any]
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    seconds: float = 0.0


@dataclass
class SearchResult:
    space: ParameterSpace
    trials: List[Trial] = field(default_factory=list)

    @property
    def best(self) -> Trial:
        ok = [t for t in self.trials if np.isfinite(t.score)]
        if not ok:
            raise ValueError("no successful trials")
        return min(ok, key=lambda t: t.score)

    def ranked(self, k: int = 5) -> List[Trial]:
        return sorted((t for t in self.trials if np.isfinite(t.score)), key=lambda t: t.score)[:k]

    def summary(self, top: int = 5) -> str:
        n_fail = sum(1 for t in self.trials if t.error)
        lines = [f"{len(self.trials)} trials ({n_fail} failed), best score {self.best.score:.4g}"]
        for i, t in enumerate(self.ranked(top)):
            ps = ", ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in t.params.items())
            lines.append(f"  #{i + 1}  score {t.score:9.4g}   {ps}")
        return "\n".join(lines)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([t.__dict__ for t in self.trials], indent=1, default=_jsonable))

    def to_csv(self, path: str | Path) -> None:
        names = [p.name for p in self.space.params]
        mkeys = sorted({k for t in self.trials for k, v in t.metrics.items() if np.isscalar(v)})
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(names + ["score"] + mkeys + ["error"])
            for t in self.trials:
                w.writerow([t.params.get(n) for n in names] + [t.score] + [t.metrics.get(k) for k in mkeys] + [t.error])


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _evaluate(objective: Objective, params: Dict[str, Any]) -> Trial:
    t0 = time.time()
    try:
        out = objective(params)
        if isinstance(out, dict):
            score, metrics = float(out["score"]), {k: v for k, v in out.items() if k != "score"}
        elif isinstance(out, tuple):
            score, metrics = float(out[0]), dict(out[1])
        else:
            score, metrics = float(out), {}
        if not np.isfinite(score):
            raise ValueError(f"non-finite score {score}")
        return Trial(params, score, metrics, None, time.time() - t0)
    except Exception as e:  # invalid geometry, singular model, ... -> recorded, never fatal
        return Trial(params, float("inf"), {}, f"{type(e).__name__}: {e}", time.time() - t0)


def search(objective: Objective, space: ParameterSpace, *, method: str = "lhs", n: int = 30, seed: int = 0,
           workers: int = 1, on_trial: Optional[Callable[[int, Trial], None]] = None,
           de_options: Optional[dict] = None) -> SearchResult:
    """Run a search.  ``n`` is the evaluation budget (``de``: max function evaluations, approximately)."""
    res = SearchResult(space)

    def record(t: Trial):
        res.trials.append(t)
        if on_trial:
            on_trial(len(res.trials), t)

    if method == "de":
        cache: Dict[Tuple, float] = {}

        def f(u):
            key = tuple(np.round(u, 6))
            if key in cache:
                return cache[key]
            t = _evaluate(objective, space.decode(u))
            record(t)
            cache[key] = t.score if np.isfinite(t.score) else 1e12
            return cache[key]

        popsize = 5
        opts = dict(maxiter=max(1, n // (popsize * len(space)) - 1), popsize=popsize, seed=seed, tol=0.0,
                    polish=False, init="sobol", updating="immediate", workers=1)
        opts.update(de_options or {})
        differential_evolution(f, [(0, 1)] * len(space), **opts)
        return res

    U = space.unit_samples(method, n, seed)
    plist = [space.decode(u) for u in U]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for t in ex.map(_evaluate, itertools.repeat(objective), plist):
                record(t)
    else:
        for p in plist:
            record(_evaluate(objective, p))
    return res
