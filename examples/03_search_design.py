"""Search a configuration space: sound-post position and f-hole length.

Each evaluation rebuilds the model (~7 s).  Latin-hypercube first, then refine
around the best with differential evolution.

    python examples/03_search_design.py            # quick (24 evaluations)
    python examples/03_search_design.py 120        # bigger budget
"""
import sys
from pathlib import Path

from luthier.objectives import ViolinObjective
from luthier.search import Param, ParameterSpace, search
from luthier.violin import ViolinSpec

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 24
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)

space = ParameterSpace(
    Param("post__x", 0.018, 0.030),                    # sound-post sideways position [m]
    Param("post__y", 0.150, 0.168),                    # ... and along the body
    Param("plates__fhole__length", 0.066, 0.084),      # f-hole length [m]
)
# Aim the air resonance at A0 = 280 Hz and avoid wolf-prone notes.
objective = ViolinObjective(base=ViolinSpec(), a0_target=280.0, wolf_weight=1.0)

result = search(objective, space, method="lhs", n=budget, seed=0,
                on_trial=lambda i, t: print(f"[{i:3d}] score {t.score:8.3f}  A0 {t.metrics.get('A0_hz', float('nan')):6.1f} Hz  "
                                            f"wolves {t.metrics.get('n_wolf', '-')}  {t.error or ''}"))
print("\n" + result.summary(5))
result.to_csv(out / "search_trials.csv")
result.to_json(out / "search_trials.json")
print("trial log written to", out)
