"""Use your own CAD mesh instead of the procedural violin.

Export each plate as a *mid-surface sheet* (not a closed solid) into one OBJ with a
group per part (``g top`` / ``g back`` / ``g ribs``).  Here we fabricate such a file
from the procedural generator so the example is self-contained -- replace
``demo.obj`` with your own export.

    python examples/04_cad_import.py [your_body.obj]
"""
import sys
from pathlib import Path

import numpy as np

from luthier import Model, modal_analysis
from luthier.materials import MAPLE, SPRUCE
from luthier.mesh import load_parts, save_obj
from luthier.violin_geometry import PlateGeometry, build_plates

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    pl = build_plates(PlateGeometry())
    path = out / "demo.obj"
    save_obj({"top": pl.top, "back": pl.back, "ribs": pl.ribs}, path, units="mm")

parts = load_parts(path, units="mm")
print("parts found:", {k: f"{len(v.faces)} tris" for k, v in parts.items()})

# Attach what CAD does not carry: thickness, material, grain direction.
# Thickness may be a number, a per-triangle array, or a function of the triangle centroids.
def graded(centroids):              # example: thicker toward the centre line
    return 0.0025 + 0.0006 * np.exp(-(centroids[:, 0] / 0.05) ** 2)

m = Model()
m.add_shell_part("top", parts["top"].vertices, parts["top"].faces, graded, SPRUCE, grain=(0, 1, 0))
m.add_shell_part("back", parts["back"].vertices, parts["back"].faces, 0.004, MAPLE, grain=(0, 1, 0))
m.add_shell_part("ribs", parts["ribs"].vertices, parts["ribs"].faces, 0.0011, MAPLE,
                 grain=lambda c, n: np.cross([0, 0, 1.0], n))        # grain runs round the rib
md = modal_analysis(m, 20)
print("free-body modes (Hz):", np.round(md.freqs[md.flexible()][:10], 1))
