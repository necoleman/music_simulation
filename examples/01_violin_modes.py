"""Build the reference violin, list its body modes, and plot the bridge mobility.

    python examples/01_violin_modes.py
"""
from pathlib import Path

import numpy as np

from luthier.violin import Violin, ViolinSpec

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)

v = Violin(ViolinSpec())
print(f"{v.model.n_nodes} nodes, {v.model.n_dof} dofs, body mass {v.total_mass()*1e3:.0f} g, "
      f"cavity {v.cavity_volume*1e3:.2f} L, rigid-wall Helmholtz {v.helmholtz_rigid_hz:.0f} Hz")
print(f"\n{'#':>3} {'Hz':>8}  {'air%':>5}  {'bridge x-motion':>16}")
for i, f, air, bx in v.mode_table(24)[:16]:
    print(f"{i:3d} {f:8.1f}  {100*air:5.0f}  {bx:16.3f}")

freqs = np.linspace(100, 3000, 3000)
Yx = v.bridge_admittance(freqs, (1, 0, 0))
Yz = v.bridge_admittance(freqs, (0, 0, 1))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(freqs, abs(Yx) * 1e3, label="across body (bowing direction)")
    ax.semilogy(freqs, abs(Yz) * 1e3, label="vertical", alpha=0.7)
    ax.set(xlabel="Hz", ylabel="|mobility| at bridge notch [mm/s/N]", title="Bridge mobility (model)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "bridge_mobility.png", dpi=130)
    print("\nwrote", out / "bridge_mobility.png")
except ImportError:
    print("(install matplotlib to get the plot)")
