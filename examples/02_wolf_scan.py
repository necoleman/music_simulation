"""Scan every semitone of every string for wolf-prone notes.

    python examples/02_wolf_scan.py
"""
from pathlib import Path

from luthier.strings import wolf_scan, wolf_summary
from luthier.violin import Violin, ViolinSpec

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)

v = Violin(ViolinSpec())
res = wolf_scan(v.modes(40), v.bridge_top, semitones=range(0, 15))
print("notes with noticeable string/body hybridisation (mixing > 0.15):\n")
for r in res:
    if r.mixing > 0.15:
        print(r.row())
print("\n", wolf_summary(res))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in sorted({r.string for r in res}):
        rr = [r for r in res if r.string == name]
        ax.plot([r.f_note for r in rr], [r.mixing for r in rr], "o-", label=f"{name} string", ms=4)
    ax.axhline(0.5, color="r", ls="--", lw=1, label="wolf threshold")
    ax.set(xscale="log", xlabel="note [Hz]", ylabel="string/body mixing (0..1)", title="Wolf-proneness by note")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "wolf_scan.png", dpi=130)
    print("wrote", out / "wolf_scan.png")
except ImportError:
    pass
