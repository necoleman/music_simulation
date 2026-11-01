"""Recorder: 3-D air-column model, intonation report, hole-position tuning, then a search.

    python examples/06_recorder_tuning.py

Each fingering solve is a full 3-D Helmholtz eigenproblem on a tetrahedral mesh of the air
(bore + chimneys + window), ~0.5 s; building the mesh takes ~5 s.
"""
from luthier.jet import Jet
from luthier.recorder import (BAROQUE_SOPRANO, Recorder, RecorderSpec, ToneHole, tune_hole_positions)

bore = ((0.0, 0.0063), (0.10, 0.0055), (0.20, 0.0047), (0.308, 0.0040))      # (z, radius) [m], z=0 at the block
z0 = {"T": 0.087, "1": 0.108, "2": 0.128, "3": 0.150, "4": 0.182, "5": 0.208, "6": 0.234, "7": 0.262}
holes = tuple(ToneHole(n, z, 0.0045, angle_deg=180 if n == "T" else 0) for n, z in z0.items())
spec = RecorderSpec(bore, holes)                        # breath-warmed air, 26 C

print("Initial guess:")
for r in Recorder(spec).analyse():
    print(" ", r.row())

print("\nTuning hole positions (3-D solve each iteration):")
spec, hist = tune_hole_positions(spec, iterations=4,
                                 on_iter=lambda i, s, rep: print(f"  iter {i}: " + " ".join(f"{r.note}{r.cents:+.0f}" for r in rep)))
print("\nTuned hole centres [mm]:", {h.name: round(1000 * h.z, 1) for h in spec.holes})
print("\nFinal report:")
for r in hist[-1]:
    print(" ", r.row())

# Playing pitches: close the air column with the air jet (windway + labium).  300 Pa at C5, harder for higher notes.
jet = Jet(blowing_pressure=300.0, pressure_ref_hz=523.25, flue_height=0.001)
print("\nPlaying pitches with the jet (first register):")
rec = Recorder(spec)
for r in rec.analyse(jet=jet):
    print(" ", r.row())

print("\nRe-tuning the holes for *playing* pitch:")
spec, hist = tune_hole_positions(spec, iterations=3, jet=jet,
                                 on_iter=lambda i, s, rep: print(f"  iter {i}: " + " ".join(f"{r.note}{r.cents:+.0f}" for r in rep)))

print("\nC5 vs blowing pressure (register changes):")
for P, f, g, mode in Recorder(spec).blow_range((), [150, 250, 400, 700, 1200, 2000]):
    print(f"  {P:6.0f} Pa  -> {f:7.1f} Hz   gain {g:4.2f}   air-column mode {mode + 1}")

# Next steps with the generic search layer, e.g. vary hole diameters / undercut / bore taper:
#   from luthier.objectives import RecorderObjective
#   from luthier.search import Param, ParameterSpace, search
#   space = ParameterSpace(Param("holes__3__diameter", 0.004, 0.006), Param("holes__3__undercut", 0.0, 0.0015))
#   search(RecorderObjective(base=spec), space, method="lhs", n=20)
