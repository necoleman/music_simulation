"""Overtones of a note on this violin, and a WAV you can listen to (indicative only).

    python examples/05_note_sound.py
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from luthier.strings import VIOLIN_STRINGS, note_partials, render_pluck
from luthier.violin import Violin, ViolinSpec

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
v = Violin(ViolinSpec())
md = v.modes(40)
for string, semi in (("A", 0), ("G", 3), ("D", 8)):
    st = VIOLIN_STRINGS[string]
    parts = note_partials(md, v.bridge_top, st, semi, n_harm=10)
    print(f"\n{string} string +{semi} ({st.note(semi)[0]:.1f} Hz)")
    for p in parts:
        print(f"  n={p.n:2d}  ideal {p.f_ideal:8.2f}  coupled {p.f:8.2f} Hz  {p.shift_cents:+6.1f} cents   Q {p.q:6.0f}")
    y = render_pluck(parts, duration=2.5)
    wavfile.write(out / f"note_{string}{semi:+d}.wav", 44100, (y * 32000).astype(np.int16))
print("\nWAV files in", out)
