# luthier

Simulate a musical instrument **before it is built**: predict its resonances, overtones and
problem notes from its geometry and the properties of the wood, and search a space of
design parameters for the best configuration.

**Status: v0.2 — violin and recorder.** The shared core (materials, mesh import, finite
elements, modal analysis, search) is in place and validated. The violin is a shell-FEM model;
the recorder is a **full 3-D acoustic FEM** of the air inside the instrument (bore, chimneys,
window) — no 1-D waveguide assumption anywhere.

## What works today

| Your use case | Status | How |
|---|---|---|
| Given a violin, find its **wolf tones** | ✅ working (model-level) | `wolf_scan` — every semitone of every string, [example 02](examples/02_wolf_scan.py) |
| Violin: best **f-hole** shape/position | ✅ search works; absolute accuracy needs calibration | `Param("plates__fhole__length", …)` + `search` — [example 03](examples/03_search_design.py) |
| Violin: best **sound-post / bass-bar / bridge** placement | ✅ same | `post__x`, `post__y`, `bass_bar__x`, `bridge__y`, … |
| Simulate an instrument from a **CAD mesh** | ✅ OBJ/STL mid-surface sheets | `load_parts` — [example 04](examples/04_cad_import.py) |
| **Overtones / sound** of a note | ✅ partials, body-induced shifts & decays; indicative WAV | `note_partials`, `render_pluck` — [example 05](examples/05_note_sound.py) |
| Recorder: interior shape, finger-hole placement & shape | ✅ 3-D model; intonation, overtones, Q per fingering; hole-position tuning | `Recorder`, `tune_hole_positions`, `RecorderObjective` — [example 06](examples/06_recorder_tuning.py) |

## Install and run

```bash
pip install -e ".[dev]"          # numpy, scipy, shapely (+ matplotlib, trimesh, gmsh, pytest)
pytest                           # violin + recorder tests
python examples/01_violin_modes.py
python examples/02_wolf_scan.py
python examples/03_search_design.py 24
python examples/06_recorder_tuning.py     # needs `pip install gmsh`
```

## A first session

```python
from luthier.violin import Violin, ViolinSpec
from luthier.strings import wolf_scan, wolf_summary

spec = ViolinSpec()                                   # frozen dataclass: everything is a parameter
spec = spec.with_(post__x=0.024, plates__fhole__length=0.079)   # nested update by "__" path
v = Violin(spec)

for i, f, air, bridge in v.mode_table(20)[:8]:        # body modes; 'air' = share of energy in the f-hole air plug
    print(f"{f:7.1f} Hz   air {air:.0%}")

res = wolf_scan(v.modes(40), v.bridge_top)            # string/body coupled-mode analysis
print(wolf_summary(res))
```

## Architecture

```
luthier/
  materials.py        orthotropic wood (spruce, maple, ebony …) + isotropic metals; tap-test calibration helper
  mesh.py             OBJ (grouped parts) / STL import, OBJ export, structured test meshes
  fem.py              shell (CST + DSG3 Mindlin), 3-D beams with rigid offsets, pin links, masses,
                      air-cavity mixed formulation, sparse assembly, shift-invert modal solver
  violin_geometry.py  parametric violin plates/ribs/f-holes (stand-in for CAD; used by the search)
  violin.py           ViolinSpec + Violin: plates, ribs, bass bar, sound post, bridge, Helmholtz cavity
  strings.py          string/body coupling (complex roots), wolf criterion, partials, indicative rendering
  search.py           ParameterSpace, grid/random/LHS/Sobol/differential-evolution, trial log (JSON/CSV)
  objectives.py       ViolinObjective (A0 target, mode targets, wolf avoidance), RecorderObjective (intonation, octave)
  airmesh.py          tetrahedral air meshes + planar acoustic openings found geometrically
  acoustics.py        3-D Helmholtz FEM, Robin end-correction radiation, modal Q (radiation/viscous/thermal)
  recorder.py         RecorderSpec -> gmsh/OpenCASCADE air mesh, fingerings, note reports, hole tuning
  specutil.py         nested "a__b__0__c" updates of frozen specs
```

Data flow: **geometry → FE model → modes → bridge mobility → string coupling → notes/wolves → score → search**.

## How far to trust it (read this)

Verified against closed-form solutions (all in `tests/`, run on every change):

* simply-supported isotropic plate: 0.2 % error at 32×32 elements, converging monotonically
* free square plate vs Leissa tables: 13.47 / 19.60 / 24.27 → within 0.5 % at 40×40
* orthotropic (spruce) simply-supported plate vs exact series: < 4 %; grain-direction rotation invariance
* cantilever beam (both bending planes) < 1 %; eccentric beam-on-plate vs composite-section theory < 3 %
* free violin body: six true zero-energy rigid-body modes
* air cavity vs rigid-wall Helmholtz formula: < 3 %
* string–body coupled roots vs two-oscillator theory: all roots recovered in 300 randomised cases; frequencies to ~0.1 %
* OBJ round trip preserves part names and geometry

**Not validated against a real violin.** The built-in violin is a *generic* shape with handbook wood
properties: its A0 is ~225 Hz (real violins ≈ 270–290) because its cavity is ~2.5 L (real ≈ 2 L).
Use it to compare *designs against each other*; for absolute predictions, feed your own CAD, measured
wood properties (`Orthotropic.from_measured`) and calibrate against a tap-test or measured bridge
admittance before trusting numbers in Hz.

Known simplifications: body modelled *in vacuo* (no air loading/radiation from outside); no corner blocks,
linings, varnish, neck/fingerboard/tailpiece; bridge is a parametric portal frame whose defaults put its
rocking resonance near 2.5 kHz; air cavity is lumped (valid below ~600 Hz); one string polarisation;
wolf flag is a two-oscillator criterion (nearest body mode) so it is least reliable where body modes crowd
together (above ~800 Hz); the wolf *count* changes in steps as designs change, so prefer the continuous
`mixing` values when judging a design.

### Recorder: what it is and is not

Each fingering is one generalised eigenproblem `(K + Σ B_open/ℓ) p = k² M p` on ~65 k tetrahedra
(~5 s to mesh once, ~0.5 s per fingering). A finger closing a hole just turns that opening into rigid
wall, so all fingerings share one mesh. Closed-hole volume, chimney height, undercut, bore taper and
hole shape are real geometry, not fitted parameters. Verified: straight-pipe pitches ≤ 0.15 %, Kirchhoff
wall-loss Q ≤ 0.5 %, Helmholtz-neck resonator ≈ 1.4 % (textbook end-correction limit), mesh refinement
≈ 2 cents.

**Playing pitch:** `Recorder.analyse(jet=Jet(...))` closes the air column with an air-jet feedback loop
(`luthier.jet`): you get the playing pitch, whether each note speaks, and which register it sounds in at a given
blowing pressure (`Recorder.blow_range`). The loop *structure* is textbook (Fletcher/Coltman); its three
coupling constants (`gain`, `window_inertance`, `growth`) are empirical and need calibrating on a real instrument.

**Not modelled:** nonlinear jet saturation (so no loudness/timbre), the 3-D windway and block (flue height is a
parameter), wall vibration, and radiation beyond a frequency-independent
end-correction impedance. Facetted hole outlines are corrected in area but not in shape. **Not yet compared
with a measured recorder** — use it to compare designs and to find which hole needs to move, then calibrate
`jet_cents` and `air_temp_c` against your first prototype. Double holes (F, F♯ on a baroque soprano) are
single holes here, so those cross-fingerings are only indicative.
