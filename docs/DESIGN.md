# Design notes

## Goals

1. Predict an instrument's resonances/overtones from geometry + materials **before** building it.
2. Prevent construction mistakes (hole positions, plate thickness, post placement…).
3. Let the user define a configuration space and search it.

## Shape of the engine

Every instrument is a pipeline with the same two ends and a different middle:

```
geometry + materials ──► physical model ──► resonances / transfer functions ──► playable notes ──► score ──► search
```

* **Shared:** materials, mesh import, sparse FE kernels, modal analysis, parameter spaces, search, objectives.
* **Per family:** the physical model. Strings/plates → shell FEM + string coupling (done).
  Winds → 3-D Helmholtz FEM of the enclosed air (done, recorder).

A design is always a *frozen dataclass spec* (`ViolinSpec`); search parameters are `"__"`-paths into it,
so any field — a hole position, a thickness, a material scale factor — is searchable without new code.

## Body model: why shells

Violin plates are 2.5–4 mm thick on a ~350 mm body. Solid tetrahedra lock in such thin parts and need
millions of elements; a mid-surface shell model captures arching, graded thickness and orthotropy with a few
thousand triangles, which is what makes *searching* affordable (~3 s per eigen-solve here).

Element: flat triangle with 6 dofs/node.

* membrane: constant-strain triangle, orthotropic plane stress rotated to each element's grain direction
* bending + shear: Mindlin–Reissner, shear strains from the **discrete shear gap** (DSG3) — no shear locking
* drilling dof: penalty to the element's mean in-plane rotation (`DRILL_ALPHA`)
* lumped mass (translational + rotary inertia)

Discrete members: Euler–Bernoulli 3-D beams with **rigid end offsets** (bass bar sits on the plate surface,
not on the mid-plane) and **barycentric attachment** (`End(nodes, weights, offset)`), so a post/bar/foot
position is a *continuous* parameter instead of snapping to the nearest mesh node (which would add ±½ element
of noise to every search).

Pitfalls found and fixed while building (kept as regression tests):

* a constant lateral spring on the sound post is not rotation-invariant → spurious "rigid body" resonances;
  links are now pin-ended axial trusses with the offset's `θ×o` term included
* open f-holes make the wall-volume weights non-closed → rigid translation changed the cavity volume; the
  f-hole "lid" is now distributed on the rim nodes so both translations and rotations are volume-neutral
* ARPACK drops copies of the 6-fold zero eigenvalue → rigid modes are deflated analytically

### Mesh requirements

Plan on ≥ 8–10 elements per bending wavelength at the highest frequency of interest. The default 9 mm mesh
(≈ 1 500 triangles per plate) is adequate to ~1.5 kHz; halve it for results above that (cost grows ~8×).
f-holes are meshed at 0.45× the global size.

## Air cavity

Lumped uniform-pressure cavity (valid while the cavity is small against the wavelength, i.e. below ~600 Hz)
with a Helmholtz plug through the f-holes:

```
M u'' + K u − b p = 0 ;   m_q q'' + p = 0 ;   (V/κ) p + b·u − q = 0
```
`b` = outward area weights of the walls, `V` cavity volume, `κ = ρc²`, `m_q = ρ/C` with `C` the Rayleigh
conductance of the openings (equivalent-ellipse model, `C = πa/K(e)`, plus thickness correction). The mixed
(p, q) form keeps K sparse; a direct `κ/V · bbᵀ` term would be dense. Hole *shape* enters through `C`, so
f-hole length/width/position change A0 and its coupling to the plates.

## String–body coupling and wolf tones

String end mobility `Ys = (j/Zs) tan(kL)`, body mobility `Yb(ω)` (modal sum, valid for complex ω). Free
vibration of the joined system: `sin(kL) − j Zs Yb cos(kL) = 0`. Roots `ω = ω_r + jω_i` give frequency and Q of
each coupled mode.

Numerics: near a body resonance `g` has a pole next to the hybrid root and plain Newton misses it. The solver
multiplies through by the denominators of nearby body modes so the function is entire in the search window,
then runs vectorised Newton from a dense grid of complex starting points.

**Wolf criterion.** With `ω_s`, `ω_b` the uncoupled string and nearest-body-mode complex frequencies and
`ω₁, ω₂` the two coupled roots: body participation `pᵢ = |ωᵢ−ω_s| / (|ωᵢ−ω_s| + |ωᵢ−ω_b|)`,
`mixing = 1 − |p₁ − p₂|` (0 = decoupled, 1 = 50/50 hybrid). A note is flagged when mixing ≥ 0.5 and the root
separation (the beat rate) is 0.4–25 Hz. A first version based on "splitting vs. damping difference" was
rejected because it flagged weakly coupled pairs (string Q 380 beside body Q 35).

## Search

`ParameterSpace` → decoded dicts → `Objective(params) → score | (score, metrics)`. Failures (bad geometry,
singular model) are recorded as `inf` and never abort a run. Methods: grid, random, Latin hypercube, Sobol,
differential evolution. Typical cost: ~7 s per violin evaluation (3 s FE, 4 s wolf scan).

Advice: first run `lhs`/`sobol` to learn the landscape and see which parameters matter, then `de` on the
important ones. Prefer continuous terms (`mixing`, cents error) over counts (`n_wolf`) in objectives.

## Recorder (3-D air-column model)

Air pressure obeys `∇²p + k²p = 0`. Walls are rigid (`∂p/∂n = 0`); every opening (window, foot, open tone hole) is a
planar patch recognised **geometrically** on the mesh. Radiation uses the reactive end correction as a
frequency-independent Robin condition `∂p/∂n = −p/ℓ`, `ℓ = ε·a_eq·shape` (ε = 0.6133 unflanged … 0.8216 flanged, 0.7
default for a hole in a tube wall; `shape` from the Rayleigh conductance of the ellipse), which keeps the problem a
**real symmetric** generalised eigenproblem. Losses are added by perturbation from each mode's own field:
Levine–Schwinger radiation resistance, viscous and thermal boundary layers; `Q = ω/2α`, `α = P/2E`.
`robin_scale = true area / meshed area` compensates the polygonal outline of curved openings.

Geometry: bore is a revolved radius profile; chimneys are lofted ellipses (outer size → bore size via `undercut`);
the window is a box. Meshing: gmsh/OpenCASCADE; `curvature_points=16` is the default speed/accuracy trade
(≈ 2 cents against 24, 3× cheaper); use 24 for a final check.

Playing pitch (`jet.py`): the window impedance `Z(ω)` of the 3-D modes (modal sum, window-face-averaged mode shapes)
is closed into the standard jet feedback loop (Fletcher/Coltman/Verge): pressure → transverse velocity at the flue exit
(window inertance) → jet delay `W/c_ph` with `c_ph ≈ 0.4U`, spatial growth `e^{μW}` and 1/(jω) → flow into the pipe
`Q = gain·b·U·η`. Loop gain `L = −Z·gain·b·U·e^{(μ−jω/c_ph)W}/(ρ l_w ω²)`; `arg L = 0` gives the playing frequency (it sits
above a resonance for a fast jet — pitch rises with blowing pressure — and the jet delay is 0.25–0.75 periods), `|L| ≥ 1` is
the start-up condition, and the strongest candidate sets the register (hard blowing → octave, twelfth). Physical: structure
and scalings. Empirical: `gain`, `window_inertance`, `growth`. No nonlinear saturation, so pitch/register/ease of speaking
only — not loudness or timbre. `Recorder(spec, jet_cents=…)` remains as a flat empirical offset.

Intonation tuning: `tune_hole_positions` moves each governing hole by the 1-D sensitivity `df/f = −dz/(c/2f)`
and re-solves in 3-D (converges to ≲ 5 cents in 3–4 iterations on the test design; cross-fingerings need hole
diameters/undercut, which the search layer can vary).

## Calibration (do this before trusting absolute numbers)

1. Measure the wood: tap-test a strip along/across the grain → `Orthotropic.from_measured`.
2. Weigh and measure the actual plates; use the real thickness map (`thickness` accepts arrays/callables).
3. Compare predicted vs measured free-plate Chladni/tap modes of a single plate (build a `Model` with one part).
4. Compare the assembled body's bridge mobility to a measurement (shaker or impact hammer at the bridge notch).
   Tune global factors (`scaled(E_L=…)`, `Bridge.leg_thickness`, damping `zeta`) until it matches.

## Roadmap

1. **Recorder follow-ups.** *Parked:* double holes (twin-opening chimneys `6a/6b`, half-covered holes as partial-area openings, a pair end-correction for neighbouring radiators, and a checked baroque fingering chart); keyed flutes; calibrate `Jet` (gain, window inertance, growth) against a measured instrument; nonlinear jet saturation for amplitude/timbre; windway geometry (3-D flue) instead of the flue-height parameter; import of a CAD *air-volume* mesh (the `AirMesh.from_tets` path already accepts one); complex-frequency radiation (BEM or PML) in place of the Robin end correction; calibration against a measured instrument.
2. **Solid CAD → mid-surface.** Today CAD must be exported as sheets. Add automatic mid-surface extraction
   from closed plate solids (medial-axis / offset-surface pairing) or a quadratic-tet solid element path
   for thick parts.
3. **Measured-admittance coupling** (`strings.admittance_from_measurement`): vector-fit a measured bridge
   mobility and use it in place of the FE body — useful for diagnosing wolves on an *existing* violin.
4. **Radiation and air loading:** BEM/Rayleigh-integral radiation to get pressure vs frequency/direction;
   added-mass correction for plates in air.
5. **Corner blocks, linings, neck/fingerboard/tailpiece**, detailed bridge (kidney cut-outs) as meshes.
6. **Time-domain synthesis** with bowed-string friction models driving the coupled modes.
7. **Speed:** cache the stiffness of unchanged parts between search trials; reduced-order models for the
   post/bar/bridge parameters (they touch few dofs); Numba for the wolf scan.
