# AGENTS.md

## Purpose

This repository is an algorithm benchmark and autoresearch playground for **sub-aperture stitching / interferometric reconstruction**. Agents working in this repo must optimize reconstruction quality **without breaking the experimental harness**.

The current codebase compares several editable baselines:

* `GLS Standard`
* `GLS Robust (Huber)`
* `SCS (Simultaneous Calibration and Stitching)`
* `SIAC (Alternating Calibration)`
* `PSO (stochastic refinement over GLS)`

A representative stress scenario is `scenarios/s17_highres_circular.yaml`, which combines:

* high resolution (`256x256`)
* annular scan pattern
* circular truth + detector pupils
* detector-fixed reference bias field
* pose bias / drift / jitter
* geometric retrace error
* low-frequency noise and mid-spatial ripple
* sparse outliers
* randomized alignment nuisances per sub-aperture

This means many naive stitching approaches fail because they confound:

* object surface content
* detector-fixed calibration bias
* per-subaperture nuisance terms
* overlap outliers / edge effects

---

## Repository map

### Scenarios

* `scenarios/*.yaml`
* `scenarios/template_full.yaml` is the exhaustive reference for scenario configuration.
* Scenario YAMLs define geometry, truth generation, digital twin effects, instrument errors, nuisances, and evaluation policy.

### Editable baselines

* `src/stitching/editable/gls/baseline.py`
* `src/stitching/editable/gls_robust/baseline.py`
* `src/stitching/editable/scs/baseline.py`
* `src/stitching/editable/siac/baseline.py`
* `src/stitching/editable/pso/baseline.py`

These are the main files agents are expected to modify during algorithmic work.

### Analysis / reporting

* `src/stitching/analysis/leaderboard.py`
* `scripts/compare_baselines.py`

`compare_baselines.py` is the fastest visual sanity-check entry point for the provided scenario. It expects optional calibration maps in reconstruction metadata.

---

## Core interface contract

All candidate algorithms must preserve the `CandidateStitcher` interface:

```python
class CandidateStitcher:
    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        ...
```

Return type must remain `ReconstructionSurface`.

Required fields in the returned object:

* `z`: reconstructed global surface (`np.ndarray`)
* `valid_mask`: global valid mask
* `source_observation_ids`: tuple of input observation ids
* `observed_support_mask`: support mask showing where data contributed

Optional but important metadata keys:

* `method`: short algorithm name
* `instrument_calibration`: detector-fixed estimated map if the algorithm estimates one
* any diagnostic scalar values useful for analysis

Do **not** break these metadata conventions because `scripts/compare_baselines.py` already consumes them.

---

## Domain model and terminology

### Observations

Each `SubApertureObservation` contains a local tile measurement (`obs.z`) with:

* `tile_shape`
* `global_shape`
* `center_xy`
* `valid_mask`

The algorithm must map local detector pixels into a global reconstruction frame.

### Nuisance terms

Current baselines mostly solve for per-observation low-order nuisance terms:

* piston
* tip
* tilt
* sometimes a placeholder fourth slot (`focus`) is carried in arrays for compatibility, even if not actively solved

Important: several baselines only solve **3 actual parameters** and store them into an array of shape `(n_obs, 4)` with the last column unused.

### Reference / calibration map

Some scenarios include a detector-fixed bias field. Algorithms like `SCS` and `SIAC` explicitly try to estimate it.

This map lives in detector coordinates, not global object coordinates.

### Gauge freedoms

The stitching problem has unobservable modes unless explicitly constrained. Typical gauges removed in this repo:

* global piston
* global tip
* global tilt
* sometimes defocus / astigmatism-like modes for detector calibration maps

When changing a solver, always check whether you are reintroducing gauge degeneracies.

---

## What matters most when editing algorithms

### 1. Preserve numerical robustness

The stress scenarios contain:

* sparse outliers
* detector edge roll-off
* pose drift / jitter
* structured bias fields

Therefore, algorithms that assume perfect overlap consistency are fragile.

Preferred strategies already present in the repo:

* IRLS / Huber reweighting
* Tukey / robust residual weighting
* cross-fade weights near overlap borders
* light Tikhonov regularization
* explicit gauge constraints
* mild low-pass smoothing of estimated calibration maps

### 2. Do not mix surface truth with detector bias

A common failure mode is absorbing detector-fixed structure into per-subaperture nuisance terms or into the reconstructed surface.

If you estimate a reference map, make sure to:

* accumulate it in detector coordinates
* use robust averaging / weighting
* remove degenerate low-order modes from the detector map to avoid crosstalk with nuisance terms

### 3. Respect overlap geometry

Most solvers build equations from pixels landing on the same global coordinates.

If you modify overlap assembly:

* keep local-to-global indexing exact
* guard all cropping carefully (`gy_s`, `gy_e`, `gx_s`, `gx_e`, etc.)
* preserve behavior near image borders

### 4. Keep fusion logic explicit

Most baselines follow this pattern:

1. estimate nuisance terms (and maybe calibration)
2. correct each observation
3. fuse corrected observations into the global grid

Do not hide fusion assumptions inside the alignment solver unless necessary.

---

## Baseline-specific notes

### GLS (`src/stitching/editable/gls/baseline.py`)

* Standard sparse least-squares global alignment
* Uniform fusion weights
* Good minimal reference implementation
* No robustification beyond light regularization and gauge constraints

Use this as the simplest baseline for structural comparisons.

### Robust GLS (`src/stitching/editable/gls_robust/baseline.py`)

* Same general structure as GLS
* Adds IRLS + Huber weighting
* Uses radial cross-fading to downweight edges during fusion

This is a good template when introducing robust statistics incrementally.

### SCS (`src/stitching/editable/scs/baseline.py`)

* Simultaneously solves nuisance terms and a detector-fixed calibration map
* Adds gauge constraints on both nuisance block and reference map block
* Uses iterative robust weighting
* Smooths the recovered reference map slightly

This is the most structurally complete “joint solve” baseline in the provided code.

### SIAC (`src/stitching/editable/siac/baseline.py`)

* Alternates between global reconstruction, detector-map estimation, and nuisance re-solve
* Uses cross-fade weighting and robust reference-map estimation
* Explicitly removes low-order degenerate modes from the detector map

When editing SIAC, watch for:

* oscillatory alternation
* leakage between nuisance block and calibration block
* over-smoothing of the reference map
* under-constrained updates in low-overlap regions

### PSO (`src/stitching/editable/pso/baseline.py`)

* Starts from GLS
* Applies simplified stochastic random-search refinement
* Currently closer to stochastic hill-climbing than true PSO

Treat this as experimental. Avoid making it much slower unless the gain is substantial.

---

## Known practical invariants

Agents should preserve the following practical invariants unless the task explicitly asks otherwise:

1. **No interface breaks** in editable baselines
2. **No change to scenario schema** unless required by the task
3. **Return finite, masked outputs** rather than silent shape mismatches
4. **Prefer deterministic behavior** when randomness is introduced
5. **Keep runtime reasonable** on `s17_highres_circular`
6. **Do not remove metadata used by analysis scripts**

---

## Recommended workflow for coding agents

### For algorithm changes

1. Read the target baseline and compare it to `gls` and `scs`
2. Identify whether the issue is in:

   * overlap equation assembly
   * gauge constraints
   * robust weighting
   * calibration-map estimation
   * fusion weighting
3. Make the smallest coherent change first
4. Run a focused scenario check
5. Inspect both reconstruction error and calibration-map quality

### For debugging poor performance

Check these first:

* Is the algorithm estimating a detector-fixed map at all?
* Are low-order detector modes being removed?
* Are overlap equations using all useful pairings or just a weak anchor?
* Are edge pixels overweighted?
* Are robust weights updated from residuals computed in the correct system?
* Is smoothing erasing meaningful calibration content?

---

## Useful commands

### Compare baselines on the provided stress scenario

```bash
python scripts/compare_baselines.py --scenario scenarios/s17_highres_circular.yaml
```

This generates:

* detrended ground truth
* estimated reconstructions
* reconstruction error maps
* estimated instrument calibration maps
* calibration error maps

Output path:

```text
artifacts/comparison_all_algos_s17.png
```

### Scenario template reference

Use this when adding or extending a scenario:

```text
scenarios/template_full.yaml
```

---

## Guidance for agents editing math-heavy code

### Prefer explicit comments for these cases only

Add comments when the code is doing one of the following:

* fixing a gauge ambiguity
* defining robust weighting rules
* mapping between local and global coordinates
* removing degenerate modes from a detector map
* stabilizing alternating updates

Avoid adding noisy comments for obvious NumPy slicing.

### Be careful with silent mathematical regressions

Small changes can degrade performance without causing exceptions. In particular:

* changing residual definitions
* changing normalization of `x_norm` / `y_norm`
* changing constraint rows
* changing smoothing sigma
* changing regularization scale
* changing overlap clique construction

All of these can substantially change RMS without any crash.

---

## Preferred coding style in this repo

* Keep functions numerically explicit and locally readable
* Prefer small helper methods when logic becomes multi-stage
* Use NumPy / SciPy idioms already present in neighboring baselines
* Avoid introducing heavyweight abstractions around sparse solves
* Preserve deterministic seeds when stochastic refinement is used

For arrays:

* use `dtype=float` explicitly where relevant
* return `np.nan` outside valid support when appropriate
* always guard empty masks / empty overlaps

---

## If you add a new baseline

Follow the existing layout:

```text
src/stitching/editable/<new_method>/baseline.py
```

And ensure:

* class name is still `CandidateStitcher`
* `reconstruct(...)` signature is unchanged
* returned metadata includes a `method` key
* optional calibration maps use `instrument_calibration`

---

## Leaderboard expectations

`src/stitching/analysis/leaderboard.py` stores accepted candidates and ranks them by `aggregate_rms`.

Implication for agents:

* the primary optimization target is reconstruction quality, especially aggregate RMS
* but runtime still matters and is tracked
* hypothesis text and commit metadata should remain meaningful in experiment logs

---

## What not to do

* Do not rename public keys consumed by scripts unless all consumers are updated
* Do not assume square pupils everywhere; scenarios can be circular / annular
* Do not hardcode `s17` behavior into generic solver logic
* Do not remove gauge constraints “because the solver still runs”
* Do not trust visual smoothness alone; calibration leakage can look visually clean and still worsen RMS

---

## Minimal mission statement for agents

When working in this repo, optimize stitching algorithms for **robust reconstruction under realistic digital-twin nuisances** while preserving:

* the harness contract
* scenario compatibility
* analysis script compatibility
* numerical stability
* reproducibility

When in doubt, prefer:

1. correct physics / geometry separation
2. explicit constraints
3. robust statistics
4. small verifiable changes over large rewrites
