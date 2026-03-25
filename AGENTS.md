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


