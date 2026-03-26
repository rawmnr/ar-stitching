# AR Stitching - Autonomous Optimization of optimized_stitching_algo.py

This is an experiment to autonomously optimize a single stitching candidate for
a robustness suite centered on the S17 digital-twin family rather than one
fixed scenario.

The editable candidate is:
- `src/stitching/editable/optimized_stitching_algo.py`

The recommended frozen evaluator is:
- `autoresearch/eval_multi_scenario.py`

That evaluator should be based on `src/stitching/harness/evaluator.py`,
evaluate exactly one candidate file across a fixed multi-scenario suite, and
print a small parseable metric block plus per-scenario diagnostics. Once the
loop starts, that evaluator is read-only.

## Evaluation Protocol

### 1. Verification Suite
The standard verification suite consists of 5 scenarios:
- `scenarios/s17_highres_circular.yaml`: Baseline circular pupil, 256x256.
- `scenarios/s17_highres_high_overlap.yaml`: High overlap (0.35).
- `scenarios/s17_highres_low_overlap.yaml`: Low overlap (0.15).
- `scenarios/s17_highres_square.yaml`: Square detector pupil, high overlap.
- `scenarios/s17_lowres_circular.yaml`: Low resolution grid (128x128), 64x64 tiles.

### 2. Generalizability Rules
The algorithm MUST derive resolution-dependent settings from the geometry:
- `sigma_filter` = clip((min(tile_shape) / c_sigma) / sqrt(max(redundancy, 1.0) / 3.0), 0.8, 3.0) (default `c_sigma=110.0`).
- `edge_erosion_px` = max(1, min(tile_shape) // 100).
- `calibration_bp_sigma` = max(0.3, min(tile_shape) / 256).
- `calibration_mf_min_obs` = max(3, int(n_obs * 0.35)).

Do NOT hardcode any smoothing sigma, erosion, or gate threshold as a constant.

### 3. Convergence-Based Loops
All alternating refinement loops (SIAC) MUST use convergence criteria:
- Primary stop: absolute change in calibration map < `siac_convergence_tol`.
- Secondary stop: relative stagnation detection (5 iterations with <1% change).
- Timeout: `max_siac_iter` as a safety guard.

### 4. Batch Evaluation & Overrides
The candidate supports config overrides via the `STITCH_CONFIG` environment variable (JSON dictionary).
Use this to test multiple hyperparameters in parallel or to perform sensitivity analysis.

Example:
```bash
STITCH_CONFIG='{"nuisance_reg_lambda": 15.0, "c_sigma": 80.0}' uv run autoresearch/eval_multi_scenario.py ...
```

### Sweep Protocol (fast iteration)

When testing scalar hyperparameters, DO NOT edit the code repeatedly.
Instead, use STITCH_CONFIG overrides to sweep values in a single session:

1. Make ONE structural code change and commit.
2. Sweep hyperparameters via env var on ONE fast scenario:
   ```bash
   for cs in 75 80 82.5 85 90; do
     STITCH_CONFIG="{\"c_sigma\":$cs}" uv run autoresearch/eval_s17_single.py \
       --scenario scenarios/s17_highres_square.yaml 2>/dev/null | grep aggregate_rms
   done
   ```
3. Pick the best value, update the default in StitchingConfig.
4. Validate on the full 5-scenario suite.
5. Log the FULL suite result in results.tsv.

## Review Criteria
- **accepted_all**: Must be 1.
- **aggregate_rms**: Target < 1.0 nm.
- **Worst-case RMS**: Must be < 1.5 nm on low-overlap scenarios.
- **Generalization**: No regression allowed when switching between circular and square pupils.

## Setup

To set up a new experiment:

1. Agree on a run tag based on today's date, for example `mar24-s17-opt`.
   The branch `autoresearch/<tag>` must not already exist.
2. Create the branch: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files:
   - `src/stitching/editable/optimized_stitching_algo.py` - the only editable file. This is the current candidate stitcher.
   - `autoresearch/eval_multi_scenario.py` - frozen evaluation harness. Do not modify after setup.
   - `src/stitching/harness/evaluator.py` - trusted evaluation entry point and guardrails. Read-only.
   - `src/stitching/trusted/eval/metrics.py` - trusted metric definitions, especially detrended RMS. Read-only.
   - `scenarios/s17_highres_circular.yaml` - primary high-resolution annular circular stress case. Read-only.
   - `scenarios/s17_lowres_circular.yaml` - low-resolution circular variant. Read-only.
   - `scenarios/s17_highres_square.yaml` - square-pupil high-resolution variant. Read-only.
   - `scenarios/s17_highres_low_overlap.yaml` - sparse-overlap high-resolution variant. Read-only.
   - `scenarios/s17_highres_high_overlap.yaml` - dense-overlap high-resolution variant. Read-only.
   - `src/stitching/editable/gls/baseline.py` - simplest structural baseline. Read-only.
   - `src/stitching/editable/gls_robust/baseline.py` - robust GLS reference. Read-only.
   - `src/stitching/editable/scs/baseline.py` - simultaneous calibration + stitching reference. Read-only.
   - `src/stitching/editable/siac/baseline.py` - alternating calibration reference. Read-only.
   - `src/stitching/editable/siac_reg/baseline.py` - pose-aware SIAC reference. Read-only.
   - `src/stitching/editable/pso/baseline.py` - stochastic refinement reference. Read-only.
   - `docs/stitching_implementation_guide.md` - local implementation roadmap. Read-only.
   - `docs/Optimisation_Robuste_Stitching_Optique_Metrologie_wrapped.txt` - domain strategy notes on robust stitching, calibration separation, IRLS, Tukey/Huber, drift and retrace handling. Read-only.
   - `docs/recallage_subpixel.md` - constrained sub-pixel registration notes. Read-only.
4. Verify prerequisites:
   - `autoresearch/eval_multi_scenario.py` exists.
   - It loads `src/stitching/editable/optimized_stitching_algo.py`.
   - It evaluates the fixed robustness suite:
     - `scenarios/s17_highres_circular.yaml`
     - `scenarios/s17_lowres_circular.yaml`
     - `scenarios/s17_highres_square.yaml`
     - `scenarios/s17_highres_low_overlap.yaml`
     - `scenarios/s17_highres_high_overlap.yaml`
   - It runs successfully with:
     `python autoresearch/eval_multi_scenario.py --candidate src/stitching/editable/optimized_stitching_algo.py --scenarios scenarios/s17_highres_circular.yaml scenarios/s17_lowres_circular.yaml scenarios/s17_highres_square.yaml scenarios/s17_highres_low_overlap.yaml scenarios/s17_highres_high_overlap.yaml`
   - If imports fail because `stitching` is not on `sys.path`, the frozen evaluator must fix that internally by prepending the repo `src/` directory before the loop begins.
5. Initialize local experiment state:
   - Create `autoresearch/results.tsv` with just the header row.
   - Create `autoresearch/insights.md` if missing.
   - Do not commit either file.
6. Confirm setup looks good.

Once setup is complete, kick off experimentation.

## Experimentation

This loop optimizes robust reconstruction quality on a hard digital-twin case
that mixes detector-fixed reference bias, pose drift/jitter, retrace error,
mid-spatial ripple, and edge effects. The primary failure mode is crosstalk:
the candidate can easily absorb detector bias into nuisance terms or smooth the
surface in a way that looks visually nice but worsens trusted RMS.

## Perturbations in the measurement

The S17 robustness suite contains multiple realistic perturbations. The
original `s17_highres_circular` case still carries the hardest coupled digital
twin effects, and the new variants probe whether the same solver survives
changes in resolution, overlap, and pupil geometry. Treat these as separate
error sources that must be mitigated without mixing them together.

- Pose bias and pose jitter.
  The sub-aperture centers are not perfectly where the nominal scan says they
  are. Literature usually mitigates this with constrained sub-pixel
  registration, joint pose optimization, and strict gauge-centering of pose
  corrections.
- Pose drift across the scan.
  The position error can evolve from one acquisition to the next. Literature
  usually mitigates this with global overlap-consistent pose solves instead of
  purely sequential alignment.
- Detector-fixed calibration bias.
  The detector can imprint a repeatable additive map in detector coordinates.
  Literature usually mitigates this with simultaneous or alternating
  calibration, robust detector-frame residual averaging, and explicit removal
  of low-order degenerate modes from the calibration map.
- Calibration and nuisance crosstalk.
  Piston, tip, tilt, defocus, and detector bias can explain the same residuals
  if gauges are weak. Literature usually mitigates this with explicit gauge
  constraints and low-order mode projection on the detector map.
- Geometric retrace error.
  The measurement can contain slope-dependent or geometry-dependent distortion.
  Literature usually mitigates this with better geometric modeling,
  projection-aware alignment, and by not letting low-order nuisance terms absorb
  everything.
- Low-frequency drift.
  Slow thermal or system drift can bias the whole reconstruction. Literature
  usually mitigates this with damped alternating updates, low-order drift
  modeling, and stable reference tracking rather than aggressive one-shot
  solves.
- Mid-spatial ripple and structured noise.
  The observations contain non-white spatial structure, not only simple Gaussian
  noise. Literature usually mitigates this with robust weighting and with
  selective smoothing of the detector calibration map instead of smoothing the
  final surface indiscriminately.
- Sparse outliers, dust, and edge roll-off.
  Border pixels and isolated artifacts can dominate least-squares residuals.
  Literature usually mitigates this with IRLS, Huber initialization, optional
  Tukey-style redescending rejection, mask erosion, and cross-fade edge
  weighting.
- Circular pupils and partial support.
  Valid support is not rectangular and overlap geometry matters near borders.
  Literature usually mitigates this with careful local-to-global indexing,
  support-aware cropping, and fusion rules that respect the physical aperture.

Use the trusted evaluator, not `scripts/compare_baselines.py`, as the judge.
`compare_baselines.py` is useful for human visualization, but it is not the
frozen scalar metric for the loop.

**What you CAN do:**
- Modify `src/stitching/editable/optimized_stitching_algo.py` only.
- Everything in that file is fair game:
  overlap equation assembly, robust weighting, clique construction, nuisance
  model parameterization, detector calibration estimation, gauge constraints,
  smoothing/regularization, fusion weights, bounded pose correction logic,
  helper functions, and internal control flow.

**What you CANNOT do:**
- Do not modify `autoresearch/eval_multi_scenario.py` after setup. It is frozen.
- Do not modify `src/stitching/harness/evaluator.py`.
- Do not modify `src/stitching/trusted/eval/metrics.py`.
- Do not modify any frozen suite scenario YAML used by the evaluator.
- Do not modify any baseline files in `src/stitching/editable/`.
- Do not install new packages or add dependencies.
- Do not change the `CandidateStitcher.reconstruct(...)` signature.
- Do not change the return type. It must remain `ReconstructionSurface`.
- Do not remove required metadata conventions:
  `metadata["method"]` and, when estimated, `metadata["instrument_calibration"]`.
- Do not break `observed_support_mask`. The evaluator requires it to match the
  physical support implied by the original observations.
- Do not silently return malformed arrays, changed shapes, or non-finite data
  inside valid support.

**The goal is simple: get the lowest `aggregate_rms`.**

For this program, `aggregate_rms` should come from
`evaluate_candidate_on_suite(...)` across the frozen multi-scenario suite. The
suite may mix detrended and raw acceptance depending on each scenario's
metadata, and the evaluator must preserve that trusted policy per scenario.

Why this metric matters:
- It measures reconstruction error against known truth using the trusted harness.
- It already respects scenario-level detrending rules.
- It penalizes fragile improvements that only work on one scan geometry.
- It aligns with the repository leaderboard and accepted-candidate ranking.

Metric policy for this loop:
- Primary metric: `aggregate_rms`
- Hard gate: `accepted_all: 1`
- Runtime guardrail: prefer `total_runtime_sec <= 30.0`. Treat runs above 30 s
  as `discard` unless they deliver a clearly material RMS improvement over the
  current best. Treat runs above 100 s as automatic `discard` unless you are
  explicitly running a one-off diagnosis.
- Side diagnostics only: per-scenario RMS/MAE/HF retention, Zernike residual,
  and any optional calibration diagnostics you inspect manually

Do not switch to a weighted composite score unless you can defend the weights
mathematically. In this project the safest decision rule is primary metric plus
hard gate plus soft constraint.

**`total_runtime_sec`** is a practical guardrail. The normal target band is
roughly 20-30 seconds. If a candidate materially exceeds that band, treat the
extra runtime as a regression unless the RMS gain is large enough to justify it.

**Guardrail constraint**: `num_accepted` must equal `num_scenarios`.
If the candidate fails acceptance or any hard evaluator guardrail, treat the
experiment as a failure even if the raw RMS line looks better.

**Simplicity criterion**: All else being equal, simpler is better. A tiny RMS
gain is not worth a fragile or unreadable solver. Equal RMS with cleaner logic
or safer constraints is a valid simplification win.

**The first run**: The very first run always establishes the baseline with no
edits to `src/stitching/editable/optimized_stitching_algo.py`.

## Frozen evaluator contract

The best eval strategy for this repo is a small frozen Python script:
`autoresearch/eval_multi_scenario.py`

That script should:
- insert the repo `src/` directory into `sys.path`;
- load the candidate with `load_candidate_module(...)`;
- call `evaluate_candidate_on_suite(...)` on the fixed multi-scenario suite;
- print a compact parseable summary and exit non-zero on crashes.

Recommended printed lines:

```text
---
aggregate_rms: 0.12345678
aggregate_mae: 0.10123456
max_rms: 0.18223344
min_rms: 0.08221100
std_rms: 0.03112233
total_runtime_sec: 18.42
num_accepted: 5
num_scenarios: 5
accepted_all: 1
worst_scenario: s17_highres_square
best_scenario: s17_lowres_circular
=== SCENARIO_DETAILS ===
[01] s17_highres_circular
  status: ACCEPT
  rms_detrended: 0.12345678
  hf_retention: 0.81234567
=== END_SCENARIO_DETAILS ===
=== AGENT_SUMMARY ===
performance_breakdown:
  good_rms_lt_1nm: 5/5
=== END_AGENT_SUMMARY ===
```

Keep the evaluator minimal. Do not generate plots. Do not call
`compare_baselines.py`. Do not mix in any human-only diagnostics that make the
metric harder to parse.

## Output format

Once the evaluation finishes it should print a summary like this:

```text
---
aggregate_rms: 0.12345678
aggregate_mae: 0.10123456
max_rms: 0.18223344
min_rms: 0.08221100
std_rms: 0.03112233
total_runtime_sec: 18.42
num_accepted: 5
num_scenarios: 5
accepted_all: 1
worst_scenario: s17_highres_square
best_scenario: s17_lowres_circular
=== SCENARIO_DETAILS ===
[01] s17_highres_circular
  status: ACCEPT
  rms_detrended: 0.12345678
  hf_retention: 0.81234567
=== END_SCENARIO_DETAILS ===
=== AGENT_SUMMARY ===
performance_breakdown:
  good_rms_lt_1nm: 5/5
=== END_AGENT_SUMMARY ===
```

Required completion checklist after every run:

1. Read `aggregate_rms` and `accepted_all` from the eval output.
2. Append exactly one row to `autoresearch/results.tsv`.
3. Verify the row contains the commit hash, metric, runtime, status, and a
   short description.
4. Append a brief note to `autoresearch/insights.md` if the run changed
   direction, exposed a failure mode, or produced a new useful lesson.
5. Only then mark the run complete or move to the next edit.

Extract the key metric from the log file:

```text
grep "^aggregate_rms:" autoresearch/run.log
```

Also check acceptance:

```text
grep "^accepted_all:" autoresearch/run.log
```

Also inspect the current limiter:

```text
grep "^worst_scenario:" autoresearch/run.log
```

## Logging results

When an experiment is done, log it to `autoresearch/results.tsv`
(tab-separated, not comma-separated).

**Mandatory post-evaluation gate**: an evaluation is not complete until its
`results.tsv` row has been appended. Do not start the next experiment, edit the
candidate again, or summarize the run as finished until the TSV entry is
written. This applies even for crashes or rejected runs.

The TSV has a header row and 5 columns:

```text
commit	aggregate_rms	total_runtime_sec	status	description
```

1. git commit hash (short, 7 chars)
2. `aggregate_rms` achieved - use `inf` for crashes
3. `total_runtime_sec` - use `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what the experiment tried

Example:

```text
commit	aggregate_rms	total_runtime_sec	status	description
a1b2c3d	0.18234567	14.20	keep	baseline
b2c3d4e	0.17690123	15.04	keep	stronger overlap clique weighting
c3d4e5f	0.17944444	17.82	discard	tukey switch too early in IRLS
d4e5f6g	inf	0.0	crash	support mask mismatch after pose update
```

## Insights memory

Maintain `autoresearch/insights.md` as untracked working memory.

After every experiment, add a short note when the run produces anything worth
remembering, especially:

- a new improvement;
- a regression or crash;
- a boundary condition;
- a dead end worth avoiding.

After every 10 experiments, update it with:
- what worked;
- what failed;
- dead ends not worth retrying;
- promising next directions;
- crash boundaries or guardrail boundaries.

If 5 consecutive experiments fail to improve, read `autoresearch/insights.md`
before trying the next direction.

Exploration rule when progress stalls:
- If the best `aggregate_rms` has not improved after 5 kept experiments, the
  next experiment must change the solver family or the error model, not just a
  scalar hyperparameter.
- Prefer moving to a different axis of change: overlap assembly, robust loss,
  fusion weighting, calibration estimation, pose correction, gauge handling, or
  regularization.
- Do not spend more than 2 consecutive experiments on the same local variant if
  the metric stays flat or regresses.
- If the running best is near a plateau, try a qualitatively new idea even if
  previous tweaks were still reducing runtime.

## The experiment loop

The experiment runs on a dedicated branch such as `autoresearch/<tag>`.

LOOP FOREVER:

1. Look at the git state and current best metric.
2. Read `autoresearch/results.tsv` and `autoresearch/insights.md`.
3. Modify `src/stitching/editable/optimized_stitching_algo.py` with one coherent experimental idea.
4. Commit the change with a short hypothesis-focused message.
5. Run the frozen evaluator:
   `python autoresearch/eval_multi_scenario.py --candidate src/stitching/editable/optimized_stitching_algo.py --scenarios scenarios/s17_highres_circular.yaml scenarios/s17_lowres_circular.yaml scenarios/s17_highres_square.yaml scenarios/s17_highres_low_overlap.yaml scenarios/s17_highres_high_overlap.yaml > autoresearch/run.log 2>&1`
6. Read out:
   - `grep "^aggregate_rms:" autoresearch/run.log`
   - `grep "^accepted_all:" autoresearch/run.log`
   - `grep "^total_runtime_sec:" autoresearch/run.log`
   - `grep "^worst_scenario:" autoresearch/run.log`
7. If the metric line is missing, the run crashed.
   Read `tail -n 50 autoresearch/run.log`, fix trivial issues if appropriate,
   otherwise discard the idea and move on.
8. Record the result in `autoresearch/results.tsv`.
9. Keep the commit only if all of the following are true:
   - `accepted_all: 1`
   - `aggregate_rms` is strictly lower than the current best
   - the runtime increase is reasonable for the gain
   - the implementation remains coherent and not obviously overfit
10. If the experiment is equal or worse, or fails acceptance, revert to the
    previous good commit.

When the loop gets stuck near the current best, treat that as a signal to
explore a new family of ideas rather than another small parameter sweep. The
goal is not to tunnel indefinitely on one local neighborhood of the design
space.

Do not commit `autoresearch/results.tsv`, `autoresearch/run.log`, or
`autoresearch/insights.md`.

**Timeout**: one run should usually stay well below the evaluator guardrail.
Treat anything above 8 minutes wall-clock as failure.

**Crashes**: fix trivial syntax/import mistakes. Skip fundamentally broken or
numerically unstable ideas. If a change violates support-mask guardrails or
starts returning malformed outputs, revert immediately.

**NEVER STOP**: once the loop begins, do not ask whether to continue. Keep
running until manually interrupted. If you plateau, re-read the docs and
baselines, combine near-miss ideas, or try a different algorithmic layer.

## Strategy hints

Read the local docs and baselines as design priors, then optimize for the
trusted metric, not for visual smoothness.

### Perturbation model

The S17 scenario is not a single error source—it is a superposition of four
distinct, entangled disruption vectors:

| Vector | Physical origin | Mitigation principle |
|--------|-----------------|----------------------|
| High-frequency noise / phase singularities | Optical propagation, detector noise | Robust M-estimators; do not let 2π jumps dominate the solve |
| Rigid-body positioning errors | Multi-axis kinematic stage tolerances | Global pose-graph solve; do not chain sequential pairwise alignment |
| Systematic reference error | Transmission sphere aberrations | Alternating or simultaneous calibration; remove low-order gauge modes |
| Time-dependent surface drift | Thermal expansion, mechanical lever effects | Non-rigid registration or spatiotemporal parameterization |

Crosstalk is the primary failure mode: piston/tip/tilt can masquerade as
detector bias, and vice versa. Every strategy below must be evaluated for
whether it exacerbates or mitigates this coupling.

### High-value algorithmic directions

1. **Overlap equation assembly and row weighting**
   This is the biggest lever in S17 because outliers and edge corruption enter
   here directly. Prefer global clique construction over pairwise chains.
   Weight rows by inverse-distance or local coherence metrics.

2. **Detector-fixed calibration decoupling**
   The detector imprint is additive in detector coordinates and moves relative to
   the surface. Use simultaneous or alternating calibration (SIAC-style),
   aggregate residuals in detector frame, and explicitly project out low-order
   degenerate modes (piston, tip, tilt, defocus, astigmatism) from the
   calibration map to prevent crosstalk with nuisance terms.

3. **Robust loss schedule**
   Start with Huber initialization for stability. Introduce a Tukey or
   redescending stage only after the linear solve is stable. Never switch
   loss functions mid-iteration without guardrails.

4. **Fusion weights and edge handling**
   Cosine cross-fade near overlap borders. Erode noisy borders aggressively
   enough to exclude phase singularities. Do not let border pixels dominate
   overlap equations.

5. **Pose refinement (tightly bounded)**
   Small, physically plausible corrections to sub-aperture centers are acceptable
   only if they clearly improve `aggregate_rms`. Prefer center-shift
   corrections over mask-distorting resampling. Revert immediately if
   `observed_support_mask` drifts.

### Advanced strategies from the research curriculum

The following are legitimate algorithmic directions; use them when simpler
approaches plateau:

- **Zernike / Q-polynomial basis** — If you model the surface or reference error
  in a polynomial basis, prefer orthogonal bases. Zernike polynomials are
  natural for circular apertures; Forbes Q-polynomials are better for strong
  aspheric departure because they are orthogonal in surface gradient rather than
  sag. For non-circular overlaps, Gram-Schmidt orthonormalization is acceptable.

- **Spectral stitching** — Transform overlapping regions into the 2D Fourier
  domain. Stitch amplitude and phase spectra directly before transforming back.
  This preserves mid-spatial frequency manufacturing ripples that spatial-domain
  polynomial fitting either filters out or aliases.

- **Alternating calibration (SIAC / CS / SC)** — SIAC alternates between a
  positioning-only solve (reference frozen) and a reference-only solve (poses
  frozen). CS first calibrates then stitches; SC first stitches then calibrates.
  All three are valid; the autoresearch loop should empirically select among
  them based on which yields the lowest robust suite RMS rather than the best
  single-scenario number.

- **Biconvex lifting (SparseLift)** — When alternating minimization stalls due to
  severe non-convexity, "lift" the bilinear problem into a higher-dimensional
  convex problem: define X := xz^T, then solve L1,2-norm minimization subject
  to data fidelity. Recover the surface via best rank-one SVD approximation.
  This bypasses local minima that plague standard gradient descent.

- **Graph-SLAM / Bundle Adjustment** — Frame the stitching problem as a
  pose-graph with nodes = sub-aperture poses and edges = overlap constraints.
  Solve the entire graph simultaneously with nonlinear sparse optimization,
  enforcing loop closures where scan rings close. Use robust M-estimators
  (Huber, Cauchy, Tukey) inside the bundle adjustment to downweight phase
  singularities.

- **Coherent Point Drift (CPD)** — For thermal drift that violates
  rigid-body assumptions, treat one point cloud as GMM centroids and the other
  as data points. Maximize likelihood while forcing coherent group motion.
  Tune the rigidity regularization to find the stiffness that fits the drift
  without overfitting to noise.

- **Spatiotemporal drift parameterization** — If timestamps are available,
  model r(x,y,t) = r0(x,y) + c·t + d·t² and solve a maximum likelihood
  estimation over the full spatiotemporal dataset jointly.

- **Compound loss function** — The objective is not plain MSE. Optimize a
  weighted sum:
  L_total = L_overlap + λ1·L_smoothness + λ2·L_drift + λ3·L_sparse
  where L_overlap uses a robust M-estimator on overlap residuals,
  L_smoothness applies Total Variation or gradient-domain penalization,
  L_drift penalizes high-velocity jumps in sequential pose parameters, and
  L_sparse applies L1 or L1,2 penalization to calibration coefficients.

### What to avoid

- Estimating a detector map without removing low-order gauge modes.
- Over-smoothing the calibration map until real structure is lost.
- Aggressive pose resampling that changes `observed_support_mask`.
- Letting edge pixels dominate overlap equations.
- Replacing trusted RMS metrics with visual smoothness proxies.
- Breaking deterministic behavior with uncontrolled randomness.
- Overfitting to one S17 variant rather than the shared underlying problem.

### If progress stalls

After 10+ iterations with diminishing returns, pivot to one of:
- Richer overlap clique construction (global graph vs. pairwise)
- Adaptive calibration smoothing sigma, preferably coarse-to-fine schedules
  rather than fixed small sigma from iteration 1
- Two-stage robust loss schedule (Huber init → Tukey refinement)
- Improved detector-mode projection (which modes to remove, in what order)
- Fusion weighting redesign (coherence-based, distance-based, or hybrid)

## Search strategy

The target file exposes or hardcodes these tunable levers:

- `EDGE_EROSION_PX = 2` — aggressive border exclusion of noisy pixels.
- `FEATHER_WIDTH = 0.20` — cosine cross-fade width at overlap borders.
- `lambda_reg = 1e-6` — Tikhonov stabilization of the simultaneous solve.
- `sigma_filter` — low-pass smoothing of the detector calibration map.
- `n_params = 3` — nuisance basis (piston, tip, tilt; 4th slot unused).
- IRLS iteration count = 5 — number of robust reweighting passes.
- Huber scale = `1.345 * sigma` — robustness threshold.
- Overlap assembly pattern — currently pairwise-local, not global clique.
- Calibration gauge constraints — piston, tip, tilt, defocus, astigmatism.

### Immediate priority

Do not spend the next cycle retrying end-stage HF injection tricks. The current
evidence says the remaining HF residual is not mainly caused by the final
fusion blend.

The next experiments should test whether the residual comes from detector-map
structure that a fixed-sigma SIAC loop cannot capture cleanly.

Ordered next experiments:

1. Keep deep SIAC (`n_siac=96`) and test a smaller fixed calibration smoother
   such as `sigma_filter=1.0`.
2. If that regresses, implement annealed calibration smoothing inside the SIAC
   loop, for example geometric decay from `2.5` to `0.8` across the full
   alternation schedule.
3. If annealing helps, sweep the terminal sigma over a compact range such as
   `{0.5, 0.7, 1.0, 1.3}`.
4. Only after the calibration path is stabilized, revisit richer nuisance terms
   with explicit shrinkage on the added quadratic modes.
5. Treat phase-correlation or other sub-pixel pose refinement as a later
   fallback, not the first response to the current HF plateau.

Evaluation rule for this phase:

- prefer runs that reduce both `aggregate_rms` and the high-order residual;
- if total RMS improves but `scenario_hf_retention` stays pinned at `0.0` and
  the high-order residual does not move, treat that as low-order cleanup rather
  than a real HF breakthrough;
- prioritize structural calibration changes over cosmetic fusion changes.

### Curriculum phases

The search is organized into four ordered phases. Do not skip to Phase 3
or 4 without establishing that Phases 1 and 2 have plateaued.

**Phase 1 — Foundation and sensitivity scan (~20% of budget)**
Establish a solid, stable baseline and characterize sensitivity to the
primary numeric levers. This phase has effectively already established the
current basin: deep SIAC plus stronger calibration smoothing than the shallow
regime originally suggested.

- Run the unmodified baseline to get the initial `aggregate_rms`.
- Sweep `EDGE_EROSION_PX` in {1, 2, 3, 4} — edge exclusion has outsized
  impact when phase singularities are present.
- Sweep `FEATHER_WIDTH` in {0.10, 0.20, 0.30} — too much feather degrades
  resolution; too little permits border artifacts.
- Treat the old shallow-SIAC `sigma_filter` sweep as historical context only.
  Re-check calibration smoothing under deep SIAC rather than assuming the old
  optimum still applies.
- Sweep `lambda_reg` in {1e-7, 1e-6, 1e-5} — too much regularization
  biases the solve; too little permits ill-conditioning.
- Record which lever moves `aggregate_rms` most. Lock the rest and
  concentrate on the highest-impact lever.

**Phase 2 — Core algorithmic improvements (~30% of budget)**
With Phase 1 winners fixed, tackle the structural algorithm components.

- **Overlap assembly** — Compare the current pairwise-local assembly against
  a stronger global clique construction (all observations contributing to
  all global pixels simultaneously). If switching to a clique, ensure row
  weighting still downweights edge pixels.
- **Robust loss schedule** — Try increasing IRLS iterations from 5 to 8–10.
  If stable, experiment with a two-stage schedule: start with Huber, then
  switch to Tukey/ redescending after the third iteration. The switch point
  matters—too early causes instability, too late misses gains.
- **Calibration gauge handling** — Add or remove specific gauge modes
  (e.g., keep piston+tip+tilt; test adding defocus or astigmatism to the
  projection set). Test whether projecting out 4 modes vs. 5 modes changes
  the RMS meaningfully.
- **Calibration smoothing adaptivity** — This is now the top Phase 2 item.
  Prefer coarse-to-fine smoothing within the SIAC loop over purely spatial
  adaptivity. Start broad for stability, then narrow once the alternation has
  substantially converged.

**Phase 3 — Combination and interaction testing (~30% of budget)**
Stack the best configurations from Phase 2. Watch for regressions caused by
interactions:

- Combine the best overlap assembly with the best calibration handling.
- Combine the best fusion weights with the best robust loss schedule.
- Verify that combining two independently good changes does not regress RMS.
  If it does, one of the two is an overfit to its isolated test conditions.
- If pose refinement is attempted, apply it only to observation center
  offsets (not mask resampling) and bound corrections to ±0.5 pixels.
  Revert immediately if `observed_support_mask` changes or RMS regresses.

**Phase 4 — Structural exploration (~20% of budget)**
Only if Phases 1–3 have genuinely plateaued (fewer than 0.5% RMS
improvement across 5 consecutive experiments), attempt one of:

- Switch to a different calibration strategy: CS vs. SC vs. full SIAC
  alternating. Empirically determine which operational order fits this
  scenario's noise profile.
- Implement biconvex lifting (SparseLift): reformulate the joint
  calibration-surface problem as X := xz^T and solve L1,2 convex
  optimization. This is a significant structural change—document the
  hypothesis clearly before testing.
- Add a graph-SLAM or bundle-adjustment layer for global pose consistency.
  This is high-risk/high-reward; revert if it fails guardrails or
  dramatically increases runtime.

### Search rules

- **Numeric sweeps**: for any scalar parameter, try 0.5× and 2× first,
  then bisect within the promising range.
- **Lock-and-move**: when a lever shows diminishing returns for 3
  consecutive tests, lock it and move to the next.
- **Revert on interaction regression**: if combining two independently good
  changes regresses RMS, revert both and keep only the best single change.
- **Pose gate**: if pose-related changes fail evaluator guardrails 3 times
  in a row, stop forcing pose work and return to calibration / overlap /
  fusion improvements.
- **Timeout gate**: runs exceeding 8 minutes wall-clock are treated as
  failures. Penalize runtime proportionally when evaluating whether an RMS
  gain is worth a runtime increase.
- **Simplicity wins**: if two configurations yield equivalent RMS, prefer
  the simpler one (fewer iterations, fewer parameters, more legible code).
- **Crash log**: after any crash, record the error signature in
  `autoresearch/insights.md` and avoid repeating the same structural
  hypothesis without a targeted fix.

### What not to tune

- The return type, `ReconstructionSurface` schema, and metadata keys.
- The scenario file or evaluator script.
- Any baseline file in `src/stitching/editable/`.
- The `CandidateStitcher.reconstruct(...)` signature.
- `observed_support_mask` semantics (the evaluator requires it to
  match physical support of the original observations).
