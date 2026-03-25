# AR Stitching - Autonomous Optimization of optimized_stitching_algo.py

This is an experiment to autonomously optimize a single stitching candidate for
the stress scenario `scenarios/s17_highres_circular.yaml`.

The editable candidate is:
- `src/stitching/editable/optimized_stitching_algo.py`

The recommended frozen evaluator is:
- `autoresearch/eval_s17_single.py`

That evaluator should be based on `src/stitching/harness/evaluator.py`,
evaluate exactly one candidate file on exactly one scenario, and print a small
parseable metric block. Once the loop starts, that evaluator is read-only.

## Setup

To set up a new experiment:

1. Agree on a run tag based on today's date, for example `mar24-s17-opt`.
   The branch `autoresearch/<tag>` must not already exist.
2. Create the branch: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files:
   - `src/stitching/editable/optimized_stitching_algo.py` - the only editable file. This is the current candidate stitcher.
   - `autoresearch/eval_s17_single.py` - frozen evaluation harness. Do not modify after setup.
   - `src/stitching/harness/evaluator.py` - trusted evaluation entry point and guardrails. Read-only.
   - `src/stitching/trusted/eval/metrics.py` - trusted metric definitions, especially detrended RMS. Read-only.
   - `scenarios/s17_highres_circular.yaml` - the fixed stress scenario. Read-only.
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
   - `autoresearch/eval_s17_single.py` exists.
   - It loads `src/stitching/editable/optimized_stitching_algo.py`.
   - It uses `scenarios/s17_highres_circular.yaml`.
   - It runs successfully with:
     `python autoresearch/eval_s17_single.py --candidate src/stitching/editable/optimized_stitching_algo.py --scenario scenarios/s17_highres_circular.yaml`
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

The fixed scenario `scenarios/s17_highres_circular.yaml` contains multiple
realistic perturbations at once. Treat them as separate error sources that must
be mitigated without mixing them together.

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
- Do not modify `autoresearch/eval_s17_single.py` after setup. It is frozen.
- Do not modify `src/stitching/harness/evaluator.py`.
- Do not modify `src/stitching/trusted/eval/metrics.py`.
- Do not modify `scenarios/s17_highres_circular.yaml`.
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
`evaluate_candidate_on_suite(...)` with a one-scenario suite containing only
`scenarios/s17_highres_circular.yaml`. Because that scenario has
`ignore_tilt: true`, the trusted stack will effectively optimize detrended RMS,
which is the right target here.

Why this metric matters:
- It measures reconstruction error against known truth using the trusted harness.
- It already respects scenario-level detrending rules.
- It aligns with the repository leaderboard and accepted-candidate ranking.

Metric policy for this loop:
- Primary metric: `aggregate_rms`
- Hard gate: `accepted_all: 1`
- Soft constraint: `total_runtime_sec`
- Side diagnostics only: `scenario_mae_detrended`, `scenario_hf_retention`, and
  any optional calibration diagnostics you inspect manually

Do not switch to a weighted composite score unless you can defend the weights
mathematically. In this project the safest decision rule is primary metric plus
hard gate plus soft constraint.

**`total_runtime_sec`** is a soft constraint. Some increase is acceptable for a
meaningful RMS gain, but runtime should not blow up. A change that is
materially slower with negligible RMS gain is not worth keeping.

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
`autoresearch/eval_s17_single.py`

That script should:
- insert the repo `src/` directory into `sys.path`;
- load the candidate with `load_candidate_module(...)`;
- call `evaluate_candidate_on_suite(...)` on a one-element list containing
  `Path("scenarios/s17_highres_circular.yaml")`;
- print a compact parseable summary and exit non-zero on crashes.

Recommended printed lines:

```text
---
aggregate_rms: 0.12345678
aggregate_mae: 0.10123456
max_rms: 0.12345678
total_runtime_sec: 18.42
num_accepted: 1
num_scenarios: 1
accepted_all: 1
scenario_id: s17_highres_circular
scenario_rms_detrended: 0.12345678
scenario_hf_retention: 0.81234567
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
max_rms: 0.12345678
total_runtime_sec: 18.42
num_accepted: 1
num_scenarios: 1
accepted_all: 1
scenario_id: s17_highres_circular
scenario_rms_detrended: 0.12345678
scenario_hf_retention: 0.81234567
```

Extract the key metric from the log file:

```text
grep "^aggregate_rms:" autoresearch/run.log
```

Also check acceptance:

```text
grep "^accepted_all:" autoresearch/run.log
```

## Logging results

When an experiment is done, log it to `autoresearch/results.tsv`
(tab-separated, not comma-separated).

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

After every 10 experiments, update it with:
- what worked;
- what failed;
- dead ends not worth retrying;
- promising next directions;
- crash boundaries or guardrail boundaries.

If 5 consecutive experiments fail to improve, read `autoresearch/insights.md`
before trying the next direction.

## The experiment loop

The experiment runs on a dedicated branch such as `autoresearch/<tag>`.

LOOP FOREVER:

1. Look at the git state and current best metric.
2. Read `autoresearch/results.tsv` and `autoresearch/insights.md`.
3. Modify `src/stitching/editable/optimized_stitching_algo.py` with one coherent experimental idea.
4. Commit the change with a short hypothesis-focused message.
5. Run the frozen evaluator:
   `python autoresearch/eval_s17_single.py --candidate src/stitching/editable/optimized_stitching_algo.py --scenario scenarios/s17_highres_circular.yaml > autoresearch/run.log 2>&1`
6. Read out:
   - `grep "^aggregate_rms:" autoresearch/run.log`
   - `grep "^accepted_all:" autoresearch/run.log`
   - `grep "^total_runtime_sec:" autoresearch/run.log`
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
  them based on which yields the lowest RMS on this specific scenario.

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
- Overfitting to S17 noise rather than the true underlying surface.

### If progress stalls

After 10+ iterations with diminishing returns, pivot to one of:
- Richer overlap clique construction (global graph vs. pairwise)
- Adaptive calibration smoothing sigma (search 0.3–1.5 range)
- Two-stage robust loss schedule (Huber init → Tukey refinement)
- Improved detector-mode projection (which modes to remove, in what order)
- Fusion weighting redesign (coherence-based, distance-based, or hybrid)

## Search strategy

The target file exposes or hardcodes these tunable levers:

- `EDGE_EROSION_PX = 2` — aggressive border exclusion of noisy pixels.
- `FEATHER_WIDTH = 0.20` — cosine cross-fade width at overlap borders.
- `lambda_reg = 1e-6` — Tikhonov stabilization of the simultaneous solve.
- `sigma_filter = 0.7` — low-pass smoothing of the detector calibration map.
- `n_params = 3` — nuisance basis (piston, tip, tilt; 4th slot unused).
- IRLS iteration count = 5 — number of robust reweighting passes.
- Huber scale = `1.345 * sigma` — robustness threshold.
- Overlap assembly pattern — currently pairwise-local, not global clique.
- Calibration gauge constraints — piston, tip, tilt, defocus, astigmatism.

### Curriculum phases

The search is organized into four ordered phases. Do not skip to Phase 3
or 4 without establishing that Phases 1 and 2 have plateaued.

**Phase 1 — Foundation and sensitivity scan (~20% of budget)**
Establish a solid, stable baseline and characterize sensitivity to the
primary numeric levers.

- Run the unmodified baseline to get the initial `aggregate_rms`.
- Sweep `EDGE_EROSION_PX` in {1, 2, 3, 4} — edge exclusion has outsized
  impact when phase singularities are present.
- Sweep `FEATHER_WIDTH` in {0.10, 0.20, 0.30} — too much feather degrades
  resolution; too little permits border artifacts.
- Sweep `sigma_filter` in {0.3, 0.5, 0.7, 1.0, 1.5} — capture the
  under-smoothing and over-smoothing regimes for calibration.
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
- **Calibration smoothing adaptivity** — Instead of fixed `sigma_filter`,
  try spatially adaptive smoothing: stronger smoothing near edges, weaker
  near overlap centers where signal is strongest.

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
