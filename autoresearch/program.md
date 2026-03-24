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
   - `docs/recallage_subpixel_amellioré.md` - constrained sub-pixel registration notes. Read-only.
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

High-value directions:
- Better separation of detector-fixed calibration from object surface content.
- More robust overlap equations and weighting under ripple, edge roll-off, and
  sparse outliers.
- Safer gauge handling so nuisance terms and calibration modes do not leak into
  one another.
- Edge-aware fusion that suppresses border corruption without erasing real
  structure.
- Very tightly bounded pose refinement only if it clearly helps and preserves
  evaluator guardrails.

From the docs, prefer:
- Huber-style IRLS for stable initial solves.
- A Tukey or redescending stage only after the initialization is already stable.
- Detector-frame calibration estimation with robust aggregation.
- Removal of detector low-order modes to limit crosstalk with nuisance terms.
- Explicit damping in alternating updates.
- Spatial cross-fade weights near overlap borders.
- Small, testable changes before ambitious rewrites.

Common pitfalls in this repo:
- Estimating a detector map but forgetting to remove low-order gauge modes.
- Over-smoothing the calibration map and losing real detector structure.
- Using aggressive pose resampling that changes `observed_support_mask`.
- Letting edge pixels dominate overlap equations.
- Replacing trusted metrics with visual proxies.
- Breaking deterministic behavior with uncontrolled random search.

If you add pose correction logic:
- keep corrections small and physically plausible;
- prefer corrections to observation centers over mask-distorting resampling when possible;
- never let the returned `observed_support_mask` drift away from the physical
  support of the original observations;
- require clear evaluator improvement before keeping pose machinery.

If you have made small tweaks for 10+ iterations with little progress, try one
radically different but still principled direction:
- richer overlap clique construction;
- adaptive calibration smoothing;
- a two-stage robust loss schedule;
- leave-one-out or gradient-informed pose correction;
- improved detector-mode projection;
- fusion weighting redesign.

## Search strategy

The target file currently exposes or hardcodes these tunable levers:
- `EDGE_EROSION_PX = 2` - how aggressively noisy borders are excluded.
- `FEATHER_WIDTH = 0.20` - width of cosine cross-fade in fusion.
- `lambda_reg = 1e-6` - Tikhonov stabilization of the simultaneous solve.
- `sigma_filter = 0.7` - low-pass smoothing of the detector calibration map.
- `n_params = 3` - nuisance basis limited to piston, tip, tilt.
- IRLS iteration count = 5 - number of robust reweighting passes.
- Huber scale multiplier = `1.345 * sigma` - robustness threshold.
- Overlap assembly pattern - currently close to local pairwise equations, not an
  aggressively weighted global clique.
- Calibration gauge constraints - piston, tip, tilt, defocus, astigmatism.

Search priority from most likely to matter to least:
1. Overlap equation assembly and row weighting.
   This is usually the biggest lever for s17 because overlap corruption,
   outliers, and edge effects directly enter here.
2. Detector calibration estimation and low-order mode projection.
   The docs strongly suggest calibration separation and gauge removal are core.
3. Robust loss schedule.
   Huber is stable; a later Tukey stage may help if introduced carefully.
4. Fusion weights and edge erosion.
   Border handling matters, but it should not compensate for a weak solve.
5. Pose refinement.
   Potentially valuable, but only if tightly bounded and evaluator-safe.
6. Minor numeric tolerances and iteration counts.
   Useful only after the structural pieces are solid.

**Phase 1** (first ~20%): sensitivity scan
- Try 2-3 values for `EDGE_EROSION_PX`, `FEATHER_WIDTH`, `sigma_filter`,
  and `lambda_reg`.
- Compare current overlap assembly against a stronger clique or better row
  weighting if the change is local and readable.
- Establish whether runtime is dominated by the solve or by extra preprocessing.

**Phase 2** (next ~30%): focused range search on the highest-impact levers
- If stronger edge suppression helps, test nearby values rather than only more.
- If calibration smoothing helps, search for the smallest useful sigma.
- If robust weighting helps, tune thresholding rather than immediately adding
  more complexity.

**Phase 3** (next ~30%): combine winners
- Stack the best overlap weighting, best calibration handling, and best fusion
  weights.
- Check interactions carefully; many combinations that look individually good
  can reintroduce crosstalk or oversmoothing together.

**Phase 4** (final ~20%): structural improvements
- Improve calibration-map estimation logic.
- Add a second robust stage or better residual normalization.
- Add bounded pose-refinement logic if the docs and baselines suggest a safe
  route and earlier phases have plateaued.
- Simplify anything that is no longer carrying its weight.

Rules:
- For numeric parameters, try 0.5x and 2x first, then bisect.
- Record crash and guardrail boundaries in `autoresearch/insights.md`.
- If 3 consecutive tweaks to the same lever show diminishing returns, lock it.
- If pose-related changes fail guardrails 3 times, stop forcing pose work and
  return to calibration / overlap / fusion improvements.
- If two ideas appear independent, you may combine them in one test, but revert
  both if the combination regresses.
- If tuning stops helping, switch from parameter search to algorithmic changes.
