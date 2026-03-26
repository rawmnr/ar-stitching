# Autoresearch Insights

## Current Best
- **0.92080593** at 56.23s (`044c8aa` - 56 SIAC passes, low-order detrend enabled for the 128x128 low-res case)
- **0.94760672** at 31.71s (`044c8aa` - 12 SIAC passes, low-order detrend enabled for the 128x128 low-res case)
- **0.92055912** at 32.16s (`f2a6108` - per-observation band-passed detector MF calibration, `sigma_filter=1.55`, `n_siac=149`, `lambda=14`, `mf_alpha=0.35`)
- follow-up calibration restructuring reached **0.92108941** before alpha retuning; the final committed best remained `0.92055912`
- **0.92167848** at 29.41s (same structural line before per-observation MF accumulation; `sigma_filter=1.55`)
- **0.95289226** at 12.05s (`work-20260325-z` - first regularized 6-term nuisance refinement breakthrough, `lambda=20`)
- **0.97434873** at 13.38s (`work-20260325-o` - `n_siac=96`, `sigma_filter=2.1`)
- **0.97441900** at 12.25s (`work-20260325-n` - `n_siac=96`, `sigma_filter=2.2`)
- Earlier sub-1nm milestone: **0.98874800** (`work-20260325-i`)

## Major Breakthroughs

### CALIBRATION_BLOCK=1: Full-Resolution Calibration Map
- **Block=1 (full resolution) is dramatically better than block=2 (coarse grid)**
- block=2 destroys mid/high-frequency content through 2x2 block averaging before gaussian smoothing
- block=1 preserves full detector resolution through the calibration filter

### Long SIAC Alternation Is The Real Lever
- `n_siac` is the first lever with monotonic, meaningful gains:
- 16 -> 1.0679
- 32 -> 1.0411
- 48 -> 1.0195
- 64 -> 1.0049
- 72 -> 0.9829
- 80 -> 0.9788
- 96 -> 0.9743
- Stronger calibration smoothing becomes beneficial once alternation is allowed to converge:
- `n_siac=64, sigma_filter=1.5` -> **0.9987**
- `n_siac=64, sigma_filter=2.0` -> **0.9887**
- `n_siac=96, sigma_filter=2.1` -> **0.9743**
- Interpretation: the current loop was under-converged more than fundamentally mis-specified.

### HF Retention Still Zero
- `scenario_hf_retention` remains 0.0 across all experiments
- the Zernike >36 residual still dominates the remaining error budget
- final-surface HF split did not help; all tested `HF_SPLIT_SIGMA` values regressed RMS
- this implies the high-frequency loss is upstream of the final fusion blend

### Detector MF Calibration Is Real But Delicate
- the calibration diagnostic plots were directionally correct: the missing structure is detector-fixed mid-frequency content, not just a slightly-too-smooth final fused surface
- naive MF injection into the calibration map regressed because it re-injected surface HF leakage
- extracting detector MF from each observation's band-passed detector-space residual before cross-observation averaging materially improved stability
- this raised the stable MF-injection ceiling from tiny values to roughly `mf_alpha ~= 0.35`
- the best filtered-MF configuration remained count-gated (`min_obs=8`); looser gating reintroduced contaminated MF and regressed quickly

## What Worked
- 12 SIAC passes on the multi-scenario suite, combined with low-order detrend only for the 128x128 low-res scenario
- CALIBRATION_BLOCK=1 (full resolution calibration map)
- n_irls=1 remains a good runtime/quality tradeoff
- much deeper SIAC than 16 iterations
- in the current architecture, the best basin moved to `n_siac ~= 149`
- the current best low-frequency calibration smoother is much lower than the old basin: `sigma_filter ~= 1.55`
- `SOLVE_FEATHER_WIDTH=0.51` remains competitive
- keeping the global solve at 3 nuisance terms while allowing SIAC refinement to fit quadratic nuisance modes with explicit shrinkage
- selective damping of the added quadratic nuisance modes (`NUISANCE_QUADRATIC_DAMPING = 0.125`)
- the current shrinkage optimum is lower than the first quadratic basin: `NUISANCE_REG_LAMBDA ~= 14.0`
- detector MF recovery works best when it is:
  band-passed,
  projected to remove low-order detector modes,
  accumulated per observation before averaging,
  and gated by observation count in detector coordinates

## What Failed
- final HF split fusion regressed by 0.13-0.17 nm
- pose-shift sweeps had no measurable effect in the current implementation
- edge erosion above 1 regressed
- block-based coarse calibration (`CALIBRATION_BLOCK > 1`) still regressed badly
- median filtering for calibration regressed
- per-step detector gauge projection plus damped nuisance updates regressed to 1.0981
- normalized convolution for the calibration smoother regressed across the full sigma sweep; best retuned point was still ~1.0013
- naive 6-parameter nuisance expansion catastrophically reopened low-order gauge/crosstalk and blew up to 12.55 nm
- projecting detector modes directly on every estimated reference map also regressed to ~1.09 nm
- fixed `sigma_filter=1.0` with `n_siac=96` regressed to `1.0245`
- annealed detector smoothing from `2.5` to `0.8` over the SIAC loop regressed to `0.9910`
- raw MF calibration injection from the averaged residual map regressed unless the gain was kept very small
- loosening the MF gate from `min_obs=8` to `min_obs=6` reopened too much contaminated structure and regressed badly
- variance gating for MF injection did not beat the simpler count gate
- the current refactor with basis-aware projection and `c_sigma=110.0`, `lambda=15.0`, `sigma_floor=0.8` improved the circular baseline to `1.0767`, the low-res circular case to `1.4686`, and the low-overlap case to `1.4812`; the full 5-scenario aggregate was `1.1239` with `57.1s` total runtime
- the main runtime cost is now deep SIAC convergence in the square and high-overlap cases (`~175` and `~25` iterations respectively), not the circular baseline
- late-iteration de-damping of `R_map` updates regressed
- raw phase-correlation pose refinement catastrophically failed (`~14.9` nm RMS, rejected)
- conservative gradient-domain local pose refinement was safe but completely inert
- two-stage coarse+fine pose refinement, even when fed back through short post-pose alternation, remained inert on the metric

## Structural Findings
- the multi-scenario basin is better than the old single-scenario basin once low-res calibration leakage is reduced explicitly
- the low-res circular case was the main quality limiter before the low-order detrend change
- the old 1.07 nm floor was mostly an alternation-depth issue
- calibration smoothing interacts positively with deeper SIAC, even though it looked weak in the shallow regime
- the current pose-refinement path is effectively inactive on this scenario
- high-frequency loss is still real, but the final fusion stage is not where the current implementation is losing the most performance
- the old basin near `n_siac=96` and `sigma_filter around 2.1-2.2` is no longer the right reference point once quadratic nuisance damping and detector-MF calibration are enabled
- the current solver relies on implicit edge damping from zero-filled smoothing; removing that bias cleanly without losing stability needs a different calibration model, not just normalized convolution
- richer nuisance terms will need explicit shrinkage or a different gauge construction; the unconstrained 6-term extension is not usable
- the useful path is not expanding the global simultaneous solve; it is adding regularized quadratic cleanup only in the per-observation nuisance refinement
- the shrinkage sweep is architecture-dependent: once quadratic damping and deeper SIAC were added, the best `lambda` moved down from `~20-30` to `~14`
- the most useful calibration change so far is not "add more HF after averaging"; it is "extract detector MF per observation, then average those detector-space MF pieces"
- this new MF path produced a real but still modest HF improvement:
  Zernike >36 residual moved from `~0.845` in the old basin to `~0.8373` in the current best
- the basis-aware detector-mode projection fix is now active in the code path, so the square-pupil nuisance basis no longer projects through the wrong radial low-order modes
- most of the total RMS gain still comes from low-order cleanup, so the project remains fundamentally HF-limited
- pose is not the active bottleneck in the current implementation:
  the scenario injects whole-pixel pose error scales, but both coarse and fine pose searches failed to move the accepted metric once calibration was improved
- if pose is revisited later, it needs stronger instrumentation first (for example logging recovered shifts and validating they are nonzero / physically plausible) rather than another blind optimizer swap

## Promising Next Directions
1. Keep the per-observation MF calibration architecture and improve detector/surface separation inside that path rather than returning to raw averaged-MF injection
2. Investigate stronger detector-space consensus estimators for MF accumulation, since the current count gate still limits usable MF gain
3. Consider calibration-side confidence models that separate detector-fixed MF from leaked surface MF before averaging, rather than tuning pose again immediately
4. Investigate whether low-overlap robustness can be improved without reopening the low-order leakage that was fixed for the low-res case

## Crash Boundaries
