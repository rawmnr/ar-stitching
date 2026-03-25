# Autoresearch Insights

## Current Best
- **0.92055912** at 32.16s (`f2a6108` - per-observation band-passed detector MF calibration, `sigma_filter=1.55`, `n_siac=149`, `lambda=14`, `mf_alpha=0.35`)
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

## What Worked
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
- late-iteration de-damping of `R_map` updates regressed

## Structural Findings
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
- most of the total RMS gain still comes from low-order cleanup, so the project remains fundamentally HF-limited

## Promising Next Directions
1. Keep the per-observation MF calibration architecture and improve detector/surface separation inside that path rather than returning to raw averaged-MF injection
2. Investigate stronger detector-space consensus estimators for MF accumulation, since the current count gate still limits usable MF gain
3. Rework pose refinement structurally; the current discrete shift path is not moving the metric
4. Investigate why HF retention stays exactly zero even as RMS improves below 0.93 nm

## Crash Boundaries
