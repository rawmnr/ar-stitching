# Autoresearch Insights

## Current Best
- **0.95289226** at 12.05s (`work-20260325-z` - regularized 6-term nuisance refinement, `lambda=20`)
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

## What Worked
- CALIBRATION_BLOCK=1 (full resolution calibration map)
- n_irls=1 remains a good runtime/quality tradeoff
- much deeper SIAC than 16 iterations
- `sigma_filter` above 1.346 once SIAC is deep enough; 2.0 is current best
- `SOLVE_FEATHER_WIDTH=0.51` remains competitive
- keeping the global solve at 3 nuisance terms while allowing SIAC refinement to fit quadratic nuisance modes with explicit shrinkage
- the best shrinkage seen so far is `NUISANCE_REG_LAMBDA=20.0`

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

## Structural Findings
- the old 1.07 nm floor was mostly an alternation-depth issue
- calibration smoothing interacts positively with deeper SIAC, even though it looked weak in the shallow regime
- the current pose-refinement path is effectively inactive on this scenario
- high-frequency loss is still real, but the final fusion stage is not where the current implementation is losing the most performance
- the best basin appears near `n_siac=96` and `sigma_filter` around `2.1-2.2`; pushing SIAC to 112-128 did not improve further
- the current solver relies on implicit edge damping from zero-filled smoothing; removing that bias cleanly without losing stability needs a different calibration model, not just normalized convolution
- richer nuisance terms will need explicit shrinkage or a different gauge construction; the unconstrained 6-term extension is not usable
- the useful path is not expanding the global simultaneous solve; it is adding regularized quadratic cleanup only in the per-observation nuisance refinement
- the lambda sweep is non-monotonic: `10` under-regularizes, `100` over-regularizes, and the current best band is around `20-30`
- `lambda=20` improved both total RMS and Zernike >36 residual (`0.84196` vs `0.84541` baseline) without changing the HF retention metric from zero

## Promising Next Directions
1. Refine the new quadratic-nuisance shrinkage band around `lambda=20` and test whether selective damping of nuisance updates improves stability further
2. Rework pose refinement structurally; the current discrete shift path is not moving the metric
3. Separate detector-map low-order content from object content without projecting the full map every SIAC step
4. Investigate why HF retention stays exactly zero even as RMS improves below 1 nm

## Crash Boundaries
