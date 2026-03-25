# Autoresearch Insights

## What Worked
- Edge-aware weighting of the overlap equations improved aggregate RMS versus the uniform baseline.
- Broad calibration smoothing (sigma=1.2-1.25) significantly outperforms narrow (0.9) - captures more mid-spatial structure.
- Geometric mean row weighting for overlap equations (sqrt(w_i * w_j)) massively outperforms arithmetic mean (0.5*(w_i + w_j))). This was the single biggest improvement.
- With geometric mean, FEWER IRLS iterations = BETTER RMS AND FASTER runtime. The geometric mean naturally provides robust weighting so fewer refinement passes are needed.
- SIAC damping 0.75/0.25 is optimal; 0.7/0.3 and 0.8/0.2 both regress.
- SOLVE_FEATHER=0.495 near-edge-only is better than 0.45.
- lambda_reg in 2e-5 to 5e-5 range gives same RMS, doesn't meaningfully change runtime for geometric mean config.
- SIAC alternating iterations (n_siac=6) with damping 0.75/0.25 balances convergence speed and quality.
- SOLVE_FEATHER=0.495 (near-edge-only weighting) is better than 0.45.
- IRLS=8 is the sweet spot; IRLS=10 gives marginal RMS gain but ~36s runtime (>30s guardrail).
- Damping 0.75/0.25 is optimal; 0.8/0.2 is slower and worse.

## What Failed
- A weighted overlap clique with feather-based row weights regressed aggregate RMS from the baseline (this was with arithmetic mean).
- The simultaneous SCS-style solve still looks too coupled for s17 if the detector map and nuisances are fit in one block.
- A plain alternating SIAC-style loop without pose correction regressed even further than the baseline.
- Bounded pose refinement can shave RMS a little, but the extra solve cost was too high for the gain.
- Applying pose corrections only in fusion was not enough to preserve the RMS improvement.
- A second Tukey stage and post-smoothing low-order projection on the calibration map did not improve RMS.
- Eroding the masks before building the alignment system is too aggressive for s17; it hurts RMS badly even though runtime drops.
- Geometric mean with arithmetic mean row weights was previously untested - it works! The arithmetic mean was the bottleneck.
- n_siac=7 or 8 regresses vs n_siac=6.
- Sigma 1.15 or 1.35 slightly worse than 1.25.
- n_siac=7 regressed vs n_siac=6.
- Calibration smoothing below 0.9 regresses RMS.

## Dead Ends
- Replacing the joint solve with a simple alternating calibration loop is not enough on its own.
- Pose refinement needs either a cheaper update path or a stronger objective than the current one-step registration.
- Calibration-map cleanup after the simultaneous solve is not sufficient by itself.
- Sigma values outside 1.1-1.3 range: too narrow loses structure, too broad costs runtime.
- Damping values outside 0.7-0.75 range: too aggressive or too conservative both hurt.

## Promising Directions
- Try sigma=1.15 as middle ground between 1.2 (best RMS) and 1.1 (faster).
- Explore adaptive spatially-varying sigma: stronger near edges, weaker at overlap centers.
- SIAC-6 + sigma=1.2 is the current best (1.11194321 at 29.98s). Focus on runtime reduction for this config.
- If sigma=1.25-1.3 can reduce n_siac needed, it may beat the runtime constraint.
- Consider whether the fusion step can use tighter sigma while solve uses broader.

## Crash Boundaries
