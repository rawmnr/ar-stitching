# ar-stitching

`ar-stitching` is a scientific Python repository for optical sub-aperture stitching autoresearch.

Current phase: **Autoresearch Framework & Agent-Driven Optimization**.

## Repository Overview

The project is structured around a **Trusted/Editable** boundary to ensure scientific integrity while allowing autonomous agents to evolve stitching algorithms.

- `src/stitching/trusted/`: **The Scientific Reference**. Contains the simulation bench, sub-pixel transforms, nuisance models (drift, retrace, structured bias), and rigorous evaluation metrics.
- `src/stitching/editable/`: **The Agent Playground**. Contains the algorithm "genome" (`candidate_current.py`) and archived successful candidates.
- `src/stitching/harness/`: **The Engine**. Orchestrates the autoresearch loop, git-based versioning, resource budgeting, and performance ledger.
- `src/stitching/agents/`: **The Backends**. Routes optimization requests to AI models (Codex, OpenCode/Anthropic) via structured protocols.

## Key Features

### 1. Autoresearch Framework (New)
- **Closed-Loop Optimization**: A full "Propose → Evaluate → Decide" cycle powered by `src/stitching/harness/loop.py`.
- **GitOps Integration**: Isolated execution in temporary worktrees with automatic commits for accepted improvements and reverts for regressions.
- **Scientific Ledger**: Append-only log of every iteration, including hypotheses, diffs, metrics, and aggregate results.
- **Multi-Agent Support**: Interchangeable backends for OpenAI Codex and OpenCode (local or cloud models).
- **Leaderboard**: Automated ranking of the best stitching candidates based on aggregate RMS error.

### 2. High-Fidelity Simulation (Digital Twin)
- **Metrology Realism**: 
    - **Optical PSF**: Gaussian blurring for optical smoothing and pixel fill factor.
    - **Surface Non-stationarity**: Multi-mode bending drift (Zernike Z4-Z8).
    - **Mid-Spatial Ripples**: Periodic polishing marks fixed in the piece frame.
    - **Edge Roll-off**: Signal attenuation and noise boost at pupil boundaries.
- **Advanced Scanning**: Automated `grid` and `annular` scan plans with guaranteed coverage and sub-pixel registration.
- **Instrument Modeling**: Support for square/circular pupils with **NaN-based** invalid area semantics.

### 3. Evaluation & Diagnostics
- **Rigorous Metrics**: RMS and MAE calculated strictly on valid intersections, geometry gates (0.99 IoU), and HF retention monitoring.
- **Mismatch Analysis**: Per-pixel standard deviation maps to identify internal consistency errors and stitching artifacts.
- **Automated Reporting**: 6-view PNG reports providing deep visual insight into reconstruction quality.

## Project Status

- [x] **Trusted Stack**: Complete high-fidelity simulator and evaluation engine.
- [x] **Harness**: Fully operational Autoresearch loop with GitOps and Ledger.
- [x] **Agents**: Integrated Codex and OpenCode backends with structured prompting.
- [x] **Baseline**: Piston-corrected mean baseline established and ready for optimization.
- [ ] **Phase 2**: Agent-driven evolution toward GLS (Global Least Squares) and Robust M-Estimators.
- [ ] **Phase 3**: Self-calibration (CS/SC) for stationary reference bias removal.

## Getting Started

### 1. Initial Evaluation
Run a baseline evaluation on existing scenarios:
```bash
python scripts/run_single_loop.py --experiment-id baseline_eval --max-iterations 0
```

### 2. Start Autoresearch
Launch the optimization loop using the Codex backend:
```bash
python scripts/run_single_loop.py --experiment-id exp_01 --backend codex --budget-profile fast
```

### 3. Promote Best Candidate
Archive the top performer and update the leaderboard:
```bash
python scripts/promote_best.py
```

## Scientific Mandate (AGENTS.md)
All agent modifications must follow the mathematical hypotheses described in `AGENTS.md`. The trusted boundary is strictly enforced; agents can only modify `src/stitching/editable/`.
