# ar-stitching

`ar-stitching` is a scientific Python repository for optical sub‑aperture stitching autoresearch.

## Current Phase

Current phase: **Autoresearch Framework & Agent‑Driven Optimization**.

## Table of Contents

- [Repository Overview](#repository-overview)
- [Key Features](#key-features)
- [Project Status](#project-status)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

## Repository Overview

The project is structured around a **trusted/editable** boundary to ensure scientific integrity while allowing autonomous agents to evolve stitching algorithms.

- **Trusted** – `src/stitching/trusted/` – contains the scientific reference model (simulation bench, sub‑pixel transforms, drift models, evaluation metrics).
- **Editable** – `src/stitching/editable/` – the “genome” (`candidate_current.py`) and archived successful candidates.
- **Harness** – `src/stitching/harness/` – orchestrates the autoresearch loop, GitOps, resource budgeting, and performance ledger.
- **Agents** – `src/stitching/agents/` – routes optimisation requests to AI back‑ends (Codex, OpenCode/Anthropic).

## Key Features

1. **Autoresearch Framework (New)**
   * **Closed‑Loop Optimization** – “Propose → Evaluate → Decide” cycle powered by `src/stitching/harness/loop.py`.
   * **GitOps Integration** – isolated execution in temporary work‑trees with automatic commits for accepted improvements and reverts for regressions.
   * **Scientific Ledger** – append‑only log of every iteration (hypotheses, diffs, metrics, aggregate results).
   * **Multi‑Agent Support** – interchangeable back‑ends for OpenAI Codex and OpenCode (local or cloud models).
   * **Leaderboard** – automated ranking of best stitching candidates based on aggregate RMS error.
2. **High‑Fidelity Simulation (Digital Twin)**
   * **Optical PSF** – Gaussian blurring for optical smoothing and pixel fill factor.
   * **Surface Non‑Stationarity** – multi‑mode bending drift (Zernike Z4‑Z8).
   * **Mid‑Spatial Ripples** – periodic polishing marks fixed in the piece frame.
   * **Edge Roll‑off** – signal attenuation and noise boost at pupil boundaries.
   * **Advanced Scanning** – automated `grid` and `annular` scan plans with guaranteed coverage and sub‑pixel registration.
   * **Instrument Modeling** – support for square/circular pupils with NaN‑based invalid area semantics.
3. **Evaluation & Diagnostics**
   * **Rigorous Metrics** – RMS and MAE calculated strictly on valid intersections, geometry gates (IoU ≥ 0.99), and high‑frequency retention monitoring.
   * **Mismatch Analysis** – per‑pixel standard deviation maps to identify internal consistency errors and stitching artifacts.
   * **Automated Reporting** – 6‑view PNG reports providing deep visual insight into reconstruction quality.

## Project Status

- [x] Trusted Stack – complete high‑fidelity simulator and evaluation engine.
- [x] Harness – fully operational autoresearch loop with GitOps and Ledger.
- [x] Agents – integrated Codex and OpenCode back‑ends with structured prompting.
- [x] Baseline – piston‑corrected mean baseline established and ready for optimisation.
- [ ] Phase 2 – agent‑driven evolution toward GLS (Global Least Squares) and Robust M‑Estimators.
- [ ] Phase 3 – self‑calibration (CS/SC) for stationary reference bias removal.

## Getting Started

1. **Initial Evaluation** – run a baseline evaluation on existing scenarios:
   ```bash
   python scripts/run_single_loop.py --experiment-id baseline_eval --max-iterations 0
   ```
2. **Start Autoresearch** – launch the optimisation loop using the Codex backend:
   ```bash
   python scripts/run_single_loop.py --experiment-id exp_01 --backend codex --budget-profile fast
   ```
3. **Promote Best Candidate** – archive the top performer and update the leaderboard:
   ```bash
   python scripts/promote_best.py
   ```

## Scientific Mandate (`AGENTS.md`)
All agent modifications must follow the mathematical hypotheses described there. The trusted boundary is strictly enforced; agents can only modify `src/stitching/editable/`.

## License
MIT License

## Contributing
- Follow the pull‑request template.
- Commit style guidelines: run `black` and `isort` before commit.
- Linting: run `flake8`.

