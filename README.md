# ar-stitching

`ar-stitching` is a scientific Python repository for optical sub-aperture stitching autoresearch.

Current phase: **High-Fidelity Simulation Bench & Evaluation Foundation**.

## Repository intent

- `src/stitching/trusted/` contains the scientific reference: simulation, sub-pixel transforms, nuisance models (drift, retrace, structured bias), and evaluation metrics.
- `src/stitching/editable/` is reserved for future agent-editable stitching algorithms and comparison solvers.
- `src/stitching/harness/` orchestrates scenario loading, evaluation, and visualization.

## Key Features (Trusted Stack)

- **Digital Twin Realism**:
    - **Optical PSF**: Gaussian blurring to simulate optical smoothing and pixel fill factor.
    - **Surface Non-stationarity**: Multi-mode bending drift (Zernike Z4-Z8) over time.
    - **Mid-Spatial Ripples**: Periodic perturbations (polishing marks) fixed in piece XY frame.
    - **Edge Roll-off**: Signal attenuation and noise boost at pupil boundaries.
- **Sub-pixel & Rotation**: High-fidelity sampling with **Bicubic interpolation** and arbitrary rotation angles.
- **Automated Scan Plans**: Dynamic generation of `grid` and `annular` patterns with guaranteed edge-to-edge coverage and target overlap.
- **Instrument Modeling**: Square or circular detector pupils with **NaN-based** invalid area semantics.
- **Advanced Nuisance Models**:
    - **Pose Error**: Systematic bias, correlated drift (random walk), and independent jitter.
    - **Structured Bias**: Tip, tilt, and focus (Z1-Z4) as per-detector nuisances.
    - **Low-Frequency Noise**: Zernike-based (Fringe Z1-Z15) spectral perturbations.
    - **Complex Retrace**: Geometric field distortion (coordinate-based).
- **Visualization**: Comprehensive reporting harness generating 6-view PNG reports for any scenario.
- **Mismatch Diagnostics**: Per-pixel standard deviation maps and statistics (mean, median, p95) to assess internal consistency.

## Current status

- Typed contracts for core data flowing through the simulator and evaluator.
- 114+ unit and integration tests ensuring deterministic behavior and scientific sanity.
- Evaluation metrics with relaxed geometry gates (0.99 IoU) to support interpolation artifacts.
- Support for Legendre (square) and Zernike (circular) basis-driven surface generation.

## Not implemented yet

- Advanced optimization solvers (GLS, CS, SC) beyond the baseline scaffold.
- Adaptive mesh or non-uniform global grids.
