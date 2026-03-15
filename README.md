# ar-stitching

`ar-stitching` is a scientific Python repository for optical sub-aperture stitching autoresearch.

Current phase: **Simulation Bench & Evaluation Foundation**.

## Repository intent

- `src/stitching/trusted/` contains the scientific reference: simulation, sub-pixel transforms, nuisance models (drift, retrace, structured bias), and evaluation metrics.
- `src/stitching/editable/` is reserved for future agent-editable stitching algorithms and comparison solvers.
- `src/stitching/harness/` orchestrates scenario loading and evaluation runs.

## Key Features (Trusted Stack)

- **Digital Twin Realism**:
    - **Optical PSF**: Gaussian blurring to simulate optical smoothing and pixel fill factor.
    - **Surface Non-stationarity**: Bending drift (focus/flexion) of the piece over time.
    - **Mid-Spatial Ripples**: Periodic perturbations simulating polishing marks or machining traces.
    - **Edge Roll-off**: Signal attenuation and noise boost at pupil boundaries.
- **Sub-pixel & Rotation**: High-fidelity sampling with **Bicubic interpolation** and arbitrary rotation angles.
- **Instrument Modeling**: Choice of square or circular detector pupils with soft edges.
- **Advanced Nuisance Models**:
    - **Pose Error**: Systematic bias, correlated drift (random walk), and independent jitter.
    - **Structured Bias**: Tip, tilt, and focus (Z1-Z4) as per-detector nuisances.
    - **Low-Frequency Noise**: Zernike-based (Fringe Z1-Z15) spectral perturbations.
    - **Complex Retrace**: Geometric field distortion (coordinate-based).
- **Mismatch Diagnostics**: Per-pixel standard deviation maps and statistics (mean, median, p95) to assess internal consistency without ground truth.
- **Scenario-Driven**: 15+ canonical YAML scenarios covering noise, outliers, drift, and realistic metrology conditions.

## Current status

- Typed contracts for core data flowing through the simulator and evaluator.
- 105+ unit and integration tests ensuring deterministic behavior and scientific sanity.
- Evaluation metrics with relaxed geometry gates (0.99 IoU) to support interpolation artifacts.
- Support for Legendre (square) and Zernike (circular) basis-driven surface generation.
- Mean and Median tile-fusion baselines as primary reference points.

## Not implemented yet

- Advanced optimization solvers (GLS, CS, SC) beyond the baseline scaffold.
- Higher-order interpolation schemes (bicubic, splines).
- Arbitrary rotation resampling (currently limited to 90-degree increments).
