# ar-stitching

`ar-stitching` is a scientific Python repository for optical sub-aperture stitching autoresearch.

Current phase: foundation only.

## Repository intent

- `src/stitching/trusted/` contains trusted simulation, masking, transforms, nuisance models, and evaluation code.
- `src/stitching/editable/` is reserved for future agent-editable stitching algorithms.
- `src/stitching/harness/` orchestrates scenario loading and evaluation runs.

## Current status

- Typed contracts for the core data flowing through the simulator and evaluator
- Placeholder trusted modules with testable default behavior
- Canonical scenario YAML files for early experiments
- Initial identity-focused tests and geometry invariants
- Mean tile-fusion baseline as the primary reference, with an experimental median baseline kept for comparison only

## Not implemented yet

- Advanced optimization methods such as GLS, CS, or SC
- Full reconstruction pipelines beyond an identity baseline scaffold
