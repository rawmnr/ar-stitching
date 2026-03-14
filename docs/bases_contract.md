# Bases Contract

## Scope

This document defines how trusted basis-driven surface generation is structured in the repository.

## Design Rule

- The repository core depends only on `numpy` and `scipy` for basis logic.
- Optical basis backends such as Zernike/Fringe are optional adapters behind internal APIs.
- Trusted simulator and evaluator code must call internal basis functions, not external packages directly.

## Current Internal API

- `generate_legendre_surface(coefficients, shape)`
- `sample_legendre_basis_2d(shape, max_degree_y, max_degree_x)`
- `surface_from_basis(shape, pixel_size, basis_name, coefficients, units="arb")`
- `generate_zernike_surface(...)` for optional circular-pupil backends

## Basis Semantics

- Legendre basis is implemented as a tensor product of 1D Legendre polynomials on normalized axes in `[-1, 1]`.
- Zernike/Fringe generation is treated as circular-pupil-only logic and remains optional.
- Returned surfaces must always be plain NumPy arrays in repo-owned contracts.

## Backend Policy

- `legendre.py` is mandatory and trusted.
- `zernike.py` is an adapter layer only.
- `internal` Zernike generation is always available in the default install.
- `optiland` is the only optional backend currently considered by `backend="auto"`.
- `prysm` is not yet a supported public backend; requests for it should fail explicitly until the adapter is fully wired.

## Current Limitations

- Only Legendre and internal Zernike generation are guaranteed to be available in the default install.
- No Zernike fit pipeline is wired into the simulator yet.
- No backend-specific objects are allowed to leak outside `trusted/bases/`.
- Prysm remains an installation extra, not an active trusted backend.
