from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stitching.contracts import ScenarioConfig
from stitching.editable._legacy_basis import (
    basis_term_stack,
    overlap_support_mask,
    project_global_mask_to_tile,
    remove_low_order_modes,
)
from stitching.editable.neoss.baseline import CandidateStitcher
from stitching.trusted.simulator.identity import simulate_identity_observations


def _plane_detrend(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    effective_mask = np.asarray(mask, dtype=bool) & np.isfinite(data)
    if not np.any(effective_mask):
        return np.full_like(data, np.nan, dtype=float)
    yy, xx = np.indices(data.shape, dtype=float)
    A = np.column_stack([xx[effective_mask], yy[effective_mask], np.ones(np.count_nonzero(effective_mask))])
    coeffs, *_ = np.linalg.lstsq(A, data[effective_mask], rcond=None)
    out = np.full_like(data, np.nan, dtype=float)
    out[mask] = data[mask] - (np.column_stack([xx[mask], yy[mask], np.ones(np.count_nonzero(mask))]) @ coeffs)
    return out


def _scenario_neoss_mode(config: ScenarioConfig) -> tuple[str, str]:
    tp_mode = str(config.metadata.get("neoss_tp_mode", config.metadata.get("truth_basis", "Z"))).strip().upper()
    if tp_mode not in {"L", "Z"}:
        tp_mode = "Z"
    cs_mode = str(config.metadata.get("neoss_cs_mode", "L" if tp_mode == "L" else "Z")).strip().upper()
    if cs_mode not in {"L", "Z"}:
        cs_mode = "Z"
    return tp_mode, cs_mode


def _resolve_terms(config: ScenarioConfig, key: str, align_terms: tuple[int, ...], default_count: int) -> tuple[int, ...]:
    configured = config.metadata.get(key)
    if configured is not None:
        return tuple(int(v) for v in configured)
    start = (max(align_terms) + 1) if align_terms else 0
    return tuple(range(start, start + default_count))


def _build_common_metadata(config: ScenarioConfig, tp_mode: str, cs_mode: str) -> dict[str, object]:
    align_terms = tuple(int(v) for v in config.metadata.get("alignment_term", (0, 1, 2)))
    tp_terms = _resolve_terms(config, "neoss_tp_terms", align_terms, int(config.metadata.get("neoss_tp_default_count", 36)))
    cs_terms = _resolve_terms(config, "neoss_cs_terms", align_terms, int(config.metadata.get("neoss_cs_default_count", 36)))
    tp_terms = tuple(term for term in tp_terms if term not in align_terms)
    cs_terms = tuple(term for term in cs_terms if term not in align_terms and term not in {3, 4, 5})

    global_shape = tuple(int(v) for v in config.grid_shape)
    tile_shape = tuple(int(v) for v in config.tile_shape or config.grid_shape)
    rpupille_cs = float(config.metadata.get("neoss_rpupille_cs", global_shape[0] / 2.0))
    rpupille_tp = float(config.metadata.get("neoss_rpupille_tp", tile_shape[0] / 2.0))
    sigma_meta = config.metadata.get("neoss_sigma_px", config.metadata.get("neoss_sigma"))
    sigma_px = float(sigma_meta) if sigma_meta is not None else max(0.72 * (tile_shape[0] * float(config.metadata.get("detector_radius_fraction", 0.48))), 1.0)

    return {
        "align_terms": align_terms,
        "tp_terms": tp_terms,
        "cs_terms": cs_terms,
        "global_shape": global_shape,
        "tile_shape": tile_shape,
        "rpupille_cs": rpupille_cs,
        "rpupille_tp": rpupille_tp,
        "sigma_px": sigma_px,
        "coord_system": str(config.metadata.get("neoss_coordinate_system", "IRIDE")),
        "radius_fraction": float(config.metadata.get("detector_radius_fraction", 0.48)),
        "zernike_indexing": str(config.metadata.get("neoss_zernike_indexing", "iso")).lower(),
    }


def _build_matlab_input(
    observations,
    config: ScenarioConfig,
    tp_mode: str,
    cs_mode: str,
    python_metadata: dict[str, object],
) -> dict[str, object]:
    table_data = np.stack([np.asarray(obs.z, dtype=float).ravel(order="F") for obs in observations], axis=0)
    coords = []
    for obs in observations:
        dx, dy = obs.translation_xy
        coords.append((float(dx), float(dy)))
    coord_arr = np.asarray(coords, dtype=float)

    meta = _build_common_metadata(config, tp_mode, cs_mode)
    cfg = {
        "resolutionCS": float(meta["global_shape"][0]),
        "resolutionTP": float(meta["tile_shape"][0]),
        "RpupilleCS": float(meta["rpupille_cs"]),
        "RpupilleTP": float(meta["rpupille_tp"]),
        "lambda": float(config.metadata.get("neoss_lambda", 1.0)),
        "nb_cartes": float(len(observations)),
        "sigma": float(meta["sigma_px"]),
        "mismatch": float(config.metadata.get("neoss_mismatch", 0.0)),
        "mode_TP": np.array(tp_mode),
        "mode_CS": np.array(cs_mode),
        "indice_alignement": np.asarray([term + 1 for term in meta["align_terms"]], dtype=float),
        "indice_CS": np.asarray([term + 1 for term in meta["cs_terms"]], dtype=float),
        "indice_TP": np.asarray([term + 1 for term in meta["tp_terms"]], dtype=float),
        "limit": float(config.metadata.get("neoss_limit", 1.0)),
        "supportage": float(config.metadata.get("neoss_supportage", 0.0)),
        "pathSupportage": np.array(str(config.metadata.get("neoss_path_supportage", ""))),
        "SystemeCoordonnees": np.array(meta["coord_system"]),
        "Coord1": coord_arr[:, 0],
        "Coord2": coord_arr[:, 1],
        "use_random_map": bool(not bool(python_metadata.get("neoss_disable_random_map", True))),
    }
    return {"TableData": table_data, "cfg": cfg}


def _run_matlab_bridge(matlab_exe: str, input_path: Path, output_path: Path, *, obs_idx: int) -> None:
    bridge = "ar_stitching_debug_neoss_mlr"
    input_arg = str(input_path).replace("\\", "/")
    output_arg = str(output_path).replace("\\", "/")
    scripts_dir = str((REPO_ROOT / "scripts" / "matlab").resolve()).replace("\\", "/")
    cmd = [
        matlab_exe,
        "-batch",
        f"addpath(genpath('{scripts_dir}')); {bridge}('{input_arg}','{output_arg}',{obs_idx})",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _matlab_to_dict(value):
    if hasattr(value, "_fieldnames"):
        return {field: _matlab_to_dict(getattr(value, field)) for field in value._fieldnames}
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_matlab_to_dict(v) for v in value.ravel()]
    return value


def _array_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float | tuple[int, ...]]:
    if sparse.issparse(a):
        a = a.toarray()
    if sparse.issparse(b):
        b = b.toarray()
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return {"shape_a": a.shape, "shape_b": b.shape, "max_abs": float("nan"), "rms": float("nan")}
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"shape": a.shape, "max_abs": float("nan"), "rms": float("nan")}
    diff = a - b
    return {
        "shape": a.shape,
        "max_abs": float(np.max(np.abs(diff[mask]))),
        "rms": float(np.sqrt(np.mean(np.square(diff[mask])))),
        "mean": float(np.mean(diff[mask])),
    }


def _mask_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    if sparse.issparse(a):
        a = a.toarray()
    if sparse.issparse(b):
        b = b.toarray()
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        if a.size == b.size:
            a = a.reshape(-1)
            b = b.reshape(-1)
        else:
            return {"shape_a": a.shape, "shape_b": b.shape, "diff_count": None}
    return {
        "shape": a.shape,
        "diff_count": int(np.count_nonzero(a ^ b)),
        "agree_ratio": float(np.mean(a == b)),
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonicalize_svd_columns(U: np.ndarray) -> np.ndarray:
    U = np.array(U, copy=True, dtype=float)
    if U.size == 0:
        return U
    for col in range(U.shape[1]):
        column = U[:, col]
        if not np.any(np.isfinite(column)):
            continue
        anchor = int(np.argmax(np.abs(column)))
        if column[anchor] < 0:
            U[:, col] *= -1.0
    return U


def compute_python_debug(observations, config: ScenarioConfig) -> dict[str, object]:
    candidate = CandidateStitcher()
    meta = _build_common_metadata(config, _scenario_neoss_mode(config)[0], _scenario_neoss_mode(config)[1])
    tp_mode, cs_mode = _scenario_neoss_mode(config)

    overlap_support = overlap_support_mask(observations, meta["global_shape"])
    disable_random_map = bool(config.metadata.get("neoss_disable_random_map", True))
    radius_fraction = meta["radius_fraction"] if tp_mode == "Z" or cs_mode == "Z" else None
    zernike_indexing = meta["zernike_indexing"] if tp_mode == "Z" or cs_mode == "Z" else "fringe"

    if disable_random_map:
        detector_cal = np.zeros(meta["tile_shape"], dtype=float)
    else:
        detector_cal = candidate._initial_detector_calibration(  # pylint: disable=protected-access
            observations=observations,
            tile_shape=meta["tile_shape"],
            mode=tp_mode,
            terms=meta["tp_terms"],
            radius_fraction=radius_fraction,
            zernike_indexing=zernike_indexing,
            limit=float(config.metadata.get("neoss_limit", config.metadata.get("limit", 1.0))),
        )

    tp_rho = 1.0
    cs_rho = float(meta["tile_shape"][0]) / float(max(meta["global_shape"][0], 1))
    coord1_tp, coord2_tp = candidate._legacy_grille(  # pylint: disable=protected-access
        meta["tile_shape"],
        tp_rho,
        tp_mode,
        (0.0, 0.0, 0.0, 0.0),
        meta["coord_system"],
    )

    n_align = len(meta["align_terms"])
    n_tp = len(meta["tp_terms"])
    n_cs = len(meta["cs_terms"])
    n_ha = n_tp + n_cs
    n_obs = len(observations)
    n_elmt_y_i = n_ha + n_align

    M = np.zeros((n_obs * n_elmt_y_i, n_ha + n_align * n_obs), dtype=float)
    y = np.zeros(n_obs * n_elmt_y_i, dtype=float)
    obs_debug: list[dict[str, object]] = []

    for obs_idx, obs in enumerate(observations):
        block_rows = slice(obs_idx * n_elmt_y_i, (obs_idx + 1) * n_elmt_y_i)
        detector_view = np.asarray(detector_cal, dtype=float)
        local_obs = np.asarray(obs.z, dtype=float) - detector_view
        local_mask = np.asarray(obs.valid_mask, dtype=bool)
        local_overlap = project_global_mask_to_tile(
            overlap_support,
            meta["global_shape"],
            obs.tile_shape,
            obs.center_xy,
        )
        fit_mask = local_mask & local_overlap & np.isfinite(local_obs) & np.isfinite(detector_view)
        if not np.any(fit_mask):
            fit_mask = local_mask & np.isfinite(local_obs) & np.isfinite(detector_view)

        param = candidate._legacy_param(obs, meta["global_shape"], meta["coord_system"])  # pylint: disable=protected-access
        coord1_cs, coord2_cs = candidate._legacy_grille(  # pylint: disable=protected-access
            obs.tile_shape,
            cs_rho,
            cs_mode,
            param,
            meta["coord_system"],
        )

        coords = {
            "coord1_TP": coord1_tp[fit_mask],
            "coord2_TP": coord2_tp[fit_mask],
            "coord1_CS": coord1_cs[fit_mask],
            "coord2_CS": coord2_cs[fit_mask],
        }
        T = candidate._remplissage_matrice_fit(  # pylint: disable=protected-access
            tp_mode=tp_mode,
            cs_mode=cs_mode,
            align_terms=meta["align_terms"],
            tp_terms=meta["tp_terms"],
            cs_terms=meta["cs_terms"],
            coords=coords,
            zernike_indexing=zernike_indexing,
        )

        carte = local_obs[fit_mask]
        skipped = bool(T.size == 0 or carte.size == 0 or T.shape[0] < T.shape[1])
        if skipped:
            y_i = np.zeros((n_elmt_y_i,), dtype=float)
            M_i = np.zeros((n_elmt_y_i, n_elmt_y_i), dtype=float)
        else:
            U, _, _ = np.linalg.svd(T, full_matrices=False)
            U = _canonicalize_svd_columns(U)
            y_i = U.T @ carte
            M_i = U.T @ T
            y[block_rows] = y_i
            M[block_rows, :n_ha] = M_i[:, n_align:]
            align_col = n_ha + obs_idx * n_align
            M[block_rows, align_col : align_col + n_align] = M_i[:, :n_align]

        obs_debug.append(
            {
                "obs_idx": obs_idx + 1,
                "translation_xy": (float(obs.translation_xy[0]), float(obs.translation_xy[1])),
                "param": tuple(float(v) for v in param),
                "carte_raw": np.asarray(obs.z, dtype=float),
                "carte_masked": np.where(fit_mask, local_obs, np.nan),
                "carte_masked_full": local_obs,
                "carte_fit": carte,
                "coord1_TP_full": coord1_tp,
                "coord2_TP_full": coord2_tp,
                "coord1_CS_full": coord1_cs,
                "coord2_CS_full": coord2_cs,
                "coord1_TP": coords["coord1_TP"],
                "coord2_TP": coords["coord2_TP"],
                "coord1_CS": coords["coord1_CS"],
                "coord2_CS": coords["coord2_CS"],
                "T": T,
                "y_i": y_i,
                "M_i": M_i,
                "mask": fit_mask,
                "fit_mask": np.ones_like(carte, dtype=bool),
                "skipped": skipped,
            }
        )

    try:
        x = np.linalg.solve(M, y)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(M, y, rcond=None)

    tp_coeffs = x[:n_tp]
    cs_coeffs = x[n_tp:n_ha]
    align_coeffs = x[n_ha:].reshape(n_obs, n_align) if n_align else np.zeros((n_obs, 0), dtype=float)

    tp_basis_stack, tp_mask = basis_term_stack(
        tp_mode,
        meta["tp_terms"],
        meta["tile_shape"],
        radius_fraction=radius_fraction,
        zernike_indexing=zernike_indexing,
    )
    detector_map = np.asarray(detector_cal, dtype=float)
    if tp_basis_stack.size:
        detector_map = detector_map + np.tensordot(tp_coeffs, tp_basis_stack, axes=(0, 0))
    if tp_mode == "Z":
        detector_map = np.where(tp_mask, detector_map, np.nan)

    return {
        "nb_obs": n_obs,
        "resolutionTP": meta["tile_shape"][0],
        "resolutionCS": meta["global_shape"][0],
        "nb_elmt_y_i": n_elmt_y_i,
        "nb_term_HA": n_ha,
        "nb_term_alignement": n_align,
        "align_terms": meta["align_terms"],
        "tp_terms": meta["tp_terms"],
        "cs_terms": meta["cs_terms"],
        "carte_random": np.asarray(detector_cal, dtype=float),
        "M": M,
        "y": y,
        "x": x,
        "tp_coeffs": tp_coeffs,
        "cs_coeffs": cs_coeffs,
        "align_coeffs": align_coeffs,
        "detector_map": detector_map,
        "overlap_support": overlap_support,
        "obs_data": obs_debug,
    }


def _compare_obs(matlab_obs: dict[str, object], python_obs: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "obs_idx": int(matlab_obs.get("obs_idx", python_obs.get("obs_idx", 0))),
        "issues": [],
    }
    for key in ("translation_xy", "param"):
        if key in matlab_obs and key in python_obs:
            a = np.asarray(matlab_obs[key], dtype=float)
            b = np.asarray(python_obs[key], dtype=float)
            metrics = _array_metrics(a, b)
            result[f"{key}_metrics"] = metrics
            if np.isfinite(metrics.get("rms", np.nan)) and metrics["rms"] > 1e-9:
                result["issues"].append(f"{key} mismatch rms={metrics['rms']:.6e}")

    for key in (
        "carte_raw",
        "carte_masked",
        "carte_masked_full",
        "carte_fit",
        "coord1_TP_full",
        "coord2_TP_full",
        "coord1_CS_full",
        "coord2_CS_full",
        "coord1_TP",
        "coord2_TP",
        "coord1_CS",
        "coord2_CS",
        "y_i",
        "T",
        "M_i",
    ):
        if key in matlab_obs and key in python_obs:
            metrics = _array_metrics(matlab_obs[key], python_obs[key])
            result[f"{key}_metrics"] = metrics
            if np.isfinite(metrics.get("rms", np.nan)) and metrics["rms"] > 1e-9:
                result["issues"].append(f"{key} mismatch rms={metrics['rms']:.6e}")
            if key == "T" and np.isfinite(metrics.get("rms", np.nan)) and metrics["rms"] > 1e-9:
                a = np.asarray(matlab_obs[key], dtype=float)
                b = np.asarray(python_obs[key], dtype=float)
                if a.shape == b.shape and a.ndim == 2:
                    diff = a - b
                    col_rms = np.sqrt(np.nanmean(np.square(diff), axis=0))
                    row_rms = np.sqrt(np.nanmean(np.square(diff), axis=1))
                    result["T_column_rms"] = col_rms.tolist()
                    result["T_row_rms"] = row_rms.tolist()

    if "fit_mask" in matlab_obs and "fit_mask" in python_obs:
        result["fit_mask_metrics"] = _mask_metrics(matlab_obs["fit_mask"], python_obs["fit_mask"])
        if result["fit_mask_metrics"].get("diff_count"):
            result["issues"].append(f"fit_mask differs on {result['fit_mask_metrics']['diff_count']} pixels")
    if "mask" in matlab_obs and "mask" in python_obs:
        result["mask_metrics"] = _mask_metrics(matlab_obs["mask"], python_obs["mask"])
        if result["mask_metrics"].get("diff_count"):
            result["issues"].append(f"mask differs on {result['mask_metrics']['diff_count']} pixels")

    result["has_issues"] = bool(result["issues"])
    return result


def compare_debug_data(matlab_debug: dict[str, object], python_debug: dict[str, object], output_dir: Path) -> dict[str, object]:
    results: dict[str, object] = {}

    results["M_metrics"] = _array_metrics(matlab_debug["M"], python_debug["M"])
    results["y_metrics"] = _array_metrics(matlab_debug["y"], python_debug["y"])
    results["x_metrics"] = _array_metrics(matlab_debug["x"], python_debug["x"])
    if "carte_random" in matlab_debug and "carte_random" in python_debug:
        results["carte_random_metrics"] = _array_metrics(matlab_debug["carte_random"], python_debug["carte_random"])
    results["detector_map_metrics"] = _array_metrics(matlab_debug["carte_Instrument"], python_debug["detector_map"])
    overlap_key = "overlap_support" if "overlap_support" in matlab_debug else "masqueRecouvrement"
    results["overlap_support_metrics"] = _mask_metrics(matlab_debug[overlap_key], python_debug["overlap_support"])

    obs_results = []
    n_obs = min(int(matlab_debug["nb_obs"]), len(python_debug["obs_data"]))
    for i in range(n_obs):
        obs_key = f"obs_{i + 1:02d}"
        if obs_key not in matlab_debug:
            continue
        obs_results.append(_compare_obs(matlab_debug[obs_key], python_debug["obs_data"][i]))
    results["observations"] = obs_results

    print("\n" + "=" * 88)
    print("NEOSS MLR BLOCK-BY-BLOCK COMPARISON")
    print("=" * 88)
    print(f"M      : rms={results['M_metrics'].get('rms', np.nan):.6e}  max={results['M_metrics'].get('max_abs', np.nan):.6e}")
    print(f"y      : rms={results['y_metrics'].get('rms', np.nan):.6e}  max={results['y_metrics'].get('max_abs', np.nan):.6e}")
    print(f"x      : rms={results['x_metrics'].get('rms', np.nan):.6e}  max={results['x_metrics'].get('max_abs', np.nan):.6e}")
    if "carte_random_metrics" in results:
        print(
            f"rand   : rms={results['carte_random_metrics'].get('rms', np.nan):.6e}  "
            f"max={results['carte_random_metrics'].get('max_abs', np.nan):.6e}"
        )
    print(f"calib  : rms={results['detector_map_metrics'].get('rms', np.nan):.6e}  max={results['detector_map_metrics'].get('max_abs', np.nan):.6e}")
    print(
        f"overlap: diff={results['overlap_support_metrics'].get('diff_count', 'N/A')} "
        f"agree={results['overlap_support_metrics'].get('agree_ratio', np.nan):.6f}"
    )

    bad_obs = [item for item in obs_results if item.get("has_issues")]
    if bad_obs:
        print("\nPer-observation issues:")
        for item in bad_obs[:8]:
            print(f"  obs {item['obs_idx']:02d}: {', '.join(item['issues'][:4])}")
            if "T_column_rms" in item:
                col_rms = np.asarray(item["T_column_rms"], dtype=float)
                top = np.argsort(col_rms)[::-1][:5]
                print(
                    "    T columns: "
                    + ", ".join(f"{int(idx)}={col_rms[int(idx)]:.3e}" for idx in top)
                )
    else:
        print("\nPer-observation issues: none")

    output_path = output_dir / "neoss_mlr_block_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nResults saved to: {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug NEOSS MLR block comparison between MATLAB and Python.")
    parser.add_argument("--scenario", type=str, default="scenarios/s17_highres_circular.yaml")
    parser.add_argument("--matlab-exe", type=str, default=None)
    parser.add_argument("--obs-idx", type=int, default=0, help="Observation index to export from MATLAB (0=all)")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    config = ScenarioConfig.from_yaml(scenario_path)
    tp_mode, cs_mode = _scenario_neoss_mode(config)

    python_config = ScenarioConfig(
        scenario_id=config.scenario_id,
        description=config.description,
        grid_shape=config.grid_shape,
        pixel_size=config.pixel_size,
        scan_offsets=config.scan_offsets,
        tile_shape=config.tile_shape,
        baseline_name=config.baseline_name,
        rotation_deg=config.rotation_deg,
        reference_bias=config.reference_bias,
        gaussian_noise_std=config.gaussian_noise_std,
        outlier_fraction=config.outlier_fraction,
        retrace_error=config.retrace_error,
        seed=config.seed,
        metadata={
            **config.metadata,
            "neoss_tp_mode": tp_mode,
            "neoss_cs_mode": cs_mode,
            "neoss_disable_random_map": False,
        },
    )

    _, observations = simulate_identity_observations(python_config)
    python_debug = compute_python_debug(observations, python_config)

    matlab_exe = args.matlab_exe or shutil.which("matlab") or str(Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"))
    if not Path(matlab_exe).exists():
        raise FileNotFoundError(f"Cannot find MATLAB executable: {matlab_exe}")

    with tempfile.TemporaryDirectory(prefix="neoss_mlr_debug_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "neoss_debug_input.mat"
        output_path = tmp_path / "neoss_debug_output.mat"
        payload = _build_matlab_input(observations, python_config, tp_mode, cs_mode, python_debug)
        savemat(input_path, payload, do_compression=True)
        _run_matlab_bridge(matlab_exe, input_path, output_path, obs_idx=args.obs_idx)
        matlab_payload = loadmat(output_path, squeeze_me=True, struct_as_record=False)

    matlab_debug = {key: _matlab_to_dict(value) for key, value in matlab_payload.items() if not key.startswith("__")}
    results = compare_debug_data(matlab_debug, python_debug, REPO_ROOT / "artifacts")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
