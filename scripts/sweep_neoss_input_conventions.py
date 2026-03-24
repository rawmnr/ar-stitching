from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stitching.contracts import ScenarioConfig
from stitching.harness.evaluator import load_candidate_module
from stitching.trusted.simulator.identity import simulate_identity_observations


@dataclasses.dataclass(frozen=True)
class ConventionSpec:
    name: str
    tile_transform: callable
    coord_transform: callable


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


def _build_base_cfg(config: ScenarioConfig, tp_mode: str, cs_mode: str, python_metadata: dict[str, object]) -> dict[str, object]:
    align_terms = tuple(int(v) for v in python_metadata.get("alignment_terms", config.metadata.get("alignment_term", (0, 1, 2))))
    tp_terms = tuple(int(v) for v in python_metadata.get("tp_terms", config.metadata.get("neoss_tp_terms", ())))
    cs_terms = tuple(int(v) for v in python_metadata.get("cs_terms", config.metadata.get("neoss_cs_terms", ())))

    if not tp_terms:
        tp_terms = tuple(range(3, 3 + int(config.metadata.get("neoss_tp_default_count", 36))))
    if not cs_terms:
        cs_terms = tuple(range(6, 6 + int(config.metadata.get("neoss_cs_default_count", 36))))

    global_shape = tuple(int(v) for v in config.grid_shape)
    tile_shape = tuple(int(v) for v in config.tile_shape or config.grid_shape)
    rpupille_cs = float(config.metadata.get("neoss_rpupille_cs", global_shape[0] / 2.0))
    rpupille_tp = float(config.metadata.get("neoss_rpupille_tp", tile_shape[0] / 2.0))
    sigma_meta = config.metadata.get("neoss_sigma_px", config.metadata.get("neoss_sigma"))
    sigma_px = float(sigma_meta) if sigma_meta is not None else max(0.72 * (tile_shape[0] * float(config.metadata.get("detector_radius_fraction", 0.48))), 1.0)

    cfg = {
        "resolutionCS": float(global_shape[0]),
        "resolutionTP": float(tile_shape[0]),
        "RpupilleCS": rpupille_cs,
        "RpupilleTP": rpupille_tp,
        "lambda": float(config.metadata.get("neoss_lambda", 1.0)),
        "nb_cartes": float(0),
        "sigma": sigma_px,
        "mismatch": float(config.metadata.get("neoss_mismatch", 0.0)),
        "mode_TP": np.array(tp_mode),
        "mode_CS": np.array(cs_mode),
        "indice_alignement": np.asarray([term + 1 for term in align_terms], dtype=float),
        "indice_CS": np.asarray([term + 1 for term in cs_terms], dtype=float),
        "indice_TP": np.asarray([term + 1 for term in tp_terms], dtype=float),
        "limit": float(config.metadata.get("neoss_limit", 1.0)),
        "supportage": float(config.metadata.get("neoss_supportage", 0.0)),
        "pathSupportage": np.array(str(config.metadata.get("neoss_path_supportage", ""))),
        "SystemeCoordonnees": np.array(str(config.metadata.get("neoss_coordinate_system", "IRIDE"))),
        "use_random_map": bool(not bool(config.metadata.get("neoss_disable_random_map", True))),
    }
    return cfg


def _run_matlab_bridge(matlab_exe: str, input_path: Path, output_path: Path) -> None:
    bridge = "ar_stitching_run_neoss_bridge"
    input_arg = str(input_path).replace("\\", "/")
    output_arg = str(output_path).replace("\\", "/")
    scripts_dir = str((REPO_ROOT / "scripts" / "matlab").resolve()).replace("\\", "/")
    cmd = [
        matlab_exe,
        "-batch",
        f"addpath(genpath('{scripts_dir}')); {bridge}('{input_arg}','{output_arg}')",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _build_conventions() -> tuple[ConventionSpec, ...]:
    return (
        ConventionSpec(
            name="raw_xy",
            tile_transform=lambda obs: np.asarray(obs.z, dtype=float),
            coord_transform=lambda dx, dy: (float(dx), float(dy)),
        ),
        ConventionSpec(
            name="raw_negxy",
            tile_transform=lambda obs: np.asarray(obs.z, dtype=float),
            coord_transform=lambda dx, dy: (-float(dx), -float(dy)),
        ),
        ConventionSpec(
            name="raw_swap",
            tile_transform=lambda obs: np.asarray(obs.z, dtype=float),
            coord_transform=lambda dx, dy: (float(dy), float(dx)),
        ),
        ConventionSpec(
            name="raw_swap_neg",
            tile_transform=lambda obs: np.asarray(obs.z, dtype=float),
            coord_transform=lambda dx, dy: (-float(dy), -float(dx)),
        ),
        ConventionSpec(
            name="flipudF_xy",
            tile_transform=lambda obs: np.flipud(np.asarray(obs.z, dtype=float)),
            coord_transform=lambda dx, dy: (float(dx), float(dy)),
        ),
        ConventionSpec(
            name="flipudF_negxy",
            tile_transform=lambda obs: np.flipud(np.asarray(obs.z, dtype=float)),
            coord_transform=lambda dx, dy: (-float(dx), -float(dy)),
        ),
        ConventionSpec(
            name="fliplr_xy",
            tile_transform=lambda obs: np.fliplr(np.asarray(obs.z, dtype=float)),
            coord_transform=lambda dx, dy: (float(dx), float(dy)),
        ),
        ConventionSpec(
            name="fliplr_negxy",
            tile_transform=lambda obs: np.fliplr(np.asarray(obs.z, dtype=float)),
            coord_transform=lambda dx, dy: (-float(dx), -float(dy)),
        ),
    )


def _serialize_convention_input(
    observations,
    config: ScenarioConfig,
    tp_mode: str,
    cs_mode: str,
    convention: ConventionSpec,
    python_metadata: dict[str, object],
) -> dict[str, object]:
    base_cfg = _build_base_cfg(config, tp_mode, cs_mode, python_metadata)
    table_rows = []
    coords = []
    for obs in observations:
        transformed = convention.tile_transform(obs)
        if transformed.ndim == 2:
            table_rows.append(np.asarray(transformed, dtype=float).ravel(order="F"))
        else:
            table_rows.append(np.asarray(transformed, dtype=float).ravel(order="F"))
        dx, dy = obs.translation_xy
        cx, cy = convention.coord_transform(dx, dy)
        coords.append((cx, cy))

    base_cfg["nb_cartes"] = float(len(observations))
    coord_arr = np.asarray(coords, dtype=float)
    base_cfg["Coord1"] = coord_arr[:, 0]
    base_cfg["Coord2"] = coord_arr[:, 1]
    return {"TableData": np.stack(table_rows, axis=0), "cfg": base_cfg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep NEOSS MATLAB input conventions and rank them.")
    parser.add_argument("--scenario", type=str, default="scenarios/s17_highres_circular.yaml")
    parser.add_argument("--matlab-exe", type=str, default=None)
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
        metadata={**config.metadata, "neoss_tp_mode": tp_mode, "neoss_cs_mode": cs_mode, "neoss_disable_random_map": False},
    )

    _, observations = simulate_identity_observations(python_config)
    candidate = load_candidate_module(REPO_ROOT / "src" / "stitching" / "editable" / "neoss" / "baseline.py")
    python_recon = candidate.reconstruct(observations, python_config)
    python_map = np.asarray(python_recon.z, dtype=float)
    python_cal = np.asarray(python_recon.metadata.get("instrument_calibration"), dtype=float)

    matlab_exe = args.matlab_exe or shutil.which("matlab") or str(Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"))
    if not Path(matlab_exe).exists():
        raise FileNotFoundError(f"Cannot find MATLAB executable: {matlab_exe}")

    conventions = _build_conventions()
    rows = []

    with tempfile.TemporaryDirectory(prefix="neoss_conventions_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for convention in conventions:
            input_path = tmp_path / f"input_{convention.name}.mat"
            output_path = tmp_path / f"output_{convention.name}.mat"
            payload = _serialize_convention_input(observations, python_config, tp_mode, cs_mode, convention, python_recon.metadata)
            savemat(input_path, payload, do_compression=True)
            _run_matlab_bridge(matlab_exe, input_path, output_path)
            matlab_payload = loadmat(output_path, squeeze_me=True, struct_as_record=False)

            matlab_map = np.asarray(matlab_payload["map"], dtype=float)
            matlab_cal = np.asarray(matlab_payload["carte_Instrument"], dtype=float)

            map_mask = np.isfinite(matlab_map) & np.isfinite(python_map)
            cal_mask = np.isfinite(matlab_cal) & np.isfinite(python_cal)
            if not np.any(map_mask) or not np.any(cal_mask):
                map_rms = np.nan
                map_rms_detrended = np.nan
                cal_rms = np.nan
                cal_rms_detrended = np.nan
            else:
                map_diff = python_map - matlab_map
                cal_diff = python_cal - matlab_cal
                map_diff_detrended = _plane_detrend(map_diff, map_mask)
                cal_diff_detrended = _plane_detrend(cal_diff, cal_mask)
                map_rms = float(np.sqrt(np.nanmean(np.square(map_diff[map_mask]))))
                map_rms_detrended = float(np.sqrt(np.nanmean(np.square(map_diff_detrended[map_mask]))))
                cal_rms = float(np.sqrt(np.nanmean(np.square(cal_diff[cal_mask]))))
                cal_rms_detrended = float(np.sqrt(np.nanmean(np.square(cal_diff_detrended[cal_mask]))))

            rows.append(
                {
                    "name": convention.name,
                    "map_rms": map_rms,
                    "map_rms_detrended": map_rms_detrended,
                    "cal_rms": cal_rms,
                    "cal_rms_detrended": cal_rms_detrended,
                }
            )

    rows.sort(key=lambda row: (np.inf if np.isnan(row["cal_rms_detrended"]) else row["cal_rms_detrended"]))

    print(f"Scenario: {scenario_path}")
    print(f"TP mode: {tp_mode}")
    print(f"CS mode: {cs_mode}")
    print(f"{'Convention':<18} {'Map RMS':>10} {'Map RMS dt':>12} {'Cal RMS':>10} {'Cal RMS dt':>12}")
    print("-" * 66)
    for row in rows:
        print(
            f"{row['name']:<18} "
            f"{row['map_rms']:>10.6f} "
            f"{row['map_rms_detrended']:>12.6f} "
            f"{row['cal_rms']:>10.6f} "
            f"{row['cal_rms_detrended']:>12.6f}"
        )

    best = rows[0]
    print("-" * 66)
    print(f"Best convention by calibration detrended RMS: {best['name']}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    artifact_dir = REPO_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    names = [row["name"] for row in rows]
    values = [row["cal_rms_detrended"] for row in rows]
    ax.bar(names, values, color="#4c72b0")
    ax.set_ylabel("Calibration RMS detrended")
    ax.set_title(f"NEOSS input convention sweep: {scenario_path.stem}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out_fig = artifact_dir / f"neoss_input_conventions_{scenario_path.stem}.png"
    plt.savefig(out_fig, dpi=160)
    print(f"Comparison figure: {out_fig}")


if __name__ == "__main__":
    main()
