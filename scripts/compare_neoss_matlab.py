from __future__ import annotations

import argparse
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
from stitching.editable._legacy_basis import place_tile_in_global_frame
from stitching.harness.evaluator import load_candidate_module
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


def _build_matlab_input(observations, config: ScenarioConfig, tp_mode: str, cs_mode: str, python_metadata: dict[str, object]) -> dict[str, object]:
    # Use the raw tile layout; the convention sweep script identifies this as the
    # best bridge for the NEOSS MATLAB legacy calibration map.
    table_data = np.stack([np.asarray(obs.z, dtype=float).ravel(order="F") for obs in observations], axis=0)
    align_terms = tuple(int(v) for v in python_metadata.get("alignment_terms", config.metadata.get("alignment_term", (0, 1, 2))))
    tp_terms = tuple(int(v) for v in python_metadata.get("tp_terms", config.metadata.get("neoss_tp_terms", ())))
    cs_terms = tuple(int(v) for v in python_metadata.get("cs_terms", config.metadata.get("neoss_cs_terms", ())))

    if not tp_terms:
        tp_terms = tuple(range(3, 3 + int(config.metadata.get("neoss_tp_default_count", 36))))
    if not cs_terms:
        cs_terms = tuple(range(6, 6 + int(config.metadata.get("neoss_cs_default_count", 36))))

    detector_radius_fraction = float(config.metadata.get("detector_radius_fraction", 0.48))
    global_shape = tuple(int(v) for v in config.grid_shape)
    tile_shape = tuple(int(v) for v in config.tile_shape or config.grid_shape)
    rpupille_cs = float(config.metadata.get("neoss_rpupille_cs", global_shape[0] / 2.0))
    rpupille_tp = float(config.metadata.get("neoss_rpupille_tp", tile_shape[0] / 2.0))
    sigma_meta = config.metadata.get("neoss_sigma_px", config.metadata.get("neoss_sigma"))
    sigma_px = float(sigma_meta) if sigma_meta is not None else max(0.72 * (tile_shape[0] * detector_radius_fraction), 1.0)

    coords = []
    for obs in observations:
        dx, dy = obs.translation_xy
        coords.append((float(dx), float(dy)))
    coord_arr = np.asarray(coords, dtype=float)

    cfg = {
        "resolutionCS": float(global_shape[0]),
        "resolutionTP": float(tile_shape[0]),
        "RpupilleCS": rpupille_cs,
        "RpupilleTP": rpupille_tp,
        "lambda": float(config.metadata.get("neoss_lambda", 1.0)),
        "nb_cartes": float(len(observations)),
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
        "Coord1": coord_arr[:, 0],
        "Coord2": coord_arr[:, 1],
        "use_random_map": bool(not bool(config.metadata.get("neoss_disable_random_map", True))),
    }
    return {"TableData": table_data, "cfg": cfg}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NEOSS MATLAB vs Python on a scenario.")
    parser.add_argument("--scenario", type=str, default="scenarios/s17_highres_circular.yaml")
    parser.add_argument("--matlab-exe", type=str, default=None)
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

    matlab_exe = args.matlab_exe or shutil.which("matlab") or str(Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"))
    if not Path(matlab_exe).exists():
        raise FileNotFoundError(f"Cannot find MATLAB executable: {matlab_exe}")

    with tempfile.TemporaryDirectory(prefix="neoss_matlab_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "neoss_input.mat"
        output_path = tmp_path / "neoss_output.mat"
        payload = _build_matlab_input(observations, python_config, tp_mode, cs_mode, python_recon.metadata)
        savemat(input_path, payload, do_compression=True)
        _run_matlab_bridge(matlab_exe, input_path, output_path)
        matlab_payload = loadmat(output_path, squeeze_me=True, struct_as_record=False)

    matlab_map = np.asarray(matlab_payload["map"], dtype=float)
    matlab_cal = np.asarray(matlab_payload["carte_Instrument"], dtype=float)
    python_map = np.asarray(python_recon.z, dtype=float)
    python_cal = np.asarray(python_recon.metadata.get("instrument_calibration"), dtype=float)

    map_mask = np.isfinite(matlab_map) & np.isfinite(python_map)
    cal_mask = np.isfinite(matlab_cal) & np.isfinite(python_cal)
    if not np.any(map_mask):
        raise RuntimeError("No finite overlap between MATLAB and Python NEOSS maps.")
    if not np.any(cal_mask):
        raise RuntimeError("No finite overlap between MATLAB and Python NEOSS calibration maps.")

    map_diff = python_map - matlab_map
    map_diff_detrended = _plane_detrend(map_diff, map_mask)
    cal_diff = python_cal - matlab_cal
    cal_diff_detrended = _plane_detrend(cal_diff, cal_mask)

    map_rms = float(np.sqrt(np.nanmean(np.square(map_diff[map_mask]))))
    map_rms_detrended = float(np.sqrt(np.nanmean(np.square(map_diff_detrended[map_mask]))))
    cal_rms = float(np.sqrt(np.nanmean(np.square(cal_diff[cal_mask]))))
    cal_rms_detrended = float(np.sqrt(np.nanmean(np.square(cal_diff_detrended[cal_mask]))))

    print(f"Scenario: {scenario_path}")
    print(f"TP mode: {tp_mode}")
    print(f"CS mode: {cs_mode}")
    print(f"Python map shape: {python_map.shape}")
    print(f"MATLAB map shape: {matlab_map.shape}")
    print(f"Python vs MATLAB map RMS: {map_rms:.6f}")
    print(f"Python vs MATLAB map RMS detrended: {map_rms_detrended:.6f}")
    print(f"Python calibration shape: {python_cal.shape}")
    print(f"MATLAB calibration shape: {matlab_cal.shape}")
    print(f"Python vs MATLAB calibration RMS: {cal_rms:.6f}")
    print(f"Python vs MATLAB calibration RMS detrended: {cal_rms_detrended:.6f}")

    artifact_dir = REPO_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    map_vmin = np.nanpercentile(np.concatenate([python_map[map_mask], matlab_map[map_mask]]), 2)
    map_vmax = np.nanpercentile(np.concatenate([python_map[map_mask], matlab_map[map_mask]]), 98)
    cal_vmin = np.nanpercentile(np.concatenate([python_cal[cal_mask], matlab_cal[cal_mask]]), 2)
    cal_vmax = np.nanpercentile(np.concatenate([python_cal[cal_mask], matlab_cal[cal_mask]]), 98)

    im0 = axes[0, 0].imshow(python_map, origin="lower", cmap="viridis", vmin=map_vmin, vmax=map_vmax)
    axes[0, 0].set_title("Python NEOSS\n(complete)")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(matlab_map, origin="lower", cmap="viridis", vmin=map_vmin, vmax=map_vmax)
    axes[0, 1].set_title("MATLAB NEOSS\n(complete)")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(map_diff_detrended, origin="lower", cmap="RdBu_r")
    axes[0, 2].set_title("Python - MATLAB\n(map detrended)")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(python_cal, origin="lower", cmap="viridis", vmin=cal_vmin, vmax=cal_vmax)
    axes[1, 0].set_title("Python NEOSS\n(calibration)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(matlab_cal, origin="lower", cmap="viridis", vmin=cal_vmin, vmax=cal_vmax)
    axes[1, 1].set_title("MATLAB NEOSS\n(calibration)")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(cal_diff_detrended, origin="lower", cmap="RdBu_r")
    axes[1, 2].set_title("Python - MATLAB\n(calibration detrended)")
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    output_fig = artifact_dir / f"neoss_matlab_vs_python_{scenario_path.stem}.png"
    plt.savefig(output_fig, dpi=160)
    print(f"Comparison figure: {output_fig}")


if __name__ == "__main__":
    main()
