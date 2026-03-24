from __future__ import annotations

import argparse
import os
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
from stitching.editable._legacy_basis import place_tile_in_global_frame
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


def _scenario_subaper_mode(config: ScenarioConfig) -> str:
    mode = str(config.metadata.get("subaper_mode", config.metadata.get("truth_basis", "LM"))).strip().upper()
    if mode not in {"L", "Z", "LM"}:
        mode = "LM"
    return mode


def _center_crop(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = values.shape[:2]
    target_rows, target_cols = target_shape
    if target_rows > rows or target_cols > cols:
        raise ValueError("Target crop shape larger than source shape.")
    top = (rows - target_rows) // 2
    left = (cols - target_cols) // 2
    return values[top : top + target_rows, left : left + target_cols]


def _stitch_tiles_to_global(
    observations,
    tiles: np.ndarray,
    global_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch per-observation tiles into the global frame.

    The legacy MATLAB core returns one corrected SSPP per observation. To
    compare complete reconstructions, we place those tiles back into the global
    scan frame using the same rounded placement convention as the Python port.
    """

    sum_z = np.zeros(global_shape, dtype=float)
    count = np.zeros(global_shape, dtype=float)

    for obs_idx, obs in enumerate(observations):
        if obs_idx >= tiles.shape[2]:
            break
        placed_values, placed_mask = place_tile_in_global_frame(
            tiles[:, :, obs_idx],
            obs.valid_mask,
            global_shape,
            obs.center_xy,
        )
        sum_z += placed_values
        count += placed_mask.astype(float)

    stitched = np.full(global_shape, np.nan, dtype=float)
    valid_mask = count > 0
    stitched[valid_mask] = sum_z[valid_mask] / count[valid_mask]
    return stitched, valid_mask


def _build_matlab_input(observations, config: ScenarioConfig, mode: str) -> dict[str, object]:
    table_data = np.stack([np.asarray(obs.z, dtype=float) for obs in observations], axis=2)
    align_terms = tuple(int(v) for v in config.metadata.get("alignment_term", (0, 1, 2)))
    compensateurs = np.asarray([term + 1 for term in align_terms], dtype=float)[None, :]
    return {
        "TableData": table_data,
        "Type": np.array(mode),
        "Compensateurs": compensateurs,
    }


def _run_matlab_bridge(matlab_exe: str, input_path: Path, output_path: Path) -> None:
    bridge = "ar_stitching_run_subaper_bridge"
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
    parser = argparse.ArgumentParser(description="Compare Subaper MATLAB vs Python on a scenario.")
    parser.add_argument("--scenario", type=str, default="scenarios/s17_highres_circular.yaml")
    parser.add_argument("--matlab-exe", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, help="Override Subaper mode (L, Z, LM).")
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    config = ScenarioConfig.from_yaml(scenario_path)
    mode = (args.mode or _scenario_subaper_mode(config)).strip().upper()
    if mode not in {"L", "Z", "LM"}:
        raise ValueError(f"Unsupported mode '{mode}'.")

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
        metadata={**config.metadata, "subaper_mode": mode},
    )

    _, observations = simulate_identity_observations(python_config)
    candidate = load_candidate_module(REPO_ROOT / "src" / "stitching" / "editable" / "subaper" / "baseline.py")
    python_recon = candidate.reconstruct(observations, python_config)

    matlab_exe = args.matlab_exe or shutil.which("matlab") or str(Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"))
    if not Path(matlab_exe).exists():
        raise FileNotFoundError(f"Cannot find MATLAB executable: {matlab_exe}")

    with tempfile.TemporaryDirectory(prefix="subaper_matlab_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "subaper_input.mat"
        output_path = tmp_path / "subaper_output.mat"
        savemat(input_path, _build_matlab_input(observations, python_config, mode), do_compression=True)
        _run_matlab_bridge(matlab_exe, input_path, output_path)
        matlab_payload = loadmat(output_path, squeeze_me=True, struct_as_record=False)

        matlab_map = np.asarray(matlab_payload["map"], dtype=float)
        matlab_mismatch = np.asarray(matlab_payload["mismatch"], dtype=float)
        matlab_tiles = np.asarray(matlab_payload["sppAdjMatrix"], dtype=float)
        python_map = np.asarray(python_recon.metadata.get("raw_global_map", python_recon.z), dtype=float)
        matlab_global_map, matlab_global_mask = _stitch_tiles_to_global(observations, matlab_tiles, python_config.grid_shape)

    common_mask = np.isfinite(matlab_global_map) & np.isfinite(python_map) & matlab_global_mask
    if not np.any(common_mask):
        raise RuntimeError("No finite overlap between MATLAB and Python maps.")

    diff = python_map - matlab_global_map
    diff_detrended = _plane_detrend(diff, common_mask)
    cal_rms = float(np.sqrt(np.nanmean(np.square(diff[common_mask]))))
    cal_rms_detrended = float(np.sqrt(np.nanmean(np.square(diff_detrended[common_mask]))))
    mismatch_rms = float(np.sqrt(np.nanmean(np.square(matlab_mismatch[np.isfinite(matlab_mismatch)]))))

    print(f"Scenario: {scenario_path}")
    print(f"Mode: {mode}")
    print(f"Python RMS: {float(np.sqrt(np.nanmean(np.square(python_map[common_mask])))):.6f}")
    print(f"MATLAB complete RMS: {float(np.sqrt(np.nanmean(np.square(matlab_global_map[common_mask])))):.6f}")
    print(f"Python map shape: {python_map.shape}")
    print(f"MATLAB complete map shape: {matlab_global_map.shape}")
    print(f"Python vs MATLAB complete RMS: {cal_rms:.6f}")
    print(f"Python vs MATLAB complete RMS detrended: {cal_rms_detrended:.6f}")
    print(f"MATLAB mismatch RMS: {mismatch_rms:.6f}")

    artifact_dir = REPO_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin = np.nanpercentile(np.concatenate([python_map[common_mask], matlab_global_map[common_mask]]), 2)
    vmax = np.nanpercentile(np.concatenate([python_map[common_mask], matlab_global_map[common_mask]]), 98)
    im0 = axes[0].imshow(python_map, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Python Subaper\n(complete)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(matlab_global_map, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("MATLAB Subaper\n(complete)")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(diff_detrended, origin="lower", cmap="RdBu_r")
    axes[2].set_title("Python - MATLAB\n(complete, detrended)")
    plt.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    output_fig = artifact_dir / f"subaper_matlab_vs_python_{scenario_path.stem}.png"
    plt.savefig(output_fig, dpi=160)
    print(f"Comparison figure: {output_fig}")


if __name__ == "__main__":
    main()
