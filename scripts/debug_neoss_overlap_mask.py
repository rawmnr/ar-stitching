from __future__ import annotations

import argparse
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
from scipy import sparse
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stitching.contracts import ScenarioConfig
from stitching.editable._legacy_basis import overlap_support_mask
from stitching.trusted.simulator.identity import simulate_identity_observations


def _scenario_neoss_mode(config: ScenarioConfig) -> tuple[str, str]:
    tp_mode = str(config.metadata.get("neoss_tp_mode", config.metadata.get("truth_basis", "Z"))).strip().upper()
    if tp_mode not in {"L", "Z"}:
        tp_mode = "Z"
    cs_mode = str(config.metadata.get("neoss_cs_mode", "L" if tp_mode == "L" else "Z")).strip().upper()
    if cs_mode not in {"L", "Z"}:
        cs_mode = "Z"
    return tp_mode, cs_mode


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_matlab_input(observations, config: ScenarioConfig, tp_mode: str, cs_mode: str) -> dict[str, object]:
    table_data = np.stack([np.asarray(obs.z, dtype=float).ravel(order="F") for obs in observations], axis=0)
    coords = []
    for obs in observations:
        dx, dy = obs.translation_xy
        coords.append((float(dx), float(dy)))
    coord_arr = np.asarray(coords, dtype=float)

    global_shape = tuple(int(v) for v in config.grid_shape)
    tile_shape = tuple(int(v) for v in config.tile_shape or config.grid_shape)
    cfg = {
        "resolutionCS": float(global_shape[0]),
        "resolutionTP": float(tile_shape[0]),
        "RpupilleCS": float(config.metadata.get("neoss_rpupille_cs", global_shape[0] / 2.0)),
        "RpupilleTP": float(config.metadata.get("neoss_rpupille_tp", tile_shape[0] / 2.0)),
        "nb_cartes": float(len(observations)),
        "mode_TP": np.array(tp_mode),
        "mode_CS": np.array(cs_mode),
        "Coord1": coord_arr[:, 0],
        "Coord2": coord_arr[:, 1],
    }
    return {"TableData": table_data, "cfg": cfg}


def _run_matlab_bridge(matlab_exe: str, input_path: Path, output_path: Path) -> None:
    bridge = "ar_stitching_debug_overlap_mask"
    input_arg = str(input_path).replace("\\", "/")
    output_arg = str(output_path).replace("\\", "/")
    scripts_dir = str((REPO_ROOT / "scripts" / "matlab").resolve()).replace("\\", "/")
    cmd = [
        matlab_exe,
        "-batch",
        f"addpath(genpath('{scripts_dir}')); {bridge}('{input_arg}','{output_arg}')",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _mask_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    if sparse.issparse(a):
        a = a.toarray()
    if sparse.issparse(b):
        b = b.toarray()
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        return {"shape_a": a.shape, "shape_b": b.shape, "diff_count": None}
    diff = a ^ b
    return {
        "shape": a.shape,
        "diff_count": int(np.count_nonzero(diff)),
        "agree_ratio": float(np.mean(a == b)),
        "python_count": int(np.count_nonzero(b)),
        "matlab_count": int(np.count_nonzero(a)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NEOSS overlap masks between MATLAB and Python.")
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
        metadata={**config.metadata, "neoss_tp_mode": tp_mode, "neoss_cs_mode": cs_mode},
    )

    _, observations = simulate_identity_observations(python_config)
    python_overlap = overlap_support_mask(observations, tuple(int(v) for v in config.grid_shape))

    matlab_exe = args.matlab_exe or shutil.which("matlab") or str(Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"))
    if not Path(matlab_exe).exists():
        raise FileNotFoundError(f"Cannot find MATLAB executable: {matlab_exe}")

    with tempfile.TemporaryDirectory(prefix="neoss_overlap_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "neoss_overlap_input.mat"
        output_path = tmp_path / "neoss_overlap_output.mat"
        payload = _build_matlab_input(observations, python_config, tp_mode, cs_mode)
        savemat(input_path, payload, do_compression=True)
        _run_matlab_bridge(matlab_exe, input_path, output_path)
        matlab_payload = loadmat(output_path, squeeze_me=True, struct_as_record=False)

    matlab_overlap = np.asarray(matlab_payload["overlap_mask"], dtype=bool)
    matlab_count = matlab_payload.get("overlap_count", np.zeros_like(matlab_overlap, dtype=float))

    metrics = _mask_metrics(matlab_overlap, python_overlap)
    print("\n" + "=" * 80)
    print("NEOSS OVERLAP MASK COMPARISON")
    print("=" * 80)
    print(f"Scenario: {scenario_path}")
    print(f"MATLAB overlap pixels: {metrics['matlab_count']}")
    print(f"Python overlap pixels: {metrics['python_count']}")
    print(f"Different pixels: {metrics['diff_count']}")
    print(f"Agree ratio: {metrics['agree_ratio']:.6f}")

    output_dir = REPO_ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)

    diff = matlab_overlap ^ python_overlap
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(matlab_overlap.astype(float), origin="lower", cmap="gray")
    axes[0].set_title("MATLAB overlap")
    axes[1].imshow(python_overlap.astype(float), origin="lower", cmap="gray")
    axes[1].set_title("Python overlap")
    axes[2].imshow(diff.astype(float), origin="lower", cmap="hot")
    axes[2].set_title("Difference")
    axes[3].imshow(np.asarray(matlab_count, dtype=float), origin="lower", cmap="viridis")
    axes[3].set_title("MATLAB overlap count")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig_path = output_dir / f"neoss_overlap_mask_{scenario_path.stem}.png"
    plt.savefig(fig_path, dpi=160)
    print(f"Figure saved: {fig_path}")

    results = {
        "scenario": str(scenario_path),
        "tp_mode": tp_mode,
        "cs_mode": cs_mode,
        "metrics": metrics,
    }
    json_path = output_dir / f"neoss_overlap_mask_{scenario_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"JSON saved: {json_path}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
