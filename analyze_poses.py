"""Analyze ground truth pose errors vs estimated corrections."""
import numpy as np
from pathlib import Path
from stitching.contracts import ScenarioConfig
from stitching.harness.evaluator import load_candidate_module, evaluate_candidate_on_scenario

scenario_path = Path('scenarios/s17_highres_circular.yaml')
config = ScenarioConfig.from_yaml(scenario_path)

def compute_ground_truth_poses(config: ScenarioConfig) -> np.ndarray:
    """Compute the realized pose errors for each observation."""
    n_obs = 10
    poses = np.zeros((n_obs, 2))
    
    metadata = config.metadata
    seed = config.seed
    
    for idx in range(n_obs):
        cx, cy = 0.0, 0.0
        
        if "realized_pose_bias_xy" in metadata:
            bias = metadata["realized_pose_bias_xy"]
            cx += float(bias[0])
            cy += float(bias[1])
        
        drift_std = float(metadata.get("realized_pose_drift_std", 0.0))
        if drift_std > 0.0:
            rng_drift = np.random.default_rng(seed + 20_000)
            steps = rng_drift.normal(0.0, drift_std, size=(idx + 1, 2))
            drift_xy = np.sum(steps, axis=0)
            cx += drift_xy[0]
            cy += drift_xy[1]
        
        noise_std = float(metadata.get("realized_pose_error_std", 0.0))
        if noise_std > 0.0:
            rng_noise = np.random.default_rng(seed + idx + 10_000)
            cx += rng_noise.normal(0.0, noise_std)
            cy += rng_noise.normal(0.0, noise_std)
        
        poses[idx] = [cy, cx]
    
    poses -= poses.mean(axis=0)
    
    return poses

print("=" * 60)
print("SCENARIO S17 POSE ERROR ANALYSIS")
print("=" * 60)

print("\nScenario parameters:")
print(f"  realized_pose_bias_xy: {config.metadata.get('realized_pose_bias_xy')}")
print(f"  realized_pose_drift_std: {config.metadata.get('realized_pose_drift_std')}")
print(f"  realized_pose_error_std: {config.metadata.get('realized_pose_error_std')}")

gt_poses = compute_ground_truth_poses(config)

print("\n" + "-" * 60)
print("Ground Truth Pose Errors (pixels):")
print("-" * 60)
print(f"{'Obs':>4} {'dy':>10} {'dx':>10}")
for i, (dy, dx) in enumerate(gt_poses):
    print(f"{i:>4} {dy:>10.4f} {dx:>10.4f}")

print(f"\nGround Truth Statistics:")
print(f"  Max: {np.max(np.abs(gt_poses)):.4f} px")
print(f"  RMS: {np.sqrt(np.mean(gt_poses**2)):.4f} px")

print("\n" + "=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

results = {}
for name, path in [
    ('GLS', 'src/stitching/editable/gls/baseline.py'), 
    ('SCS', 'src/stitching/editable/scs/baseline.py'), 
    ('SIAC', 'src/stitching/editable/siac/baseline.py'), 
    ('SIAC+Reg', 'src/stitching/editable/siac_reg/baseline.py')
]:
    candidate = load_candidate_module(Path(path))
    report = evaluate_candidate_on_scenario(candidate, scenario_path)
    rms = report.signal_metrics.get("rms_detrended", float('nan'))
    results[name] = rms
    print(f"{name:>12}: RMS = {rms:.6f}")

print("\n" + "=" * 60)
print("POSE CORRECTION ANALYSIS")
print("=" * 60)

candidate = load_candidate_module(Path('src/stitching/editable/siac_reg/baseline.py'))
report = evaluate_candidate_on_scenario(candidate, scenario_path)

if hasattr(report, 'reconstruction') and report.reconstruction and 'pose_corrections' in report.reconstruction.metadata:
    est_poses = report.reconstruction.metadata['pose_corrections']
    
    print("\nEstimated Pose Corrections (pixels):")
    print(f"{'Obs':>4} {'GT dy':>10} {'GT dx':>10} {'Est dy':>10} {'Est dx':>10} {'Err':>10}")
    
    errors = []
    for i in range(len(gt_poses)):
        gt_dy, gt_dx = gt_poses[i]
        est_dy, est_dx = est_poses[i]
        err = np.sqrt((gt_dy - est_dy)**2 + (gt_dx - est_dx)**2)
        errors.append(err)
        print(f"{i:>4} {gt_dy:>10.4f} {gt_dx:>10.4f} {est_dy:>10.4f} {est_dx:>10.4f} {err:>10.4f}")
    
    print(f"\nPose Estimation Error:")
    print(f"  Mean error: {np.mean(errors):.4f} px")
    print(f"  Max error:  {np.max(errors):.4f} px")
    print(f"  RMS error:  {np.sqrt(np.mean(np.array(errors)**2)):.4f} px")
    
    pose_corr = np.sqrt(np.mean(est_poses**2))
    pose_gt = np.sqrt(np.mean(gt_poses**2))
    print(f"\n  Estimated pose RMS: {pose_corr:.4f} px")
    print(f"  Ground truth RMS:   {pose_gt:.4f} px")
    print(f"  Recovery ratio:     {pose_corr/pose_gt*100:.1f}%")
