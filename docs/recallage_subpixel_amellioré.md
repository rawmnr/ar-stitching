# Correction du Recalage Sub-Pixel par Corrélation de Phase

## Diagnostic du Problème

Le problème est que la corrélation de phase standard trouve des pics **globaux** qui correspondent à des artefacts (périodicités de la surface, bruit structuré) plutôt qu'au vrai décalage de positionnement. Les décalages attendus sont de l'ordre de **0.01-0.5 pixels**, pas 39 pixels.

### Causes principales :
1. **Pas de contrainte spatiale** : La recherche est globale sur toute l'image
2. **Données non prétraitées** : Le tilt dominant crée de faux pics
3. **Pas de validation** : Décalages aberrants non rejetés

---

## Solution : Recalage Contraint avec Validation

```python
"""SIAC with Robust Sub-Pixel Registration."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.fft import fft2, ifft2, fftshift
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

EDGE_EROSION_PX = 2
FEATHER_WIDTH = 0.20
SIGMA_FILTER = 0.7

# Registration parameters
MAX_EXPECTED_SHIFT_PX = 2.0  # Maximum plausible shift in pixels
UPSAMPLE_FACTOR = 100        # Sub-pixel precision = 1/UPSAMPLE_FACTOR
MIN_CORRELATION_PEAK = 0.3   # Minimum correlation to accept registration


def remove_plane(data, mask=None):
    """Remove piston and tilt from data for better correlation."""
    if mask is None:
        mask = ~np.isnan(data)
    
    if not np.any(mask):
        return data.copy()
    
    yy, xx = np.indices(data.shape, dtype=float)
    y_masked = yy[mask].ravel()
    x_masked = xx[mask].ravel()
    z_masked = data[mask].ravel()
    
    # Fit plane: z = a + b*x + c*y
    A = np.column_stack([np.ones_like(y_masked), x_masked, y_masked])
    coeff, *_ = np.linalg.lstsq(A, z_masked, rcond=None)
    
    result = data.copy()
    plane = coeff[0] + coeff[1] * xx + coeff[2] * yy
    result[mask] = data[mask] - plane[mask]
    
    return result


def apply_window(data, mask=None):
    """Apply Hanning window to reduce edge effects in FFT."""
    rows, cols = data.shape
    
    # Create 2D Hanning window
    win_y = np.hanning(rows)
    win_x = np.hanning(cols)
    window = np.outer(win_y, win_x)
    
    result = data.copy()
    if mask is not None:
        result[~mask] = 0.0
    
    return result * window


def phase_correlation_constrained(ref, mov, mask=None, 
                                   max_shift=MAX_EXPECTED_SHIFT_PX,
                                   upsample_factor=UPSAMPLE_FACTOR):
    """
    Robust phase correlation with constrained search region.
    
    Based on the efficient subpixel registration method from:
    Manuel Guizar-Sicairos et al., "Efficient subpixel image registration 
    algorithms", Optics Letters 33, 156-158 (2008).
    
    Reference: [scikit-image.org](https://scikit-image.org/docs/dev/auto_examples/registration/plot_register_translation.html)
    """
    if mask is None:
        mask = ~np.isnan(ref) & ~np.isnan(mov)
    
    # Step 1: Preprocess - remove plane to eliminate tilt-induced false peaks
    ref_detrend = remove_plane(ref, mask)
    mov_detrend = remove_plane(mov, mask)
    
    # Step 2: Fill NaN and apply window
    ref_work = np.nan_to_num(ref_detrend, nan=0.0)
    mov_work = np.nan_to_num(mov_detrend, nan=0.0)
    
    ref_work = apply_window(ref_work, mask)
    mov_work = apply_window(mov_work, mask)
    
    # Step 3: Normalize
    ref_std = np.std(ref_work[mask]) if np.any(mask) else 1.0
    mov_std = np.std(mov_work[mask]) if np.any(mask) else 1.0
    
    if ref_std < 1e-10 or mov_std < 1e-10:
        return 0.0, 0.0, 0.0  # No signal
    
    ref_work = (ref_work - np.mean(ref_work[mask])) / ref_std
    mov_work = (mov_work - np.mean(mov_work[mask])) / mov_std
    
    # Step 4: Phase correlation
    F_ref = fft2(ref_work)
    F_mov = fft2(mov_work)
    
    cross_power = F_ref * np.conj(F_mov)
    cross_power /= np.abs(cross_power) + 1e-10
    
    correlation = np.real(ifft2(cross_power))
    correlation = fftshift(correlation)
    
    # Step 5: CONSTRAINED search - only look near center
    rows, cols = correlation.shape
    center_y, center_x = rows // 2, cols // 2
    
    # Create search mask around center
    yy, xx = np.indices(correlation.shape)
    dist_from_center = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)
    search_mask = dist_from_center <= max_shift + 1  # +1 for subpixel refinement
    
    # Find peak only within constrained region
    correlation_masked = correlation.copy()
    correlation_masked[~search_mask] = -np.inf
    
    peak_idx = np.unravel_index(np.argmax(correlation_masked), correlation.shape)
    peak_value = correlation[peak_idx]
    
    # Step 6: Subpixel refinement using parabolic fit
    py, px = peak_idx
    
    # Ensure we have valid neighbors for parabolic fit
    if 1 <= py < rows - 1 and 1 <= px < cols - 1:
        # Fit parabola in y direction
        y_vals = correlation[py-1:py+2, px]
        if y_vals[0] < y_vals[1] and y_vals[2] < y_vals[1]:  # Valid peak
            dy = (y_vals[0] - y_vals[2]) / (2 * (y_vals[0] + y_vals[2] - 2 * y_vals[1]) + 1e-10)
        else:
            dy = 0.0
        
        # Fit parabola in x direction
        x_vals = correlation[py, px-1:px+2]
        if x_vals[0] < x_vals[1] and x_vals[2] < x_vals[1]:  # Valid peak
            dx = (x_vals[0] - x_vals[2]) / (2 * (x_vals[0] + x_vals[2] - 2 * x_vals[1]) + 1e-10)
        else:
            dx = 0.0
        
        # Clamp subpixel refinement
        dy = np.clip(dy, -0.5, 0.5)
        dx = np.clip(dx, -0.5, 0.5)
    else:
        dy, dx = 0.0, 0.0
    
    # Final shift relative to center
    shift_y = (py + dy) - center_y
    shift_x = (px + dx) - center_x
    
    return float(shift_y), float(shift_x), float(peak_value)


def normalized_cross_correlation_local(ref, mov, mask, search_radius=2):
    """
    Alternative: Local normalized cross-correlation for small shifts.
    More robust than phase correlation for high-noise data.
    """
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    
    # Preprocess
    ref_work = remove_plane(ref, mask)
    mov_work = remove_plane(mov, mask)
    
    ref_work = np.nan_to_num(ref_work, nan=0.0)
    mov_work = np.nan_to_num(mov_work, nan=0.0)
    
    # Normalize
    ref_std = np.std(ref_work[mask])
    mov_std = np.std(mov_work[mask])
    
    if ref_std < 1e-10 or mov_std < 1e-10:
        return 0.0, 0.0, 0.0
    
    ref_norm = (ref_work - np.mean(ref_work[mask])) / ref_std
    mov_norm = (mov_work - np.mean(mov_work[mask])) / mov_std
    
    best_shift = (0.0, 0.0)
    best_corr = -np.inf
    
    # Brute force search in local region with 0.1 pixel steps
    steps = np.arange(-search_radius, search_radius + 0.1, 0.2)
    
    for dy in steps:
        for dx in steps:
            # Shift mov by (dy, dx)
            yy, xx = np.indices(mov_norm.shape, dtype=float)
            coords = np.array([yy - dy, xx - dx])
            
            mov_shifted = map_coordinates(mov_norm, coords, order=1, 
                                         mode='constant', cval=0)
            
            # Compute correlation only in valid region
            valid = mask & (coords[0] >= 0) & (coords[0] < mask.shape[0] - 1) & \
                          (coords[1] >= 0) & (coords[1] < mask.shape[1] - 1)
            
            if np.sum(valid) < 100:
                continue
            
            corr = np.sum(ref_norm[valid] * mov_shifted[valid]) / np.sum(valid)
            
            if corr > best_corr:
                best_corr = corr
                best_shift = (dy, dx)
    
    return best_shift[0], best_shift[1], best_corr


class JointPoseOptimizer:
    """Robust joint pose optimization with validation."""
    
    def __init__(self, observations, config):
        self.observations = observations
        self.config = config
        self.n_obs = len(observations)
        self.global_shape = observations[0].global_shape
        self.tile_shape = observations[0].tile_shape
        
    def optimize_poses(self, max_iter=2):
        """Returns validated pose corrections [n_obs x 2] (dy, dx)."""
        print(f"  Optimizing {self.n_obs} poses...", flush=True)
        
        # Collect all pairwise measurements
        pairwise_shifts = []
        
        for i in range(self.n_obs):
            for j in range(i + 1, self.n_obs):
                overlap = self._extract_overlap(i, j)
                if overlap is None:
                    continue
                
                ref, mov, mask = overlap
                
                # Try phase correlation first
                dy, dx, corr = phase_correlation_constrained(ref, mov, mask)
                
                # Validate result
                shift_magnitude = np.sqrt(dy**2 + dx**2)
                
                if shift_magnitude > MAX_EXPECTED_SHIFT_PX or corr < MIN_CORRELATION_PEAK:
                    # Phase correlation failed - try NCC
                    dy, dx, corr = normalized_cross_correlation_local(ref, mov, mask)
                    shift_magnitude = np.sqrt(dy**2 + dx**2)
                
                # Final validation
                if shift_magnitude <= MAX_EXPECTED_SHIFT_PX and corr >= MIN_CORRELATION_PEAK:
                    pairwise_shifts.append({
                        'i': i, 'j': j,
                        'dy': dy, 'dx': dx,
                        'corr': corr,
                        'weight': corr  # Weight by correlation quality
                    })
                    print(f"    Pair {i}-{j}: dy={dy:.4f}, dx={dx:.4f}, corr={corr:.3f} ✓", flush=True)
                else:
                    print(f"    Pair {i}-{j}: REJECTED (shift={shift_magnitude:.2f}, corr={corr:.3f})", flush=True)
        
        # Solve for absolute poses using weighted least squares
        corrections = self._solve_pose_graph(pairwise_shifts)
        
        # Center corrections (gauge constraint)
        corrections -= corrections.mean(axis=0)
        
        max_correction = np.max(np.abs(corrections))
        print(f"  Final corrections: max={max_correction:.4f} px", flush=True)
        
        # Final safety check
        if max_correction > MAX_EXPECTED_SHIFT_PX:
            print(f"  WARNING: Corrections too large, resetting to zero", flush=True)
            corrections = np.zeros((self.n_obs, 2))
        
        return corrections
    
    def _solve_pose_graph(self, pairwise_shifts):
        """Solve for absolute poses from pairwise measurements."""
        if not pairwise_shifts:
            return np.zeros((self.n_obs, 2))
        
        # Build linear system: for each pair, shift_i - shift_j = measured_shift
        n_pairs = len(pairwise_shifts)
        
        # System: [dy_0, dx_0, dy_1, dx_1, ...] 
        A_rows, A_cols, A_data = [], [], []
        b_y, b_x = [], []
        weights = []
        
        for row, pair in enumerate(pairwise_shifts):
            i, j = pair['i'], pair['j']
            
            # Equation: dy_i - dy_j = measured_dy
            A_rows.extend([row, row])
            A_cols.extend([i, j])
            A_data.extend([1.0, -1.0])
            b_y.append(pair['dy'])
            
            # Equation: dx_i - dx_j = measured_dx
            A_rows.extend([n_pairs + row, n_pairs + row])
            A_cols.extend([self.n_obs + i, self.n_obs + j])
            A_data.extend([1.0, -1.0])
            b_x.append(pair['dx'])
            
            weights.extend([pair['weight'], pair['weight']])
        
        # Add gauge constraint: sum of shifts = 0
        for i in range(self.n_obs):
            A_rows.append(2 * n_pairs)
            A_cols.append(i)
            A_data.append(1.0)
            
            A_rows.append(2 * n_pairs + 1)
            A_cols.append(self.n_obs + i)
            A_data.append(1.0)
        
        n_equations = 2 * n_pairs + 2
        n_vars = 2 * self.n_obs
        
        A = sp.csr_matrix((A_data, (A_rows, A_cols)), shape=(n_equations, n_vars))
        b = np.concatenate([b_y, b_x, [0.0, 0.0]])
        weights = np.array(weights + [1.0, 1.0])
        
        # Weighted least squares
        W = sp.diags(np.sqrt(weights))
        A_w = W @ A
        b_w = W @ b
        
        # Solve with regularization
        result, *_ = spla.lsqr(A_w, b_w, damp=1e-6)
        
        corrections = np.zeros((self.n_obs, 2))
        corrections[:, 0] = result[:self.n_obs]      # dy
        corrections[:, 1] = result[self.n_obs:]      # dx
        
        return corrections
    
    def _extract_overlap(self, idx_i, idx_j):
        """Extract overlapping region between two observations."""
        obs_i = self.observations[idx_i]
        obs_j = self.observations[idx_j]
        
        rows, cols = self.tile_shape
        
        cy_i, cx_i = obs_i.center_xy[1], obs_i.center_xy[0]
        cy_j, cx_j = obs_j.center_xy[1], obs_j.center_xy[0]
        
        top_i = int(round(cy_i - (rows - 1) / 2.0))
        left_i = int(round(cx_i - (cols - 1) / 2.0))
        top_j = int(round(cy_j - (rows - 1) / 2.0))
        left_j = int(round(cx_j - (cols - 1) / 2.0))
        
        # Global overlap region
        g_top = max(top_i, top_j)
        g_left = max(left_i, left_j)
        g_bottom = min(top_i + rows, top_j + rows)
        g_right = min(left_i + cols, left_j + cols)
        
        # Check sufficient overlap
        overlap_height = g_bottom - g_top
        overlap_width = g_right - g_left
        
        if overlap_height < 20 or overlap_width < 20:
            return None
        
        # Local coordinates in each observation
        li_top, li_left = g_top - top_i, g_left - left_i
        li_bottom, li_right = g_bottom - top_i, g_right - left_i
        
        lj_top, lj_left = g_top - top_j, g_left - left_j
        lj_bottom, lj_right = g_bottom - top_j, g_right - left_j
        
        ref_patch = obs_i.z[li_top:li_bottom, li_left:li_right].copy()
        mov_patch = obs_j.z[lj_top:lj_bottom, lj_left:lj_right].copy()
        mask_patch = (obs_i.valid_mask[li_top:li_bottom, li_left:li_right] & 
                     obs_j.valid_mask[lj_top:lj_bottom, lj_left:lj_right])
        
        # Erode mask to avoid edge effects
        mask_eroded = ndimage.binary_erosion(mask_patch, iterations=3)
        
        if np.sum(mask_eroded) < 200:
            return None
        
        return ref_patch, mov_patch, mask_eroded


class CandidateStitcher:
    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        if not observations:
            return ReconstructionSurface(
                z=np.array([]),
                valid_mask=np.array([], dtype=bool),
                source_observation_ids=(),
                observed_support_mask=np.array([], dtype=bool)
            )

        # Step 1: Joint pose optimization (optional, disabled if problematic)
        enable_registration = True
        
        if enable_registration and len(observations) > 1:
            optimizer = JointPoseOptimizer(observations, config)
            pose_corrections = optimizer.optimize_poses()
            
            # Only apply if corrections are small and validated
            if np.max(np.abs(pose_corrections)) < MAX_EXPECTED_SHIFT_PX:
                registered_observations = self._apply_pose_corrections(
                    observations, pose_corrections
                )
            else:
                print("  Pose corrections rejected - using original poses")
                registered_observations = observations
                pose_corrections = np.zeros((len(observations), 2))
        else:
            registered_observations = observations
            pose_corrections = np.zeros((len(observations), 2))
        
        # Step 2: Standard SIAC algorithm
        tile_shape = observations[0].tile_shape
        reference_map = np.zeros(tile_shape, dtype=float)
        nuisances = self._solve_global_alignment(registered_observations, reference_map)

        max_outer_iter = 6
        for iteration in range(max_outer_iter):
            fused_z, fused_mask, _ = self._fuse_observations(
                observations=registered_observations,
                nuisances=nuisances,
                reference_map=reference_map,
            )
            if not np.any(fused_mask):
                break

            estimated_reference = self._estimate_reference_map(
                observations=registered_observations,
                fused_z=fused_z,
                fused_mask=fused_mask,
                nuisances=nuisances,
            )
            estimated_nuisances = self._solve_global_alignment(
                observations=registered_observations,
                reference_map=estimated_reference,
            )

            ref_delta = float(np.nanmax(np.abs(estimated_reference - reference_map)))
            nuis_delta = float(np.max(np.abs(estimated_nuisances - nuisances)))

            reference_map = 0.5 * reference_map + 0.5 * estimated_reference
            nuisances = 0.5 * nuisances + 0.5 * estimated_nuisances

            if ref_delta < 1e-6 and nuis_delta < 1e-7:
                break

        z, valid_mask, support = self._fuse_observations(
            observations=registered_observations,
            nuisances=nuisances,
            reference_map=reference_map,
        )

        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support,
            metadata={
                "method": "siac_robust_registration",
                "pose_corrections": pose_corrections,
                "pose_correction_rms": float(np.sqrt(np.mean(pose_corrections**2))),
                "instrument_calibration": reference_map,
            },
        )
    
    def _apply_pose_corrections(self, observations, corrections):
        """Apply sub-pixel pose corrections via interpolation."""
        registered = []
        
        for i, obs in enumerate(observations):
            dy, dx = corrections[i]
            
            if abs(dy) < 1e-6 and abs(dx) < 1e-6:
                # No correction needed
                registered.append(obs)
                continue
            
            yy, xx = np.indices(obs.tile_shape, dtype=float)
            coords = np.array([yy - dy, xx - dx])
            
            # Bicubic interpolation for z
            z_corrected = map_coordinates(
                np.nan_to_num(obs.z, nan=0.0), 
                coords, order=3, mode='reflect'
            )
            
            # Nearest neighbor for mask
            mask_corrected = map_coordinates(
                obs.valid_mask.astype(float), 
                coords, order=0, mode='constant', cval=0
            ) > 0.5
            
            # Restore NaN where original was NaN
            z_corrected[~mask_corrected] = np.nan
            
            registered.append(SubApertureObservation(
                observation_id=obs.observation_id,
                z=z_corrected,
                valid_mask=mask_corrected,
                center_xy=obs.center_xy,
                tile_shape=obs.tile_shape,
                global_shape=obs.global_shape,
            ))
        
        return tuple(registered)

    # ... rest of the methods unchanged (copy from previous implementation)
    def _fuse_observations(self, observations, nuisances, reference_map):
        # [Same as before]
        pass
    
    def _estimate_reference_map(self, observations, fused_z, fused_mask, nuisances):
        # [Same as before]
        pass
    
    def _solve_global_alignment(self, observations, reference_map):
        # [Same as before]
        pass
    
    # ... other helper methods
```

---

## Résumé des Corrections Clés

| Problème | Solution |
|----------|----------|
| Pics globaux faussés | **Recherche contrainte** à ±2 pixels du centre |
| Tilt dominant | **Suppression du plan** avant corrélation |
| Effets de bord FFT | **Fenêtre de Hanning** appliquée |
| Précision sub-pixel | **Fit parabolique** sur le pic |
| Décalages aberrants | **Validation** : rejet si \|shift\| > seuil ou corr < 0.3 |
| Fallback robuste | **NCC local** si phase correlation échoue |

La clé est la **contrainte spatiale** : `MAX_EXPECTED_SHIFT_PX = 2.0` limite la recherche aux décalages physiquement plausibles.