# Stitching Implementation Guide: Advanced Robust Optimization

## 1. Mathematical Model with Reference Bias

Each observation $S_i(\mathbf{r})$ includes a stationary instrument bias $B(x, y)$:

$$S_i(\mathbf{r}) = W(\mathbf{r}) + \text{Model}_i(x, y) + B(x, y) + \epsilon$$

Where:
- $W(\mathbf{r})$ is the object truth.
- $\text{Model}_i(x, y) = p_i + tx_i x + ty_i y + f_i (x^2+y^2)$ (Alignment).
- $B(x, y)$ is the **Reference Bias** (Stationary in detector frame).

## 2. Simultaneous Calibration and Stitching (SCS)

To solve for both alignment and bias, use an iterative approach:

### Step A: Global Alignment (Fixed Bias)
Solve for all $\mathbf{x}_i$ by minimizing:
$$\sum_{i,j} \iint_{O_{i,j}} [ (S_i - B - \text{Model}_i) - (S_j - B - \text{Model}_j) ]^2$$
*This is the current GLS solver, just subtract the current estimate of $B$ from $S_i$ first.*

### Step B: Bias Estimation (Fixed Alignment)
The bias $B(x, y)$ is the average residual across all observations in the detector frame:
1. For each observation $i$, compute the local residual: 
   $R_i(x, y) = S_i(x, y) - \text{Model}_i(\hat{\mathbf{x}}_i) - W_{\text{recon}}(\mathbf{r})$
2. Average all $R_i(x, y)$ to get a raw bias map $B_{\text{raw}}(x, y)$.
3. **Regularize**: Fit Zernike polynomials (e.g., up to Z15) to $B_{\text{raw}}$ to get a smooth $B(x, y)$.

## 3. Robustness (Handling Outliers)

To ignore dead pixels and spikes, replace standard Least Squares with **Iteratively Reweighted Least Squares (IRLS)**:
1. Compute residuals $r = Ax - b$.
2. Compute weights $w = \text{Huber}(r)$:
   - $w = 1.0$ if $|r| < \sigma$
   - $w = \sigma / |r|$ if $|r| \ge \sigma$
3. Solve $A^T W A x = A^T W b$.

## 4. Implementation Curriculum (The Roadmap)

**Do not try to implement everything at once.**

- **Milestone 1**: Robust GLS. Implement IRLS with Huber weights in `_solve_global_alignment` to handle outliers.
- **Milestone 2**: Scalar Self-Calib. Estimate a single constant value for $B$ (the average detector offset).
- **Milestone 3**: Zernike Self-Calib. Implement the full SCS loop to estimate the field $B(x,y)$.
