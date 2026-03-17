# Stitching Implementation Guide: Least-Squares Alignment

## Mathematical Model

Each observation $S_i(\mathbf{r})$ is modeled as the sum of the true surface $W(\mathbf{r})$ and a set of sub-aperture specific alignment nuisances (piston, tip, tilt, focus):

$$S_i(\mathbf{r}) = W(\mathbf{r}) + p_i + tx_i \cdot x + ty_i \cdot y + f_i \cdot (x^2 + y^2) + \epsilon$$

where $(x, y)$ are normalized detector coordinates in $[-1, 1]$.

## Objective Function

To stitch the sub-apertures, we minimize the squared difference in all overlapping regions $O_{i,j}$:

$$\chi^2 = \sum_{i,j} \iint_{O_{i,j}} w(\mathbf{r}) [ (S_i(\mathbf{r}) - \text{model}_i(\mathbf{r})) - (S_j(\mathbf{r}) - \text{model}_j(\mathbf{r})) ]^2 d\mathbf{r}$$

where $\text{model}_i(\mathbf{r}) = p_i + tx_i x + ty_i y + f_i (x^2+y^2)$.

## Implementation Strategy (Global Solver)

1. **Overlap Detection**: Identify pairs of observations $(i, j)$ that overlap.
2. **Design Matrix $A$**: For each pixel $k$ in an overlap $O_{i,j}$, create a row in the sparse matrix $A$:
   - The row represents the constraint: $(\text{model}_i - \text{model}_j) = (S_i - S_j)$
   - Columns correspond to the unknown parameters for all sub-apertures: $\mathbf{x} = [p_0, tx_0, ty_0, f_0, p_1, tx_1, \dots]^T$.
   - For a pixel at $(x, y)$ in $O_{i,j}$, the row has:
     - $+1, +x, +y, +(x^2+y^2)$ at indices for sub-aperture $i$.
     - $-1, -x, -y, -(x^2+y^2)$ at indices for sub-aperture $j$.
3. **Data Vector $b$**: The value for this row is $S_i(x,y) - S_j(x,y)$.
4. **Weighted Least Squares**: Solve $(A^T W A) \mathbf{x} = A^T W b$.
   - Use `scipy.sparse.linalg.lsqr` for efficiency.
   - **Crucial**: Fix one sub-aperture (e.g., $p_0=0, tx_0=0, ty_0=0, f_0=0$) to remove the global rank deficiency (singular matrix).
5. **Regularization (Damping)**: To prevent over-fitting (e.g., estimating focus where none exists), use the `damp` parameter in `lsqr`. 
   - A value of `1e-4` to `1e-2` forces parameters toward zero if they don't improve the residuals significantly.
   - Equation solved: $(A^T A + \lambda I) \mathbf{x} = A^T b$.
6. **Reconstruction**: Apply the estimated parameters to correct each observation:
   $S_i^{\text{corrected}} = S_i - \text{model}_i(\hat{\mathbf{x}})$
   Then perform a simple mean or weighted average of the corrected observations.

## Recommended Libraries
- `numpy`
- `scipy.sparse` (for the matrix $A$)
- `scipy.sparse.linalg` (for the `lsqr` solver)
