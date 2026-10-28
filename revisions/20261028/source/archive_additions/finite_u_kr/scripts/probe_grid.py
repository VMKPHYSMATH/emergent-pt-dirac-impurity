from pathlib import Path
"""Re-run the archived saddle solver with the same equations but a frequency grid that also resolves
the logarithmic region between the pole window and the band (log-spaced supplement)."""
import sys, math, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import saddle_gate_legacy as G
orig = G.adaptive_omega_grid
def patched(r, delta_c, beta, refine, p=G.P):
    omega, grid = orig(r, delta_c, beta, refine, p)
    lg = np.logspace(-10, math.log10(4.0), 6000)
    omega = np.unique(np.concatenate([omega, lg, -lg]))
    omega = omega[(omega >= p.band_lo) & (omega <= p.band_hi)]
    grid = dict(grid); grid["omega_points"] = float(len(omega))
    return omega, grid
# residuals of the archived solution on the original and on the supplemented grid
r_a, dc_a = 0.009807411733055882, 1.0000001191596621
v0 = G.evaluate_saddle(r_a, dc_a, 0.5)
G.adaptive_omega_grid = patched
v1 = G.evaluate_saddle(r_a, dc_a, 0.5)
print("archived (r, delta_c) at beta0=0.5: constraint/stationarity residuals  original grid: %.2e / %.2e   supplemented grid: %.2e / %.2e"
      % (v0["constraint_residual"], v0["stationarity_divided_residual"], v1["constraint_residual"], v1["stationarity_divided_residual"]))
for beta in (0.35, 0.5, 0.55):
    out = G.solve_seed(beta, G.SEEDS[0])
    print("solver with supplemented grid: beta0=%.2f status=%s r=%.6f delta_c=%.9f" % (beta, out.get("status"), out.get("r", float("nan")), out.get("delta_c", float("nan"))))
