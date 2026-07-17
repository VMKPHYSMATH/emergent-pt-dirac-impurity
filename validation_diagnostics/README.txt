Numerical validation diagnostics
====================================

These files are direct outputs from the released prb_all_in_one_final.py code using the final manuscript parameters.

Condition number:
- Fig1_kappa.dat: beta0, kappa_imp. The origin is 1.000000; the discrete-grid maximum is 23.13469982 at beta0=0.5025.
- The independently tracked projected double-root value quoted in the manuscript is beta0_proj_EP=0.4975. The 0.5025 maximum is the adjacent point on a grid with spacing 0.0075.

FDR:
- Fig4_FDR_residual_impurity.dat and Fig4_FDR_residual_bath.dat: absolute residuals for beta0=0.45 and eta_FDR=1e-3.
- The maximum residual is 5.61e-14.
- Fig. 4(d) uses the separately stated spectral resolution eta_spec=0.015 and is not used to compute the FDR residual.

