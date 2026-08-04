#!/usr/bin/env julia
include(joinpath(@__DIR__, "..", "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG

# Small metallic equilibrium control.  This verifies complete-basis closure and
# density-matrix normalization; it is intentionally too small to demand a
# converged Kondo scale.
model = ModelParams(U=2.0, eps_d=-1.0, beta0=0.0, delta_coh=0.0,
                    frame=:relative, bath_exponent=0.0, gamma_edge=0.12)
nrg = NRGParams(Lambda=3.0, z_shift=0.5, iterations=6, nkeep=96,
                min_keep_per_charge=1, star_intervals=40, save_levels=12)
leh = LehmannParams(enabled=false)
thermo = ThermoParams(enabled=true, temperature_min=1e-4,
                      temperature_max=1.0, temperature_points=81)
cfg = RunConfig(model=model, nrg=nrg, lehmann=leh, thermo=thermo,
                output_dir=mktempdir())
result = run_nrg(cfg)
fdm = complete_basis_thermodynamics(result)
@assert fdm.complete_basis_count_pass
@assert fdm.spectrum_real
@assert fdm.density_matrix_normalization_error < 1e-10
@assert all(isfinite, fdm.entropy_full)
println("FDM thermodynamics smoke PASS")
println("complete basis = $(fdm.complete_basis_count)")
println("max centered Im(E) = $(fdm.max_centered_imaginary_energy)")
