using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG

model = ModelParams(
    U=2.0,
    eps_d=-1.0,
    delta0=0.075,
    c_delta=0.05,
    g_gamma=0.20,
    beta0=0.49,
    delta_coh=1e-3,
    frame=:relative,
    bandwidth=pi/4,
    bath_exponent=1.0,
    gamma_edge=0.12,
)
nrg = NRGParams(
    Lambda=3.0,
    z_shift=0.5,
    iterations=4,
    nkeep=32,
    min_keep_per_charge=2,
    star_intervals=30,
    save_levels=8,
)
lehmann = LehmannParams(enabled=true, omega_min=-0.5, omega_max=0.5,
                          omega_points=101, eta=0.02)
config = RunConfig(model=model, nrg=nrg, lehmann=lehmann,
                   output_dir=mktempdir())
result = run_nrg(config)
@assert length(result.records) == nrg.iterations + 1
@assert all(isfinite(real(e)) && isfinite(imag(e)) for e in result.state.energies)
@assert result.state.max_residual < 1e-7
leh = kept_space_lehmann(result)
@assert size(leh.GR) == (lehmann.omega_points, 2, 2)
@assert all(isfinite, leh.spectral_trace)

# Analytic SOC-overlap checks: det(S_lambda)=F(lambda) and
# V_lambda V_lambda^dagger / V^2 = S_lambda.
soc_model = ModelParams(
    U=2.0, eps_d=-1.0, beta0=0.49, delta_coh=1e-3,
    frame=:relative, bandwidth=pi/4, bath_exponent=1.0, gamma_edge=0.12,
    soc_mode=:overlap, soc_lambda=0.5, soc_kmax=pi/4,
)
Ssoc = soc_overlap_matrix(soc_model)
Fsoc = soc_form_factor(soc_model)
Vsoc = impurity_hybridization_matrix(soc_model)
V0 = physical_hybridization(soc_model)
@assert abs(real(det(Ssoc)) - Fsoc) < 1e-12
@assert norm(Vsoc * adjoint(Vsoc) / V0^2 - Ssoc) < 1e-12

write_run_outputs(result; output_dir=config.output_dir)
@assert isfile(joinpath(config.output_dir, "complex_level_flow.csv"))
@assert isfile(joinpath(config.output_dir, "lehmann_sumrule.csv"))
@assert isfile(joinpath(config.output_dir, "soc_hybridization_matrix.csv"))
println("Julia smoke test PASS")
