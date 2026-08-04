using TOML

project_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(project_root, "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG
include(joinpath(project_root, "benchmarks", "prb_complete", "scripts",
                 "dmnrg_response_core.jl"))
using .PTDiracDMNRGResponse

mktempdir() do tmp
    out = joinpath(tmp, "run")
    cfgpath = joinpath(tmp, "dmnrg_smoke.toml")
    open(cfgpath, "w") do io
        write(io, """
[model]
U = 1.0
eps_d = -0.5
delta0 = 0.08
c_delta = 0.05
g_gamma = 0.20
beta0 = 0.20
delta_coh = 0.0
gamma_common = 0.12
frame = "relative"
bandwidth = 0.7853981633974483
bath_exponent = 0.0
gamma_edge = 0.12
hybridization_V = nan
bath_split = 0.0
reciprocal_hybridization = true
soc_mode = "none"
soc_lambda = 0.0
soc_kmax = 0.7853981633974483

[nrg]
Lambda = 3.0
z_shift = 0.5
iterations = 2
nkeep = 256
min_keep_per_charge = 0
sort_type = "LowRe"
star_intervals = 30
overlap_floor = 1.0e-10
degeneracy_tolerance = 1.0e-10
residual_tolerance = 1.0e-8
save_levels = 12

[lehmann]
enabled = false
omega_min = -1.0
omega_max = 1.0
omega_points = 101
eta = 0.01

[thermodynamics]
enabled = false
temperature_min = 1.0e-7
temperature_max = 1.0
temperature_points = 31
imag_tolerance = 1.0e-8
entropy_target = 0.34657359027997264
low_entropy_max = 0.2079441541679836
moment_entropy_min = 0.4505456673639645

[dmnrg_response]
omega_min = 0.0
omega_max = 0.5
omega_points = 201
eta = 0.005
imag_tolerance = 1.0e-7
components = ["x", "z"]
weight_fraction = 1.0
weight_floor = 0.0
max_transitions_per_shell = 200000
time_max = 50.0
time_points = 101
exploratory_complex = false

[output]
directory = "$(out)"
""")
    end

    response = run_dmnrg_response(cfgpath)
    @assert response.response_computed
    @assert response.equilibrium_gate
    @assert response.density_trace_error < 1.0e-9
    @assert !isempty(response.transitions)
    @assert length(response.omega) == 201
    @assert all(isfinite, real.(response.spectra[(:x,:x)]))
    @assert all(isfinite, imag.(response.spectra[(:x,:x)]))
    @assert maximum(abs.(response.spectra[(:x,:x)])) > 0.0
    # With nkeep equal to the exact final Hilbert-space dimension, no state is
    # discarded before the terminal shell; the complete-basis partition should
    # therefore place all nonzero transition support at the last iteration.
    nonzero_iterations = unique([line.iteration for line in response.transitions])
    @assert nonzero_iterations == [2]

    write_run_outputs(response.nrg_result; output_dir=out)
    write_dmnrg_response_outputs(response; output_dir=out)
    @assert isfile(joinpath(out, "dmnrg_response_spectra.csv"))
    @assert isfile(joinpath(out, "dmnrg_response_time.csv"))
    @assert isfile(joinpath(out, "dmnrg_response_summary.toml"))
end

println("DM-NRG response smoke PASS")
