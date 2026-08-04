#!/usr/bin/env julia

# Fine U-detuning scan for Jordan-vector and quartic-scaling tests.
#
# Usage:
#   julia --project=. benchmarks/prb_complete/scripts/run_jordan_quartic_scaling_scan.jl pilot scalar
#   julia --project=. benchmarks/prb_complete/scripts/run_jordan_quartic_scaling_scan.jl pilot both /custom/output/root
#
# mode = scalar | soc | both

profile = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :pilot
profile in (:pilot, :production) || error("profile must be pilot or production")
mode = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :scalar
mode in (:scalar, :soc, :both) || error("mode must be scalar, soc, or both")

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(repo_root, "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG
using Printf

output_root = length(ARGS) >= 3 ? abspath(expanduser(ARGS[3])) :
              joinpath(repo_root, "output", "jordan_quartic_scaling")
mkpath(output_root)

if profile == :pilot
    nrg = NRGParams(
        Lambda=3.0, z_shift=0.5, iterations=8, nkeep=240,
        min_keep_per_charge=6, sort_type=:LowRe, star_intervals=90,
        overlap_floor=1.0e-11, degeneracy_tolerance=1.0e-11,
        residual_tolerance=1.0e-10, save_levels=48,
    )
    U_values = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
    delta_values = (3.0e-4, 1.0e-4, 3.0e-5)
else
    nrg = NRGParams(
        Lambda=2.5, z_shift=0.5, iterations=10, nkeep=400,
        min_keep_per_charge=8, sort_type=:LowRe, star_intervals=150,
        overlap_floor=1.0e-12, degeneracy_tolerance=1.0e-12,
        residual_tolerance=1.0e-11, save_levels=72,
    )
    U_values = (0.0, 0.0025, 0.005, 0.010, 0.015, 0.020, 0.030,
                0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.400,
                0.600, 0.800)
    delta_values = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5)
end

lehmann = LehmannParams(enabled=false)
const KMAX = pi / 4
const SOC_LAMBDA = 0.50

function model_for(; U=0.0, delta_coh=1.0e-4, soc_mode=:none, lambda=0.0)
    ModelParams(
        U=U,
        eps_d=-U/2,
        delta0=0.075,
        c_delta=0.050,
        g_gamma=0.200,
        beta0=0.5,
        delta_coh=delta_coh,
        gamma_common=0.120,
        frame=:relative,
        bandwidth=KMAX,
        bath_exponent=1.0,
        gamma_edge=0.120,
        hybridization_V=NaN,
        bath_split=0.0,
        reciprocal_hybridization=true,
        soc_mode=soc_mode,
        soc_lambda=lambda,
        soc_kmax=KMAX,
    )
end

function run_one(label::String; kwargs...)
    out = joinpath(output_root, label)
    complete = joinpath(out, "impurity_transition_residue_matrix.csv")
    if isfile(complete)
        @printf("[skip] %s already complete\n", label)
        return out
    end
    mkpath(out)
    cfg = RunConfig(model=model_for(; kwargs...), nrg=nrg, lehmann=lehmann,
                    output_dir=out)
    @printf("\n===== %s =====\n", label)
    @printf("U=%0.8g delta=%0.3e lambda=%0.8g F=%0.8g mode=%s\n",
            cfg.model.U, cfg.model.delta_coh, cfg.model.soc_lambda,
            soc_form_factor(cfg.model), String(cfg.model.soc_mode))
    result = run_nrg(cfg)
    write_run_outputs(result; output_dir=out)
    return out
end

modes = mode == :both ? (:scalar, :soc) : (mode,)
for which in modes
    if which == :scalar
        prefix = "scalar"
        smode = :none
        lambda = 0.0
    else
        prefix = @sprintf("soc_lambda_%0.6f", SOC_LAMBDA)
        smode = :overlap
        lambda = SOC_LAMBDA
    end
    for delta in delta_values
        for U in U_values
            label = @sprintf("%s/delta_%0.1e/U_%0.6f", prefix, delta, U)
            run_one(label; U=U, delta_coh=delta, soc_mode=smode, lambda=lambda)
        end
    end
end

println("\nJordan/quartic scaling scan complete: $output_root")
println("Analyze with:")
println("python3 benchmarks/prb_complete/scripts/analyze_jordan_quartic_scaling.py \\")
println("  $output_root --out $(joinpath(output_root, "analysis")) --iteration 5")
