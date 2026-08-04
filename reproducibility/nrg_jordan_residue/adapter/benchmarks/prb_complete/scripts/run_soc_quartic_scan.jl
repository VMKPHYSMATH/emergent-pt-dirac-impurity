#!/usr/bin/env julia

# SOC-overlap scan for the quartic/Jordan diagnostic.
#
# Minimal low-energy embedding:
#   S_lambda = [1 x; x 1],  x=lambda/kmax,
#   det(S_lambda)=F(lambda)=max(1-x^2,0),
#   V_lambda = V sqrt(S_lambda).
#
# Usage:
#   julia --project=. benchmarks/prb_complete/scripts/run_soc_quartic_scan.jl pilot
#   julia --project=. benchmarks/prb_complete/scripts/run_soc_quartic_scan.jl production /custom/output/root

profile = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :pilot
profile in (:pilot, :production) || error("profile must be pilot or production")

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(repo_root, "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG
using Printf

output_root = length(ARGS) >= 2 ? abspath(expanduser(ARGS[2])) :
              joinpath(repo_root, "output", "soc_quartic_benchmark")
mkpath(output_root)

if profile == :pilot
    nrg = NRGParams(
        Lambda=3.0, z_shift=0.5, iterations=20, nkeep=180,
        min_keep_per_charge=4, sort_type=:LowRe, star_intervals=80,
        overlap_floor=1.0e-10, degeneracy_tolerance=1.0e-10,
        residual_tolerance=1.0e-9, save_levels=36,
    )
else
    nrg = NRGParams(
        Lambda=2.5, z_shift=0.5, iterations=36, nkeep=360,
        min_keep_per_charge=6, sort_type=:LowRe, star_intervals=140,
        overlap_floor=1.0e-11, degeneracy_tolerance=1.0e-11,
        residual_tolerance=1.0e-10, save_levels=56,
    )
end

lehmann = LehmannParams(enabled=false)
const KMAX = pi / 4

function model_for(; U=2.0, beta0=0.5, delta_coh=1.0e-4,
                    lambda=0.0, soc_mode=:overlap,
                    bath_exponent=1.0, gamma_edge=0.120)
    return ModelParams(
        U=U,
        eps_d=-U/2,
        delta0=0.075,
        c_delta=0.050,
        g_gamma=0.200,
        beta0=beta0,
        delta_coh=delta_coh,
        gamma_common=0.120,
        frame=:relative,
        bandwidth=KMAX,
        bath_exponent=bath_exponent,
        gamma_edge=gamma_edge,
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
    if isfile(joinpath(out, "impurity_transition_residue_matrix.csv")) &&
       isfile(joinpath(out, "soc_hybridization_matrix.csv"))
        @printf("[skip] %s already complete\n", label)
        return out
    end
    mkpath(out)
    cfg = RunConfig(model=model_for(; kwargs...), nrg=nrg, lehmann=lehmann,
                    output_dir=out)
    @printf("\n===== %s =====\n", label)
    @printf("lambda=%0.8f  F(lambda)=%0.8f  mode=%s\n",
            cfg.model.soc_lambda, soc_form_factor(cfg.model), String(cfg.model.soc_mode))
    result = run_nrg(cfg)
    write_run_outputs(result; output_dir=out)
    return out
end

# 0. Exact scalar-control identity at lambda=0.
run_one("controls/scalar_U2"; U=2.0, lambda=0.0, soc_mode=:none)
run_one("controls/overlap_lambda0_U2"; U=2.0, lambda=0.0, soc_mode=:overlap)

# 1. Main lambda/F scan at the smallest causal detuning.
lambda_values = profile == :pilot ?
    (0.0, 0.15, 0.30, 0.45, 0.50, 0.60, 0.70, 0.75, 0.99*KMAX) :
    (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.775, 0.79*KMAX, 0.90*KMAX, 0.99*KMAX)
for lambda in lambda_values
    label = @sprintf("lambda_scan/lambda_%0.6f_U2", lambda)
    run_one(label; U=2.0, lambda=lambda, delta_coh=1.0e-4)
end

# 2. Noninteracting controls separate one-body SOC overlap from U effects.
for lambda in (0.0, 0.30, 0.50, 0.70, 0.99*KMAX)
    label = @sprintf("u0_control/lambda_%0.6f_U0", lambda)
    run_one(label; U=0.0, lambda=lambda, delta_coh=1.0e-4)
end

# 3. Detuning approaches at representative F values.
for lambda in (0.0, 0.50, 0.70)
    for delta in (1.0e-2, 1.0e-3, 1.0e-4)
        label = @sprintf("detuning/lambda_%0.3f_delta_%0.1e_U2", lambda, delta)
        run_one(label; U=2.0, lambda=lambda, delta_coh=delta)
    end
end

println("\nSOC quartic/Jordan scan complete: $output_root")
