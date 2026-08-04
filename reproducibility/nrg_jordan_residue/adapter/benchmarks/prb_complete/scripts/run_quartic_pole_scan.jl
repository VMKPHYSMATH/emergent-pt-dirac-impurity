#!/usr/bin/env julia

# Standalone scan driver for the quartic/Jordan diagnostics extension.
# Usage:
#   julia --project=. benchmarks/prb_complete/scripts/run_quartic_pole_scan.jl pilot
#   julia --project=. benchmarks/prb_complete/scripts/run_quartic_pole_scan.jl production /custom/output/root

profile = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :pilot
profile in (:pilot, :production) || error("profile must be pilot or production")

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(repo_root, "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG
using Printf

output_root = length(ARGS) >= 2 ? abspath(expanduser(ARGS[2])) :
              joinpath(repo_root, "output", "quartic_pole_benchmark")
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

function model_for(; U=2.0, beta0=0.5, delta_coh=1.0e-3,
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
        bandwidth=pi/4,
        bath_exponent=bath_exponent,
        gamma_edge=gamma_edge,
        hybridization_V=NaN,
        bath_split=0.0,
        reciprocal_hybridization=true,
    )
end

function run_one(label::String; kwargs...)
    out = joinpath(output_root, label)
    if isfile(joinpath(out, "impurity_transition_residue_matrix.csv"))
        @printf("[skip] %s already complete\n", label)
        return out
    end
    mkpath(out)
    cfg = RunConfig(model=model_for(; kwargs...), nrg=nrg, lehmann=lehmann,
                    output_dir=out)
    @printf("\n===== %s =====\n", label)
    result = run_nrg(cfg)
    write_run_outputs(result; output_dir=out)
    return out
end

# 1. Detuning approach to the bare core.  The exact Jordan point is excluded.
for U in (0.0, 2.0)
    for delta in (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
        label = @sprintf("detuning/U_%0.3g_delta_%0.1e", U, delta)
        run_one(label; U=U, beta0=0.5, delta_coh=delta)
    end
end

# 2. U scaling at fixed small causal detuning.
for U in (0.0, 0.5, 1.0, 2.0, 4.0)
    label = @sprintf("u_scan/U_%0.3g_delta_1e-4", U)
    run_one(label; U=U, beta0=0.5, delta_coh=1.0e-4)
end

# 3. Side-point scan to distinguish a local minimum from a true coalescence.
for beta0 in (0.490, 0.495, 0.500, 0.505, 0.510)
    label = @sprintf("beta_scan/beta_%0.3f_U_2", beta0)
    run_one(label; U=2.0, beta0=beta0, delta_coh=1.0e-4)
end

println("\nQuartic/Jordan scan complete: $output_root")
