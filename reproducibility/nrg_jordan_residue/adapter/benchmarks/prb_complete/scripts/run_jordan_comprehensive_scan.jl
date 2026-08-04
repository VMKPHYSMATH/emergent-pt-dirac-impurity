#!/usr/bin/env julia

# Comprehensive Jordan-channel scaling and convergence scan.
#
# Usage:
#   julia --project=. benchmarks/prb_complete/scripts/run_jordan_comprehensive_scan.jl comprehensive core scalar
#   julia --project=. benchmarks/prb_complete/scripts/run_jordan_comprehensive_scan.jl comprehensive convergence scalar
#   julia --project=. benchmarks/prb_complete/scripts/run_jordan_comprehensive_scan.jl comprehensive all both /custom/output
#
# Arguments:
#   profile = pilot | comprehensive | production
#   suite   = core | convergence | all
#   mode    = scalar | soc | both
#
# The core grid resolves U -> 0 and delta_coh -> 0.  The convergence suite uses
# a reduced grid to test Nkeep, z-shift, and Lambda sensitivity.

profile = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :comprehensive
profile in (:pilot, :comprehensive, :production) || error("profile must be pilot, comprehensive, or production")
suite = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :core
suite in (:core, :convergence, :all) || error("suite must be core, convergence, or all")
mode = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :scalar
mode in (:scalar, :soc, :both) || error("mode must be scalar, soc, or both")

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(repo_root, "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG
using Printf
using Dates

output_root = length(ARGS) >= 4 ? abspath(expanduser(ARGS[4])) :
              joinpath(repo_root, "output", "jordan_comprehensive")
mkpath(output_root)

const KMAX = pi / 4
const SOC_LAMBDA = 0.50
const CORE_U = profile == :pilot ?
    (0.0, 0.005, 0.010, 0.020, 0.050) :
    (0.0, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200, 0.0300, 0.0500, 0.0750, 0.1000)
const CORE_DELTA = profile == :pilot ?
    (1.0e-4, 4.0e-5, 1.0e-5) :
    (1.0e-4, 7.0e-5, 4.0e-5, 2.0e-5, 1.0e-5)
const CONV_U = (0.0, 0.005, 0.010, 0.020, 0.050)
const CONV_DELTA = (1.0e-4, 4.0e-5, 1.0e-5)

function nrg_config(; Lambda=3.0, z_shift=0.5, nkeep=320)
    if profile == :pilot
        return NRGParams(
            Lambda=Lambda, z_shift=z_shift, iterations=7, nkeep=min(nkeep, 240),
            min_keep_per_charge=6, sort_type=:LowRe, star_intervals=90,
            overlap_floor=1.0e-11, degeneracy_tolerance=1.0e-11,
            residual_tolerance=1.0e-10, save_levels=64,
        )
    elseif profile == :comprehensive
        return NRGParams(
            Lambda=Lambda, z_shift=z_shift, iterations=8, nkeep=nkeep,
            min_keep_per_charge=8, sort_type=:LowRe, star_intervals=130,
            overlap_floor=3.0e-12, degeneracy_tolerance=3.0e-12,
            residual_tolerance=3.0e-11, save_levels=80,
        )
    else
        return NRGParams(
            Lambda=Lambda, z_shift=z_shift, iterations=10, nkeep=nkeep,
            min_keep_per_charge=10, sort_type=:LowRe, star_intervals=190,
            overlap_floor=1.0e-12, degeneracy_tolerance=1.0e-12,
            residual_tolerance=1.0e-11, save_levels=112,
        )
    end
end

# label, Lambda, z, nkeep, grid
reference_nkeep = profile == :pilot ? 240 : (profile == :production ? 480 : 320)
configs = NamedTuple[]
if suite in (:core, :all)
    push!(configs, (label="reference", Lambda=3.0, z=0.5, nkeep=reference_nkeep, grid=:core))
end
if suite in (:convergence, :all)
    # The reference subset is included explicitly so every convergence comparison
    # is made on exactly the same U/delta grid.
    push!(configs, (label="conv_ref", Lambda=3.0, z=0.5, nkeep=reference_nkeep, grid=:conv))
    if profile != :pilot
        push!(configs, (label="nkeep_240", Lambda=3.0, z=0.5, nkeep=240, grid=:conv))
        push!(configs, (label="nkeep_400", Lambda=3.0, z=0.5, nkeep=400, grid=:conv))
        push!(configs, (label="z_0p25", Lambda=3.0, z=0.25, nkeep=reference_nkeep, grid=:conv))
        push!(configs, (label="z_0p75", Lambda=3.0, z=0.75, nkeep=reference_nkeep, grid=:conv))
        push!(configs, (label="Lambda_2p5", Lambda=2.5, z=0.5, nkeep=reference_nkeep, grid=:conv))
    end
end

lehmann = LehmannParams(enabled=false)

function model_for(; U=0.0, delta_coh=1.0e-5, soc_mode=:none, lambda=0.0)
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

function write_scan_metadata(out, cfg, model_label, U, delta)
    stamp = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
    open(joinpath(out, "SCAN_METADATA.txt"), "w") do io
        println(io, "created_utc=$stamp")
        println(io, "profile=$profile")
        println(io, "suite=$suite")
        println(io, "config_label=$(cfg.label)")
        println(io, "model_label=$model_label")
        println(io, "Lambda=$(cfg.Lambda)")
        println(io, "z_shift=$(cfg.z)")
        println(io, "nkeep=$(cfg.nkeep)")
        println(io, "grid=$(cfg.grid)")
        println(io, "U=$U")
        println(io, "delta_coh=$delta")
    end
end

function run_one(cfg, model_label::String; U, delta_coh, soc_mode, lambda)
    out = joinpath(output_root, cfg.label, model_label,
                   @sprintf("delta_%0.1e", delta_coh), @sprintf("U_%0.6f", U))
    complete = joinpath(out, "impurity_transition_residue_matrix.csv")
    if isfile(complete)
        @printf("[skip] %s\n", relpath(out, output_root))
        return out
    end
    mkpath(out)
    nrg = nrg_config(Lambda=cfg.Lambda, z_shift=cfg.z, nkeep=cfg.nkeep)
    config = RunConfig(
        model=model_for(U=U, delta_coh=delta_coh, soc_mode=soc_mode, lambda=lambda),
        nrg=nrg, lehmann=lehmann, output_dir=out,
    )
    @printf("\n===== %s =====\n", relpath(out, output_root))
    @printf("U=%0.8g delta=%0.3e Lambda=%0.3g z=%0.3g nkeep=%d lambda=%0.5g F=%0.8g\n",
            U, delta_coh, cfg.Lambda, cfg.z, cfg.nkeep, lambda,
            soc_form_factor(config.model))
    result = run_nrg(config)
    write_run_outputs(result; output_dir=out)
    write_scan_metadata(out, cfg, model_label, U, delta_coh)
    return out
end

modes = mode == :both ? (:scalar, :soc) : (mode,)
for cfg in configs
    U_values = cfg.grid == :core ? CORE_U : CONV_U
    delta_values = cfg.grid == :core ? CORE_DELTA : CONV_DELTA
    for which in modes
        if which == :scalar
            model_label = "scalar"
            smode = :none
            lambda = 0.0
        else
            model_label = @sprintf("soc_overlap_lambda_%0.6f", SOC_LAMBDA)
            smode = :overlap
            lambda = SOC_LAMBDA
        end
        for delta in delta_values, U in U_values
            run_one(cfg, model_label; U=U, delta_coh=delta, soc_mode=smode, lambda=lambda)
        end
    end
end

println("\nComprehensive scan complete: $output_root")
println("Analyze with:")
println("python3 benchmarks/prb_complete/scripts/analyze_jordan_comprehensive.py \\")
analysis_dir = joinpath(output_root, "analysis")
println("  $output_root --out $analysis_dir --iterations 4 5 6")
