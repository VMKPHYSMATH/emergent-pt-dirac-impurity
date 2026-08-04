#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG

function usage()
    println("Usage:")
    println("  julia --project=. run_pt_dirac_nhnrg.jl --config config/example.toml")
    println("  julia --project=. run_pt_dirac_nhnrg.jl --config config/example.toml --beta 0.49")
    println("  julia --project=. run_pt_dirac_nhnrg.jl --config config/example.toml --scan 0.45,0.48,0.49,0.495,0.505,0.51,0.52")
end

function parse_args(args)
    config_path = joinpath(@__DIR__, "config", "example.toml")
    beta_override = nothing
    scan = nothing
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--config"
            i += 1; config_path = args[i]
        elseif arg == "--beta"
            i += 1; beta_override = parse(Float64, args[i])
        elseif arg == "--scan"
            i += 1; scan = parse.(Float64, split(args[i], ','))
        elseif arg in ("-h", "--help")
            usage(); exit(0)
        else
            error("unknown argument $arg")
        end
        i += 1
    end
    return config_path, beta_override, scan
end

config_path, beta_override, scan = parse_args(ARGS)
config = load_config(config_path)

if beta_override !== nothing
    m = config.model
    model = ModelParams(
        U=m.U, eps_d=m.eps_d, delta0=m.delta0, c_delta=m.c_delta,
        g_gamma=m.g_gamma, beta0=beta_override, delta_coh=m.delta_coh,
        gamma_common=m.gamma_common, frame=m.frame, bandwidth=m.bandwidth,
        bath_exponent=m.bath_exponent, gamma_edge=m.gamma_edge,
        hybridization_V=m.hybridization_V, bath_split=m.bath_split,
        reciprocal_hybridization=m.reciprocal_hybridization,
        soc_mode=m.soc_mode, soc_lambda=m.soc_lambda, soc_kmax=m.soc_kmax,
    )
    config = RunConfig(model=model, nrg=config.nrg, lehmann=config.lehmann,
                       thermo=config.thermo, output_dir=config.output_dir)
end

if scan === nothing
    result = run_nrg(config)
    path = write_run_outputs(result)
    println("Wrote output to $path")
else
    paths = run_scan(config, scan)
    println("Wrote $(length(paths)) scan directories under $(config.output_dir)")
end
