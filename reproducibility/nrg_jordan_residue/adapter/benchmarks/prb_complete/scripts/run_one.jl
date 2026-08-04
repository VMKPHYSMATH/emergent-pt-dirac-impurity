project_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(project_root, "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG

length(ARGS) == 1 || error("usage: julia --project=. benchmarks/prb_complete/scripts/run_one.jl CONFIG.toml")
cfg = load_config(ARGS[1])
result = run_nrg(cfg)
write_run_outputs(result; output_dir=cfg.output_dir)
println("Wrote output to $(cfg.output_dir)")
