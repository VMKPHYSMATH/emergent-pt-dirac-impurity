project_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(project_root, "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG
include(joinpath(@__DIR__, "dmnrg_response_core.jl"))
using .PTDiracDMNRGResponse

length(ARGS) == 1 || error("usage: julia --project=. benchmarks/prb_complete/scripts/run_dmnrg_response_one.jl CONFIG.toml")
response = run_dmnrg_response(ARGS[1])
write_run_outputs(response.nrg_result;
                  output_dir=response.nrg_result.config.output_dir)
write_dmnrg_response_outputs(response;
                             output_dir=response.nrg_result.config.output_dir)
println("Wrote DM-NRG response to $(response.nrg_result.config.output_dir)")
println("equilibrium_gate=$(response.equilibrium_gate) response_computed=$(response.response_computed)")
