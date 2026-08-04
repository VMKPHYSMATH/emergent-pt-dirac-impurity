module PTDiracDMNRGResponse

using LinearAlgebra
using Printf
using TOML
using Statistics
import Main.PTDiracNHNRG
const M = PTDiracNHNRG

export DMResponseParams, DMResponseResult, run_dmnrg_response,
       write_dmnrg_response_outputs, run_nrg_with_shell_capture

Base.@kwdef struct DMResponseParams
    omega_min::Float64 = 0.0
    omega_max::Float64 = 0.30
    omega_points::Int = 801
    eta::Float64 = 0.003
    imag_tolerance::Float64 = 1.0e-8
    components::Vector{Symbol} = Symbol[:x, :z]
    weight_fraction::Float64 = 0.9999
    weight_floor::Float64 = 1.0e-12
    max_transitions_per_shell::Int = 4000
    time_max::Float64 = 400.0
    time_points::Int = 1001
    exploratory_complex::Bool = true
end

struct ShellData
    iteration::Int
    scale::Float64
    energies_scaled::Vector{ComplexF64}
    energies_physical::Vector{ComplexF64}
    charges::Vector{Int}
    left::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    keep::Vector{Int}
    discarded::Vector{Int}
end

struct TransitionLine
    iteration::Int
    component_a::Symbol
    component_b::Symbol
    gap::ComplexF64
    weight::ComplexF64
end

struct PairQuality
    iteration::Int
    component_a::Symbol
    component_b::Symbol
    candidate_count::Int
    selected_count::Int
    total_abs_weight::Float64
    selected_abs_weight::Float64
    selected_fraction::Float64
    full_commutator_weight::ComplexF64
    selected_commutator_weight::ComplexF64
end

struct DMResponseResult
    params::DMResponseParams
    nrg_result::M.NRGResult
    transitions::Vector{TransitionLine}
    quality::Vector{PairQuality}
    omega::Vector{Float64}
    spectra::Dict{Tuple{Symbol,Symbol},Vector{ComplexF64}}
    time::Vector{Float64}
    time_response::Dict{Tuple{Symbol,Symbol},Vector{ComplexF64}}
    equilibrium_gate::Bool
    response_computed::Bool
    max_centered_imaginary_energy::Float64
    ground_energy::ComplexF64
    density_trace_error::Float64
end

function _response_params(path::AbstractString)::DMResponseParams
    raw = TOML.parsefile(path)
    r = get(raw, "dmnrg_response", Dict{String,Any}())
    components_raw = get(r, "components", ["x", "z"])
    components = Symbol[Symbol(lowercase(String(x))) for x in components_raw]
    allowed = Set([:x, :y, :z])
    all(x -> x in allowed, components) ||
        error("dmnrg_response.components must contain only x, y, z")
    length(unique(components)) == length(components) ||
        error("dmnrg_response.components contains duplicates")
    return DMResponseParams(
        omega_min = Float64(get(r, "omega_min", 0.0)),
        omega_max = Float64(get(r, "omega_max", 0.30)),
        omega_points = Int(get(r, "omega_points", 801)),
        eta = Float64(get(r, "eta", 0.003)),
        imag_tolerance = Float64(get(r, "imag_tolerance", 1.0e-8)),
        components = components,
        weight_fraction = Float64(get(r, "weight_fraction", 0.9999)),
        weight_floor = Float64(get(r, "weight_floor", 1.0e-12)),
        max_transitions_per_shell = Int(get(r, "max_transitions_per_shell", 4000)),
        time_max = Float64(get(r, "time_max", 400.0)),
        time_points = Int(get(r, "time_points", 1001)),
        exploratory_complex = Bool(get(r, "exploratory_complex", true)),
    )
end

function _validate_params(p::DMResponseParams)
    p.omega_max > p.omega_min || error("omega_max must exceed omega_min")
    p.omega_points >= 3 || error("omega_points must be at least 3")
    p.eta > 0.0 || error("eta must be positive")
    0.0 < p.weight_fraction <= 1.0 || error("weight_fraction must lie in (0,1]")
    p.weight_floor >= 0.0 || error("weight_floor must be nonnegative")
    p.max_transitions_per_shell > 0 || error("max_transitions_per_shell must be positive")
    p.time_max > 0.0 || error("time_max must be positive")
    p.time_points >= 3 || error("time_points must be at least 3")
end

function _diagonalize_capture(H::Matrix{ComplexF64}, charges::Vector{Int},
                              nrg::M.NRGParams)
    dim = size(H, 1)
    all_values = ComplexF64[]
    all_charges = Int[]
    all_left = zeros(ComplexF64, dim, dim)
    all_right = zeros(ComplexF64, dim, dim)
    cursor = 0
    maxres = 0.0
    maxbio = 0.0
    minoverlap = Inf

    for q in sort(unique(charges))
        idx = findall(==(q), charges)
        block = Matrix{ComplexF64}(H[idx, idx])
        eigb = M.lr_eigensystem(block, nrg.overlap_floor,
                                nrg.degeneracy_tolerance)
        nb = length(idx)
        cols = (cursor + 1):(cursor + nb)
        append!(all_values, eigb.values)
        append!(all_charges, fill(q, nb))
        all_left[idx, cols] .= eigb.left
        all_right[idx, cols] .= eigb.right
        cursor += nb
        maxres = max(maxres, eigb.max_residual)
        maxbio = max(maxbio, eigb.biorth_error)
        minoverlap = min(minoverlap, eigb.min_pair_overlap)
    end

    keep, reference = M._select_kept(all_values, all_charges, nrg)
    keep_set = Set(keep)
    discarded = Int[i for i in eachindex(all_values) if !(i in keep_set)]
    shifted = all_values .- reference
    return shifted, all_charges, all_left, all_right, keep, discarded,
           reference, maxres, maxbio, minoverlap
end

function _initialize_capture(config::M.RunConfig, alpha0::Float64)
    nrg, model = config.nrg, config.model
    D = model.bandwidth
    Himp = M.build_impurity_hamiltonian(model) ./ D
    H = kron(Himp, M.ILOCAL)
    H .+= kron(M.ILOCAL,
               M.site_hamiltonian(alpha0 / D, model.bath_split / D))
    Vmat = M.impurity_hybridization_matrix(model) ./ D
    local_cre = ntuple(a -> Matrix{ComplexF64}(adjoint(M.CLOCAL[a])), M.NFLAV)
    H .+= M.coupling_term(M.CLOCAL, local_cre, M.QLOCAL, Vmat)

    full_charges = Int[qimp + qsite for qimp in M.QLOCAL for qsite in M.QLOCAL]
    shifted, charges, L, R, keep, discarded, reference,
        maxres, maxbio, minoverlap = _diagonalize_capture(H, full_charges, nrg)

    values = shifted[keep]
    qkeep = charges[keep]
    Pimp = M.parity_from_charges(M.QLOCAL)
    last_ann_full = ntuple(a -> kron(Pimp, M.CLOCAL[a]), M.NFLAV)
    last_cre_full = ntuple(a -> kron(Pimp, adjoint(M.CLOCAL[a])), M.NFLAV)
    imp_ann_full = ntuple(a -> kron(M.CLOCAL[a], M.ILOCAL), M.NFLAV)
    imp_cre_full = ntuple(a -> kron(adjoint(M.CLOCAL[a]), M.ILOCAL), M.NFLAV)
    Lk = L[:, keep]
    Rk = R[:, keep]
    last_ann_new = ntuple(a -> M.transform_operator(Lk, last_ann_full[a], Rk), M.NFLAV)
    last_cre_new = ntuple(a -> M.transform_operator(Lk, last_cre_full[a], Rk), M.NFLAV)
    imp_ann_new = ntuple(a -> M.transform_operator(Lk, imp_ann_full[a], Rk), M.NFLAV)
    imp_cre_new = ntuple(a -> M.transform_operator(Lk, imp_cre_full[a], Rk), M.NFLAV)

    energy_offset = D * reference
    discarded_physical = ComplexF64[energy_offset + D*x for x in shifted[discarded]]
    state = M.NRGState(0, values, qkeep, last_ann_new, last_cre_new,
                       imp_ann_new, imp_cre_new, energy_offset,
                       discarded_physical, charges[discarded],
                       maxres, maxbio, minoverlap)
    shell = ShellData(0, D, shifted,
                      ComplexF64[energy_offset + D*x for x in shifted],
                      charges, L, R, keep, discarded)
    return state, shell
end

function _add_capture(old::M.NRGState, site_index::Int, alpha::Float64,
                      hopping::Float64, config::M.RunConfig)
    nrg, model = config.nrg, config.model
    D = model.bandwidth
    omega_old = D * nrg.Lambda^(-old.iteration / 2)
    omega_new = D * nrg.Lambda^(-(old.iteration + 1) / 2)
    ratio = omega_old / omega_new
    nold = length(old.energies)

    H = ratio .* kron(Diagonal(old.energies) |> Matrix, M.ILOCAL)
    H .+= kron(Matrix{ComplexF64}(I, nold, nold),
               M.site_hamiltonian(alpha / omega_new,
                                  model.bath_split / omega_new))
    T = Matrix{ComplexF64}(I, M.NFLAV, M.NFLAV) .* (hopping / omega_new)
    H .+= M.coupling_term(old.last_ann, old.last_cre, old.charges, T)

    full_charges = Int[qold + qsite for qold in old.charges for qsite in M.QLOCAL]
    shifted, charges, L, R, keep, discarded, reference,
        maxres, maxbio, minoverlap = _diagonalize_capture(H, full_charges, nrg)

    values = shifted[keep]
    qkeep = charges[keep]
    Pold = M.parity_from_charges(old.charges)
    last_ann_full = ntuple(a -> kron(Pold, M.CLOCAL[a]), M.NFLAV)
    last_cre_full = ntuple(a -> kron(Pold, adjoint(M.CLOCAL[a])), M.NFLAV)
    imp_ann_full = ntuple(a -> kron(old.impurity_ann[a], M.ILOCAL), M.NFLAV)
    imp_cre_full = ntuple(a -> kron(old.impurity_cre[a], M.ILOCAL), M.NFLAV)
    Lk = L[:, keep]
    Rk = R[:, keep]
    last_ann_new = ntuple(a -> M.transform_operator(Lk, last_ann_full[a], Rk), M.NFLAV)
    last_cre_new = ntuple(a -> M.transform_operator(Lk, last_cre_full[a], Rk), M.NFLAV)
    imp_ann_new = ntuple(a -> M.transform_operator(Lk, imp_ann_full[a], Rk), M.NFLAV)
    imp_cre_new = ntuple(a -> M.transform_operator(Lk, imp_cre_full[a], Rk), M.NFLAV)

    energy_offset = old.energy_offset + omega_new * reference
    discarded_physical = ComplexF64[
        energy_offset + omega_new*x for x in shifted[discarded]
    ]
    state = M.NRGState(site_index, values, qkeep, last_ann_new, last_cre_new,
                       imp_ann_new, imp_cre_new, energy_offset,
                       discarded_physical, charges[discarded],
                       maxres, maxbio, minoverlap)
    shell = ShellData(site_index, omega_new, shifted,
                      ComplexF64[energy_offset + omega_new*x for x in shifted],
                      charges, L, R, keep, discarded)
    return state, shell
end

function run_nrg_with_shell_capture(config::M.RunConfig)
    alpha, beta = M.make_chain(config)
    state, shell0 = _initialize_capture(config, alpha[1])
    shells = ShellData[shell0]
    records = M.IterationRecord[M.iteration_record(state, config)]
    discarded_shells = M.DiscardedShell[
        M.DiscardedShell(0, records[1].scale,
                         copy(state.discarded_physical),
                         copy(state.discarded_charges))
    ]
    for site in 1:config.nrg.iterations
        state, shell = _add_capture(state, site, alpha[site + 1], beta[site], config)
        push!(shells, shell)
        push!(records, M.iteration_record(state, config))
        push!(discarded_shells,
              M.DiscardedShell(site, records[end].scale,
                               copy(state.discarded_physical),
                               copy(state.discarded_charges)))
        if state.max_residual > config.nrg.residual_tolerance
            @warn "eigensolver residual exceeds tolerance" iteration=site residual=state.max_residual
        end
        @printf("iteration=%3d kept=%4d scale=%10.3e residual=%8.2e bio=%8.2e minLR=%8.2e\n",
                site, length(state.energies), records[end].scale,
                state.max_residual, state.biorth_error,
                state.min_pair_overlap)
    end
    result = M.NRGResult(config, state, records, alpha, beta, discarded_shells)
    return result, shells
end

function _spin_operators_impurity()
    d1, d2 = M.CLOCAL
    c1, c2 = adjoint(d1), adjoint(d2)
    sx = 0.5 .* (c1*d2 + c2*d1)
    sy = (-0.5im) .* (c1*d2 - c2*d1)
    sz = 0.5 .* (M.NLOCAL[1] - M.NLOCAL[2])
    return Dict{Symbol,Matrix{ComplexF64}}(
        :x => Matrix{ComplexF64}(sx),
        :y => Matrix{ComplexF64}(sy),
        :z => Matrix{ComplexF64}(sz),
    )
end

function _build_operator_shells(shells::Vector{ShellData},
                                components::Vector{Symbol})
    imp = _spin_operators_impurity()
    out = [Dict{Symbol,Matrix{ComplexF64}}() for _ in eachindex(shells)]
    kept_previous = Dict{Symbol,Matrix{ComplexF64}}()
    for (ishell, shell) in enumerate(shells)
        for comp in components
            product_operator = if ishell == 1
                kron(imp[comp], M.ILOCAL)
            else
                kron(kept_previous[comp], M.ILOCAL)
            end
            full_operator = Matrix{ComplexF64}(
                adjoint(shell.left) * product_operator * shell.right
            )
            out[ishell][comp] = full_operator
        end
        for comp in components
            kept_previous[comp] = out[ishell][comp][shell.keep, shell.keep]
        end
    end
    return out
end

function _partial_trace_last_site(rho_product::Matrix{ComplexF64})
    dim = size(rho_product, 1)
    dim % M.LOCAL_DIM == 0 || error("product dimension is not divisible by LOCAL_DIM")
    nold = div(dim, M.LOCAL_DIM)
    reduced = zeros(ComplexF64, nold, nold)
    for i in 1:nold, j in 1:nold
        value = 0.0 + 0.0im
        for s in 1:M.LOCAL_DIM
            ii = (i - 1)*M.LOCAL_DIM + s
            jj = (j - 1)*M.LOCAL_DIM + s
            value += rho_product[ii, jj]
        end
        reduced[i, j] = value
    end
    return reduced
end

function _zero_temperature_density_matrices(shells::Vector{ShellData})
    nshell = length(shells)
    densities = Vector{Matrix{ComplexF64}}(undef, nshell)
    final = shells[end]
    ground = final.keep[argmin([
        (real(final.energies_physical[i]),
         abs(imag(final.energies_physical[i])),
         abs(final.energies_physical[i])) for i in final.keep
    ])]
    rho = zeros(ComplexF64, length(final.energies_physical),
                length(final.energies_physical))
    rho[ground, ground] = 1.0 + 0.0im
    densities[end] = rho

    for n in nshell:-1:2
        shell = shells[n]
        previous = shells[n - 1]
        rho_product = shell.right * densities[n] * adjoint(shell.left)
        rho_kept_previous = _partial_trace_last_site(rho_product)
        rho_previous = zeros(ComplexF64,
                             length(previous.energies_physical),
                             length(previous.energies_physical))
        rho_previous[previous.keep, previous.keep] .= rho_kept_previous
        trrho = tr(rho_previous)
        if abs(trrho) > 100*eps(Float64)
            rho_previous ./= trrho
        end
        densities[n - 1] = rho_previous
    end
    trace_error = maximum(abs(tr(rho) - 1.0) for rho in densities)
    return densities, ground, final.energies_physical[ground], trace_error
end

function _complete_basis_imaginary_gate(result::M.NRGResult, tolerance::Float64)
    energies, _, _ = M._complete_basis_levels(result)
    reference = energies[argmin(real.(energies))]
    maximag = maximum(abs.(imag.(energies .- reference)); init=0.0)
    threshold = tolerance * max(result.config.model.bandwidth, 1.0)
    gate = result.config.model.frame == :relative && maximag <= threshold
    return gate, maximag
end

function _select_transition_lines(shell::ShellData,
                                  operators::Dict{Symbol,Matrix{ComplexF64}},
                                  rho::Matrix{ComplexF64},
                                  p::DMResponseParams,
                                  final_shell::Bool)
    lines = TransitionLine[]
    qualities = PairQuality[]
    nkeep_set = Set(shell.keep)
    nstate = length(shell.energies_physical)

    for bcomp in p.components
        B = operators[bcomp]
        comm = B*rho - rho*B
        for acomp in p.components
            A = operators[acomp]
            candidates = Tuple{Float64,Int,Int,ComplexF64,ComplexF64}[]
            total_abs = 0.0
            full_sum = 0.0 + 0.0im
            max_abs = 0.0
            for a in 1:nstate, b in 1:nstate
                if !final_shell && (a in nkeep_set) && (b in nkeep_set)
                    continue
                end
                weight = A[a,b] * comm[b,a]
                aw = abs(weight)
                if aw == 0.0 || !isfinite(aw)
                    continue
                end
                gap = shell.energies_physical[b] - shell.energies_physical[a]
                full_sum += weight
                total_abs += aw
                max_abs = max(max_abs, aw)
                push!(candidates, (aw, a, b, gap, weight))
            end

            if isempty(candidates) || total_abs == 0.0
                push!(qualities, PairQuality(shell.iteration, acomp, bcomp,
                                             0, 0, 0.0, 0.0, 1.0,
                                             full_sum, 0.0 + 0.0im))
                continue
            end
            threshold = p.weight_floor * max_abs
            filter!(x -> x[1] >= threshold, candidates)
            sort!(candidates, by=x -> -x[1])
            target = p.weight_fraction * total_abs
            selected_abs = 0.0
            selected_sum = 0.0 + 0.0im
            selected_count = 0
            for item in candidates
                selected_count >= p.max_transitions_per_shell && break
                aw, _, _, gap, weight = item
                push!(lines, TransitionLine(shell.iteration, acomp, bcomp,
                                            gap, weight))
                selected_abs += aw
                selected_sum += weight
                selected_count += 1
                selected_abs >= target && break
            end
            fraction = total_abs > 0.0 ? min(selected_abs/total_abs, 1.0) : 1.0
            push!(qualities, PairQuality(shell.iteration, acomp, bcomp,
                                         length(candidates), selected_count,
                                         total_abs, selected_abs, fraction,
                                         full_sum, selected_sum))
        end
    end
    return lines, qualities
end

function _evaluate_response(lines::Vector{TransitionLine}, p::DMResponseParams)
    omega = collect(range(p.omega_min, p.omega_max, length=p.omega_points))
    spectra = Dict{Tuple{Symbol,Symbol},Vector{ComplexF64}}()
    for a in p.components, b in p.components
        spectra[(a,b)] = zeros(ComplexF64, length(omega))
    end
    for line in lines
        key = (line.component_a, line.component_b)
        values = spectra[key]
        for iw in eachindex(omega)
            values[iw] += line.weight /
                          (complex(omega[iw], p.eta) - line.gap)
        end
    end
    time = collect(range(0.0, p.time_max, length=p.time_points))
    time_response = Dict{Tuple{Symbol,Symbol},Vector{ComplexF64}}()
    for a in p.components, b in p.components
        time_response[(a,b)] = zeros(ComplexF64, length(time))
    end
    for line in lines
        key = (line.component_a, line.component_b)
        values = time_response[key]
        for it in eachindex(time)
            values[it] += -1im * line.weight *
                          exp(-1im*line.gap*time[it] - p.eta*time[it])
        end
    end
    return omega, spectra, time, time_response
end

function run_dmnrg_response(config_path::AbstractString)
    cfg = M.load_config(config_path)
    p = _response_params(config_path)
    _validate_params(p)
    result, shells = run_nrg_with_shell_capture(cfg)
    gate, maximag = _complete_basis_imaginary_gate(result, p.imag_tolerance)
    response_computed = gate || p.exploratory_complex
    if !response_computed
        return DMResponseResult(p, result, TransitionLine[], PairQuality[],
                                Float64[],
                                Dict{Tuple{Symbol,Symbol},Vector{ComplexF64}}(),
                                Float64[],
                                Dict{Tuple{Symbol,Symbol},Vector{ComplexF64}}(),
                                gate, false, maximag, NaN + NaN*im, NaN)
    end

    densities, _, ground_energy, trace_error =
        _zero_temperature_density_matrices(shells)
    operator_shells = _build_operator_shells(shells, p.components)
    transitions = TransitionLine[]
    qualities = PairQuality[]
    for ishell in eachindex(shells)
        lines, q = _select_transition_lines(shells[ishell],
                                            operator_shells[ishell],
                                            densities[ishell], p,
                                            ishell == length(shells))
        append!(transitions, lines)
        append!(qualities, q)
    end
    omega, spectra, time, time_response = _evaluate_response(transitions, p)
    return DMResponseResult(p, result, transitions, qualities,
                            omega, spectra, time, time_response,
                            gate, true, maximag, ground_energy, trace_error)
end

function _write_csv(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(row, ','))
        end
    end
end

function _toml_float(x::Real)
    if isnan(x)
        return "nan"
    elseif isinf(x)
        return signbit(x) ? "-inf" : "inf"
    end
    return repr(Float64(x))
end

function write_dmnrg_response_outputs(response::DMResponseResult;
                                      output_dir::AbstractString=
                                      response.nrg_result.config.output_dir)
    mkpath(output_dir)
    p = response.params
    transition_rows = Any[]
    for line in response.transitions
        push!(transition_rows,
              (line.iteration, String(line.component_a), String(line.component_b),
               real(line.gap), imag(line.gap), abs(line.gap),
               real(line.weight), imag(line.weight), abs(line.weight)))
    end
    _write_csv(joinpath(output_dir, "dmnrg_response_transitions.csv"),
               ["iteration", "component_a", "component_b",
                "gap_real", "gap_imag", "gap_abs",
                "weight_real", "weight_imag", "weight_abs"],
               transition_rows)

    quality_rows = Any[]
    for q in response.quality
        push!(quality_rows,
              (q.iteration, String(q.component_a), String(q.component_b),
               q.candidate_count, q.selected_count,
               q.total_abs_weight, q.selected_abs_weight, q.selected_fraction,
               real(q.full_commutator_weight), imag(q.full_commutator_weight),
               real(q.selected_commutator_weight),
               imag(q.selected_commutator_weight)))
    end
    _write_csv(joinpath(output_dir, "dmnrg_response_quality.csv"),
               ["iteration", "component_a", "component_b",
                "candidate_count", "selected_count", "total_abs_weight",
                "selected_abs_weight", "selected_fraction",
                "full_commutator_real", "full_commutator_imag",
                "selected_commutator_real", "selected_commutator_imag"],
               quality_rows)

    if response.response_computed
        pairs = Tuple{Symbol,Symbol}[(a,b) for a in p.components for b in p.components]
        header = String["omega"]
        for (a,b) in pairs
            push!(header, "ReChi_$(a)$(b)")
            push!(header, "ImChi_$(a)$(b)")
            push!(header, "minus_ImChi_$(a)$(b)_over_pi")
        end
        rows = Any[]
        for iw in eachindex(response.omega)
            row = Any[response.omega[iw]]
            for pair in pairs
                value = response.spectra[pair][iw]
                append!(row, (real(value), imag(value), -imag(value)/pi))
            end
            push!(rows, Tuple(row))
        end
        _write_csv(joinpath(output_dir, "dmnrg_response_spectra.csv"),
                   header, rows)

        time_header = String["time"]
        for (a,b) in pairs
            push!(time_header, "ReChi_$(a)$(b)")
            push!(time_header, "ImChi_$(a)$(b)")
        end
        time_rows = Any[]
        for it in eachindex(response.time)
            row = Any[response.time[it]]
            for pair in pairs
                value = response.time_response[pair][it]
                append!(row, (real(value), imag(value)))
            end
            push!(time_rows, Tuple(row))
        end
        _write_csv(joinpath(output_dir, "dmnrg_response_time.csv"),
                   time_header, time_rows)
    end

    open(joinpath(output_dir, "dmnrg_response_summary.toml"), "w") do io
        println(io, "method = \"zero-temperature complete-basis density-matrix NRG response\"")
        println(io, "peak_model_imposed = false")
        println(io, "ep_functional_form_imposed = false")
        println(io, "equilibrium_gate = $(response.equilibrium_gate)")
        println(io, "response_computed = $(response.response_computed)")
        println(io, "exploratory_complex = $(p.exploratory_complex)")
        println(io, "max_centered_imaginary_energy = $(_toml_float(response.max_centered_imaginary_energy))")
        println(io, "ground_energy_real = $(_toml_float(real(response.ground_energy)))")
        println(io, "ground_energy_imag = $(_toml_float(imag(response.ground_energy)))")
        println(io, "density_trace_error = $(_toml_float(response.density_trace_error))")
        println(io, "eta = $(_toml_float(p.eta))")
        println(io, "transition_count = $(length(response.transitions))")
        model = response.nrg_result.config.model
        nrg = response.nrg_result.config.nrg
        println(io, "beta0 = $(_toml_float(model.beta0))")
        println(io, "U = $(_toml_float(model.U))")
        println(io, "soc_mode = \"$(model.soc_mode)\"")
        println(io, "soc_lambda = $(_toml_float(model.soc_lambda))")
        println(io, "soc_kmax = $(_toml_float(model.soc_kmax))")
        println(io, "soc_ratio = $(_toml_float(model.soc_lambda/model.soc_kmax))")
        println(io, "z_shift = $(_toml_float(nrg.z_shift))")
        println(io, "Lambda = $(_toml_float(nrg.Lambda))")
        println(io, "iterations = $(nrg.iterations)")
        println(io, "nkeep = $(nrg.nkeep)")
        quoted_components = join(["\"$(x)\"" for x in p.components], ", ")
        println(io, "components = [$quoted_components]")
    end
    return output_dir
end

end # module
