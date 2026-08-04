module PTDiracNHNRG

using LinearAlgebra
using SparseArrays
using Printf
using TOML
using DelimitedFiles
using Statistics

export ModelParams, NRGParams, LehmannParams, ThermoParams, RunConfig,
       load_config, run_nrg, run_scan, write_run_outputs,
       build_impurity_hamiltonian, logarithmic_star, star_to_chain,
       kept_space_lehmann, complete_basis_thermodynamics, FDMThermoResult,
       physical_hybridization, soc_form_factor,
       soc_overlap_matrix, impurity_hybridization_matrix

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

Base.@kwdef struct ModelParams
    U::Float64 = 2.0
    eps_d::Float64 = -1.0
    delta0::Float64 = 0.075
    c_delta::Float64 = 0.050
    g_gamma::Float64 = 0.200
    beta0::Float64 = 0.490
    delta_coh::Float64 = 1.0e-3
    gamma_common::Float64 = 0.120
    frame::Symbol = :relative          # :relative or :passive
    bandwidth::Float64 = pi / 4
    bath_exponent::Float64 = 1.0       # pseudogap exponent r
    gamma_edge::Float64 = 0.120        # Γ(|ω|=D)
    hybridization_V::Float64 = NaN     # direct override if finite
    bath_split::Float64 = 0.0          # optional Wilson-site flavour splitting
    reciprocal_hybridization::Bool = true
    soc_mode::Symbol = :none           # :none or :overlap
    soc_lambda::Float64 = 0.0          # projected SOC strength
    soc_kmax::Float64 = pi / 4         # momentum cutoff entering F(lambda)
end

Base.@kwdef struct NRGParams
    Lambda::Float64 = 3.0
    z_shift::Float64 = 0.5
    iterations::Int = 36
    nkeep::Int = 300
    min_keep_per_charge::Int = 4
    sort_type::Symbol = :LowRe         # :LowRe, :LowMag, :LowReMag
    star_intervals::Int = 120
    overlap_floor::Float64 = 1.0e-10
    degeneracy_tolerance::Float64 = 1.0e-10
    residual_tolerance::Float64 = 1.0e-9
    save_levels::Int = 24
end

Base.@kwdef struct LehmannParams
    enabled::Bool = true
    omega_min::Float64 = -1.0
    omega_max::Float64 = 1.0
    omega_points::Int = 2001
    eta::Float64 = 0.01
end

"""Settings for complete-basis/FDM thermodynamics.

The thermal density matrix is constructed from all states discarded along the
Wilson chain, with the remaining environment represented by its exact local
multiplicity.  A thermodynamic result is emitted only when the complete-basis
spectrum is real within `imag_tolerance`; this excludes the broken-PT and
passive-loss regimes from an equilibrium `T_K` interpretation.
"""
Base.@kwdef struct ThermoParams
    enabled::Bool = false
    temperature_min::Float64 = 1.0e-7
    temperature_max::Float64 = 1.0
    temperature_points::Int = 241
    imag_tolerance::Float64 = 1.0e-8
    entropy_target::Float64 = 0.5 * log(2.0)
    low_entropy_max::Float64 = 0.30 * log(2.0)
    moment_entropy_min::Float64 = 0.65 * log(2.0)
end

Base.@kwdef struct RunConfig
    model::ModelParams = ModelParams()
    nrg::NRGParams = NRGParams()
    lehmann::LehmannParams = LehmannParams()
    thermo::ThermoParams = ThermoParams()
    output_dir::String = "pt_dirac_nhnrg_output"
end

function _get(tbl::AbstractDict, key::String, default)
    return haskey(tbl, key) ? tbl[key] : default
end

function load_config(path::AbstractString)::RunConfig
    raw = TOML.parsefile(path)
    m = get(raw, "model", Dict{String,Any}())
    n = get(raw, "nrg", Dict{String,Any}())
    l = get(raw, "lehmann", Dict{String,Any}())
    t = get(raw, "thermodynamics", Dict{String,Any}())
    o = get(raw, "output", Dict{String,Any}())

    model = ModelParams(
        U = Float64(_get(m, "U", 2.0)),
        eps_d = Float64(_get(m, "eps_d", -1.0)),
        delta0 = Float64(_get(m, "delta0", 0.075)),
        c_delta = Float64(_get(m, "c_delta", 0.050)),
        g_gamma = Float64(_get(m, "g_gamma", 0.200)),
        beta0 = Float64(_get(m, "beta0", 0.490)),
        delta_coh = Float64(_get(m, "delta_coh", 1.0e-3)),
        gamma_common = Float64(_get(m, "gamma_common", 0.120)),
        frame = Symbol(_get(m, "frame", "relative")),
        bandwidth = Float64(_get(m, "bandwidth", pi / 4)),
        bath_exponent = Float64(_get(m, "bath_exponent", 1.0)),
        gamma_edge = Float64(_get(m, "gamma_edge", 0.120)),
        hybridization_V = Float64(_get(m, "hybridization_V", NaN)),
        bath_split = Float64(_get(m, "bath_split", 0.0)),
        reciprocal_hybridization = Bool(_get(m, "reciprocal_hybridization", true)),
        soc_mode = Symbol(_get(m, "soc_mode", "none")),
        soc_lambda = Float64(_get(m, "soc_lambda", 0.0)),
        soc_kmax = Float64(_get(m, "soc_kmax", pi / 4)),
    )

    nrg = NRGParams(
        Lambda = Float64(_get(n, "Lambda", 3.0)),
        z_shift = Float64(_get(n, "z_shift", 0.5)),
        iterations = Int(_get(n, "iterations", 36)),
        nkeep = Int(_get(n, "nkeep", 300)),
        min_keep_per_charge = Int(_get(n, "min_keep_per_charge", 4)),
        sort_type = Symbol(_get(n, "sort_type", "LowRe")),
        star_intervals = Int(_get(n, "star_intervals", 120)),
        overlap_floor = Float64(_get(n, "overlap_floor", 1.0e-10)),
        degeneracy_tolerance = Float64(_get(n, "degeneracy_tolerance", 1.0e-10)),
        residual_tolerance = Float64(_get(n, "residual_tolerance", 1.0e-9)),
        save_levels = Int(_get(n, "save_levels", 24)),
    )

    lehmann = LehmannParams(
        enabled = Bool(_get(l, "enabled", true)),
        omega_min = Float64(_get(l, "omega_min", -1.0)),
        omega_max = Float64(_get(l, "omega_max", 1.0)),
        omega_points = Int(_get(l, "omega_points", 2001)),
        eta = Float64(_get(l, "eta", 0.01)),
    )

    thermo = ThermoParams(
        enabled = Bool(_get(t, "enabled", false)),
        temperature_min = Float64(_get(t, "temperature_min", 1.0e-7)),
        temperature_max = Float64(_get(t, "temperature_max", 1.0)),
        temperature_points = Int(_get(t, "temperature_points", 241)),
        imag_tolerance = Float64(_get(t, "imag_tolerance", 1.0e-8)),
        entropy_target = Float64(_get(t, "entropy_target", 0.5 * log(2.0))),
        low_entropy_max = Float64(_get(t, "low_entropy_max", 0.30 * log(2.0))),
        moment_entropy_min = Float64(_get(t, "moment_entropy_min", 0.65 * log(2.0))),
    )

    return RunConfig(
        model=model,
        nrg=nrg,
        lehmann=lehmann,
        thermo=thermo,
        output_dir=String(_get(o, "directory", "pt_dirac_nhnrg_output")),
    )
end

# -----------------------------------------------------------------------------
# Local fermion algebra
# -----------------------------------------------------------------------------

const NFLAV = 2
const LOCAL_DIM = 4

function annihilation_operator(norb::Int, orb::Int)::Matrix{ComplexF64}
    @assert 1 <= orb <= norb
    dim = 1 << norb
    op = zeros(ComplexF64, dim, dim)
    bit = 1 << (orb - 1)
    lower_mask = bit - 1
    for ket0 in 0:(dim - 1)
        if (ket0 & bit) != 0
            bra0 = ket0 ⊻ bit
            parity = count_ones(ket0 & lower_mask)
            op[bra0 + 1, ket0 + 1] = iseven(parity) ? 1.0 : -1.0
        end
    end
    return op
end

const CLOCAL = (annihilation_operator(2, 1), annihilation_operator(2, 2))
const NLOCAL = (adjoint(CLOCAL[1]) * CLOCAL[1], adjoint(CLOCAL[2]) * CLOCAL[2])
const ILOCAL = Matrix{ComplexF64}(I, LOCAL_DIM, LOCAL_DIM)
const QLOCAL = Int[count_ones(x) for x in 0:(LOCAL_DIM - 1)]

function parity_from_charges(charges::Vector{Int})::Matrix{ComplexF64}
    return Diagonal(ComplexF64[iseven(q) ? 1.0 : -1.0 for q in charges]) |> Matrix
end

function physical_hybridization(model::ModelParams)::Float64
    if isfinite(model.hybridization_V)
        return model.hybridization_V
    end
    r = model.bath_exponent
    D = model.bandwidth
    # Normalized rho(ω)=(r+1)/(2D)|ω/D|^r.  This choice gives
    # Γ(±D)=π V² (r+1)/(2D)=gamma_edge.
    return sqrt(2.0 * D * model.gamma_edge / (pi * (r + 1.0)))
end

function _soc_ratio(model::ModelParams)::Float64
    model.soc_kmax > 0.0 || error("soc_kmax must be positive")
    return clamp(model.soc_lambda / model.soc_kmax, -1.0, 1.0)
end

"""Projected SOC overlap factor F(lambda)=max(1-(lambda/kmax)^2,0)."""
function soc_form_factor(model::ModelParams)::Float64
    x = _soc_ratio(model)
    return max(1.0 - x^2, 0.0)
end

"""
Normalized two-channel overlap matrix.  In :overlap mode its determinant is
exactly F(lambda), while Tr(S_lambda)=2 is independent of lambda.  This is a
minimal low-energy embedding of the projected chiral-overlap factor; it is not
a momentum-resolved block-Lanczos representation of the full SOC dispersion.
"""
function soc_overlap_matrix(model::ModelParams)::Matrix{ComplexF64}
    if model.soc_mode == :none
        return Matrix{ComplexF64}(I, NFLAV, NFLAV)
    elseif model.soc_mode == :overlap
        x = _soc_ratio(model)
        return ComplexF64[1.0 x; x 1.0]
    end
    error("Unknown soc_mode $(model.soc_mode); use none or overlap")
end

"""
Impurity-to-first-Wilson-site coupling matrix V_lambda=V sqrt(S_lambda).
Consequently V_lambda*V_lambda' / V^2 = S_lambda and the normalized
hybridization determinant equals F(lambda).
"""
function impurity_hybridization_matrix(model::ModelParams)::Matrix{ComplexF64}
    V = physical_hybridization(model)
    S = soc_overlap_matrix(model)
    eigs = eigen(Hermitian(S))
    vals = max.(real.(eigs.values), 0.0)
    sqrtS = eigs.vectors * Diagonal(sqrt.(vals)) * adjoint(eigs.vectors)
    return Matrix{ComplexF64}(V .* sqrtS)
end

function projected_channels(model::ModelParams)
    delta_eff = model.delta0 + model.c_delta * model.beta0
    gamma_pt = model.g_gamma * model.beta0
    return delta_eff, gamma_pt
end

function build_impurity_hamiltonian(model::ModelParams)::Matrix{ComplexF64}
    n1, n2 = NLOCAL
    d1, d2 = CLOCAL
    delta_eff, gamma_pt = projected_channels(model)
    mix = complex(model.delta_coh, gamma_pt)

    h = complex(model.eps_d) .* (n1 + n2)
    h .+= complex(model.U) .* (n1 * n2)
    h .+= complex(delta_eff) .* (n1 - n2)
    # Complex-symmetric local channel conversion, matching the rotated kernel.
    h .+= mix .* (adjoint(d1) * d2 + adjoint(d2) * d1)

    if model.frame == :passive
        if model.gamma_common + 1.0e-14 < abs(gamma_pt)
            error("Passive frame requires gamma_common >= |gamma_pt|; got $(model.gamma_common) < $(abs(gamma_pt))")
        end
        h .+= -1im * model.gamma_common .* (n1 + n2)
    elseif model.frame != :relative
        error("Unknown frame $(model.frame); use relative or passive")
    end
    return Matrix{ComplexF64}(h)
end

function site_hamiltonian(eps_common::Real, eps_split::Real=0.0)::Matrix{ComplexF64}
    n1, n2 = NLOCAL
    return Matrix{ComplexF64}(complex(eps_common) .* (n1 + n2) + complex(eps_split) .* (n1 - n2))
end

# -----------------------------------------------------------------------------
# Logarithmic discretization and numerical star-to-chain transform
# -----------------------------------------------------------------------------

"""
Return star energies and normalized weights for the particle-hole-symmetric
power-law DOS rho(ω)=(r+1)/(2D)|ω/D|^r on [-D,D].
"""
function logarithmic_star(Lambda::Float64, z::Float64, r::Float64,
                          D::Float64, nintervals::Int)
    Lambda > 1.0 || error("Lambda must exceed one")
    0.0 < z <= 1.0 || error("z_shift must lie in (0,1]")
    r > -1.0 || error("bath exponent must exceed -1")
    energies = Float64[]
    weights = Float64[]
    for n in 0:(nintervals - 1)
        # Standard z-discretized intervals: the first interval reaches the
        # band edge, and later intervals tile logarithmically toward zero.
        if n == 0
            b = 1.0
            a = Lambda^(-z)
        else
            b = Lambda^(-(n - 1 + z))
            a = Lambda^(-(n + z))
        end
        w = 0.5 * (b^(r + 1.0) - a^(r + 1.0))
        numerator = b^(r + 2.0) - a^(r + 2.0)
        denominator = b^(r + 1.0) - a^(r + 1.0)
        xi = D * (r + 1.0) / (r + 2.0) * numerator / denominator
        push!(energies, +xi); push!(weights, w)
        push!(energies, -xi); push!(weights, w)
    end
    # Normalize the retained star exactly. The omitted tail is exponentially small.
    weights ./= sum(weights)
    return energies, weights
end

"""
Lanczos tridiagonalization of a diagonal star Hamiltonian. Full
reorthogonalization is used because logarithmic star energies become clustered.
Returns onsite energies alpha[1:nsites] and hoppings beta[1:nsites-1].
"""
function star_to_chain(energies::Vector{Float64}, weights::Vector{Float64},
                       nsites::Int)
    length(energies) == length(weights) || error("star arrays have unequal lengths")
    nsites <= length(energies) || error("need at least nsites star orbitals")
    q = sqrt.(weights)
    q ./= norm(q)
    qprev = zeros(Float64, length(q))
    basis = Vector{Vector{Float64}}()
    push!(basis, copy(q))
    alpha = zeros(Float64, nsites)
    beta = zeros(Float64, max(nsites - 1, 0))
    beta_prev = 0.0

    for n in 1:nsites
        hq = energies .* q
        alpha[n] = dot(q, hq)
        residual = hq .- alpha[n] .* q
        if n > 1
            residual .-= beta_prev .* qprev
        end
        # Full reorthogonalization; two passes improve stability.
        for _ in 1:2
            for v in basis
                residual .-= dot(v, residual) .* v
            end
        end
        if n < nsites
            bn = norm(residual)
            if !(isfinite(bn) && bn > 100 * eps(Float64))
                error("star-to-chain Lanczos terminated at site $n with beta=$bn; reduce iterations or use fewer decades")
            end
            beta[n] = bn
            qprev = q
            q = residual ./ bn
            push!(basis, copy(q))
            beta_prev = bn
        end
    end
    return alpha, beta
end

function make_chain(config::RunConfig)
    m, n = config.model, config.nrg
    nsites = n.iterations + 1
    e, w = logarithmic_star(n.Lambda, n.z_shift, m.bath_exponent,
                            m.bandwidth, n.star_intervals)
    alpha, beta = star_to_chain(e, w, nsites)
    return alpha, beta
end

# -----------------------------------------------------------------------------
# Non-Hermitian eigensystem and truncation
# -----------------------------------------------------------------------------

struct EigBlock
    values::Vector{ComplexF64}
    left::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    max_residual::Float64
    biorth_error::Float64
    min_pair_overlap::Float64
end

function _match_left_vectors(evals::Vector{ComplexF64}, evals_left::Vector{ComplexF64})
    unused = collect(eachindex(evals_left))
    match = zeros(Int, length(evals))
    for j in eachindex(evals)
        distances = [abs(evals_left[k] - conj(evals[j])) for k in unused]
        pos = argmin(distances)
        match[j] = unused[pos]
        deleteat!(unused, pos)
    end
    return match
end

"""
Connected components of eigenvalues that are indistinguishable at the declared
backward-error tolerance. Mixing is allowed only inside such a component, so
true non-degenerate eigenvectors are never altered.
"""
function _degenerate_clusters(evals::Vector{ComplexF64}, atol::Float64)
    n = length(evals)
    parent = collect(1:n)

    function root(i::Int)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        return i
    end

    function unite(i::Int, j::Int)
        ri, rj = root(i), root(j)
        ri == rj && return
        parent[rj] = ri
    end

    for i in 1:n-1, j in i+1:n
        if abs(evals[i] - evals[j]) <= atol
            unite(i, j)
        end
    end

    groups = Dict{Int,Vector{Int}}()
    for i in 1:n
        push!(get!(groups, root(i), Int[]), i)
    end
    return collect(values(groups))
end

"""
Left/right eigensystem with subspace biorthogonalization.

For an exactly or numerically degenerate but diagonalizable multiplet, LAPACK is
free to return unrelated bases for the right and left invariant subspaces.
Pairwise normalization can then produce a spurious zero <L|R>. We instead form
S=L^dagger R inside each degenerate multiplet and use its SVD to construct dual
bases. A genuinely defective subspace is detected by a vanishing smallest
singular value of S.
"""
function lr_eigensystem(A::Matrix{ComplexF64}, overlap_floor::Float64,
                        degeneracy_tolerance::Float64)::EigBlock
    fr = eigen(A)
    fl = eigen(adjoint(A))
    evals = ComplexF64.(fr.values)
    Rraw = Matrix{ComplexF64}(fr.vectors)
    match = _match_left_vectors(evals, ComplexF64.(fl.values))
    Lraw = Matrix{ComplexF64}(fl.vectors[:, match])

    scale = max(1.0, opnorm(A, Inf), maximum(abs, evals; init=0.0))
    atol = degeneracy_tolerance * scale
    clusters = _degenerate_clusters(evals, atol)

    R = similar(Rraw)
    L = similar(Lraw)
    min_overlap = Inf

    for cluster in clusters
        Rc = Rraw[:, cluster]
        Lc = Lraw[:, cluster]
        overlap = adjoint(Lc) * Rc
        F = svd(overlap)
        sigma_min = minimum(F.S)
        min_overlap = min(min_overlap, sigma_min)
        if sigma_min < overlap_floor
            center = sum(evals[cluster]) / length(cluster)
            error("near-defective invariant subspace: sigma_min(L'R)=$(sigma_min) " *
                  "below floor=$(overlap_floor), multiplicity=$(length(cluster)), " *
                  "eigenvalue_center=$(center)")
        end

        invsqrt = Diagonal(ComplexF64.(1.0 ./ sqrt.(F.S)))
        V = adjoint(F.Vt)
        R[:, cluster] .= Rc * V * invsqrt
        L[:, cluster] .= Lc * F.U * invsqrt
    end

    maxres = 0.0
    for j in eachindex(evals)
        maxres = max(maxres, norm(A * R[:, j] - evals[j] * R[:, j]))
        maxres = max(maxres, norm(adjoint(A) * L[:, j] - conj(evals[j]) * L[:, j]))
    end
    bio = norm(adjoint(L) * R - I, Inf)
    return EigBlock(evals, L, R, maxres, bio, min_overlap)
end

function _sort_key(value::ComplexF64, reference::ComplexF64, sort_type::Symbol)
    x = value - reference
    if sort_type == :LowRe
        return (real(x), abs(imag(x)), abs(x))
    elseif sort_type == :LowMag
        return (abs(x), real(x), abs(imag(x)))
    elseif sort_type == :LowReMag
        return (real(x), abs(x), abs(imag(x)))
    end
    error("unknown sort type $sort_type")
end

mutable struct NRGState
    iteration::Int
    energies::Vector{ComplexF64}
    charges::Vector{Int}
    # In a nonunitary biorthogonal basis, the matrix of c† is not the
    # ordinary adjoint of the matrix of c.  Propagate both independently:
    #   c_RL  = L† c  R,
    #   c†_RL = L† c† R.
    last_ann::NTuple{2,Matrix{ComplexF64}}
    last_cre::NTuple{2,Matrix{ComplexF64}}
    impurity_ann::NTuple{2,Matrix{ComplexF64}}
    impurity_cre::NTuple{2,Matrix{ComplexF64}}
    # Absolute physical reference accumulated from the shell-by-shell energy
    # shifts.  It is required to place discarded states from different shells
    # on one common energy axis for complete-basis/FDM thermodynamics.
    energy_offset::ComplexF64
    discarded_physical::Vector{ComplexF64}
    discarded_charges::Vector{Int}
    max_residual::Float64
    biorth_error::Float64
    min_pair_overlap::Float64
end

const ResidueMatrix4 = NTuple{4,ComplexF64}  # (11, 12, 21, 22)

struct IterationRecord
    iteration::Int
    scale::Float64
    kept::Int
    ground_charge::Int
    max_residual::Float64
    biorth_error::Float64
    min_pair_overlap::Float64
    levels::Vector{Tuple{Int,Int,ComplexF64}}
    # Highest impurity addition/removal transition weights at this iteration.
    # Tuple fields:
    # (charge, support_rank, energy, add_abs, rem_abs, total_abs,
    #  add_residue, rem_residue)
    transition_levels::Vector{
        Tuple{Int,Int,ComplexF64,Float64,Float64,Float64,ComplexF64,ComplexF64}
    }
    # Full channel-resolved residue matrices for Jordan/quartic diagnostics.
    # Tuple fields:
    # (charge, matrix_rank, energy, matrix_support_abs,
    #  addition_matrix=(11,12,21,22), removal_matrix=(11,12,21,22))
    transition_matrix_levels::Vector{
        Tuple{Int,Int,ComplexF64,Float64,ResidueMatrix4,ResidueMatrix4}
    }
end

function coupling_term(old_ann::NTuple{2,Matrix{ComplexF64}},
                       old_cre::NTuple{2,Matrix{ComplexF64}},
                       old_charges::Vector{Int},
                       coupling::Matrix{ComplexF64})
    nold = length(old_charges)
    Pold = parity_from_charges(old_charges)
    result = zeros(ComplexF64, nold * LOCAL_DIM, nold * LOCAL_DIM)
    for a in 1:NFLAV, b in 1:NFLAV
        t = coupling[a, b]
        if t != 0
            # Old modes precede the new site in the fermionic ordering.
            # IMPORTANT: after a nonunitary left/right basis change,
            # L† c† R != (L† c R)†.  Use the independently transformed
            # creation matrix rather than adjoint(old_ann[a]).
            result .+= t .* kron(old_cre[a] * Pold, CLOCAL[b])
            result .+= conj(t) .* kron(Pold * old_ann[a], adjoint(CLOCAL[b]))
        end
    end
    return result
end

function _select_kept(evals::Vector{ComplexF64}, charges::Vector{Int},
                      nrg::NRGParams)
    reference_index = argmin([(real(e), abs(imag(e)), abs(e)) for e in evals])
    reference = evals[reference_index]
    order = sortperm(eachindex(evals), by=i -> _sort_key(evals[i], reference, nrg.sort_type))
    # Reserve a small number of low states in every charge sector, then fill
    # the remaining budget in the global complex-energy order.
    mandatory = Int[]
    if nrg.min_keep_per_charge > 0
        for q in unique(charges)
            qorder = [i for i in order if charges[i] == q]
            append!(mandatory, qorder[1:min(nrg.min_keep_per_charge, length(qorder))])
        end
    end
    mandatory = unique(mandatory)
    length(mandatory) <= nrg.nkeep || error(
        "min_keep_per_charge reserves $(length(mandatory)) states, exceeding nkeep=$(nrg.nkeep)"
    )
    selected = Set(mandatory)
    for i in order
        length(selected) >= min(nrg.nkeep, length(order)) && break
        push!(selected, i)
    end

    # Do not cut through a numerically exact multiplet within one charge sector.
    # The kept count may exceed nkeep by at most the size of the boundary multiplet.
    scale = max(1.0, maximum(abs, evals; init=0.0))
    atol = nrg.degeneracy_tolerance * scale
    for q in unique(charges)
        idx = findall(==(q), charges)
        local_clusters = _degenerate_clusters(evals[idx], atol)
        for local_cluster in local_clusters
            global_cluster = idx[local_cluster]
            if any(i -> i in selected, global_cluster)
                union!(selected, global_cluster)
            end
        end
    end

    rank = Dict(i => pos for (pos, i) in enumerate(order))
    keep = sort!(collect(selected), by=i -> rank[i])
    return keep, reference
end

function _diagonalize_by_charge_complete(H::Matrix{ComplexF64}, charges::Vector{Int}, nrg::NRGParams)
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
        eigb = lr_eigensystem(block, nrg.overlap_floor, nrg.degeneracy_tolerance)
        nb = length(idx)
        cols = (cursor + 1):(cursor + nb)
        all_values = vcat(all_values, eigb.values)
        append!(all_charges, fill(q, nb))
        all_left[idx, cols] .= eigb.left
        all_right[idx, cols] .= eigb.right
        cursor += nb
        maxres = max(maxres, eigb.max_residual)
        maxbio = max(maxbio, eigb.biorth_error)
        minoverlap = min(minoverlap, eigb.min_pair_overlap)
    end

    keep, reference = _select_kept(all_values, all_charges, nrg)
    keep_set = Set(keep)
    discarded = Int[i for i in eachindex(all_values) if !(i in keep_set)]
    values = all_values[keep] .- reference
    qkeep = all_charges[keep]
    discarded_values = all_values[discarded] .- reference
    qdiscarded = all_charges[discarded]
    Lk = all_left[:, keep]
    Rk = all_right[:, keep]
    return values, qkeep, Lk, Rk, maxres, maxbio, minoverlap,
           reference, discarded_values, qdiscarded
end

# Backward-compatible public helper used by the existing regression tests.
function diagonalize_by_charge(H::Matrix{ComplexF64}, charges::Vector{Int},
                               nrg::NRGParams)
    out = _diagonalize_by_charge_complete(H, charges, nrg)
    return out[1], out[2], out[3], out[4], out[5], out[6], out[7]
end

function transform_operator(Lk::Matrix{ComplexF64}, O::Matrix{ComplexF64},
                            Rk::Matrix{ComplexF64})
    return Matrix{ComplexF64}(adjoint(Lk) * O * Rk)
end

function add_site(old::NRGState, site_index::Int, alpha::Float64, hopping::Float64,
                  config::RunConfig)::NRGState
    nrg, model = config.nrg, config.model
    D = model.bandwidth
    omega_old = D * nrg.Lambda^(-old.iteration / 2)
    omega_new = D * nrg.Lambda^(-(old.iteration + 1) / 2)
    ratio = omega_old / omega_new
    nold = length(old.energies)

    H = ratio .* kron(Diagonal(old.energies) |> Matrix, ILOCAL)
    H .+= kron(Matrix{ComplexF64}(I, nold, nold),
               site_hamiltonian(alpha / omega_new, model.bath_split / omega_new))
    T = Matrix{ComplexF64}(I, NFLAV, NFLAV) .* (hopping / omega_new)
    H .+= coupling_term(old.last_ann, old.last_cre, old.charges, T)

    full_charges = Int[qold + qsite for qold in old.charges for qsite in QLOCAL]
    (values, qkeep, Lk, Rk, maxres, maxbio, minoverlap,
     reference, discarded_scaled, qdiscarded) =
        _diagonalize_by_charge_complete(H, full_charges, nrg)

    Pold = parity_from_charges(old.charges)
    last_ann_full = ntuple(a -> kron(Pold, CLOCAL[a]), NFLAV)
    last_cre_full = ntuple(a -> kron(Pold, adjoint(CLOCAL[a])), NFLAV)
    imp_ann_full = ntuple(a -> kron(old.impurity_ann[a], ILOCAL), NFLAV)
    imp_cre_full = ntuple(a -> kron(old.impurity_cre[a], ILOCAL), NFLAV)

    last_ann_new = ntuple(a -> transform_operator(Lk, last_ann_full[a], Rk), NFLAV)
    last_cre_new = ntuple(a -> transform_operator(Lk, last_cre_full[a], Rk), NFLAV)
    imp_ann_new = ntuple(a -> transform_operator(Lk, imp_ann_full[a], Rk), NFLAV)
    imp_cre_new = ntuple(a -> transform_operator(Lk, imp_cre_full[a], Rk), NFLAV)

    energy_offset = old.energy_offset + omega_new * reference
    discarded_physical = ComplexF64[
        energy_offset + omega_new * x for x in discarded_scaled
    ]
    return NRGState(site_index, values, qkeep, last_ann_new, last_cre_new,
                    imp_ann_new, imp_cre_new, energy_offset,
                    discarded_physical, qdiscarded,
                    maxres, maxbio, minoverlap)
end

function initialize_nrg(config::RunConfig, alpha0::Float64)::NRGState
    nrg, model = config.nrg, config.model
    D = model.bandwidth
    Himp = build_impurity_hamiltonian(model) ./ D
    H = kron(Himp, ILOCAL)
    H .+= kron(ILOCAL, site_hamiltonian(alpha0 / D, model.bath_split / D))

    Vmat = impurity_hybridization_matrix(model) ./ D
    local_cre = ntuple(a -> Matrix{ComplexF64}(adjoint(CLOCAL[a])), NFLAV)
    H .+= coupling_term(CLOCAL, local_cre, QLOCAL, Vmat)

    full_charges = Int[qimp + qsite for qimp in QLOCAL for qsite in QLOCAL]
    (values, qkeep, Lk, Rk, maxres, maxbio, minoverlap,
     reference, discarded_scaled, qdiscarded) =
        _diagonalize_by_charge_complete(H, full_charges, nrg)

    Pimp = parity_from_charges(QLOCAL)
    last_ann_full = ntuple(a -> kron(Pimp, CLOCAL[a]), NFLAV)
    last_cre_full = ntuple(a -> kron(Pimp, adjoint(CLOCAL[a])), NFLAV)
    imp_ann_full = ntuple(a -> kron(CLOCAL[a], ILOCAL), NFLAV)
    imp_cre_full = ntuple(a -> kron(adjoint(CLOCAL[a]), ILOCAL), NFLAV)

    last_ann_new = ntuple(a -> transform_operator(Lk, last_ann_full[a], Rk), NFLAV)
    last_cre_new = ntuple(a -> transform_operator(Lk, last_cre_full[a], Rk), NFLAV)
    imp_ann_new = ntuple(a -> transform_operator(Lk, imp_ann_full[a], Rk), NFLAV)
    imp_cre_new = ntuple(a -> transform_operator(Lk, imp_cre_full[a], Rk), NFLAV)

    energy_offset = D * reference
    discarded_physical = ComplexF64[
        energy_offset + D * x for x in discarded_scaled
    ]
    return NRGState(0, values, qkeep, last_ann_new, last_cre_new,
                    imp_ann_new, imp_cre_new, energy_offset,
                    discarded_physical, qdiscarded,
                    maxres, maxbio, minoverlap)
end

function iteration_record(state::NRGState, config::RunConfig)::IterationRecord
    n = config.nrg
    D = config.model.bandwidth
    scale = D * n.Lambda^(-state.iteration / 2)
    ground = argmin([(real(e), abs(imag(e)), abs(e)) for e in state.energies])
    ground_charge = state.charges[ground]

    # Standard low-complex-energy level flow.
    order = sortperm(eachindex(state.energies),
                     by=i -> _sort_key(state.energies[i], 0.0 + 0im, n.sort_type))
    levels = Tuple{Int,Int,ComplexF64}[]
    counters = Dict{Int,Int}()
    for i in order[1:min(n.save_levels, length(order))]
        q = state.charges[i]
        ordinal = get(counters, q, 0)
        counters[q] = ordinal + 1
        push!(levels, (q, ordinal, state.energies[i]))
    end

    # Basis-rescaling-invariant impurity transition residues.  The products
    # <L0|d|Ri><Li|d†|R0> and <L0|d†|Ri><Li|d|R0> are invariant under
    # reciprocal left/right eigenvector rescaling.  We rank states by the sum
    # of absolute channel-resolved residues, not by a possibly cancelling sum.
    support_candidates =
        Tuple{Int,ComplexF64,Float64,Float64,Float64,ComplexF64,ComplexF64}[]
    for i in eachindex(state.energies)
        q = state.charges[i]
        add_abs = 0.0
        rem_abs = 0.0
        add_residue = 0.0 + 0.0im
        rem_residue = 0.0 + 0.0im

        if q == ground_charge + 1
            for a in 1:NFLAV
                residue = state.impurity_ann[a][ground, i] *
                          state.impurity_cre[a][i, ground]
                add_abs += abs(residue)
                add_residue += residue
            end
        elseif q == ground_charge - 1
            for a in 1:NFLAV
                residue = state.impurity_cre[a][ground, i] *
                          state.impurity_ann[a][i, ground]
                rem_abs += abs(residue)
                rem_residue += residue
            end
        end

        total_abs = add_abs + rem_abs
        if total_abs > 0.0
            push!(support_candidates,
                  (q, state.energies[i], add_abs, rem_abs, total_abs,
                   add_residue, rem_residue))
        end
    end

    sort!(support_candidates, by=x -> (-x[5], real(x[2]), abs(imag(x[2]))))
    transition_levels =
        Tuple{Int,Int,ComplexF64,Float64,Float64,Float64,ComplexF64,ComplexF64}[]
    for (rank, item) in enumerate(
            support_candidates[1:min(n.save_levels, length(support_candidates))])
        q, e, add_abs, rem_abs, total_abs, add_residue, rem_residue = item
        push!(transition_levels,
              (q, rank - 1, e, add_abs, rem_abs, total_abs,
               add_residue, rem_residue))
    end

    # Independently rank states by the full 2x2 residue-matrix support.  This
    # preserves the historical transition_weights ranking while exposing the
    # off-diagonal EP-active channels needed for pairwise Jordan recombination.
    zero4 = (0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im)
    matrix_candidates =
        Tuple{Int,ComplexF64,Float64,ResidueMatrix4,ResidueMatrix4}[]
    for i in eachindex(state.energies)
        q = state.charges[i]
        add_matrix = zeros(ComplexF64, NFLAV, NFLAV)
        rem_matrix = zeros(ComplexF64, NFLAV, NFLAV)

        if q == ground_charge + 1
            for a in 1:NFLAV, b in 1:NFLAV
                add_matrix[a, b] = state.impurity_ann[a][ground, i] *
                                   state.impurity_cre[b][i, ground]
            end
        elseif q == ground_charge - 1
            for a in 1:NFLAV, b in 1:NFLAV
                rem_matrix[a, b] = state.impurity_cre[b][ground, i] *
                                   state.impurity_ann[a][i, ground]
            end
        end

        add4 = (add_matrix[1,1], add_matrix[1,2],
                add_matrix[2,1], add_matrix[2,2])
        rem4 = (rem_matrix[1,1], rem_matrix[1,2],
                rem_matrix[2,1], rem_matrix[2,2])
        matrix_support = sum(abs, add4) + sum(abs, rem4)
        if matrix_support > 0.0
            push!(matrix_candidates,
                  (q, state.energies[i], matrix_support, add4, rem4))
        end
    end

    sort!(matrix_candidates, by=x -> (-x[3], real(x[2]), abs(imag(x[2]))))
    transition_matrix_levels =
        Tuple{Int,Int,ComplexF64,Float64,ResidueMatrix4,ResidueMatrix4}[]
    for (rank, item) in enumerate(
            matrix_candidates[1:min(n.save_levels, length(matrix_candidates))])
        q, e, matrix_support, add4, rem4 = item
        push!(transition_matrix_levels,
              (q, rank - 1, e, matrix_support, add4, rem4))
    end

    return IterationRecord(state.iteration, scale, length(state.energies),
                           ground_charge, state.max_residual,
                           state.biorth_error, state.min_pair_overlap,
                           levels, transition_levels, transition_matrix_levels)
end

struct DiscardedShell
    iteration::Int
    scale::Float64
    energies::Vector{ComplexF64}
    charges::Vector{Int}
end

struct NRGResult
    config::RunConfig
    state::NRGState
    records::Vector{IterationRecord}
    chain_alpha::Vector{Float64}
    chain_beta::Vector{Float64}
    discarded_shells::Vector{DiscardedShell}
end

function run_nrg(config::RunConfig)::NRGResult
    alpha, beta = make_chain(config)
    state = initialize_nrg(config, alpha[1])
    records = IterationRecord[iteration_record(state, config)]
    discarded_shells = DiscardedShell[
        DiscardedShell(0, records[1].scale,
                       copy(state.discarded_physical), copy(state.discarded_charges))
    ]
    if state.max_residual > config.nrg.residual_tolerance
        @warn "initial eigensolver residual exceeds tolerance" residual=state.max_residual
    end
    for site in 1:config.nrg.iterations
        state = add_site(state, site, alpha[site + 1], beta[site], config)
        push!(records, iteration_record(state, config))
        push!(discarded_shells,
              DiscardedShell(site, records[end].scale,
                             copy(state.discarded_physical),
                             copy(state.discarded_charges)))
        if state.max_residual > config.nrg.residual_tolerance
            @warn "eigensolver residual exceeds tolerance" iteration=site residual=state.max_residual
        end
        @printf("iteration=%3d kept=%4d scale=%10.3e residual=%8.2e bio=%8.2e minLR=%8.2e\n",
                site, length(state.energies), records[end].scale,
                state.max_residual, state.biorth_error, state.min_pair_overlap)
    end
    return NRGResult(config, state, records, alpha, beta, discarded_shells)
end

# -----------------------------------------------------------------------------
# Complete-basis / full-density-matrix thermodynamics
# -----------------------------------------------------------------------------

# TOML 1.0 requires lowercase nan/inf tokens, whereas Julia prints NaN/Inf.
function _toml_float(x::Real)::String
    y = Float64(x)
    if isnan(y)
        return "nan"
    elseif isinf(y)
        return signbit(y) ? "-inf" : "inf"
    end
    return repr(y)
end


struct FDMThermoResult
    temperature::Vector{Float64}
    entropy_full::Vector{Float64}
    entropy_bath::Vector{Float64}
    entropy_impurity::Vector{Float64}
    heat_capacity_full::Vector{Float64}
    heat_capacity_bath::Vector{Float64}
    heat_capacity_impurity::Vector{Float64}
    shell_weights::Matrix{Float64}  # [temperature, iteration+1]
    spectrum_real::Bool
    max_centered_imaginary_energy::Float64
    density_matrix_normalization_error::Float64
    complete_basis_count_pass::Bool
    complete_basis_count::BigInt
    complete_basis_target::BigInt
    tk_entropy::Float64
    tk_valid::Bool
    crossing_log_slope::Float64
    low_temperature_entropy::Float64
    maximum_impurity_entropy::Float64
end

function _complete_basis_dimension(result::NRGResult)
    N = result.config.nrg.iterations
    total = BigInt(0)
    for shell in result.discarded_shells
        total += BigInt(length(shell.energies)) * BigInt(LOCAL_DIM)^(N - shell.iteration)
    end
    total += BigInt(length(result.state.energies))
    target = BigInt(LOCAL_DIM)^(N + 2)  # impurity plus N+1 Wilson sites
    return total, target, total == target
end

function _complete_basis_levels(result::NRGResult)
    N = result.config.nrg.iterations
    energies = ComplexF64[]
    logdeg = Float64[]
    shell_index = Int[]
    for shell in result.discarded_shells
        environment_sites = N - shell.iteration
        lg = environment_sites * log(Float64(LOCAL_DIM))
        for E in shell.energies
            push!(energies, E)
            push!(logdeg, lg)
            push!(shell_index, shell.iteration)
        end
    end

    # At the last Wilson shell all remaining kept states are declared discarded,
    # completing the Anders--Schiller basis without double counting.
    omegaN = result.config.model.bandwidth *
             result.config.nrg.Lambda^(-N / 2)
    for e in result.state.energies
        push!(energies, result.state.energy_offset + omegaN * e)
        push!(logdeg, 0.0)
        push!(shell_index, N)
    end
    isempty(energies) && error("complete basis contains no states")
    return energies, logdeg, shell_index
end

function _thermal_stats(energies::Vector{Float64}, logdeg::Vector{Float64},
                        temperature::Float64)
    temperature > 0.0 || error("temperature must be positive")
    E0 = minimum(energies)
    erel = energies .- E0
    logs = logdeg .- erel ./ temperature
    m = maximum(logs)
    raw = exp.(logs .- m)
    denom = sum(raw)
    probs = raw ./ denom
    meanE = dot(probs, erel)
    meanE2 = dot(probs, erel .* erel)
    logZ = m + log(denom)
    entropy = logZ + meanE / temperature
    heat_capacity = max((meanE2 - meanE^2) / temperature^2, 0.0)
    return entropy, heat_capacity, probs, abs(sum(probs) - 1.0)
end

function _fermion_entropy_and_heat(x::Float64)
    if x > 40.0
        a = exp(-x)
        return (1.0 + x) * a, x^2 * a
    elseif x < -40.0
        a = exp(x)
        return (1.0 - x) * a, x^2 * a
    end
    f = 1.0 / (1.0 + exp(x))
    one_minus = 1.0 - f
    entropy = 0.0
    if f > 0.0
        entropy -= f * log(f)
    end
    if one_minus > 0.0
        entropy -= one_minus * log(one_minus)
    end
    return entropy, x^2 * f * one_minus
end

function _bath_thermodynamics(result::NRGResult, temperatures::Vector{Float64})
    alpha = result.chain_alpha
    beta = result.chain_beta
    h1 = SymTridiagonal(alpha, beta)
    base = eigvals(h1)
    split = result.config.model.bath_split
    one_particle = vcat(base .+ split, base .- split)
    entropy = zeros(Float64, length(temperatures))
    heat = zeros(Float64, length(temperatures))
    for (it, T) in enumerate(temperatures)
        for eps in one_particle
            s, c = _fermion_entropy_and_heat(eps / T)
            entropy[it] += s
            heat[it] += c
        end
    end
    return entropy, heat
end

function _log_temperature_crossing(T::Vector{Float64}, y::Vector{Float64},
                                   target::Float64)
    for i in 1:(length(T) - 1)
        y1, y2 = y[i], y[i + 1]
        if isfinite(y1) && isfinite(y2) && y2 > y1 &&
           y1 <= target <= y2
            frac = (target - y1) / (y2 - y1)
            logT = log(T[i]) + frac * (log(T[i + 1]) - log(T[i]))
            slope = (y2 - y1) / (log(T[i + 1]) - log(T[i]))
            return exp(logT), slope
        end
    end
    return NaN, NaN
end

"""Compute equilibrium complete-basis/FDM thermodynamics and an entropy scale.

The implicit biorthogonal density matrix is

    rho = Z^{-1} sum_{n,s in D_n,e} exp(-E_ns/T) |R_nse><L_nse|,

where the environment multiplicity is `LOCAL_DIM^(N-n)`.  Positive thermal
weights require a real complete-basis spectrum.  Consequently the function
refuses to label a thermodynamic scale in the broken-PT/passive-loss regime.
The reported `tk_entropy` uses the transparent spin-1/2 crossover criterion
`S_imp(T_K)=ln(2)/2` (or the configured `entropy_target`).
"""
function complete_basis_thermodynamics(result::NRGResult)::FDMThermoResult
    p = result.config.thermo
    p.temperature_min > 0.0 || error("temperature_min must be positive")
    p.temperature_max > p.temperature_min ||
        error("temperature_max must exceed temperature_min")
    p.temperature_points >= 3 || error("temperature_points must be at least 3")

    T = exp.(collect(range(log(p.temperature_min), log(p.temperature_max),
                           length=p.temperature_points)))
    energies_complex, logdeg, shell_index = _complete_basis_levels(result)
    counted_dimension, target_dimension, count_pass = _complete_basis_dimension(result)
    reference = energies_complex[argmin(real.(energies_complex))]
    centered_imag = imag.(energies_complex .- reference)
    maximag = isempty(centered_imag) ? 0.0 : maximum(abs.(centered_imag))
    tolerance = p.imag_tolerance * max(result.config.model.bandwidth, 1.0)
    spectrum_real = result.config.model.frame == :relative && maximag <= tolerance

    entropy_full = fill(NaN, length(T))
    heat_full = fill(NaN, length(T))
    entropy_bath, heat_bath = _bath_thermodynamics(result, T)
    entropy_imp = fill(NaN, length(T))
    heat_imp = fill(NaN, length(T))
    shell_weights = fill(NaN, length(T), result.config.nrg.iterations + 1)
    normerr = NaN

    if spectrum_real
        energies = real.(energies_complex)
        normerr = 0.0
        shell_weights .= 0.0
        for (it, temp) in enumerate(T)
            s, c, probs, err = _thermal_stats(energies, logdeg, temp)
            entropy_full[it] = s
            heat_full[it] = c
            entropy_imp[it] = s - entropy_bath[it]
            heat_imp[it] = c - heat_bath[it]
            normerr = max(normerr, err)
            for j in eachindex(probs)
                shell_weights[it, shell_index[j] + 1] += probs[j]
            end
        end
    end

    tk, slope = if spectrum_real
        _log_temperature_crossing(T, entropy_imp, p.entropy_target)
    else
        (NaN, NaN)
    end
    lowS = spectrum_real ? median(entropy_imp[1:min(5, length(T))]) : NaN
    maxS = spectrum_real ? maximum(entropy_imp) : NaN
    valid = spectrum_real && isfinite(tk) && isfinite(slope) && slope > 0.0 &&
            lowS <= p.low_entropy_max && maxS >= p.moment_entropy_min &&
            normerr <= 1.0e-10 && count_pass

    return FDMThermoResult(T, entropy_full, entropy_bath, entropy_imp,
                           heat_full, heat_bath, heat_imp, shell_weights,
                           spectrum_real, maximag, normerr,
                           count_pass, counted_dimension, target_dimension,
                           valid ? tk : NaN, valid, slope, lowS, maxS)
end

# -----------------------------------------------------------------------------
# Kept-space zero-temperature biorthogonal Lehmann diagnostic
# -----------------------------------------------------------------------------

struct LehmannResult
    omega::Vector{Float64}
    GR::Array{ComplexF64,3}   # [omega, alpha, beta]
    spectral_trace::Vector{Float64}
    sumrule::Matrix{ComplexF64}
    ground_index::Int
    ground_charge::Int
end

function kept_space_lehmann(result::NRGResult)::LehmannResult
    state = result.state
    p = result.config.lehmann
    omega = collect(range(p.omega_min, p.omega_max, length=p.omega_points))
    nw = length(omega)
    GR = zeros(ComplexF64, nw, NFLAV, NFLAV)
    energies = state.energies
    g = argmin([(real(e), abs(imag(e)), abs(e)) for e in energies])
    E0 = energies[g]
    Q0 = state.charges[g]
    sumrule = zeros(ComplexF64, NFLAV, NFLAV)

    plus = findall(==(Q0 + 1), state.charges)
    minus = findall(==(Q0 - 1), state.charges)

    for a in 1:NFLAV, b in 1:NFLAV
        add_residues = ComplexF64[]
        add_gaps = ComplexF64[]
        for m in plus
            residue = state.impurity_ann[a][g, m] * state.impurity_cre[b][m, g]
            push!(add_residues, residue)
            push!(add_gaps, energies[m] - E0)
            sumrule[a, b] += residue
        end
        rem_residues = ComplexF64[]
        rem_gaps = ComplexF64[]
        for m in minus
            residue = state.impurity_cre[b][g, m] * state.impurity_ann[a][m, g]
            push!(rem_residues, residue)
            push!(rem_gaps, energies[m] - E0)
            sumrule[a, b] += residue
        end
        for iw in eachindex(omega)
            z = complex(omega[iw], p.eta)
            value = 0.0 + 0.0im
            for j in eachindex(add_residues)
                value += add_residues[j] / (z - add_gaps[j])
            end
            for j in eachindex(rem_residues)
                value += rem_residues[j] / (z + rem_gaps[j])
            end
            GR[iw, a, b] = value
        end
    end

    spectral_trace = Float64[-imag(tr(GR[i, :, :])) / pi for i in 1:nw]
    return LehmannResult(omega, GR, spectral_trace, sumrule, g, Q0)
end

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

function _write_csv(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(row, ','))
        end
    end
end

function write_run_outputs(result::NRGResult; output_dir::AbstractString=result.config.output_dir)
    mkpath(output_dir)
    cfg = result.config

    chain_rows = Any[]
    for i in eachindex(result.chain_alpha)
        beta_value = i <= length(result.chain_beta) ? result.chain_beta[i] : NaN
        push!(chain_rows, (i - 1, result.chain_alpha[i], beta_value))
    end
    _write_csv(joinpath(output_dir, "wilson_chain.csv"),
               ["site", "onsite", "hopping_to_next"], chain_rows)

    V0 = physical_hybridization(cfg.model)
    Vmat0 = impurity_hybridization_matrix(cfg.model)
    Snorm = V0 > 0 ? (Vmat0 * adjoint(Vmat0)) ./ V0^2 : zeros(ComplexF64, NFLAV, NFLAV)
    soc_rows = Any[]
    for a in 1:NFLAV, b in 1:NFLAV
        push!(soc_rows, (a, b, real(Vmat0[a,b]), imag(Vmat0[a,b]),
                         real(Snorm[a,b]), imag(Snorm[a,b])))
    end
    _write_csv(joinpath(output_dir, "soc_hybridization_matrix.csv"),
               ["row", "column", "V_real", "V_imag",
                "normalized_Gamma_real", "normalized_Gamma_imag"], soc_rows)

    flow_rows = Any[]
    for rec in result.records
        for (q, ordinal, e) in rec.levels
            push!(flow_rows, (rec.iteration, rec.scale, rec.kept, rec.ground_charge,
                             q, ordinal, real(e), imag(e), abs(e),
                             rec.max_residual, rec.biorth_error, rec.min_pair_overlap))
        end
    end
    _write_csv(joinpath(output_dir, "complex_level_flow.csv"),
               ["iteration", "scale", "kept", "ground_charge", "charge", "ordinal",
                "energy_real", "energy_imag", "energy_abs", "max_residual",
                "biorth_error", "min_pair_overlap"], flow_rows)

    transition_rows = Any[]
    for rec in result.records
        for (q, support_rank, e, add_abs, rem_abs, total_abs,
             add_residue, rem_residue) in rec.transition_levels
            push!(transition_rows,
                  (rec.iteration, rec.scale, rec.ground_charge, q, support_rank,
                   real(e), imag(e), abs(e), abs(e) * rec.scale,
                   add_abs, rem_abs, total_abs,
                   real(add_residue), imag(add_residue),
                   real(rem_residue), imag(rem_residue),
                   rec.max_residual, rec.biorth_error, rec.min_pair_overlap))
        end
    end
    _write_csv(joinpath(output_dir, "impurity_transition_weights.csv"),
               ["iteration", "scale", "ground_charge", "charge", "support_rank",
                "energy_real", "energy_imag", "energy_abs",
                "physical_gap_abs", "addition_weight_abs",
                "removal_weight_abs", "total_weight_abs",
                "addition_residue_real", "addition_residue_imag",
                "removal_residue_real", "removal_residue_imag",
                "max_residual", "biorth_error", "min_pair_overlap"],
               transition_rows)

    matrix_rows = Any[]
    for rec in result.records
        for (q, matrix_rank, e, matrix_support, add4, rem4) in
                rec.transition_matrix_levels
            pole_sign = q == rec.ground_charge + 1 ? 1.0 :
                        q == rec.ground_charge - 1 ? -1.0 : 0.0
            pole = pole_sign * e * rec.scale
            push!(matrix_rows,
                  (rec.iteration, rec.scale, rec.ground_charge, q, matrix_rank,
                   real(e), imag(e), real(pole), imag(pole), matrix_support,
                   real(add4[1]), imag(add4[1]), real(add4[2]), imag(add4[2]),
                   real(add4[3]), imag(add4[3]), real(add4[4]), imag(add4[4]),
                   real(rem4[1]), imag(rem4[1]), real(rem4[2]), imag(rem4[2]),
                   real(rem4[3]), imag(rem4[3]), real(rem4[4]), imag(rem4[4]),
                   rec.max_residual, rec.biorth_error, rec.min_pair_overlap))
        end
    end
    _write_csv(joinpath(output_dir, "impurity_transition_residue_matrix.csv"),
               ["iteration", "scale", "ground_charge", "charge", "matrix_rank",
                "energy_real", "energy_imag", "pole_real", "pole_imag",
                "matrix_support_abs",
                "add11_real", "add11_imag", "add12_real", "add12_imag",
                "add21_real", "add21_imag", "add22_real", "add22_imag",
                "rem11_real", "rem11_imag", "rem12_real", "rem12_imag",
                "rem21_real", "rem21_imag", "rem22_real", "rem22_imag",
                "max_residual", "biorth_error", "min_pair_overlap"],
               matrix_rows)

    if cfg.lehmann.enabled
        leh = kept_space_lehmann(result)
        spec_rows = Any[]
        delta_eff_spec, gamma_pt_spec = projected_channels(cfg.model)
        Nspec = ComplexF64[delta_eff_spec 1im * gamma_pt_spec;
                           1im * gamma_pt_spec -delta_eff_spec]
        Nspec_norm = norm(Nspec)
        for i in eachindex(leh.omega)
            g = leh.GR[i, :, :]
            gx = 0.5 * (g[1,2] + g[2,1])
            gz = 0.5 * (g[1,1] - g[2,2])
            jordan_projection = if Nspec_norm > 0.0
                sum(conj.(Nspec) .* g) / Nspec_norm
            else
                0.0 + 0.0im
            end
            push!(spec_rows, (leh.omega[i],
                real(g[1,1]), imag(g[1,1]), real(g[1,2]), imag(g[1,2]),
                real(g[2,1]), imag(g[2,1]), real(g[2,2]), imag(g[2,2]),
                leh.spectral_trace[i],
                -imag(gx) / pi, -imag(gz) / pi,
                -imag(jordan_projection) / pi))
        end
        _write_csv(joinpath(output_dir, "kept_space_lehmann.csv"),
                   ["omega", "ReG11", "ImG11", "ReG12", "ImG12",
                    "ReG21", "ImG21", "ReG22", "ImG22", "minus_ImTrG_over_pi",
                    "minus_ImGx_over_pi", "minus_ImGz_over_pi",
                    "minus_ImJordanG_over_pi"],
                   spec_rows)
        sum_rows = Any[]
        for a in 1:NFLAV, b in 1:NFLAV
            push!(sum_rows, (a, b, real(leh.sumrule[a,b]), imag(leh.sumrule[a,b]),
                             a == b ? 1.0 : 0.0,
                             abs(leh.sumrule[a,b] - (a == b ? 1.0 : 0.0))))
        end
        _write_csv(joinpath(output_dir, "lehmann_sumrule.csv"),
                   ["alpha", "beta", "weight_real", "weight_imag", "target", "abs_error"],
                   sum_rows)
    end

    thermo_result = nothing
    if cfg.thermo.enabled
        thermo_result = complete_basis_thermodynamics(result)
        thermo_rows = Any[]
        for i in eachindex(thermo_result.temperature)
            push!(thermo_rows,
                  (thermo_result.temperature[i],
                   thermo_result.entropy_full[i],
                   thermo_result.entropy_bath[i],
                   thermo_result.entropy_impurity[i],
                   thermo_result.heat_capacity_full[i],
                   thermo_result.heat_capacity_bath[i],
                   thermo_result.heat_capacity_impurity[i]))
        end
        _write_csv(joinpath(output_dir, "fdm_thermodynamics.csv"),
                   ["temperature", "entropy_full", "entropy_bath",
                    "entropy_impurity", "heat_capacity_full",
                    "heat_capacity_bath", "heat_capacity_impurity"],
                   thermo_rows)

        shell_rows = Any[]
        for i in eachindex(thermo_result.temperature), n in 0:cfg.nrg.iterations
            push!(shell_rows, (thermo_result.temperature[i], n,
                               thermo_result.shell_weights[i, n + 1]))
        end
        _write_csv(joinpath(output_dir, "fdm_shell_weights.csv"),
                   ["temperature", "iteration", "density_matrix_weight"],
                   shell_rows)

        open(joinpath(output_dir, "fdm_tk_summary.toml"), "w") do io
            println(io, "method = \"Anders-Schiller complete-basis FDM thermodynamics\"")
            println(io, "criterion = \"S_imp(T_K)=entropy_target\"")
            println(io, "entropy_target = $(_toml_float(cfg.thermo.entropy_target))")
            println(io, "spectrum_real = $(thermo_result.spectrum_real)")
            println(io, "max_centered_imaginary_energy = $(_toml_float(thermo_result.max_centered_imaginary_energy))")
            println(io, "density_matrix_normalization_error = $(_toml_float(thermo_result.density_matrix_normalization_error))")
            println(io, "complete_basis_count_pass = $(thermo_result.complete_basis_count_pass)")
            println(io, "complete_basis_count = $(thermo_result.complete_basis_count)")
            println(io, "complete_basis_target = $(thermo_result.complete_basis_target)")
            println(io, "low_temperature_entropy = $(_toml_float(thermo_result.low_temperature_entropy))")
            println(io, "maximum_impurity_entropy = $(_toml_float(thermo_result.maximum_impurity_entropy))")
            println(io, "crossing_log_slope = $(_toml_float(thermo_result.crossing_log_slope))")
            println(io, "tk_valid = $(thermo_result.tk_valid)")
            println(io, "T_K_entropy = $(_toml_float(thermo_result.tk_entropy))")
        end
    end

    m = cfg.model
    delta_eff, gamma_pt = projected_channels(m)
    open(joinpath(output_dir, "RUN_SUMMARY.txt"), "w") do io
        println(io, "PT-DIRAC NON-HERMITIAN NRG ADAPTER")
        println(io, "beta0=$(m.beta0)")
        println(io, "U=$(m.U); eps_d=$(m.eps_d)")
        println(io, "Delta_eff=$delta_eff; Gamma_PT=$gamma_pt; Delta_coh=$(m.delta_coh)")
        println(io, "frame=$(m.frame); gamma_common=$(m.gamma_common)")
        println(io, "bath exponent=$(m.bath_exponent); D=$(m.bandwidth)")
        println(io, "V=$(physical_hybridization(m)); Gamma_edge=$(m.gamma_edge)")
        println(io, "soc_mode=$(m.soc_mode); soc_lambda=$(m.soc_lambda); soc_kmax=$(m.soc_kmax); F_lambda=$(soc_form_factor(m))")
        Vlambda = impurity_hybridization_matrix(m)
        Vscalar = physical_hybridization(m)
        normalized_det = Vscalar > 0.0 ? det(Vlambda * adjoint(Vlambda)) / Vscalar^4 : 0.0 + 0.0im
        println(io, "soc_normalized_hybridization_det=$(real(normalized_det))")
        println(io, "Lambda=$(cfg.nrg.Lambda); iterations=$(cfg.nrg.iterations); nkeep=$(cfg.nrg.nkeep)")
        println(io, "final kept=$(length(result.state.energies))")
        println(io, "final max residual=$(result.state.max_residual)")
        println(io, "final biorth error=$(result.state.biorth_error)")
        println(io, "final minimum paired overlap=$(result.state.min_pair_overlap)")
        if thermo_result !== nothing
            println(io, "fdm_spectrum_real=$(thermo_result.spectrum_real)")
            println(io, "fdm_T_K_entropy=$(thermo_result.tk_entropy)")
            println(io, "fdm_T_K_valid=$(thermo_result.tk_valid)")
        end
        println(io)
        println(io, "IMPORTANT SCOPE")
        println(io, "When enabled, fdm_thermodynamics.csv uses the complete Anders-Schiller discarded-state basis.")
        println(io, "Its T_K label is the entropy midpoint S_imp=ln(2)/2 and is emitted only for a real-spectrum relative-frame run.")
        println(io, "It is an equilibrium thermodynamic crossover definition, not an FDM spectral linewidth.")
        println(io, "The exported Lehmann curve is a kept-space zero-temperature diagnostic.")
        println(io, "It is not yet a complete-basis/FDM-NRG spectral function and must be judged by lehmann_sumrule.csv.")
        println(io, "kept_space_lehmann.csv also exports channel-even Gx, channel-odd Gz,")
        println(io, "and the normalized Jordan projection Tr[N_EP^dagger G]/||N_EP||_F.")
        println(io, "impurity_transition_weights.csv contains iteration-resolved, biorthogonal")
        println(io, "addition/removal residue products used only to identify impurity-supported crossovers.")
        println(io, "impurity_transition_residue_matrix.csv contains the full 2x2 complex")
        println(io, "transition-residue matrices for non-circular pole-pair and Jordan diagnostics.")
        println(io, "soc_hybridization_matrix.csv records V_lambda and V_lambda V_lambda^dagger/V^2.")
        println(io, "In soc_mode=overlap, det(V_lambda V_lambda^dagger/V^2)=F_lambda.")
        println(io, "Run away from an exact Jordan point; the biorthogonal basis is singular at exact defectiveness.")
    end
    return output_dir
end

function run_scan(config::RunConfig, betas::Vector{Float64}; root::AbstractString=config.output_dir)
    outputs = String[]
    for beta in betas
        m = config.model
        model = ModelParams(
            U=m.U, eps_d=m.eps_d, delta0=m.delta0, c_delta=m.c_delta,
            g_gamma=m.g_gamma, beta0=beta, delta_coh=m.delta_coh,
            gamma_common=m.gamma_common, frame=m.frame, bandwidth=m.bandwidth,
            bath_exponent=m.bath_exponent, gamma_edge=m.gamma_edge,
            hybridization_V=m.hybridization_V, bath_split=m.bath_split,
            reciprocal_hybridization=m.reciprocal_hybridization,
            soc_mode=m.soc_mode, soc_lambda=m.soc_lambda, soc_kmax=m.soc_kmax,
        )
        cfg = RunConfig(model=model, nrg=config.nrg, lehmann=config.lehmann,
                        thermo=config.thermo,
                        output_dir=joinpath(root, @sprintf("beta_%0.6f", beta)))
        @printf("\n===== beta0=%0.6f =====\n", beta)
        result = run_nrg(cfg)
        push!(outputs, write_run_outputs(result; output_dir=cfg.output_dir))
    end
    return outputs
end

end # module
