using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "PTDiracNHNRG.jl"))
using .PTDiracNHNRG

const M = PTDiracNHNRG

function relerr(A, B)
    return norm(A - B) / max(norm(B), eps(Float64))
end

function main()
model = ModelParams(
    U=2.0,
    eps_d=-1.0,
    delta0=0.10,
    c_delta=0.0,
    g_gamma=0.20,
    beta0=0.50,
    delta_coh=3.0e-3,
    gamma_common=0.0,
    frame=:relative,
    bandwidth=pi / 4,
    bath_exponent=1.0,
    gamma_edge=0.12,
    bath_split=0.017,
    soc_mode=:none,
)

nrg = NRGParams(
    Lambda=3.0,
    z_shift=0.5,
    iterations=1,
    nkeep=256,                 # no truncation: 16 states initially, 64 after one added site
    min_keep_per_charge=0,
    sort_type=:LowRe,
    star_intervals=48,
    overlap_floor=1.0e-12,
    degeneracy_tolerance=1.0e-11,
    residual_tolerance=1.0e-9,
    save_levels=8,
)

cfg = RunConfig(
    model=model,
    nrg=nrg,
    lehmann=LehmannParams(enabled=false),
    output_dir="output/biorth_operator_regression",
)

alpha, beta = M.make_chain(cfg)
D = model.bandwidth

# Reconstruct the full impurity + first-site block without truncation.
Himp = M.build_impurity_hamiltonian(model) ./ D
H0 = kron(Himp, M.ILOCAL)
H0 .+= kron(M.ILOCAL,
           M.site_hamiltonian(alpha[1] / D, model.bath_split / D))
Vmat = M.impurity_hybridization_matrix(model) ./ D
Cdag = ntuple(a -> Matrix{ComplexF64}(adjoint(M.CLOCAL[a])), M.NFLAV)
H0 .+= M.coupling_term(M.CLOCAL, Cdag, M.QLOCAL, Vmat)

q0 = Int[qimp + qsite for qimp in M.QLOCAL for qsite in M.QLOCAL]
values0, qkeep0, L0, R0, maxres0, maxbio0, _ =
    M.diagonalize_by_charge(H0, q0, nrg)

@assert length(values0) == 16
@assert maxres0 < 1.0e-9
@assert maxbio0 < 1.0e-9

Pimp = M.parity_from_charges(M.QLOCAL)
last_ann_orig = ntuple(a -> kron(Pimp, M.CLOCAL[a]), M.NFLAV)
last_cre_orig = ntuple(a -> kron(Pimp, adjoint(M.CLOCAL[a])), M.NFLAV)
last_ann_rl = ntuple(a -> M.transform_operator(L0, last_ann_orig[a], R0), M.NFLAV)
last_cre_rl = ntuple(a -> M.transform_operator(L0, last_cre_orig[a], R0), M.NFLAV)

# In a genuinely nonunitary left/right basis, c†_RL must not be reconstructed
# as the ordinary matrix adjoint of c_RL.
adjoint_mismatch = maximum(
    relerr(adjoint(last_ann_rl[a]), last_cre_rl[a]) for a in 1:M.NFLAV
)
@assert adjoint_mismatch > 1.0e-3

# With no truncation, the independently transformed operators retain the CAR.
Iold = Matrix{ComplexF64}(I, length(values0), length(values0))
car_error = 0.0
for a in 1:M.NFLAV, b in 1:M.NFLAV
    target = a == b ? Iold : zeros(ComplexF64, size(Iold))
    car_error = max(car_error,
                    norm(last_ann_rl[a] * last_cre_rl[b] +
                         last_cre_rl[b] * last_ann_rl[a] - target, Inf))
end
@assert car_error < 1.0e-9

# Compare the recursively constructed hopping term with an explicit basis
# transformation of the same operator in the original Fock basis.
T = ComplexF64[0.23 + 0.07im  0.04 - 0.02im;
              -0.03 + 0.01im  0.19 - 0.05im]
Pold_orig = M.parity_from_charges(q0)
Hhop_orig = zeros(ComplexF64, 16 * M.LOCAL_DIM, 16 * M.LOCAL_DIM)
for a in 1:M.NFLAV, b in 1:M.NFLAV
    t = T[a, b]
    Hhop_orig .+= t .* kron(last_cre_orig[a] * Pold_orig, M.CLOCAL[b])
    Hhop_orig .+= conj(t) .* kron(Pold_orig * last_ann_orig[a], adjoint(M.CLOCAL[b]))
end

Lext = kron(adjoint(L0), M.ILOCAL)
Rext = kron(R0, M.ILOCAL)
Hhop_exact = Lext * Hhop_orig * Rext
Hhop_fixed = M.coupling_term(last_ann_rl, last_cre_rl, qkeep0, T)
fixed_error = relerr(Hhop_fixed, Hhop_exact)
@assert fixed_error < 1.0e-9

# Demonstrate that the old shortcut is not equivalent.
Pold_rl = M.parity_from_charges(qkeep0)
Hhop_old = zeros(ComplexF64, size(Hhop_fixed))
for a in 1:M.NFLAV, b in 1:M.NFLAV
    t = T[a, b]
    Hhop_old .+= t .* kron(adjoint(last_ann_rl[a]) * Pold_rl, M.CLOCAL[b])
    Hhop_old .+= conj(t) .* kron(Pold_rl * last_ann_rl[a], adjoint(M.CLOCAL[b]))
end
old_shortcut_error = relerr(Hhop_old, Hhop_exact)
@assert old_shortcut_error > 1.0e-3

# Exercise the actual recursive path for one full, untruncated iteration.
state0 = M.initialize_nrg(cfg, alpha[1])
@assert hasproperty(state0, :last_cre)
state1 = M.add_site(state0, 1, alpha[2], beta[1], cfg)
@assert length(state1.energies) == 64
@assert state1.max_residual < 1.0e-9
@assert state1.biorth_error < 1.0e-8

println("PASS: independent biorthogonal creation/annihilation propagation")
println("  transformed c† vs adjoint(c) mismatch = ", adjoint_mismatch)
println("  CAR error (untruncated)             = ", car_error)
println("  corrected hopping relative error   = ", fixed_error)
println("  old shortcut relative error         = ", old_shortcut_error)
end

main()
