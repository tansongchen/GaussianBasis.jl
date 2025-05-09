using GaussianBasis, Molecules, StaticArrays
using GaussianBasis: ACSint, num_basis
using ForwardDiff
using Enzyme, BenchmarkTools, LinearAlgebra
import Enzyme.EnzymeRules: forward, reverse, augmented_primal
using Enzyme.EnzymeRules
import CommonSolve: init, solve!

abstract type MeanFieldAlgorithm end
struct RHF <: MeanFieldAlgorithm end
struct RHFCache{Mtype,Btype,Htype,Stype,Otype,ERItype,Ftype,Ptype,Ctype,εtype}
    molecule::Mtype
    basisset::Btype
    H::Htype
    S::Stype
    O::Otype
    ERI::ERItype
    # cache
    F::Ftype
    P::Ptype
    C::Ctype
    ε::εtype
end

struct RHFSolution{Etype,Ctype,εtype}
    E::Etype
    C::Ctype
    ε::εtype
end

function make_hydrogen(rs)
    A = 1.008
    Z = 1
    atoms = map(r -> Molecules.Atom(Z, A, SVector{3}(r)), eachcol(rs))
    Molecule(atoms)
end

function make_hydrogen_basis(H2)
    atoms = H2.atoms
    coefs = SVector(0.2769343550790519, 0.26783885160947885, 0.08347367112984118)
    exps = SVector(3.425250914, 0.6239137298, 0.168855404)
    basis = [
        CartesianShell(0, coefs, exps, atoms[1]),
        CartesianShell(0, coefs, exps, atoms[2])
    ]
    natm = length(atoms)
    nshells = length(basis)
    bpa = zeros(Int, natm)
    spa = zeros(Int, natm)
    for a in 1:natm
        for b in basis
            if atoms[a] == b.atom
                spa[a] += 1
                bpa[a] += num_basis(b)
            end
        end
    end
    nbas = sum(bpa)
    BasisSet("sto-3g", atoms, basis, bpa, spa, natm, nbas, nshells, ACSint())
end

hf_energy(P, H, F) = 0.5 * sum(P .* (H + F))

function build_fock!(F, P, H, ERI)
    F .= H
    for μ in axes(F, 1)
        for ν in axes(F, 2)
            for λ in axes(P, 1)
                for σ in axes(P, 2)
                    F[μ, ν] += P[λ, σ] * (ERI[μ, ν, λ, σ] - ERI[μ, λ, ν, σ] / 2)
                end
            end
        end
    end
end

function init(molecule, basisset, ::RHF)
    N = molecule.Nα + molecule.Nβ
    S = Symmetric(overlap(basisset))
    T = kinetic(basisset)
    V = nuclear(basisset)
    H = Symmetric(T + V)
    ERI = ERI_2e4c(basisset)
    O = Diagonal([i <= N / 2 ? 2.0 : 0.0 for i in 1:size(S, 1)])
    P = zeros(size(S))
    C = zeros(size(S))
    F = zeros(size(S))
    ε = zeros(size(S, 1))
    RHFCache(molecule, basisset, H, S, O, ERI, F, P, C, ε)
end

function scf_solve!(H, S, O, ERI, P, C, F, ε)
    E_prev = Inf
    E = 0.0
    d, U = eigen(S)
    X = U * Diagonal(1.0 ./ sqrt.(d))
    while abs(E - E_prev) > 1e-5
        E_prev = E
        build_fock!(F, P, H, ERI)
        Ft = Symmetric(X' * F * X)
        et, Ct = eigen(Ft)
        ε .= et
        C .= X * Ct
        P .= C * O * C'
        E = hf_energy(P, H, F)
    end
    return E
end

function solve!(cache::RHFCache)
    (; molecule, H, S, O, ERI, P, C, F, ε) = cache
    Vnuc = Molecules.nuclear_repulsion(molecule.atoms)
    Eelec = scf_solve!(H, S, O, ERI, P, C, F, ε)
    energy = Eelec + Vnuc
    RHFSolution(energy, C, ε)
end

# Calculate H2 energy with bond length 1.0 Angstrom
rs = [0.0 0.0; 0.0 0.0; 0.0 1.0]
H2 = make_hydrogen(rs)
basis = make_hydrogen_basis(H2)
cache = init(H2, basis, RHF())
sol = solve!(cache)

function condition!(f1, f2, C, e, H, S, ERI, O)
    P = C * O * C'
    F = zeros(size(H))
    build_fock!(F, P, H, ERI)
    f1 .= F * C - S * C * e
    f2 .= (C' * S * C) - I
end

function vjp(dC_de, df, p, t)
    (C, e, f1, f2, H, S, ERI, O) = p
    df1 = reshape(df[1:length(f1)], size(f1))
    df2 = reshape(df[length(f1)+1:end], size(f2))
    dC, de = zeros(size(C)), zeros(size(e))
    autodiff(Reverse, condition!, Const, Duplicated(f1, df1), Duplicated(f2, df2),
        Duplicated(C, dC), Duplicated(e, de), Const(H), Const(S), Const(ERI), Const(O))
    dC_de[1:length(dC)] .= dC
    dC_de[length(dC)+1:end] .= de
    dC_de
end

function augmented_primal(::RevConfigWidth{1}, ::Const{typeof(scf_solve!)}, ::Type{<:Active},
    H::Duplicated, S::Duplicated, O::Const, ERI::Duplicated, P::Duplicated, C::Duplicated, F::Duplicated, ε::Duplicated)
    println("In custom augmented primal rule.")
    E = scf_solve!(H.val, S.val, O.val, ERI.val, P.val, C.val, F.val, ε.val)
    return AugmentedReturn(E, nothing, nothing)
end

function reverse(::RevConfigWidth{1}, ::Const{typeof(scf_solve!)}, dret::Active, tape,
    H::Duplicated, S::Duplicated, O::Const, ERI::Duplicated, P::Duplicated, C::Duplicated, F::Duplicated, ε::Duplicated)
    println("In custom reverse rule.")
    autodiff(Reverse, hf_energy, Active, P, H, F)
    autodiff(Reverse, build_fock!, Const, F, P, H, ERI)
    # transfer P to C
    proto = zeros(length(C.val) + length(ε.val))
    f1 = zeros(size(C.val))
    f2 = zeros(size(C.val))
    op = FunctionOperator(vjp, proto; p = (C, ε, f1, f2, H, S, ERI, O))
    b = [vec(C.dval); vec(ε.dval)]
    prob = LinearProblem(op, b)
    sol = solve(prob)
    df1 = sol[1:length(C.val)]
    df2 = sol[length(C.val)+1:end]
    autodiff(Reverse, condition!, Const, Duplicated(f1, df1), Duplicated(f2, df2),
        Const(C.val), Const(ε.val), H, S, ERI, O)
    return (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

(; H, S, O, ERI, P, C, F, ε) = init(H2, basis, RHF())
dH, dS, dERI, dP, dC, dF, dε = Symmetric(zeros(size(H))), Symmetric(zeros(size(S))), zeros(size(ERI)), zeros(size(P)), zeros(size(C)), zeros(size(F)), zeros(size(ε))
autodiff(ReverseWithPrimal, scf_solve!, Active, Duplicated(H, dH), Duplicated(S, dS), Const(O), Duplicated(ERI, dERI),
    Duplicated(P, dP), Duplicated(C, dC), Duplicated(F, dF), Duplicated(ε, dε))
dH, dP, dF

δ = [0.0 0.0; 0.0 0.0; 0.0 1e-5]
H2_ = make_hydrogen(rs + δ)
basis_ = make_hydrogen_basis(H2_)
E_, ε_, P_, C_ = rhf(H2_, basis_)
force_fd = (E_ - E) / norm(δ, Inf)
println("Force (Finite Difference): ", force_fd)

function rhf_feynman(rs, ε, P, C)
    H2 = make_hydrogen(rs)
    N = H2.Nα + H2.Nβ
    basisset = make_hydrogen_basis(H2)
    dS = overlap(basisset)
    dT = kinetic(basisset)
    dV = nuclear(basisset)
    dERI = ERI_2e4c(basisset)
    dVnuc = Molecules.nuclear_repulsion(molecule.atoms)
    dH = dT + dV
    M = diagm([i <= N / 2 ? 2 * ε[i] : 0.0 for i in 1:size(dS, 1)])
    Q = C * M * C'
    dE = dVnuc + sum(P .* dH) - sum(Q .* dS)
    for μ in axes(P, 1)
        for ν in axes(P, 2)
            for λ in axes(P, 1)
                for σ in axes(P, 2)
                    dE += P[ν, μ] * P[λ, σ] * (dERI[μ, ν, σ, λ] - dERI[μ, λ, σ, ν] / 2) / 2
                end
            end
        end
    end
    return dE
end

force_fwd = ForwardDiff.gradient(_rs -> rhf_feynman(_rs, ε, P, C), rs)
println("Force (ForwardDiff): ", force_fwd)

force_enzyme = [0.0 0.0; 0.0 0.0; 0.0 0.0]
@btime autodiff(Reverse, rhf_feynman, Duplicated(rs, force_enzyme), Const(ε), Const(P), Const(C))
println("Force (Enzyme, Reverse Mode): ", force_enzyme)
