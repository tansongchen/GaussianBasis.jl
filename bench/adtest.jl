using GaussianBasis, Molecules, StaticArrays
using GaussianBasis: ACSint, num_basis
using ForwardDiff: ForwardDiff, derivative, gradient, jacobian
using Enzyme, BenchmarkTools, LinearAlgebra

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

function build_fock(P, S, T, V, ERI)
    F = T + V
    for μ in axes(F, 1)
        for ν in axes(F, 2)
            for λ in axes(P, 1)
                for σ in axes(P, 2)
                    F[μ, ν] += P[λ, σ] * (ERI[μ, ν, λ, σ] - ERI[μ, λ, ν, σ] / 2)
                end
            end
        end
    end
    return F
end

function rhf(molecule, basisset)
    N = molecule.Nα + molecule.Nβ
    S = overlap(basisset)
    T = kinetic(basisset)
    V = nuclear(basisset)
    ERI = ERI_2e4c(basisset)
    Vnuc = Molecules.nuclear_repulsion(molecule.atoms)
    # Initialize
    X = S^(-1 / 2)
    H = T + V
    P = zeros(eltype(S), size(S))
    M = diagm([i <= N / 2 ? 2.0 : 0.0 for i in 1:size(S, 1)])
    # HF loop
    E_prev = Inf
    E = 0.0
    ε = zeros(eltype(S), size(S, 1))
    C = zeros(eltype(S), size(S))
    while abs(E - E_prev) > 1e-5
        E_prev = E
        F = build_fock(P, S, T, V, ERI)
        Ft = Symmetric(X' * F * X)
        ε, Ct = eigen(Ft, sortby=identity)
        C = X * Ct
        P = C * M * C'
        E = hf_energy(P, H, F) + Vnuc
    end
    return E, ε, P, C
end

# Calculate H2 energy with bond length 1.0 Angstrom
rs = [0.0 0.0; 0.0 0.0; 0.0 1.0]
H2 = make_hydrogen(rs)
basis = make_hydrogen_basis(H2)
E, ε, P, C = rhf(H2, basis)
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
