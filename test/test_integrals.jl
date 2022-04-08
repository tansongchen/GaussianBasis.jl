test_file = joinpath(@__DIR__, "CH4_631g.h5")

atoms = Molecules.parse_string("""
C   -2.131551124300    2.286168823700    0.000000000000
H   -1.061551124300    2.286168823700    0.000000000000
H   -2.488213906200    1.408104616400    0.496683911300
H   -2.488218762100    2.295059432700   -1.008766153900
H   -2.488220057000    3.155340844300    0.512081313000""")

bs = BasisSet("6-31g", atoms)
bs2 = BasisSet("sto-3g", atoms)

@testset "One-Electron Integrals" begin

    @test overlap(bs) ≈ h5read(test_file, "overlap")
    @test kinetic(bs) ≈ h5read(test_file, "kinetic")
    @test nuclear(bs) ≈ h5read(test_file, "nuclear")

    @test overlap(bs, bs2) ≈ h5read(test_file, "overlap_sto3g")
    @test kinetic(bs, bs2) ≈ h5read(test_file, "kinetic_sto3g")
    @test nuclear(bs, bs2) ./2 ≈ h5read(test_file, "nuclear_sto3g")
end

@testset "Two-Electron Four-Center" begin

    @test ERI_2e4c(bs) ≈ h5read(test_file, "denseERI")

    idx, data = sparseERI_2e4c(bs)

    uniform = zeros(Int16, length(idx)*4)
    for i in eachindex(idx)
        uniform[(1+4*(i-1)):(4+4*(i-1))] .= idx[i]
    end

    @test uniform ≈ h5read(test_file, "sparseERIidx")
    @test data ≈ h5read(test_file, "sparseERIdata")
end

@testset "Two-Electron Three-Center" begin
    @test ERI_2e3c(bs, bs2) ≈ h5read(test_file, "Pqp_aux_sto3g")
end

@testset "Two-Electron Two-Center" begin
    eri = ERI_2e2c(bs) 
    @test eri ≈ h5read(test_file, "J")

    # Test individual elements
    Iacum = 0
    elemwise = true
    for A in 1:length(bs.atoms)
        for i in 1:length(bs[A]) 
            Ni = 2*bs[A,i].l+1
            ri = (Iacum+1):(Iacum+Ni)
            Jacum = 0
            for B in 1:length(bs.atoms)
                for j in 1:length(bs[B])

                    Nj = 2*bs[B,j].l+1
                    rj = (Jacum+1):(Jacum+Nj)
                    buf = zeros(Ni, Nj)
                    ERI_2e2c!(buf, bs, A, i, B, j)
                    if !(buf ≈ eri[ri, rj])
                        elemwise = false
                    end
                    Jacum += Nj
                end
            end
            Iacum += Ni
        end
    end
    @test elemwise
end

@testset "Dipole" begin
    Co = h5read(test_file, "CH4_Orbitals")
    bs3 = BasisSet("cc-pvdz", atoms)
    d = GaussianBasis.dipole(bs3) # (34, 34, 3) array
    d2 = Co' * reshape(d, (34,102)) # (5, 102) Array
    d2 = permutedims(reshape(d2, (5,34,3)), (1,3,2)) # (5,3,34) array
    d3 = reshape(d2, (15,34)) * Co #(15,5) array
    d3 = reshape(d3, 5,3,5) 
    μ = 2 .* sum(d3[i,:,i] for i=1:5)
    @test isapprox(μ, [-40.280477, 43.202326, 0.0000], atol=1e-5)
end