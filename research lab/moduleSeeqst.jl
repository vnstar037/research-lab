module SEEQSTEigenstates

using LinearAlgebra
using StatsBase
using IterTools

export GenerateSGroups,
       GenerateEigenstatesE,
       GenerateEigenstatesO,
       ProjectorsFromEigenstates,
       PauliStringEigenvalues,
       MatrixElementsForGroup,
       DensityMatrixFromGroup,
       GenerateCombinations,
       ExpectationValuesFromCounts,
       RecreatingDensityMatrixWithSeeqst

# --------------------------------------------------------------------
# Generate all stabilizer groups
# --------------------------------------------------------------------
function GenerateSGroups(N::Int)
    groups = Vector{Vector{String}}()
    for mask in 0:(2^N - 1)
        S = Vector{String}(undef, N)
        for qubit in 1:N
            S[qubit] = ((mask >> (qubit-1)) & 1) == 0 ? "IZ" : "XY"
        end
        push!(groups, S)
    end
    return groups
end

# --------------------------------------------------------------------
# Eigenstates (E group)
# --------------------------------------------------------------------
function GenerateEigenstatesE(S::Vector{String})
    ket0 = [1.0, 0.0]
    ket1 = [0.0, 1.0]
    ketp = (ket0 .+ ket1) ./ sqrt(2)
    ketm = (ket0 .- ket1) ./ sqrt(2)

    local_bases = Vector{Vector{Vector{Float64}}}()
    for s in S
        s == "IZ" && push!(local_bases, [ket0, ket1])
        s == "XY" && push!(local_bases, [ketp, ketm])
    end

    eigenstates = Vector{Vector{Float64}}()
    for combo_rev in Iterators.product(reverse(local_bases)...)
        combo = reverse(combo_rev)
        ψ = combo[1]
        for k in combo[2:end]
            ψ = kron(ψ, k)
        end
        push!(eigenstates, ψ)
    end
    return eigenstates
end

# --------------------------------------------------------------------
# Eigenstates (O group)
# --------------------------------------------------------------------
function GenerateEigenstatesO(S::Vector{String})
    all(x -> x == "IZ", S) && return Vector{Vector{ComplexF64}}()

    ket0 = ComplexF64[1, 0]
    ket1 = ComplexF64[0, 1]
    ketIp = (ket0 .+ im .* ket1) ./ sqrt(2)
    ketIm = (ket0 .- im .* ket1) ./ sqrt(2)

    local_bases = Vector{Vector{Vector{ComplexF64}}}()
    for s in S
        s == "IZ" && push!(local_bases, [ket0, ket1])
        s == "XY" && push!(local_bases, [ketIp, ketIm])
    end

    eigenstates = Vector{Vector{ComplexF64}}()
    for combo_rev in Iterators.product(reverse(local_bases)...)
        combo = reverse(combo_rev)
        ψ = combo[1]
        for k in combo[2:end]
            ψ = kron(ψ, k)
        end
        push!(eigenstates, ψ)
    end
    return eigenstates
end

# --------------------------------------------------------------------
# Projectors
# --------------------------------------------------------------------
function ProjectorsFromEigenstates(eigs)
    isempty(eigs) && return Matrix{ComplexF64}[]
    return [ψ * ψ' for ψ in eigs]
end

# --------------------------------------------------------------------
# Pauli eigenvalues
# --------------------------------------------------------------------
PauliEigenvalues(p::Char) =
    p == 'I' ? [1.0, 1.0] :
    p in ('X','Y','Z') ? [1.0, -1.0] :
    error("Unknown Pauli operator $p")

function PauliStringEigenvalues(s::String)
    ev = [1.0]
    for p in reverse(s)
        ev = vec([a*b for a in ev, b in PauliEigenvalues(p)])
    end
    return ev
end

# --------------------------------------------------------------------
# Generate Pauli combinations
# --------------------------------------------------------------------
function GenerateCombinations(list::Vector{String})
    combos = Iterators.product(reverse(list)...)
    return vec([join(reverse(c)) for c in combos])
end

# --------------------------------------------------------------------
# Expectation values
# --------------------------------------------------------------------
function ExpectationValuesFromCounts(paulis, cE, cO)
    Dict(p => dot(PauliStringEigenvalues(p),
                  occursin('Y', p) ? cO : cE)
         for p in paulis)
end

# --------------------------------------------------------------------
# Density matrix reconstruction
# --------------------------------------------------------------------
function MatrixElementsForGroup(S)
    N = length(S)
    dim = 2^N
    bits = [reverse(digits(i, base=2, pad=N)) for i in 0:dim-1]
    [(i,j) for i in 1:dim, j in 1:dim
     if all(S[k]=="IZ" ? bits[i][k]==bits[j][k] :
            bits[i][k]!=bits[j][k] for k in 1:N)]
end

function DensityMatrixFromGroup(evs, positions, n)
    ρ = zeros(ComplexF64, 2^n, 2^n)
    for (i,j) in positions
        bi = reverse(digits(i-1, base=2, pad=n))
        bj = reverse(digits(j-1, base=2, pad=n))
        for (pstring, v) in evs
            fac = 1 + 0im
            for k in 1:n
                p = pstring[k]
                if p == 'Z'
                    fac *= bi[k] == 0 ? 1 : -1
                elseif p == 'X'
                    fac *= bi[k] == bj[k] ? 0 : 1
                elseif p == 'Y'
                    bi[k] == bj[k] && (fac = 0; break)
                    fac *= bi[k] == 0 ? -im : im
                end
            end
            ρ[i,j] += fac * v
        end
    end
    return ρ / 2^n
end

function simulateMeasurement(rho, projectors, n)
    isempty(projectors) && return Float64[]
    probs = [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), n)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n
end

function RecreatingDensityMatrixWithSeeqst(rho_true, N)

    dim = size(rho_true,1)
    n = Int(round(log2(dim)))

    rho_num = zeros(ComplexF64, dim, dim)

    for si in GenerateSGroups(n)
        SE = GenerateEigenstatesE(si)
        SO = GenerateEigenstatesO(si)

        cSE = simulateMeasurement(rho_true, ProjectorsFromEigenstates(SE), N)
        cSO = simulateMeasurement(rho_true, ProjectorsFromEigenstates(SO), N)

        ev = ExpectationValuesFromCounts(
            GenerateCombinations(si), cSE, cSO
        )

        rho_num += DensityMatrixFromGroup(ev, MatrixElementsForGroup(si), n)
    end

    return rho_num
end

end # module
