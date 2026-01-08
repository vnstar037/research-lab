module SEEQSTEigenstates

using LinearAlgebra
using StatsBase
using IterTools

export GenerateSGroups,
       generateEigenstatesE,
       generateEigenstatesO,
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


function GenerateComputationalBasis(N::Int)
    basis = Vector{Vector{Float64}}()
    dim = 2^N
    for i in 0:(dim-1)
        bits = reverse(digits(i, base=2, pad=N))
        ket = [1.0]
        for b in bits
            ket = kron(ket, b == 0 ? [1.0; 0.0] : [0.0; 1.0])
        end
        push!(basis, ket)
    end
    return basis
end

# --------------------------------------------------------------------
# Eigenstates (E group)
# --------------------------------------------------------------------
# ============================================================
# Computational basis |bits>
# ============================================================
function basisVector(bits::Vector{Int})
    v = [1.0]
    for b in bits
        v = kron(v, b == 0 ? [1.0, 0.0] : [0.0, 1.0])
    end
    return v
end

# ============================================================
bitFlip(bits::Vector{Int}) = [1 - b for b in bits]

# ============================================================
function bitStrings(n::Int)
    if n == 0
        return [Int[]]
    end
    prev = bitStrings(n - 1)
    res = Vector{Vector{Int}}()
    for p in prev
        push!(res, vcat(p, 0))
        push!(res, vcat(p, 1))
    end
    return res
end

# ============================================================
# Permutation: canonical [offdiag | diag] → physical
# ============================================================
function permuteToPhysical(state::AbstractVector, offdiagPos::Vector{Int}, diagPos::Vector{Int})
    perm = vcat(offdiagPos, diagPos)
    N = length(perm)
    reshaped = reshape(state, ntuple(_ -> 2, N)...)
    invp = invperm(perm)
    return vec(permutedims(reshaped, invp))
end

# ============================================================
# Parse Pauli strings
# ============================================================
function parsePaulis(paulis::Vector{String})
    offdiagPos = Int[]
    diagPos    = Int[]

    for (i, p) in enumerate(paulis)
        if p == "XY"
            push!(offdiagPos, i)
        elseif p == "IZ"
            push!(diagPos, i)
        else
            error("Unbekannter Pauli-String: $p")
        end
    end

    return offdiagPos, diagPos
end

# ============================================================
# MAIN FUNCTION (user-facing)
# ============================================================
function generateEigenstatesE(paulis::Vector{String})
    offdiagPos, diagPos = parsePaulis(paulis)

    M = length(offdiagPos)
    K = length(diagPos)

    states = Vector{Vector{ComplexF64}}()

    # --------------------------------------------------------
    # Case 1: M = 0 → pure computational basis
    # --------------------------------------------------------
    if M == 0
        for q in bitStrings(K)
            ψ = ComplexF64.(basisVector(q))
            push!(states, permuteToPhysical(ψ, Int[], diagPos))
        end
        return states
    end

    # --------------------------------------------------------
    # Case 2: M > 0, K = 0
    # --------------------------------------------------------
    if K == 0
        p_bits = [p for p in bitStrings(M) if p[1] == 0]

        for p in p_bits
            pbar = bitFlip(p)

            vp    = ComplexF64.(basisVector(p))
            vpbar = ComplexF64.(basisVector(pbar))

            plus  = (vp + vpbar) / sqrt(2)
            minus = (vp - vpbar) / sqrt(2)

            push!(states, permuteToPhysical(plus,  offdiagPos, Int[]))
            push!(states, permuteToPhysical(minus, offdiagPos, Int[]))
        end
        return states
    end

    # --------------------------------------------------------
    # Case 3: M > 0, K > 0
    # --------------------------------------------------------
    p_bits = [p for p in bitStrings(M) if p[1] == 0]
    q_bits = bitStrings(K)

    for p in p_bits
        pbar = bitFlip(p)

        vp    = ComplexF64.(basisVector(p))
        vpbar = ComplexF64.(basisVector(pbar))

        plus  = (vp + vpbar) / sqrt(2)
        minus = (vp - vpbar) / sqrt(2)

        for q in q_bits
            vq = ComplexF64.(basisVector(q))

            ψ_plus  = kron(plus,  vq)
            ψ_minus = kron(minus, vq)

            push!(states, permuteToPhysical(ψ_plus,  offdiagPos, diagPos))
            push!(states, permuteToPhysical(ψ_minus, offdiagPos, diagPos))
        end
    end

    return states
end

# ============================================================
function generateEigenstatesO(paulis::Vector{String})
    offdiagPos = [i for (i,p) in enumerate(paulis) if p == "XY"]
    diagPos    = [i for (i,p) in enumerate(paulis) if p == "IZ"]

    M = length(offdiagPos)
    K = length(diagPos)

    if M == 0
        return Vector{Vector{ComplexF64}}()
    end

    states = Vector{Vector{ComplexF64}}()
    q_bits = bitStrings(K)

    p_bits = [p for p in bitStrings(M) if p[1] == 0]

    for p in p_bits
        pbar = bitFlip(p)

        vp    = ComplexF64.(basisVector(p))
        vpbar = ComplexF64.(basisVector(pbar))

        # O-Basis: ungerade Anzahl σy → beide Varianten
        plus_O  = (vp + im*vpbar)/sqrt(2)
        minus_O = (vp - im*vpbar)/sqrt(2)

        for q in q_bits
            vq = ComplexF64.(basisVector(q))
            ψ_plus  = kron(plus_O, vq)
            ψ_minus = kron(minus_O, vq)
            push!(states, permuteToPhysical(ψ_plus, offdiagPos, diagPos))
            push!(states, permuteToPhysical(ψ_minus, offdiagPos, diagPos))
        end
    end

    return states
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
        SE = generateEigenstatesE(si)
        SO = generateEigenstatesO(si)

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
