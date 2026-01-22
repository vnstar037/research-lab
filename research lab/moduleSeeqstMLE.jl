module SeeqstMLE

using LinearAlgebra
using StatsBase
using IterTools
using Convex
using SCS

export GenerateSGroups,
       generateEigenstatesE,
       generateEigenstatesO,
       ProjectorsFromEigenstates,
       GenerateCombinations,
       reconstructDensityMatrixWithSeeqstMLE
# --------------------------------------------------------------------
# Generate all stabilizer groups
# --------------------------------------------------------------------

function GenerateSGroups(N::Int)
    groups = Vector{Vector{String}}()

    for mask in 0:(2^N - 1)
        S = Vector{String}(undef, N)

        for qubit in 1:N
            # höchstwertiges Bit zuerst
            bit = (mask >> (N - qubit)) & 1
            S[qubit] = bit == 0 ? "IZ" : "XY"
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


function simulateMeasurement(rho, projectors, n)
    isempty(projectors) && return Float64[]
    probs = [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), n)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n
end

function reconstructDensityMatrixWithSeeqstMLE(n::Int, rho_true::Matrix{ComplexF64}, N::Int)
    d = 2^n
    
    # 1. Gruppen generieren
    S = GenerateSGroups(n)
    
    all_projectors = ComplexF64[]
    all_counts = Float64[]
    
    # 2. Für jede Gruppe E/O-Basen erstellen und Messungen simulieren
    for si in S
        SE = generateEigenstatesE(si)
        SO = generateEigenstatesO(si)
        pSE = ProjectorsFromEigenstates(SE)
        pSO = ProjectorsFromEigenstates(SO)
        cSE = simulateMeasurement(rho_true, pSE, N) * N
        cSO = simulateMeasurement(rho_true, pSO, N) * N
        
        all_projectors = vcat(all_projectors, pSE, pSO)
        all_counts     = vcat(all_counts, cSE, cSO)
    end
    
    # 3. Variablen und Constraints für MLE
    ρ = ComplexVariable(d, d)
    constraints = [
        ρ == ρ',     # Hermitesch
        ρ ⪰ 0,       # Positiv semidefinit
        tr(ρ) == 1   # Spur = 1
    ]
    
    eps = 1e-9
    loglik = sum(
        all_counts[i] * log(real(tr(ρ * all_projectors[i])) + eps)
        for i in eachindex(all_projectors)
    )
    
    # 4. Optimierungsproblem lösen
    problem = maximize(loglik, constraints)
    solve!(problem, SCS.Optimizer; silent_solver=true)
    
    # 5. Rekonstruierte Dichtematrix zurückgeben
    rho_mle = evaluate(ρ)
    return rho_mle
end


end # module
