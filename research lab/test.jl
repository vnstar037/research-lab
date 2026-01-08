using LinearAlgebra

# ============================================================
# Computational basis |bits>
# ============================================================
function basis_vector(bits::Vector{Int})
    v = [1.0]
    for b in bits
        v = kron(v, b == 0 ? [1.0, 0.0] : [0.0, 1.0])
    end
    return v
end

# ============================================================
bitflip(bits::Vector{Int}) = [1 - b for b in bits]

# ============================================================
function bitstrings(n::Int)
    if n == 0
        return [Int[]]
    end
    prev = bitstrings(n - 1)
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
function permute_to_physical(state::AbstractVector, offdiag_pos::Vector{Int}, diag_pos::Vector{Int})
    perm = vcat(offdiag_pos, diag_pos)
    N = length(perm)
    reshaped = reshape(state, ntuple(_ -> 2, N)...)
    invp = invperm(perm)
    return vec(permutedims(reshaped, invp))
end

# ============================================================
# Parse Pauli strings
# ============================================================
function parse_paulis(paulis::Vector{String})
    offdiag_pos = Int[]
    diag_pos    = Int[]

    for (i, p) in enumerate(paulis)
        if p == "XY"
            push!(offdiag_pos, i)
        elseif p == "IZ"
            push!(diag_pos, i)
        else
            error("Unbekannter Pauli-String: $p")
        end
    end

    return offdiag_pos, diag_pos
end

# ============================================================
# MAIN FUNCTION (user-facing)
# ============================================================
function eigenbases_from_paulis(paulis::Vector{String})
    offdiag_pos, diag_pos = parse_paulis(paulis)

    M = length(offdiag_pos)
    K = length(diag_pos)

    # → Typ: ComplexF64
    states = Vector{Vector{ComplexF64}}()

    # --------------------------------------------------------
    # Case 1: M = 0 → pure computational basis
    # --------------------------------------------------------
    if M == 0
        for q in bitstrings(K)
            ψ = ComplexF64.(basis_vector(q))  # Float → Complex
            push!(states, permute_to_physical(ψ, Int[], diag_pos))
        end
        return states
    end

    # --------------------------------------------------------
    # Case 2: M > 0, K = 0
    # --------------------------------------------------------
    if K == 0
        p_bits = [p for p in bitstrings(M) if p[1] == 0]

        for p in p_bits
            pbar = bitflip(p)

            vp    = ComplexF64.(basis_vector(p))
            vpbar = ComplexF64.(basis_vector(pbar))

            plus  = (vp + vpbar) / sqrt(2)
            minus = (vp - vpbar) / sqrt(2)

            push!(states, permute_to_physical(plus,  offdiag_pos, Int[]))
            push!(states, permute_to_physical(minus, offdiag_pos, Int[]))
        end
        return states
    end

    # --------------------------------------------------------
    # Case 3: M > 0, K > 0
    # --------------------------------------------------------
    p_bits = [p for p in bitstrings(M) if p[1] == 0]
    q_bits = bitstrings(K)

    for p in p_bits
        pbar = bitflip(p)

        vp    = ComplexF64.(basis_vector(p))
        vpbar = ComplexF64.(basis_vector(pbar))

        plus  = (vp + vpbar) / sqrt(2)
        minus = (vp - vpbar) / sqrt(2)

        for q in q_bits
            vq = ComplexF64.(basis_vector(q))

            ψ_plus  = kron(plus,  vq)
            ψ_minus = kron(minus, vq)

            push!(states, permute_to_physical(ψ_plus,  offdiag_pos, diag_pos))
            push!(states, permute_to_physical(ψ_minus, offdiag_pos, diag_pos))
        end
    end

    return states
end

function eigenbases_O_from_paulis(paulis::Vector{String})
    # Off-diagonal Positionen bestimmen
    offdiag_pos = [i for (i,p) in enumerate(paulis) if p == "XY"]
    diag_pos    = [i for (i,p) in enumerate(paulis) if p == "IZ"]

    M = length(offdiag_pos)
    K = length(diag_pos)

    # Spezialfall: keine off-diagonal Elemente → leere Menge
    if M == 0
        return Vector{Vector{ComplexF64}}()
    end

    states = Vector{Vector{ComplexF64}}()
    q_bits = bitstrings(K)

    # Vertreter für (p, p̄) Paare
    p_bits = [p for p in bitstrings(M) if p[1] == 0]

    for p in p_bits
        pbar = bitflip(p)

        vp    = basis_vector(p)
        vpbar = basis_vector(pbar)

        # O-Basis: ungerade Anzahl σy → beide Varianten
        plus_O  = (vp + im*vpbar)/sqrt(2)
        minus_O = (vp - im*vpbar)/sqrt(2)

        for q in q_bits
            vq = basis_vector(q)
            ψ_plus  = kron(plus_O, vq)
            ψ_minus = kron(minus_O, vq)
            push!(states, permute_to_physical(ψ_plus, offdiag_pos, diag_pos))
            push!(states, permute_to_physical(ψ_minus, offdiag_pos, diag_pos))
        end
    end

    return states
end



function ProjectorsFromEigenstates(eigs)
    isempty(eigs) && return Matrix{ComplexF64}[]
    return [ψ * ψ' for ψ in eigs]
end




paulis = ["IZ","IZ","IZ"]
states = eigenbases_from_paulis(paulis)
states_O = eigenbases_O_from_paulis(paulis)

println("Anzahl Zustände: ", length(states))
println(states)
println(states_O)

projectors=ProjectorsFromEigenstates(states)
#println(projectors)
