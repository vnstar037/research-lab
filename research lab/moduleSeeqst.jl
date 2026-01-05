module SEEQSTEigenstates

using LinearAlgebra
using StatsBase

export GenerateSGroups, GenerateComputationalBasis, ValidPStates, ValidQStates,
       ComplementStateFromP, PermuteQubits, GenerateEigenstatesE, GenerateEigenstatesO,
       pauli_string_eigenvalues,MatrixElementsForGroup,DensityMatrixFromGroup, ProjectorsFromEigenstates,
       generate_combinations,ExpectationValuesFromCounts,pauli

# --------------------------------------------------------------------
# Generate all stabilizer groups for N qubits (strings "IZ" or "XY")
# --------------------------------------------------------------------
function GenerateSGroups(N::Int)
    groups = Vector{Vector{String}}()
    for mask in 0:(2^N - 1)
        S = Vector{String}(undef, N)
        for qubit in 1:N
            bit = (mask >> (qubit-1)) & 1
            S[qubit] = bit == 0 ? "IZ" : "XY"
        end
        push!(groups, S)
    end
    return groups
end

# --------------------------------------------------------------------
# Generate full computational basis of N qubits
# --------------------------------------------------------------------
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
# Complement p̄ (bitflip) for a p in basis_p
# --------------------------------------------------------------------
function ComplementStateFromP(basis_p::Vector{Vector{Float64}}, p::Vector{Float64})
    M = Int(log2(length(basis_p)))
    if M == 0
        return [1.0]
    end
    idx = findfirst(q -> q ≈ p, basis_p)
    idx === nothing && error("p not found in basis_p")
    idx -= 1
    bits = reverse(digits(idx, base=2, pad=M))
    flipped = 1 .- bits
    ket = [1.0]
    for b in flipped
        ket = kron(ket, b == 0 ? [1.0;0.0] : [0.0;1.0])
    end
    return ket
end

# --------------------------------------------------------------------
# Valid P states (only half, fix first XY qubit to |0⟩)
# --------------------------------------------------------------------
function ValidPStates(S::Vector{String})
    XY_pos = findall(x -> x == "XY", S)
    M = length(XY_pos)
    if M == 0
        return [[1.0]]
    end
    basis_p = GenerateComputationalBasis(M)
    valid = Vector{Vector{Float64}}()
    for i in 0:(2^M - 1)
        bits = reverse(digits(i, base=2, pad=M))
        if bits[1] == 0
            push!(valid, basis_p[i+1])
        end
    end
    return valid
end

# --------------------------------------------------------------------
# Full IZ basis
# --------------------------------------------------------------------
function ValidQStates(S::Vector{String})
    IZ_pos = findall(x -> x == "IZ", S)
    K = length(IZ_pos)
    if K == 0
        return [[1.0]]
    end
    return GenerateComputationalBasis(K)
end

# --------------------------------------------------------------------
# Permute qubits in a state vector according to π
# --------------------------------------------------------------------
function PermuteQubits(ψ::Vector{T}, π::Vector{Int}) where T
    N = length(π)
    dims = ntuple(i -> 2, N)
    ψtensor = reshape(ψ, dims)
    ψperm = permutedims(ψtensor, π)
    return reshape(ψperm, 2^N)
end

# --------------------------------------------------------------------
# Generate eigenstates of the stabilizer group (E-group, ±1)
# --------------------------------------------------------------------
function GenerateEigenstatesE(S::Vector{String})
    N = length(S)
    IZpos = findall(x -> x == "IZ", S)
    XYpos = findall(x -> x == "XY", S)
    M = length(XYpos)
    K = length(IZpos)
    basis_p_valid = ValidPStates(S)
    basis_q = ValidQStates(S)
    basis_p_full = GenerateComputationalBasis(M)
    eigenstates = Vector{Vector{Float64}}()
    π = vcat(XYpos, IZpos)

    for p in basis_p_valid
        p_ = ComplementStateFromP(basis_p_full, p)
        e_plus = (p .+ p_) ./ sqrt(2)
        e_minus = (p .- p_) ./ sqrt(2)
        for q in basis_q
            if K == 0
                push!(eigenstates, e_plus)
                push!(eigenstates, e_minus)
                continue
            end
            if M == 0
                push!(eigenstates, q)
                continue
            end
            ψplus = kron(e_plus, q)
            ψminus = kron(e_minus, q)
            push!(eigenstates, PermuteQubits(ψplus, π))
            push!(eigenstates, PermuteQubits(ψminus, π))
        end
    end

    return eigenstates
end

# --------------------------------------------------------------------
# Generate eigenstates of the O-group (±i)
# --------------------------------------------------------------------
function GenerateEigenstatesO(S::Vector{String})
    # 🔴 FALL: nur IZ → keine O-Eigenzustände
    if all(x -> x == "IZ", S)
        return Vector{Vector{ComplexF64}}()  # leere Menge
    end

    N = length(S)
    IZpos = findall(x -> x == "IZ", S)
    XYpos = findall(x -> x == "XY", S)
    M = length(XYpos)
    K = length(IZpos)

    basis_p_valid = ValidPStates(S)
    basis_q = ValidQStates(S)
    basis_p_full = GenerateComputationalBasis(M)

    eigenstates = Vector{Vector{ComplexF64}}()
    π = vcat(XYpos, IZpos)

    for p in basis_p_valid
        p_ = ComplementStateFromP(basis_p_full, p)
        e_plus  = (ComplexF64.(p) .+ im .* ComplexF64.(p_)) ./ sqrt(2)
        e_minus = (ComplexF64.(p) .- im .* ComplexF64.(p_)) ./ sqrt(2)

        for q in basis_q
            qc = ComplexF64.(q)

            if K == 0
                push!(eigenstates, e_plus)
                push!(eigenstates, e_minus)
                continue
            end

            if M == 0
                # Dieser Fall tritt jetzt praktisch nicht mehr auf,
                # aber wir lassen ihn sicherheitshalber drin
                push!(eigenstates, qc)
                continue
            end

            ψplus  = kron(e_plus, qc)
            ψminus = kron(e_minus, qc)

            push!(eigenstates, PermuteQubits(ψplus, π))
            push!(eigenstates, PermuteQubits(ψminus, π))
        end
    end

    return eigenstates
end

function ProjectorsFromEigenstates(eigs)
    isempty(eigs) && return Matrix{ComplexF64}[]
    return [ψ * ψ' for ψ in eigs]
end

function pauli_string_eigenvalues(s::String)
    ev = [1.0]
    for p in s
        ev = vec([a*b for a in ev, b in pauli_eigenvalues(p)])
    end
    return ev
end

function pauli_eigenvalues(p::Char)
    if p == 'I'
        return [1, 1]
    elseif p == 'X' || p == 'Y' || p == 'Z'
        return [1, -1]
    else
        error("Unknown Pauli operator: $p")
    end
end


function generate_combinations(S::Vector{String})
    charlists = map(collect, S)
    vec([join(t) for t in Iterators.product(charlists...)])
end

# --------------------------------------------------------------------
# Erwartungswerte
# --------------------------------------------------------------------
function ExpectationValuesFromCounts(paulis, cE, cO)
    Dict(p => dot(pauli_string_eigenvalues(p),
                  occursin('Y', p) ? cO : cE)
         for p in paulis)
end

# --------------------------------------------------------------------
# Matrixelemente & Rekonstruktion
# --------------------------------------------------------------------
function MatrixElementsForGroup(S)
    N = length(S); dim = 2^N
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
        for (p,v) in evs
            fac = prod(k->p[k]=='Y' ? (bi[k]==0 ? im : -im) :
                           p[k]=='Z' ? (bi[k]==0 ? 1 : -1) : 1, 1:n)
            ρ[i,j] += fac * v
        end
    end
    return ρ / 2^n
end

end # module

module SimulateMeasurementSeeqst

using LinearAlgebra
using StatsBase

export simulateMeasurementSeeqst

"""
    simulateMeasurement(rho, projectors, n)

Simuliert eine Messung mit Projektionen `projectors` an einem Zustand `rho`
und liefert die relativen Häufigkeiten der Ergebnisse (counts / n).

Falls `projectors` leer ist, wird eine leere Menge zurückgegeben.
"""
function simulateMeasurementSeeqst(rho, projectors, n)
    isempty(projectors) && return Float64[]

    # Wahrscheinlichkeiten: p_i = Tr(rho * P_i)
    probs = [real(tr(rho * P)) for P in projectors]

    outcomes = sample(1:length(projectors), Weights(probs), n)

    counts = [sum(outcomes .== i) for i in 1:length(projectors)]

    return counts ./ n
end

end # module


