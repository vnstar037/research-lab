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
    ket0 = [1.0, 0.0]
    ket1 = [0.0, 1.0]
    ketp = (ket0 .+ ket1) ./ sqrt(2)
    ketm = (ket0 .- ket1) ./ sqrt(2)

    # lokale Basen
    local_bases = Vector{Vector{Vector{Float64}}}()
    for s in S
        if s == "IZ"
            push!(local_bases, [ket0, ket1])
        elseif s == "XY"
            push!(local_bases, [ketp, ketm])
        else
            error("Unknown stabilizer type $s")
        end
    end

    eigenstates = Vector{Vector{Float64}}()

    # ⬇️ WICHTIG: product über reversed(local_bases)
    for combo_rev in Base.Iterators.product(reverse(local_bases)...)
        combo = reverse(combo_rev)  # zurück zur physikalischen Reihenfolge

        ψ = combo[1]
        for k in combo[2:end]
            ψ = kron(ψ, k)
        end

        push!(eigenstates, ψ)
    end

    return eigenstates
end

# --------------------------------------------------------------------
# Generate eigenstates of the O-group (±i)
# --------------------------------------------------------------------
function GenerateEigenstatesO(S::Vector{String})
    # 🔴 nur IZ → keine O-Eigenzustände
    if all(x -> x == "IZ", S)
        return Vector{Vector{ComplexF64}}()
    end

    # Basiszustände
    ket0 = ComplexF64[1, 0]
    ket1 = ComplexF64[0, 1]
    ket_ip = (ket0 .+ im .* ket1) ./ sqrt(2)   # |+i⟩
    ket_im = (ket0 .- im .* ket1) ./ sqrt(2)   # |-i⟩

    # lokale Basen pro Qubit
    local_bases = Vector{Vector{Vector{ComplexF64}}}()
    for s in S
        if s == "IZ"
            push!(local_bases, [ket0, ket1])
        elseif s == "XY"
            push!(local_bases, [ket_ip, ket_im])
        else
            error("Unknown stabilizer type $s")
        end
    end

    eigenstates = Vector{Vector{ComplexF64}}()

    # rechte Qubits laufen am schnellsten
    for combo_rev in Base.Iterators.product(reverse(local_bases)...)
        combo = reverse(combo_rev)

        ψ = combo[1]
        for k in combo[2:end]
            ψ = kron(ψ, k)
        end

        push!(eigenstates, ψ)
    end

    return eigenstates
end


function ProjectorsFromEigenstates(eigs)
    isempty(eigs) && return Matrix{ComplexF64}[]
    return [ψ * ψ' for ψ in eigs]
end

function pauli_eigenvalues(p::Char)
    if p == 'I'
        return [1.0, 1.0]
    elseif p in ('X','Y','Z')
        return [1.0, -1.0]
    else
        error("Unknown Pauli operator: $p")
    end
end

# --------------------------------------------------------------------
# Eigenwerte eines Pauli-Strings (Kronecker-artiges Produkt)
# Die Reihenfolge der Bits ist so gewählt, dass die erste Position im String
# der langsam rotierende Faktor ist
# --------------------------------------------------------------------
function pauli_string_eigenvalues(s::String)
    ev = [1.0]

    # reverse(s) → erster Buchstabe rotiert langsam
    for p in reverse(s)
        local_eigs = pauli_eigenvalues(p)
        ev = vec([a*b for a in ev, b in local_eigs])
    end

    return ev
end

function generate_combinations(liste::Vector{String})
    # Wir kehren die Liste um (reverse), damit die Zeichen des 
    # ersten Strings im Produkt am schnellsten rotieren.
    kombinationen = Iterators.product(reverse(liste)...)
    
    # Beim Zusammenfügen (join) kehren wir die Tupel wieder um, 
    # damit die ursprüngliche Zeichenfolge (1. String, 2. String) erhalten bleibt.
    return vec([join(reverse(k)) for k in kombinationen])
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
        # Bits von i und j (0 oder 1) in der Qubit-Basis
        bi = reverse(digits(i-1, base=2, pad=n))
        bj = reverse(digits(j-1, base=2, pad=n))

        for (pauli_string, v) in evs
            # Faktor für jedes Pauli-String-Element
            fac = 1.0 + 0im
            for k in 1:n
                if pauli_string[k] == 'I'
                    fac *= 1
                elseif pauli_string[k] == 'Z'
                    fac *= bi[k] == 0 ? 1 : -1
                elseif pauli_string[k] == 'X'
                    # X flippt die Basis |0> <-> |1>, also nur nicht-diagonal
                    fac *= bi[k] == bj[k] ? 0 : 1
                elseif pauli_string[k] == 'Y'
                    # Y flippt wie X, aber multipliziert mit ±i
                    if bi[k] == bj[k]
                        fac *= 0
                    elseif bi[k] == 0 && bj[k] == 1
                        fac *= -im   # <0|Y|1> = -i
                    elseif bi[k] == 1 && bj[k] == 0
                        fac *= im    # <1|Y|0> = i
                    end
                end
            end
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


