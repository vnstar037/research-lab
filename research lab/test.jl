using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using Plots

include("moduleRandomDensityMatrix.jl")

using .RandomDensityMatrix

""" Goal of this file is a function, where you can recreate a density matrix 
with the scheme of this paper Selective and efficient quantum state tomography for multi-qubit systems by Aniket Patel.
Also to create a function, where you put in the indices of the matrixelement of the density matrix that you want to get.
"""

n=2
dim=2^n
nMeasurements=1000

RhoTrue=GenerateRandomDensityMatrix(dim)

function GenerateSGroups(N)
    groups = []

    # 2^N Kombinationen
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
# Generate computational basis of size N
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
# Complement p̄ (bitflip) for p in basis_p
# --------------------------------------------------------------------
function ComplementStateFromP(basis_p, p)
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
# Reduced P-basis (only half, fixing first XY qubit to |0>)
# --------------------------------------------------------------------
function ValidPStates(S)
    XY_pos = findall(x -> x == "XY", S)
    M = length(XY_pos)
    if M == 0
        return [[1.0]]
    end

    basis_p = GenerateComputationalBasis(M)
    valid = Vector{Vector{Float64}}()

    for i in 0:(2^M - 1)
        bits = reverse(digits(i, base=2, pad=M))
        if bits[1] == 0# Generate the computational basis of a N Qubit system in this order |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩ (n=3)

function GenerateComputationalBasis(N::Int)
    basis = Vector{Vector{Float64}}()
    dim = 2^N

    for i in 0:(dim-1)
        # bitstring, z. B. "010" für N=3
        bits = reverse(digits(i, base=2, pad=N))
        
        # baue ket aus Tensorprodukten
        ket = [1.0]  # Start mit Skalar
        for b in bits
            ket = kron(ket, b == 0 ? [1.0;0.0] : [0.0;1.0])
        end
        
        push!(basis, ket)
    end

    return basis
end

B = GenerateComputationalBasis(n)

function ComplementStateFromP(B::Vector{Vector{Float64}}, x)
    N = round(Int, log2(length(B[1])))

    # ----------------------------------------------------
    # Fall 1: x ist ein Index
    # ----------------------------------------------------
    if x isa Integer
        i = x - 1
    else
        # ------------------------------------------------
        # Fall 2: x ist ein Vektor → suche Index in B
        # ------------------------------------------------
        found = findfirst(b -> b ≈ x, B)
        if found === nothing
            error("Der gegebene Zustand kommt nicht in der Basis B vor.")
        end
        i = found - 1
    end

    # Bits des Index
    bits = reverse(digits(i, base=2, pad=N))

    # Komplement bilden
    bits_ = 1 .- bits

    # Komplementären Ket bauen
    ket = [1.0]
    for b in bits_
        ket = kron(ket, b == 0 ? [1.0; 0.0] : [0.0; 1.0])
    end

    return ket
end

function GenerateEigenstates(S::Vector{String})
    N = length(S)
    
    # Finde Positionen
    IZ_positions = findall(x -> x == "IZ", S)
    XY_positions = findall(x -> x == "XY", S)
    
    # Basis für p (off-diagonal) und q (diagonal)
    basis_p = GenerateComputationalBasis(length(XY_positions))
    basis_q = GenerateComputationalBasis(length(IZ_positions))
    
    eigenstates = []

    # Iteriere über alle Kombinationen |p> und |q>
    for p in basis_p
        p_ = ComplementStateFromP(basis_p,p)
        for q in basis_q
            # e_+ und e_-
            e_plus = (p .+ p_) ./ sqrt(2)
            e_minus = (p .- p_) ./ sqrt(2)
            
            # Tensorprodukt mit q
            state_plus = kron(e_plus, q)
            state_minus = kron(e_minus, q)
            
            # Permutiere wieder auf die Originalpositionen
            final_plus = similar(state_plus)
            final_minus = similar(state_minus)
            # Map XY -> XY_positions, IZ -> IZ_positions
            all_positions = vcat(XY_positions, IZ_positions)
            indices = sortperm(all_positions) # Permutation zurück zur Originalreihenfolge
            # Achtung: indices auf Amplituden anwenden ist tricky, für kleine N okay
            final_plus .= state_plus[indices]
            final_minus .= state_minus[indices]
            
            push!(eigenstates, final_plus)
            push!(eigenstates, final_minus)
        end
    end

    return eigenstates
end
            push!(valid, basis_p[i+1])
        end
    end
    return valid
end


# --------------------------------------------------------------------
# Full IZ basis
# --------------------------------------------------------------------
function ValidQStates(S)
    IZ_pos = findall(x -> x == "IZ", S)
    K = length(IZ_pos)
    if K == 0
        return [[1.0]]
    end
    return GenerateComputationalBasis(K)
end


# --------------------------------------------------------------------
# Apply qubit permutation π to a full state vector ψ of dimension 2^N
# --------------------------------------------------------------------
function PermuteQubits(ψ, π)
    N = length(π)
    dims = ntuple(i -> 2, N)           # (2,2,...,2)
    ψtensor = reshape(ψ, dims)         # convert 2^N -> 2×2×...×2

    ψperm = permutedims(ψtensor, π)    # permute qubits

    return reshape(ψperm, 2^N)         # back to full vector
end


# --------------------------------------------------------------------
# Main: generate all eigenstates of S
# --------------------------------------------------------------------
function GenerateEigenstates(S::Vector{String})
    N = length(S)

    IZpos = findall(x -> x == "IZ", S)
    XYpos = findall(x -> x == "XY", S)

    M = length(XYpos)   # #XY qubits
    K = length(IZpos)   # #IZ qubits

    basis_p_valid = ValidPStates(S)
    basis_q       = ValidQStates(S)
    basis_p_full  = GenerateComputationalBasis(M)

    eigenstates = Vector{Vector{Float64}}()

    # permutation vector: XY into their real positions, IZ into theirs
    π = vcat(XYpos, IZpos)

    for p in basis_p_valid
        p_ = ComplementStateFromP(basis_p_full, p)
        e_plus  = (p .+ p_) ./ sqrt(2)
        e_minus = (p .- p_) ./ sqrt(2)

        for q in basis_q

            # Case: only XY
            if K == 0
                push!(eigenstates, e_plus)
                push!(eigenstates, e_minus)
                continue
            end

            # Case: only IZ
            if M == 0
                push!(eigenstates, q)
                continue
            end

            # Standard: kron(e±, q)
            ψplus  = kron(e_plus,  q)
            ψminus = kron(e_minus, q)

            # Correctly permute by qubit positions
            ψp = PermuteQubits(ψplus,  π)
            ψm = PermuteQubits(ψminus, π)

            push!(eigenstates, ψp)
            push!(eigenstates, ψm)
        end
    end

    return eigenstates
end

function GenerateEigenstatesO(S::Vector{String})
    N = length(S)

    IZpos = findall(x -> x == "IZ", S)
    XYpos = findall(x -> x == "XY", S)

    M = length(XYpos)   # #XY qubits
    K = length(IZpos)   # #IZ qubits

    basis_p_valid = ValidPStates(S)
    basis_q       = ValidQStates(S)
    basis_p_full  = GenerateComputationalBasis(M)

    eigenstates = Vector{Vector{ComplexF64}}()   # O-Gruppe: komplex

    # permutation vector: XY into their real positions, IZ into theirs
    π = vcat(XYpos, IZpos)

    for p in basis_p_valid
        p_ = ComplementStateFromP(basis_p_full, p)
        # ±i combination
        e_plus  = (ComplexF64.(p) .+ im .* ComplexF64.(p_)) ./ sqrt(2)
        e_minus = (ComplexF64.(p) .- im .* ComplexF64.(p_)) ./ sqrt(2)

        for q in basis_q
            qc = ComplexF64.(q)  # make complex if only real

            # Case: only XY
            if K == 0
                push!(eigenstates, e_plus)
                push!(eigenstates, e_minus)
                continue
            end

            # Case: only IZ
            if M == 0
                push!(eigenstates, qc)
                continue
            end

            # Standard: kron(e±, q)
            ψplus  = kron(e_plus,  qc)
            ψminus = kron(e_minus, qc)

            # Correctly permute by qubit positions
            ψp = PermuteQubits(ψplus,  π)
            ψm = PermuteQubits(ψminus, π)

            push!(eigenstates, ψp)
            push!(eigenstates, ψm)
        end
    end

    return eigenstates
end


S_groups = GenerateSGroups(n)

for S in S_groups
    #println("S = ", S)
    eig = GenerateEigenstatesO(S)
    #println("Anzahl Eigenzustände: ", length(eig))
    println(eig)   # vorsicht: sehr lang!
end
