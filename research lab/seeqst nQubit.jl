using LinearAlgebra
using StatsBase
using Distributions
using Convex
using SCS
using Plots

# ⚠️ IterTools NICHT verwenden
# using IterTools

include("moduleRandomDensityMatrix.jl")
include("moduleSeeqst.jl")
include("moduleSimulateMeasurement.jl")

using .RandomDensityMatrix
using .SEEQSTEigenstates
using .SimulateMeasurement


"""
Goal of this file is to reconstruct a density matrix
using the scheme from:

Selective and Efficient Quantum State Tomography for Multi-Qubit Systems
by Aniket Patel.

Additionally, this file provides functionality to generate
all Pauli-operator combinations belonging to a given S-group.
"""

# ---------------- PARAMETERS ----------------

n = 2
dim = 2^n
nMeasurements = 1000

RhoTrue = GenerateRandomDensityMatrix(dim)

# ---------------- S-GROUPS ------------------

S_groups = GenerateSGroups(n)

# ---------------- COMBINATION FUNCTION ------------------
# ✅ Robust against IterTools pollution
# ✅ Always returns Vector{String}

function generate_combinations(ops::Vector{String})
    char_lists = [collect(op) for op in ops]

    result = String[]

    for t in Base.Iterators.product(char_lists...)
        push!(result, join(t))
    end

    return result
end

# ---------------- MAIN LOOP ------------------

for S in S_groups
    eigE = GenerateEigenstatesE(S)    # E-group
    eigO = GenerateEigenstatesO(S)    # O-group

    expanded_ops = generate_combinations(S)

    #println("S-group = ", S)
    #println("Expanded operators = ", expanded_ops)
    #println("Type = ", typeof(expanded_ops))
    #println()
end

# ---------------- FINAL TEST ------------------

ops = ["IZ", "XY","IZ"]
all_combos = generate_combinations(ops)

println("TEST")
println("Input  = ", ops)
println("Output = ", all_combos)
println("Type   = ", typeof(all_combos))

function pauli_eigenvalues(p::Char)
    if p == 'I'
        return [1, 1]
    elseif p == 'X' || p == 'Y' || p == 'Z'
        return [1, -1]
    else
        error("Unknown Pauli operator: $p")
    end
end

function pauli_string_eigenvalues(pauli::String)
    eigvals = [1]   # neutraler Startwert

    for p in pauli
        local_eigs = pauli_eigenvalues(p)
        eigvals = [a*b for a in eigvals, b in local_eigs]
        eigvals = vec(eigvals)  # garantiert 1D-Array
    end

    return eigvals
end

println(pauli_string_eigenvalues("I"))
# [1, 1]

println(pauli_string_eigenvalues("X"))
# [1, -1]

println(pauli_string_eigenvalues("IX"))
# [1, -1, 1, -1]

println(pauli_string_eigenvalues("XZ"))
# [1, -1, -1, 1]

println(pauli_string_eigenvalues("IZX"))
# [1, -1, 1, -1, -1, 1, -1, 1]
