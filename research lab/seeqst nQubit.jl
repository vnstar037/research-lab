using LinearAlgebra
using StatsBase
using Distributions
using Convex
using SCS
using Plots

using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using QuantumInformation
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

#RhoTrue== [
#    0.3  0.1+0.05im  0.05-0.02im  0.1+0.0im;
#    0.1-0.05im  0.25  0.05+0.03im  0.05-0.01im;
#    0.05+0.02im 0.05-0.03im 0.2  0.08+0.04im;
#    0.1+0.0im 0.05+0.01im 0.08-0.04im 0.25
#]




RhoNum = zeros(ComplexF64, dim, dim)

S_groups = GenerateSGroups(n)

for S in S_groups
    eigE = GenerateEigenstatesE(S)   # Float-Zustände
    eigO = GenerateEigenstatesO(S)   # Complex-Zustände

    projectorsE = ProjectorsFromEigenstates(eigE)
    projectorsO = ProjectorsFromEigenstates(eigO)

    global countsE = simulateMeasurement(RhoTrue, projectorsE, nMeasurements)
    global countsO = simulateMeasurement(RhoTrue, projectorsO, nMeasurements)


    expanded_ops = generate_combinations(S)

    positionMatrixElements=MatrixElementsForGroup(S)

    global evs=ExpectationValuesFromCounts(expanded_ops,countsE,countsO)

    partialRho=DensityMatrixFromGroup(evs,positionMatrixElements,n)

    RhoNum .+= partialRho


end

Δρ = RhoTrue-RhoNum

F = fidelity(RhoTrue,RhoNum)

#println("Matrixelement-Differenzen:")
#println(Δρ)
println("fidelity :", F)

S1=S_groups[1]
eigE1= GenerateEigenstatesE(S1)
projE1=ProjectorsFromEigenstates(eigE1)
println(S1,eigE1)
println(projE1)