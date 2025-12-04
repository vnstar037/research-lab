using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using Plots

include("moduleRandomDensityMatrix.jl")
include("moduleSeeqst.jl")

using .RandomDensityMatrix
using .SEEQSTEigenstates

""" Goal of this file is a function, where you can recreate a density matrix 
with the scheme of this paper Selective and efficient quantum state tomography for multi-qubit systems by Aniket Patel.
Also to create a function, where you put in the indices of the matrixelement of the density matrix that you want to get.
"""

n=2
dim=2^n
nMeasurements=1000

RhoTrue=GenerateRandomDensityMatrix(dim)

S_groups = GenerateSGroups(n)

for S in S_groups
    eigE = GenerateEigenstatesE(S)    # E-group
    eigO = GenerateEigenstatesO(S)   # O-group
    println("S = ", S)
    println("E eigenstates: ", eigE)
    println("O eigenstates: ", eigO)
end