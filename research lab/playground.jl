include("moduleRandomDensityMatrix.jl")
include("moduleSimulateMeasurement.jl")
include("moduleMaximumLikelihoodEstimation.jl")
include("moduleLinearInversion.jl")


using .RandomDensityMatrix
using .QuantumMLE
using .LineareInversion
using QuantumInformation


n=3
dim=2^n
nMeasurements=10000

RhoTrue=GenerateRandomDensityMatrix(dim)
RhoRecreatedMLE=RecreatingDensityMatrixWithMaximumLikelihoodEstimation(RhoTrue,nMeasurements)
RhoRecreatedLI=RecreatingDensityMatrixWithLineareInversion(RhoTrue,nMeasurements)

DeltaRhoMLE= RhoTrue-RhoRecreatedMLE
DeltaRhoLI= RhoTrue-RhoRecreatedLI

FidelityMLE = fidelity(RhoRecreatedMLE,RhoTrue)
FidelityLI= fidelity(RhoRecreatedLI,RhoTrue)
println("Matrixelement-Differenzen:")
#println(Δρ)
println("fidelity LI :", FidelityLI)
println("fidelity MLE :", FidelityMLE)



