include("moduleRandomDensityMatrix.jl")
include("moduleSimulateMeasurement.jl")
include("moduleMaximumLikelihoodEstimation.jl")
include("moduleLinearInversion.jl")
include("moduleSeeqst.jl")
include("StructureDensityMatrix.jl")





using .RandomDensityMatrix
using .QuantumMLE
using .LineareInversion
using QuantumInformation
using .SEEQSTEigenstates

n=2
dim=2^n
nMeasurements=10000

RhoTrue=GenerateRandomDensityMatrix(dim)
RhoRecreatedMLE=RecreatingDensityMatrixWithMaximumLikelihoodEstimation(RhoTrue,nMeasurements)
RhoRecreatedLI=RecreatingDensityMatrixWithLineareInversion(RhoTrue,nMeasurements)
RhoRecreatedSeeqst=RecreatingDensityMatrixWithSeeqst(RhoTrue,nMeasurements)


StructMLE=ReconstructedDensityMatrix(RhoRecreatedMLE,RhoTrue)
StructLI=ReconstructedDensityMatrix(RhoRecreatedLI,RhoTrue)
StructSeeqst=ReconstructedDensityMatrix(RhoRecreatedSeeqst,RhoTrue)
#println("Matrixelement-Differenzen:")
#println(Δρ)
println("fidelity LI :", StructLI.fidelity)
println("fidelity MLE :", StructMLE.fidelity)
println("fidelity Seeqst :", StructSeeqst.fidelity)

println("purity LI : ", StructLI.purity)
println("purity MLE : ", StructMLE.purity)
println("purity Seeqst : ", StructSeeqst.purity)


