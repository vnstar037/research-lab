using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using QuantumInformation
using Plots


include("moduleSeeqst.jl")
using .SEEQSTEigenstates
using .SimulateMeasurementSeeqst

sigma_0= [1 0; 0 1]
sigma_x= [ 0 1; 1 0]
sigma_y= [ 0 -1im; 1im 0]
sigma_z= [ 1 0; 0 -1]

#println(kron(sigma_x,sigma_0)*kron(sigma_x,sigma_z)-kron(sigma_x,sigma_z)*kron(sigma_x,sigma_0))

function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

n=2
d=2^n
N=1000
rho_true=random_density_matrix(d)



#rho_true = [
#    0.3  0.1+0.05im  0.05-0.02im  0.1+0.0im;
#    0.1-0.05im  0.25  0.05+0.03im  0.05-0.01im;
#    0.05+0.02im 0.05-0.03im 0.2  0.08+0.04im;
#    0.1+0.0im 0.05+0.01im 0.08-0.04im 0.25
#]

#rho_true = [
#    0.28              0.10+0.07im   0.06-0.05im   0.08+0.09im;
#    0.10-0.07im       0.22          0.09+0.06im   0.07-0.04im;
#    0.06+0.05im       0.09-0.06im   0.25          0.05+0.08im;
#    0.08-0.09im       0.07+0.04im   0.05-0.08im   0.25
#]



S=GenerateSGroups(n)
println(S)
si=S[3]
SE=GenerateEigenstatesE(si)
SO=GenerateEigenstatesO(si)
pSE=ProjectorsFromEigenstates(SE)
pSO=ProjectorsFromEigenstates(SO)
cSE=simulateMeasurementSeeqst(rho_true,pSE,N)
cSO=simulateMeasurementSeeqst(rho_true,pSO,N)
SiComb=generate_combinations(si)
println(SiComb)
ev3=ExpectationValuesFromCounts(SiComb,cSE,cSO)
#println(ev3)
position=MatrixElementsForGroup(si)
p_si=DensityMatrixFromGroup(ev3,position,n)
#println(p_si)

function reconstruct_density_matrix(rho_true, n, N)
    d = 2^n
    rho_num = zeros(ComplexF64, d, d)

    S = GenerateSGroups(n)

    for si in S
        SE = GenerateEigenstatesE(si)
        SO = GenerateEigenstatesO(si)

        pSE = ProjectorsFromEigenstates(SE)
        pSO = ProjectorsFromEigenstates(SO)

        cSE = simulateMeasurementSeeqst(rho_true, pSE, N)
        cSO = simulateMeasurementSeeqst(rho_true, pSO, N)

        SiComb = generate_combinations(si)
        ev = ExpectationValuesFromCounts(SiComb, cSE, cSO)

        position = MatrixElementsForGroup(si)
        rho_num += DensityMatrixFromGroup(ev, position, n)
    end

    return rho_num
end

rho_num = reconstruct_density_matrix(rho_true, n, N)

#println("OG matrix",rho_true,"Ende")
#println(rho_num-rho_true)

#println(rho_num)

F = fidelity(rho_num,rho_true)
println("fidelity :", F)