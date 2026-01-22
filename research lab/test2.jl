using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using QuantumInformation
using Plots

include("moduleSimulateMeasurement.jl")
include("moduleSeeqst.jl")
include("moduleSeeqstMLE.jl")

using .SimulateMeasurement
using .SEEQSTEigenstates
using .SeeqstMLE



sigma_0= [1 0; 0 1]
sigma_x= [ 0 1; 1 0]
sigma_y= [ 0 -1im; 1im 0]
sigma_z= [ 1 0; 0 -1]


function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

n=4
d=2^n
N=90000
rho_true=random_density_matrix(d)





#println(S)
#si=S[4]
#SE=generateEigenstatesE(si)
#SO=generateEigenstatesO(si)



#pSE=ProjectorsFromEigenstates(SE)
#pSO=ProjectorsFromEigenstates(SO)
#println(pSE)
#println(pSO)
#cSE=simulateMeasurement(rho_true,pSE,N)*N
#cSO=simulateMeasurement(rho_true,pSO,N)*N
#SiComb=GenerateCombinations(si)


function reconstruct_density_matrix(n::Int, rho_true::Matrix{ComplexF64}, N::Int)
    d = 2^n
    
    # 1. Gruppen generieren
    S = GenerateSGroups(n)
    println(S)
    
    all_projectors = ComplexF64[]
    all_counts = Float64[]
    
    # 2. Für jede Gruppe E/O-Basen erstellen und Messungen simulieren
    for si in S
        SE = generateEigenstatesE(si)
        SO = generateEigenstatesO(si)
        pSE = ProjectorsFromEigenstates(SE)
        pSO = ProjectorsFromEigenstates(SO)
        cSE = simulateMeasurement(rho_true, pSE, N) * N
        cSO = simulateMeasurement(rho_true, pSO, N) * N
        
        all_projectors = vcat(all_projectors, pSE, pSO)
        all_counts     = vcat(all_counts, cSE, cSO)
    end
    
    # 3. Variablen und Constraints für MLE
    ρ = ComplexVariable(d, d)
    constraints = [
        ρ == ρ',     # Hermitesch
        ρ ⪰ 0,       # Positiv semidefinit
        tr(ρ) == 1   # Spur = 1
    ]
    
    eps = 1e-12
    loglik = sum(
        all_counts[i] * (-log(real(tr(ρ * all_projectors[i])) + eps))
        for i in eachindex(all_projectors)
    )
    
    # 4. Optimierungsproblem lösen
    problem = minimize(loglik, constraints)
    solve!(problem, SCS.Optimizer)#; silent_solver=true)
    
    # 5. Rekonstruierte Dichtematrix zurückgeben
    rho_mle = evaluate(ρ)
    return rho_mle
end

# ------------------------------------------------------------
# 6. Ergebnisse vergleichen (nur zur Validierung!)
# ------------------------------------------------------------

# Dichtematrix rekonstruieren
#rho_mle = reconstruct_density_matrix(n, rho_true, N)
rho_mle=reconstructDensityMatrixWithSeeqstMLE(n,rho_true,N)
println("Rekonstruierte Dichtematrix:")
#display(rho_mle)

println("Wahre Dichtematrix:")
#display(rho_true)

F = fidelity(rho_mle,rho_true)
println("fidelity :", F)




#println(SiComb)
#EV1=PauliStringEigenvalues(SiComb[1])
#println(EV1,SiComb[1])
#EV2=PauliStringEigenvalues(SiComb[2])
#println(EV2,SiComb[2])
#EV3=PauliStringEigenvalues(SiComb[3])
#println(EV3,SiComb[3])
#EV4=PauliStringEigenvalues(SiComb[4])
#println(EV4,SiComb[4])

#println(pSO)
#ev3=ExpectationValuesFromCounts(SiComb,cSE,cSO)
#println(ev3)
#position=MatrixElementsForGroup(si)
#p_si=DensityMatrixFromGroup(ev3,position,n)
#println(p_si)
#println(rho_true)


#rho_num = RecreatingDensityMatrixWithSeeqst(rho_true, N)

#println("OG matrix",rho_true,"Ende")
#println(rho_num-rho_true)

#println(rho_num)

#F = fidelity(rho_num,rho_true)
#println("fidelity :", F)

#matrixelement 2x3 ist nicht stabil