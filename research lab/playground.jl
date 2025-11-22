include("module random density matrix.jl")
include("module simulate measurement.jl")
include("module_mle.jl")
include("module_ta2.jl")

using .RandomDensityMatrix
using .QuantumMLE
using .linear_inversion
using QuantumInformation


n=2
dim=2^n
n_shots=1000

rho_true=random_density_matrix(dim)
rho_num_mle=rho_MLE(rho_true,n_shots)
rho_num_li=rho_rec_linear_inversion(rho_true,n_shots)

Δρ = rho_true-rho_num_mle

F = fidelity(rho_num_mle,rho_true)
F1= fidelity(rho_num_li,rho_true)

println("Matrixelement-Differenzen:")
println(Δρ)
println("fidelity :", F1)



