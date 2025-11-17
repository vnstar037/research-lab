using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS

function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

function simulate_measurement(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes= sample(1:length(projectors),Weights(probs), n)
    counts= [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts/n
    #return counts/n
end

function simulate_measurement2(rho, projectors, n)
    projectors = vec(projectors)  # sicherstellen, dass es ein Vektor ist
    probs = [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), n)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts
end


n=3
dim=2^n
n_shots=10000
rho_true= random_density_matrix(dim)


# Projektoren definieren

# Z-Basis
p_0 = [1, 0]
p_1 = [0, 1]
proj_0 = p_0 * p_0'
proj_1 = p_1 * p_1'
proj_z = [proj_0, proj_1]

# X-Basis
p_2 = (1 / sqrt(2)) * [1; 1]
p_3 = (1 / sqrt(2)) * [1; -1]
proj_2 = p_2 * p_2'
proj_3 = p_3 * p_3'
proj_x = [proj_2, proj_3]

# Y-Basis
p_4 = (1 / sqrt(2)) * [1; 1im]
p_5 = (1 / sqrt(2)) * [1; -1im]
proj_4 = p_4 * p_4'
proj_5 = p_5 * p_5'
proj_y = [proj_4, proj_5]


projektoren_basis = [proj_z, proj_x, proj_y, proj_z]

tensor_projectors = Matrix{ComplexF64}[]
counts=Float64[]


combi_proj = collect(Iterators.product((projektoren_basis for _ in 1:n)...))

combi_proj1=reshape(combi_proj, :)

for i in 1:length(combi_proj1)
    tupel=combi_proj1[i]
    ts=product(tupel...)
    ts3=[reduce(kron, k) for k in ts]
    ts4=reshape((ts3), 1, :)

    result=simulate_measurement2(rho_true,ts4,n_shots)
    append!(tensor_projectors, ts3)
    append!(counts, result)
end



#println(tensor_projectors,size(tensor_projectors))
#println(counts,size(counts))

rho = ComplexVariable(dim, dim)

constraints = [
    rho == rho',       # Hermitesch
    rho ⪰ 0,           # positiv semidefinit
    tr(rho) == 1       # Spur = 1
]

logL = sum(
    -counts[i] * log(real(tr(rho * tensor_projectors[i])) )
    for i in 1:length(tensor_projectors)
)

problem = minimize(logL, constraints)
solve!(problem, SCS.Optimizer)  # kein verbose Argument mehr

rho_mle = evaluate(rho)


println("\nWahre Dichtematrix:")
println(rho_true)

println("\nRekonstruktion (MLE):")
println(rho_mle)

function sqrt_hermitian(A::Matrix{ComplexF64})
    vals, vecs = eigen(A)
    sqrt_vals = sqrt.(vals)
    return vecs * Diagonal(sqrt_vals) * vecs'
end

function fidelity(rho1, rho2)
    sqrt_rho = sqrt_hermitian(rho1)
    return real(tr(sqrt_hermitian(sqrt_rho * rho2 * sqrt_rho)))^2
end

println("\nFidelity:")
println(fidelity(rho_true, rho_mle))







