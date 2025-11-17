using LinearAlgebra
using StatsBase
using Distributions
using Convex
using SCS

function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

function simulate_measurement(rho, projectors, n)
    probs = [real(tr(rho * P)) for P in projectors]
    outcomes = sample([0, 1], Weights(probs), n)
    counts = [sum(outcomes .== i) for i in [0, 1]]
    return counts[1], counts[2]
end

rho_true = random_density_matrix(2)
rho_true=[ 0.7 0.3im ;
    -0.3im  0.3]

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

projectors = vcat(proj_z, proj_x, proj_y)


n_shots = 15000
poz, p1z = simulate_measurement(rho_true, proj_z, n_shots)
p2x, p3x = simulate_measurement(rho_true, proj_x, n_shots)
p4y, p5y = simulate_measurement(rho_true, proj_y, n_shots)

counts = [poz, p1z, p2x, p3x, p4y, p5y]



# MLE mit Convex.jl
rho = ComplexVariable(2, 2)

constraints = [
    rho == rho',       # Hermitesch
    rho ⪰ 0,           # positiv semidefinit
    tr(rho) == 1       # Spur = 1
]

logL = sum(
    -counts[i] * log(real(tr(rho * projectors[i])) + 1e-10)
    for i in 1:length(projectors)
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
