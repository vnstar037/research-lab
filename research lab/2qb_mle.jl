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

function simulate_measurement(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes= sample(1:4,Weights(probs), n)
    counts= [sum(outcomes .== i) for i in 1:4]
    return counts[1]/n,counts[2]/n,counts[3]/n,counts[4]/n
    #return counts/n
end

n=2
dim=2^n
n_shots=10000

rho_true= random_density_matrix(dim)
rho_true = [
    0.280 + 0.0im    0.054 - 0.023im  -0.028 + 0.089im  0.028 + 0.010im;
    0.054 + 0.023im  0.272 + 0.0im    -0.106 + 0.007im -0.067 - 0.077im;
   -0.028 - 0.089im -0.106 - 0.007im  0.243 + 0.0im    0.015 + 0.026im;
    0.028 - 0.010im -0.067 + 0.077im  0.015 - 0.026im   0.205 + 0.0im
]


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

for i in 1:dim
    for j in 1:dim
        p1=kron(projektoren_basis[i][1], projektoren_basis[j][1])
        p2=kron(projektoren_basis[i][1], projektoren_basis[j][2])
        p3=kron(projektoren_basis[i][2], projektoren_basis[j][1])
        p4=kron(projektoren_basis[i][2], projektoren_basis[j][2])
        push!(tensor_projectors, p1)
        push!(tensor_projectors, p2)
        push!(tensor_projectors, p3)
        push!(tensor_projectors, p4)
        projectors=[p1,p2,p3,p4]
        n1,n2,n3,n4=simulate_measurement(rho_true, projectors, n_shots)
        push!(counts, n1)
        push!(counts, n2)
        push!(counts, n3)
        push!(counts, n4)
    end
end

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
