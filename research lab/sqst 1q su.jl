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

d=2
n_shots=5000
rho_true=random_density_matrix(d)
#rho_true=[0.6  0.2+0.1im;
#              0.2-0.1im  0.4]

function simulate_measurement(rho, projectors, n)
    probs = [real(tr(rho * P)) for P in projectors]
    outcomes = sample([0, 1], Weights(probs), n)
    counts = [sum(outcomes .== i) for i in [0, 1]]
    return counts[1], counts[2]
end

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

poz, p1z = simulate_measurement(rho_true, proj_z, n_shots)
p2x, p3x = simulate_measurement(rho_true, proj_x, n_shots)
p4y, p5y = simulate_measurement(rho_true, proj_y, n_shots)

counts = [poz, p1z, p2x, p3x, p4y, p5y]

n00_1z=(p_0' * p_0) * conj(p_0' * p_0)
n00_2z=(p_0' * p_1) * conj(p_0' * p_1)
n00_1x=(p_0' * p_2) * conj(p_0' * p_2)
n00_2x=(p_0' * p_3) * conj(p_0' * p_3)
n00_1y=(p_0' * p_4) * conj(p_0' * p_4)
n00_2y=(p_0' * p_5) * conj(p_0' * p_5)

n01_1z=(p_0' * p_0) * conj(p_1' * p_0)
n01_2z=(p_0' * p_1) * conj(p_1' * p_1)
n01_1x=(p_0' * p_2) * conj(p_1' * p_2)
n01_2x=(p_0' * p_3) * conj(p_1' * p_3)
n01_1y=(p_0' * p_4) * conj(p_1' * p_4)
n01_2y=(p_0' * p_5) * conj(p_1' * p_5)

n10_1z=(p_1' * p_0) * conj(p_0' * p_0)
n10_2z=(p_1' * p_1) * conj(p_0' * p_1)
n10_1x=(p_1' * p_2) * conj(p_0' * p_2)
n10_2x=(p_1' * p_3) * conj(p_0' * p_3)
n10_1y=(p_1' * p_4) * conj(p_0' * p_4)
n10_2y=(p_1' * p_5) * conj(p_0' * p_5)

n11_1z=(p_1' * p_0) * conj(p_1' * p_0)
n11_2z=(p_1' * p_1) * conj(p_1' * p_1)
n11_1x=(p_1' * p_2) * conj(p_1' * p_2)
n11_2x=(p_1' * p_3) * conj(p_1' * p_3)
n11_1y=(p_1' * p_4) * conj(p_1' * p_4)
n11_2y=(p_1' * p_5) * conj(p_1' * p_5)

rho_00=(1/(3*n_shots))*(counts[1]*n00_1z+counts[2]*n00_2z+counts[3]*n00_1x+counts[4]*n00_2x+counts[5]*n00_1y+counts[6]*n00_2y)
rho_01=(1/(3*n_shots))*(counts[1]*n01_1z+counts[2]*n01_2z+counts[3]*n01_1x+counts[4]*n01_2x+counts[5]*n01_1y+counts[6]*n01_2y)
rho_10=(1/(3*n_shots))*(counts[1]*n10_1z+counts[2]*n10_2z+counts[3]*n10_1x+counts[4]*n10_2x+counts[5]*n10_1y+counts[6]*n10_2y)
rho_11=(1/(3*n_shots))*(counts[1]*n11_1z+counts[2]*n11_2z+counts[3]*n11_1x+counts[4]*n11_2x+counts[5]*n11_1y+counts[6]*n11_2y)

rho_num = [rho_00  rho_01;
           rho_10  rho_11]

rho_num /= tr(rho_num)

function sqrt_hermitian(A::Matrix{ComplexF64})
    vals, vecs = eigen(A)
    sqrt_vals = sqrt.(vals)
    return vecs * Diagonal(sqrt_vals) * vecs'
end

function fidelity(rho1, rho2)
    sqrt_rho = sqrt_hermitian(rho1)
    return real(tr(sqrt_hermitian(sqrt_rho * rho2 * sqrt_rho)))^2
end

Δρ = rho_true-rho_num

println("Matrixelement-Differenzen:")
println(Δρ)
println("fidelity :", fidelity(rho_num,rho_true))
