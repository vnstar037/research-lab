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
ϵ = 0.01     # gewünschter Schätzfehler
δ = 0.05      # 1% Fehlerwahrscheinlichkeit
M = 4         # Anzahl der Matrixelemente, die du schätzt
#C = 5       # empirisch oder aus vorherigem Test

#n_shots = round(Int, C * (1/ϵ^2) * log(M) * log(1/δ))
#println(n_shots)
#n_shots=100000
#rho_true=random_density_matrix(d)
rho_true=[0.6  0.2+0.1im;
              0.2-0.1im  0.4]

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
proj_z1=(p_0,p_1)

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
proj=[p_0,p_1,p_2,p_3,p_4,p_5]

#poz, p1z = simulate_measurement(rho_true, proj_z, n_shots)
#p2x, p3x = simulate_measurement(rho_true, proj_x, n_shots)
#p4y, p5y = simulate_measurement(rho_true, proj_y, n_shots)

#counts = [poz, p1z, p2x, p3x, p4y, p5y]

function sqst_rho_1q(counts,n_shots)
    eta = zeros(ComplexF64, 4, 6)

    for i in 1:4
        current_proj = proj_z1[mod1(i, 2)]  # 1→p_0, 2→p_1, 3→p_0, 4→p_1
    
        for j in 1:6
            eta[i, j] = (current_proj' * proj[j]) * conj(current_proj' * proj[j])
        end
    end

    rho_num = zeros(ComplexF64, 2, 2)
    
    for i in 1:2
        for j in 1:2 
            rho_num[i,j]=(1/(3*n_shots))*(counts[1]*eta[i,1]+counts[2]*eta[i,2]+counts[3]*eta[i,3]+counts[4]*eta[i,4]+counts[5]*eta[i,5]+counts[6]*eta[i,6])
        end
    end

    return rho_num
end

function sqrt_hermitian(A::Matrix{ComplexF64})
    vals, vecs = eigen(A)
    sqrt_vals = sqrt.(vals)
    return vecs * Diagonal(sqrt_vals) * vecs'
end

function fidelity(rho1, rho2)
    sqrt_rho = sqrt_hermitian(rho1)
    return real(tr(sqrt_hermitian(sqrt_rho * rho2 * sqrt_rho)))^2
end



#for Ctest in [0.5, 1.0, 2.0, 5.0, 10.0]
#    N = required_N(ϵ, δ, M, Ctest)
#    poz, p1z = simulate_measurement(rho_true, proj_z, N)
#    p2x, p3x = simulate_measurement(rho_true, proj_x, N)
#    p4y, p5y = simulate_measurement(rho_true, proj_y, N)

#    counts = [poz, p1z, p2x, p3x, p4y, p5y]
#    rho_num=sqst_rho_1q(counts,N)#n_shots)

#    println("\nFidelity:")
#    println(fidelity(rho_true, rho_num))
    # simulate with N shots and compute fidelity (repeat several times)
#end



function required_N(ϵ, δ, C, M=4)
    return round(Int, C * (1 / ϵ^2) * log(M) * log(1 / δ))
end

function optimize_parameters(rho_true, proj_z, proj_x, proj_y; M=4)
    ϵ_values = range(0.01, 0.1, length=100)
    δ_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    C_values = range(0.5, 10, length=50)

    best_F = 0.0
    best_params = (ϵ = 0.0, δ = 0.0, C = 0.0, N = 0)

    for ϵ in ϵ_values
        for δ in δ_values
            for Ctest in C_values
                N = required_N(ϵ, δ, Ctest, M)
                if N < 50
                    continue
                end

                poz, p1z = simulate_measurement(rho_true, proj_z, N)
                p2x, p3x = simulate_measurement(rho_true, proj_x, N)
                p4y, p5y = simulate_measurement(rho_true, proj_y, N)

                counts = [poz, p1z, p2x, p3x, p4y, p5y]
                rho_num = sqst_rho_1q(counts, N)

                F = fidelity(rho_true, rho_num)

                if F > best_F
                    best_F = F
                    best_params = (ϵ = ϵ, δ = δ, C = Ctest, N = N)
                end
            end
        end
    end

    println("\n✅ Beste Parameterkombination gefunden:")
    println("   ϵ (Genauigkeit)        = $(best_params.ϵ)")
    println("   δ (Fehlerwahrscheinlichkeit) = $(best_params.δ)")
    println("   C (Konstante)           = $(best_params.C)")
    println("   N (benötigte Shots)     = $(best_params.N)")
    println("💎 Maximale Fidelity       = $(best_F)\n")

    return best_params, best_F
end

# --- Aufruf ganz unten ---
best_params, best_F = optimize_parameters(rho_true, proj_z, proj_x, proj_y; M=M)