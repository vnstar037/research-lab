using LinearAlgebra
#import Pkg; Pkg.add("StatsBase")
using StatsBase
#Pkg.add("IterTools")
using IterTools 
#import Pkg; Pkg.add("MatrixFunctions")
#using MatrixFunctions
using QuantumInformation
using Plots

function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

function simulate_measurement(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes= sample([1:length(projectors)],Weights(probs), n)
    counts= [sum(outcomes .== i) for i in 1:length(projectors)]
    #return counts[1]/n,counts[2]/n,probs
    return counts/n
end




n=3
dim=2^n
n_shots=150
rho=random_density_matrix(dim)
rho = [
    0.15  0.02+0.01im  0      0       0.01   0       0      0
    0.02-0.01im  0.20  0.01   0       0      0       0      0
    0      0.01   0.10  0.02   0      0       0      0
    0      0       0.02  0.10  0      0       0      0
    0.01   0       0     0     0.15  0.01    0      0
    0      0       0     0     0.01  0.10    0.02   0
    0      0       0     0     0     0.02    0.10   0.01
    0      0       0     0     0     0       0.01   0.10
]


omega_0=[1 0; 0 1]
omega_x= [ 0 1; 1 0]
omega_y= [ 0 -1im; 1im 0]
omega_z= [ 1 0; 0 -1]

omega= [omega_0,omega_x,omega_y,omega_z]


#projektor z basis
p_0= [1 ;0]
p_1= [0;1]
proj_z1=p_0*p_0'
proj_z2=p_1*p_1'
#projektor x-basis
p_2 = (1/sqrt(2)) * [1; 1]
p_3 = (1/sqrt(2)) * [1; -1]
proj_x1=p_2*p_2'
proj_x2=p_3*p_3'

#projektor y-basis
p_4= (1/sqrt(2)) * [1; 1im]
p_5= (1/sqrt(2)) * [1; -1im]
proj_y1=p_4*p_4'
proj_y2=p_5*p_5'

proj_0 = [proj_z1, proj_z2]
proj_z = [proj_z1, -proj_z2]
proj_x = [proj_x1, -proj_x2]
proj_y = [proj_y1, -proj_y2]

projektoren_basis = [proj_0, proj_x, proj_y, proj_z]

function simulate_measurement_signed1(rho, projectors, n)
    counts_signed = 0.0

    # Berechne die effektiven Wahrscheinlichkeiten
    probs = Float64[]
    signs = Int[]

    for P in projectors
        p = real(tr(rho * P))
        sign = p ≥ 0 ? 1 : -1
        push!(probs, abs(p))
        push!(signs, sign)
    end

    # Normiere die Wahrscheinlichkeiten, damit sample funktioniert
    norm_probs = probs / sum(probs)

    # Ziehe n Stichproben nach den normierten Wahrscheinlichkeiten
    outcomes = sample(1:length(projectors), Weights(norm_probs), n)

    # Zähle jede gezogene Kategorie entsprechend ihrem Vorzeichen
    for o in outcomes
        counts_signed += signs[o]
    end

    return counts_signed / n
end

function rho_rec(rho,n_shots)

    combi_proj = collect(Iterators.product((projektoren_basis for _ in 1:n)...))
#combi_proj1 = reshape(permutedims(combi_proj), 1, :)
    combi_proj1=reshape(combi_proj, :)

    S=zeros(1,dim^2)

    for i in 1:length(combi_proj1)
        tupel=combi_proj1[i]
        ts=product(tupel...)
        ts3=[reduce(kron, k) for k in ts]
       ts4=reshape((ts3), 1, :)
        S[i]=simulate_measurement_signed1(rho,ts4,n_shots)
    end

    og=Matrix{Matrix{ComplexF64}}(undef, 1, dim^2)
    combi_omega=collect(Iterators.product((omega for _ in 1:n)...))
    combi_omega1 = reshape((combi_omega), 1, :)

    for i in 1:length(combi_omega1)
        oTupel1=combi_omega1[i]
        omega_ts=reduce(kron,oTupel1)
        og[i]=omega_ts
    end

    rho_num1=num_p(S,og,n)
    # Eingebaute Version der "Reparatur"
    eig = eigen(rho_num1)
    Dnew = Diagonal(max.(eig.values, 0))  # Negative Eigenwerte -> 0
    rho_num2 = eig.vectors * Dnew * eig.vectors'  # Rekonstruktion
    return rho_num2
end


#println(combi_proj1[1])











function num_p(S,og,n)
    rho_num=zeros(ComplexF64, dim, dim)
    for i in 1:length(S)
        rho_num+=S[i]*og[i]
    end
    rho_num=(1/2^n)*rho_num
    return rho_num
end


#function sqrt_hermitian(A::Matrix{ComplexF64})
#    vals, vecs = eigen(A)
#    sqrt_vals = sqrt.(vals)
#    return vecs * Diagonal(sqrt_vals) * vecs'
#end

#function fidelity(rho::Matrix{ComplexF64}, sigma::Matrix{ComplexF64})
#    sqrt_rho = sqrt_hermitian(rho)
#    F = sqrt_rho * sigma * sqrt_rho
#    return real(tr(sqrt_hermitian(F)))^2
#end




n_shots_list = 100:10:10000  # von 100 bis 10000 in 100er Schritten
fidelities = Float64[]


for n_shots in n_shots_list
    rho_num2 = rho_rec(rho, n_shots)
    F = fidelity(rho, rho_num2)
    #F_adj = 1 - abs(F - 1)   # Betraglicher Abstand zu 1, dann von 1 abziehen
    #push!(fidelities, F_adj)
    push!(fidelities, F)
end

# Plot 1: gesamte Fidelity vs. Anzahl Messungen
plt1 = plot(n_shots_list, fidelities,
    xlabel="Number of measurements (m)",
    ylabel="Fidelity",
    title="Fidelity of LI reconstruction vs. number of measurements",
    legend=false,
)

display(plt1)
savefig(plt1, "fidelity_plot_LI_normal.png")

# Plot 2: Zoom auf hohe Fidelity
plt2 = plot(n_shots_list, fidelities,
    xlabel="Number of measurements (m)",
    ylabel="Fidelity",
    title="Fidelity of LI reconstruction (Zoom 0.998-1.001)",
    legend=false,
    ylim=(0.998, 1.001),
)

display(plt2)
savefig(plt2, "fidelity_plot_LI_zoom.png")

#println(fidelities)



