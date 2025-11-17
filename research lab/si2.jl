using LinearAlgebra
#import Pkg; Pkg.add("StatsBase")
using StatsBase

function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

function simulate_measurement(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes= sample([1:length(projecots)],Weights(probs), n)
    counts= [sum(outcomes .== i) for i in 1:length(projecots)]
    #return counts[1]/n,counts[2]/n,probs
    return counts/n
end

rho=random_density_matrix(2)


n=1
dim=2^n
n_shots=1000

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
combi_proj=collect(Iterators.product((projektoren_basis for _ in 1:n)...))
combi_proj2 = reshape(permutedims(combi_proj), 1, :)

S=ones(1,length(combi_proj2))

function simulate_measurement1(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    #outcomes= sample([0,length(projectors)],Weights(probs), n)
    #counts= [sum(outcomes .== i) for i in 0:length(projectors)-1]
    outcomes= sample([0,1],Weights(probs), n)
    counts= [sum(outcomes .== i) for i in [0, 1]]
    return counts/n
end  


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



for i in 1:length(combi_proj2)
    ph=simulate_measurement_signed1(rho,combi_proj2[i][1],n_shots)
    S[i]=ph
end

Id=Matrix(1I, 2, 2) 
function rho_num(S)
    rho_num=zeros(dim,dim)
    for i in 1:length(omega)
       rho_num+=omega[i]*S[i]
    end
    return 0.5*(rho_num)
end



rho_num1=rho_num(S)
println("theo:")
println(rho)
println("num:")
println(rho_num1)
#println(combi_proj2)