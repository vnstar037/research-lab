using LinearAlgebra
#import Pkg; Pkg.add("StatsBase")
using StatsBase

A=[ 0.7 0.3im ;
    -0.3im  0.3]


function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

rho=random_density_matrix(2)

n=100000



function simulate_measurement(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes= sample([0,1],Weights(probs), n)
    counts= [sum(outcomes .== i) for i in [0, 1]]
    return counts[1]/n,counts[2]/n,probs
end

#projektor z basis
p_0= [1 ,0]
p_1= [0,1]

#projektor zweite basis
p_2 = (1/sqrt(5)) * [1; 2]
p_3 = (1/sqrt(5))*[-2; 1]

#projektor y-basis
p_4= (1/sqrt(3)) * [1; sqrt(2)im]
p_5= (1/sqrt(3)) * [sqrt(2)im; 1]
    

function rho_num_nom(matrix,n,p_0,p_1,p_2,p_3,p_4,p_5)

    omega_0= [ 1 0; 0 1]
    omega_x= [ 0 1; 1 0]
    omega_y= [ 0 -1im; 1im 0]
    omega_z= [ 1 0; 0 -1]
    omega=[omega_0,omega_x,omega_y,omega_z]

    
    proj_0=p_0*p_0'
    proj_1=p_1*p_1'
    tau_1=proj_0-proj_1


    proj_2=p_2*p_2'#-p_3*p_3'
    proj_3=p_3*p_3'#-p_2*p_2'
    tau_2=proj_2-proj_3

    proj_4=p_4*p_4'
    proj_5=p_5*p_5'
    tau_3=proj_4-proj_5

    tau=[omega_0,tau_2,tau_3,tau_1]

    Id=Matrix(1I, 2, 2) 


    poz,p1z,probsz= simulate_measurement(matrix,[proj_0,proj_1],n)
    p2x,p3x,probsx= simulate_measurement(matrix,[proj_2,proj_3],n)
    p4y,p5y,probsy= simulate_measurement(matrix,[proj_4,proj_5],n)
    expected_z=poz-p1z #T_z bzw T_3
    expected_x=p2x-p3x #T_x bzw T_1
    expected_y=p4y-p5y #T_y bzw T_2

    T=[1,expected_x,expected_y,expected_z]
    probs=[probsx,probsy,probsz]


    M = zeros(ComplexF64, 4, 4)

    for i in 1:4
        for j in 1:4
            M[i, j] = tr(tau[i] * omega[j])
        end
    end

    S=2*inv(M)*T


    rho_num= 0.5 * (Id + S[2] * omega_x + S[3] * omega_y + S[4] * omega_z)

    return rho_num,T,probs
end

rho_num1,T,probs=rho_num_nom(rho,n,p_0,p_1,p_2,p_3,p_4,p_5)

println("theoretical matrix:")
println(rho)
println("nummerical matrix")
println(rho_num1)
println("Erwartungswert und Wahrscheinlichkeiten:")
println(T,probs)

