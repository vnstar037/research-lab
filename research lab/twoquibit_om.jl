using LinearAlgebra
#import Pkg; Pkg.add("StatsBase")
using StatsBase





function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

rho=random_density_matrix(4)

A=[ 1 0 0 1;
    0 0 0 0; 0 0 0 0;1 0 0 1]


n=10000


function rho_num_om(matrix,n)
    omega_0=[1 0; 0 1]
    omega_x= [ 0 1; 1 0]
    omega_y= [ 0 -1im; 1im 0]
    omega_z= [ 1 0; 0 -1]

    omega=[omega_0,omega_x,omega_y,omega_z]

#projektor z basis
    p_0= [1 ,0]
    p_1= [0,1]
    proj_0=p_0*p_0'
    proj_1=p_1*p_1'

#projektor x-basis
    p_2 = (1/sqrt(2)) * [1; 1]
    p_3 = (1/sqrt(2)) * [1; -1]
    proj_2=p_2*p_2'
    proj_3=p_3*p_3'

#projektor y-basis
    p_4= (1/sqrt(2)) * [1; 1im]
    p_5= (1/sqrt(2)) * [1; -1im]
    proj_4=p_4*p_4'
    proj_5=p_5*p_5'

    Id=Matrix(1I, 4, 4) 

    projektoren=[proj_0,proj_1,proj_2,proj_3,proj_4,proj_5]

#Projektoren für 2 quibit erstellen
    projektoren_2q = Matrix{Matrix{ComplexF64}}(undef, 6, 6)
    for i in in 1:6
        for j in 1:6
            projektoren_2q[i,j]=kron(projektoren[i],projektoren[j])
        end
    end

    counts=Matrix{Matrix{ComplexF64}}(undef, 4, 4)
    counts11=simulate_measurement(matrix,[projektoren_2q[1,1],projektoren_2q[1,2],projektoren_2q[2,1],projektoren_2q[2,2]],n)
    counts12=simulate_measurement(matrix,[projektoren_2q[1,3],projektoren_2q[1,4],projektoren_2q[2,3],projektoren_2q[2,4]],n)
    counts13=simulate_measurement(matrix,[projektoren_2q[1,5],projektoren_2q[1,6],projektoren_2q[2,5],projektoren_2q[2,6]],n)
    counts21=simulate_measurement(matrix,[projektoren_2q[3,1],projektoren_2q[3,2],projektoren_2q[4,1],projektoren_2q[4,2]],n)
    counts22=simulate_measurement(matrix,[projektoren_2q[3,3],projektoren_2q[3,4],projektoren_2q[4,3],projektoren_2q[4,4]],n)
    counts23=simulate_measurement(matrix,[projektoren_2q[3,5],projektoren_2q[3,6],projektoren_2q[4,5],projektoren_2q[4,6]],n)
    counts31=simulate_measurement(matrix,[projektoren_2q[5,1],projektoren_2q[5,2],projektoren_2q[6,1],projektoren_2q[6,2]],n)
    counts32=simulate_measurement(matrix,[projektoren_2q[5,3],projektoren_2q[5,4],projektoren_2q[6,3],projektoren_2q[6,4]],n)
    counts33=simulate_measurement(matrix,[projektoren_2q[5,5],projektoren_2q[5,6],projektoren_2q[6,5],projektoren_2q[6,6]],n)
    
    for 
    
    S=Matrix{Matrix{ComplexF64}}(undef, 4, 4)

end

function simulate_measurement(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes= sample(1:4,Weights(probs), n)
    counts= [sum(outcomes .== i) for i in 1:4]
    return counts/n
end
end