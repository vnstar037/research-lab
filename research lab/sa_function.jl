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



n=10000

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

proj_z = [proj_z1, proj_z2]
proj_x = [proj_x1, proj_x2]
proj_y = [proj_y1, proj_y2]

projektoren_basis = [proj_z, proj_x, proj_y, proj_z]

function simulate_measurement(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes= sample(1:4,Weights(probs), n)
    counts= [sum(outcomes .== i) for i in 1:4]
    #return counts[1]/n,counts[2]/n,counts[3]/n,counts[4]/n
    return counts/n
end

function rho_num(rho)
    S=ones(4,4)
    M = Matrix{Vector{Float64}}(undef, 4, 4)

    for i in 1:4
        for j in 1:4
        # Alle 4 Projektorkombinationen
            projectors = [
                kron(projektoren_basis[i][1], projektoren_basis[j][1]),
                kron(projektoren_basis[i][1], projektoren_basis[j][2]),
                kron(projektoren_basis[i][2], projektoren_basis[j][1]),
                kron(projektoren_basis[i][2], projektoren_basis[j][2])
            ]
            print(projectors)
            M[i, j] = simulate_measurement(rho, projectors, n)
        end
    end

    for i in 2:4
        for j in 2:4
            S[i,j]=M[i,j][1]-M[i,j][2]-M[i,j][3]+M[i,j][4]
        end
    end

    for i in 1:4
        S[1,i]=M[1,i][1]-M[1,i][2]+M[1,i][3]-M[1,i][4]
        S[i,1]=M[i,1][1]+M[i,1][2]-M[i,1][3]-M[i,1][4]
    end

    S[1,1]=1

    rho_num=zeros(ComplexF64, 4, 4)
    for i in 1:4
        for j in 1:4
            rho_num+=S[i,j]*kron(omega[i],omega[j])
        end
    end
    rho_num=0.25*rho_num
    return rho_num
end

rho_num1=rho_num(rho)
#println("theo:")
#println(rho)
#println("num:")
#println(rho_num1)
#println("diff")
#println(rho-rho_num1)
