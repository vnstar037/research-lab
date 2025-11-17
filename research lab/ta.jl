using LinearAlgebra
#import Pkg; Pkg.add("StatsBase")
using StatsBase





function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

rho=random_density_matrix(8)



n=100000

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
    outcomes= sample(1:length(projectors),Weights(probs), n)
    counts= [sum(outcomes .== i) for i in 1:length(projectors)]
    #return counts[1]/n,counts[2]/n,counts[3]/n,counts[4]/n
    return counts/n
end    

function rho_num(rho)
    A = zeros(Float64, 1, 64)
    for i in 1:4, j in 1:4, k in 1:4
        idx = (i-1) * 16 + (j-1) * 4 + (k-1) + 1
        projectors = [
        kron(projektoren_basis[i][1], projektoren_basis[j][1], projektoren_basis[k][1]),
        kron(projektoren_basis[i][1], projektoren_basis[j][1], projektoren_basis[k][2]),
        kron(projektoren_basis[i][1], projektoren_basis[j][2], projektoren_basis[k][1]),
        kron(projektoren_basis[i][1], projektoren_basis[j][2], projektoren_basis[k][2]),
        kron(projektoren_basis[i][2], projektoren_basis[j][1], projektoren_basis[k][1]),
        kron(projektoren_basis[i][2], projektoren_basis[j][1], projektoren_basis[k][2]),
        kron(projektoren_basis[i][2], projektoren_basis[j][2], projektoren_basis[k][1]),
        kron(projektoren_basis[i][2], projektoren_basis[j][2], projektoren_basis[k][2])
        ]
        Ai=simulate_measurement(rho, projectors, n)

        if i>1&&j>1&&k>1
            A[idx]=Ai[1]-Ai[2]-Ai[3]+Ai[4]-Ai[5]+Ai[6]+Ai[7]-Ai[8]
        #(a+b)(c+d)(g-h)=(ac+ad+bc+bd)(g-h)=acg-ach+adg-adh+bcg-bch+bdg-bdg
        end

        if i==1&&j==1&&k>1
            A[idx]=Ai[1]-Ai[2]+Ai[3]-Ai[4]+Ai[5]-Ai[6]+Ai[7]-Ai[8]
        #(a+b)(c+d)(g-h)=(ac+ad+bc+bd)(g-h)=acg-ach+adg-adh+bcg-bch+bdg-bdg
        end
        if i==1&&j>1&&k==1
        A[idx]=Ai[1]+Ai[2]-Ai[3]-Ai[4]+Ai[5]+Ai[6]-Ai[7]-Ai[8]
       #(a+b)(c-d)(g+h)=(ac-ad+bc-bd)(g+h)=acg+ach-adg-adh+bcg+bch-bdg-bdg
        end
        if i>1&&j==1&&k==1
            A[idx]=Ai[1]+Ai[2]+Ai[3]+Ai[4]-Ai[5]-Ai[6]-Ai[7]-Ai[8]
        #(a-b)(c+d)(g+h)=(ac+ad-bc-bd)(g+h)=acg+ach+adg+adh-bcg-bch-bdg-bdg
        end

        if i==1&&j>1&&k>1
            A[idx]=Ai[1]-Ai[2]-Ai[3]+Ai[4]+Ai[5]-Ai[6]-Ai[7]+Ai[8]
        #(a+b)(c-d)(g-h)=(ac-ad+bc-bd)(g-h)=acg-ach-adg+adh+bcg-bch-bdg+bdh
        end
        if i>1&&j==1&&k>1
            A[idx]=Ai[1]-Ai[2]+Ai[3]-Ai[4]-Ai[5]+Ai[6]-Ai[7]+Ai[8]
        #(a-b)(c+d)(g-h)=acg-ach+adg-adh-bcg+bch-bdg+bdh
        end
        if i>1&&j>1&&k==1
            A[idx]=Ai[1]+Ai[2]-Ai[3]-Ai[4]-Ai[5]-Ai[6]+Ai[7]-Ai[8]
        #(a-b)(c-d)(g+h)=acg+ach-adg-adh-bcg-bch+bdg+bdh
        end
    end
    A[1]=1

    rho = zeros(ComplexF64, 8, 8)

    for i in 0:3, j in 0:3, k in 0:3
        idx = i * 16 + j * 4 + k + 1
        coeff = A[idx]
        basis_element = kron(omega[i+1], omega[j+1], omega[k+1])
        rho += coeff * basis_element
    end
    return rho / 8

end


rho_num1= rho_num(rho)

println("theo:")
println(rho[1,:])
println("num:")
println(rho_num1[1,:])
println("\nFrobenius-Norm Unterschied: ", norm(rho-rho_num1))

