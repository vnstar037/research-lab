using LinearAlgebra
#import Pkg; Pkg.add("StatsBase")
using StatsBase





function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

rho=random_density_matrix(2)

rho=[ 0.7 0.3im ;
    -0.3im  0.3]

n=15000

function rho_num_om(matrix,n)

    omega_x= [ 0 1; 1 0]
    omega_y= [ 0 -1im; 1im 0]
    omega_z= [ 1 0; 0 -1]

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

    Id=Matrix(1I, 2, 2) 

    poz,p1z,probsz= simulate_measurement(matrix,[proj_0,proj_1],n)
    p2x,p3x,probsx= simulate_measurement(matrix,[proj_2,proj_3],n)
    p4y,p5y,probsy= simulate_measurement(matrix,[proj_4,proj_5],n)
    expected_z=poz-p1z
    expected_x=p2x-p3x
    expected_y=p4y-p5y

    S=[expected_x,expected_y,expected_z]
    probs=[probsx,probsy,probsz]

    rho_num= 0.5 * (Id + expected_x * omega_x + expected_y * omega_y + expected_z * omega_z)
    return rho_num,S,probs
end

function simulate_measurement(rho,projectors,n)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes= sample([0,1],Weights(probs), n)
    counts= [sum(outcomes .== i) for i in [0, 1]]
    #return counts[1]/n,counts[2]/n,probs
    return counts[1]/n,counts[2]/n,probs
end


rho_num1,S,probs=rho_num_om(rho,n)

function sqrt_hermitian(A::Matrix{ComplexF64})
    vals, vecs = eigen(A)
    sqrt_vals = sqrt.(vals)
    return vecs * Diagonal(sqrt_vals) * vecs'
end

function fidelity(rho::Matrix{ComplexF64}, sigma::Matrix{ComplexF64})
    sqrt_rho = sqrt_hermitian(rho)
    F = sqrt_rho * sigma * sqrt_rho
    return real(tr(sqrt_hermitian(F)))^2
end

fid1=fidelity(rho,rho_num1)


println("theoretical matrix:")
println(rho)
println("nummerical matrix")
println(rho_num1)
println("Erwartungswert und Wahrscheinlichkeiten:")
println(S,probs)
println("fidelity:")
#println(sqrt(rho))
println(fid1)
