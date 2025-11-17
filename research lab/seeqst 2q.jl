using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using Plots

sigma_0=[1 0; 0 1]
sigma_x= [ 0 1; 1 0]
sigma_y= [ 0 -1im; 1im 0]
sigma_z= [ 1 0; 0 -1]

function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

n=2
d=2^n
N=1000
rho_true=random_density_matrix(d)

function simulate_measurement(rho,projectors,N)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), N)
    counts= [sum(outcomes .== i) for i in 1:length(projectors)]
    #return counts[1]/n,counts[2]/n,probs
    return counts/N
end

#omega= [omega_0,omega_x,omega_y,omega_z]


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

ES1=[kron(sigma_0,sigma_0),kron(sigma_0,sigma_z),kron(sigma_z,sigma_0),kron(sigma_z,sigma_z)]
OS1=[]

ES2=[kron(sigma_0,sigma_x),kron(sigma_z,sigma_x)]
OS2=[kron(sigma_0,sigma_y),kron(sigma_z,sigma_y)]

ES3=[kron(sigma_x,sigma_0),kron(sigma_x,sigma_z)]
OS3=[kron(sigma_y,sigma_0),kron(sigma_y,sigma_z)]

ES4=[kron(sigma_x,sigma_x),kron(sigma_y,sigma_y)]
OS4=[kron(sigma_x,sigma_y),kron(sigma_y,sigma_x)] 

proj_00=[kron(proj_z1,proj_z1),kron(proj_z1,proj_z2),kron(proj_z2,proj_z1),kron(proj_z2,proj_z2)]

proj_0x=[kron(proj_z1,proj_x1),kron(proj_z1,proj_x2),kron(proj_z2,proj_x1),kron(proj_z2,proj_x2)]
proj_0y=[kron(proj_z1,proj_y1),kron(proj_z1,proj_y2),kron(proj_z2,proj_y1),kron(proj_z2,proj_y2)]

proj_x0=[kron(proj_x1,proj_z1),kron(proj_x1,proj_z2),kron(proj_x2,proj_z1),kron(proj_x2,proj_z2)]
proj_y0=[kron(proj_y1,proj_z1),kron(proj_y1,proj_z2),kron(proj_y2,proj_z1),kron(proj_y2,proj_z2)]

proj_xx=[kron(proj_x1,proj_x1),kron(proj_x1,proj_x2),kron(proj_x2,proj_x1),kron(proj_x2,proj_x2)]
proj_xy=[kron(proj_x1,proj_y1),kron(proj_x1,proj_y2),kron(proj_x2,proj_y1),kron(proj_x2,proj_y2)]

counts_00=simulate_measurement(rho_true,proj_00,N)

counts_0x=simulate_measurement(rho_true,proj_0x,N)
counts_0y=simulate_measurement(rho_true,proj_0y,N)

counts_x0=simulate_measurement(rho_true,proj_x0,N)
counts_y0=simulate_measurement(rho_true,proj_y0,N)

counts_xx=simulate_measurement(rho_true,proj_xx,N)
counts_xy=simulate_measurement(rho_true,proj_xy,N)

ev00=[1,1,1,1]

ev0x=[1,-1,1,-1]
ev0y=[1,-1,1,-1]

evx0=[1,1,-1,-1]
evy0=[1,1,-1,-1]

evxx=[1,-1,-1,1]
evxy=[1,-1,-1,1]

c00=dot(ev00,counts_00)

c0x=dot(ev0x,counts_0x)
c0y=dot(ev0y,counts_0y)

cx0=dot(evx0,counts_x0)
cy0=dot(evy0,counts_y0)

cxx=dot(evxx,counts_xx)
cxy=dot(evxy,counts_xy)

rho_0000=c00
rho_1111=rho_0000
rho_0101=(cxx+cyyim)/2
rho_1010=(cxx+cyyim)/2

rho_0001=(c0x+c0yim)/2
rho_0100=conj(rho_0001)
rho_1011=(cxx+cxyim)/2
rho_1110=conj(rho_1011)

rho_0111=(cxx+cxy)/2
rho_1101=conj(rho_0111)
rho_0010=(c0x+c0yim)/2
rho_1000=conj(rho_0010)

rho_1100=(cyy+cxxim)/2
rho_0011=(cxx+cyyim)/2
rho_1001=(cxy+cxy*im)/2
rho_0110=conj(rho_1001)