

using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using QuantumInformation
using Plots

sigma_0= [1 0; 0 1]
sigma_x= [ 0 1; 1 0]
sigma_y= [ 0 -1im; 1im 0]
sigma_z= [ 1 0; 0 -1]

#println(kron(sigma_x,sigma_0)*kron(sigma_x,sigma_z)-kron(sigma_x,sigma_z)*kron(sigma_x,sigma_0))

function random_density_matrix(d)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)  # normieren auf Spur = 1
    return rho
end

n=2
d=2^n
N=10000
rho_true=random_density_matrix(d)

#rho_true = [
#    0.3  0.1+0.05im  0.05-0.02im  0.1+0.0im;
#    0.1-0.05im  0.25  0.05+0.03im  0.05-0.01im;
#    0.05+0.02im 0.05-0.03im 0.2  0.08+0.04im;
#    0.1+0.0im 0.05+0.01im 0.08-0.04im 0.25
#]



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

println(proj_xx)

counts_00=simulate_measurement(rho_true,proj_00,N)

counts_0x=simulate_measurement(rho_true,proj_0x,N)
counts_0y=simulate_measurement(rho_true,proj_0y,N)

counts_x0=simulate_measurement(rho_true,proj_x0,N)
counts_y0=simulate_measurement(rho_true,proj_y0,N)

counts_xx=simulate_measurement(rho_true,proj_xx,N)
counts_xy=simulate_measurement(rho_true,proj_xy,N)

ev00=[1,1,1,1]
ev0z=[1,-1,1,-1]
evz0=[1,1,-1,-1]
evzz=[1,-1,-1,1]

ev0x=[1,-1,1,-1]
evzx=[1,-1,-1,1]
ev0y=[1,-1,1,-1]
evzy=[1,-1,-1,1]

evx0=[1,1,-1,-1]
evy0=[1,1,-1,-1]
evxz=[1,-1,-1,1]
evyz=[1,-1,-1,1]


evxx=[1,-1,-1,1]
evyy=[1,-1,-1,1]
evxy=[1,-1,-1,1]
evyx=[1,-1,-1,1]

c00=dot(ev00,counts_00)
c0z=dot(ev0z,counts_00)
cz0=dot(evz0,counts_00)
czz=dot(evzz,counts_00)

c0x=dot(ev0x,counts_0x)
czx=dot(evzx,counts_0x)
c0y=dot(ev0y,counts_0y)
czy=dot(evzy,counts_0y)

cx0=dot(evx0,counts_x0)
cxz=dot(evxz,counts_x0)
cy0=dot(evy0,counts_y0)
cyz=dot(evyz,counts_y0)

cxx=dot(evxx,counts_xx)
cyy=dot(evyy,counts_xx)
cxy=dot(evxy,counts_xy)
cyx=dot(evyx,counts_xy)

rho_0000=(c00+c0z+cz0+czz)*1/4
rho_0101=(c00+c0z-cz0-czz)*1/4
rho_1010=(c00-c0z+cz0-czz)*1/4
rho_1111=(c00-c0z-cz0+czz)*1/4

# rho_0001 = <00|rho|01>
rho_0001 = (c0x - im*c0y + czx - im*czy)/4
rho_0100 = conj(rho_0001)

# rho_1011 = <10|rho|11>
rho_1011 = (c0x - im*c0y - czx + im*czy)/4
rho_1110 = conj(rho_1011)

rho_0010 = (cx0 + cxz - im*cy0 - im*cyz)/4
rho_1000 = conj(rho_0010)

rho_0111 = (cx0 - cxz - im*cy0 + im*cyz)/4
rho_1101 = conj(rho_0111)

rho_0011 = (cxx - cyy - im*cxy - im*cyx)/4
rho_1100 = conj(rho_0011)

rho_0110 = (cxy - cyx - im*cxx + im*cyy)/4
rho_1001 = conj(rho_0110)

rho_num = [
    rho_0000  rho_0001  rho_0010  rho_0011;
    rho_0100  rho_0101  rho_0110  rho_0111;
    rho_1000  rho_1001  rho_1010  rho_1011;
    rho_1100  rho_1101  rho_1110  rho_1111
]

Δρ = rho_true-rho_num

F = fidelity(rho_num,rho_true)

println("Matrixelement-Differenzen:")
#println(Δρ)
println("fidelity :", F)

