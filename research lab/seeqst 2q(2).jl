include("moduleRandomDensityMatrix.jl")

using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using Plots
using .RandomDensityMatrix

sigma_0=[1 0; 0 1]
sigma_x= [ 0 1; 1 0]
sigma_y= [ 0 -1im; 1im 0]
sigma_z= [ 1 0; 0 -1]



n=2
d=2^n
N=1000
RhoTrue=GenerateRandomDensityMatrix(d)

function SimulateMeasurement(rho,projectors,N)
    probs= [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), N)
    counts= [sum(outcomes .== i) for i in 1:length(projectors)]
    #return counts[1]/n,counts[2]/n,probs
    return counts/N
end


#projektor z basis
p0 = [1 ;0]
p1 = [0;1]
projz1 = p0*p0'
projz2 = p1*p1'
#projektor x-basis
p2 = (1/sqrt(2)) * [1; 1]
p3 = (1/sqrt(2)) * [1; -1]
projx1 = p2*p2'
projx2 = p3*p3'

#projektor y-basis
p4 = (1/sqrt(2)) * [1; 1im]
p5 = (1/sqrt(2)) * [1; -1im]
projy1 = p_4*p_4'
projy2 = p_5*p_5'

proj0 = [proj_z1, proj_z2]
projz = [proj_z1, -proj_z2]
projx = [proj_x1, -proj_x2]
projy = [proj_y1, -proj_y2]

projektoren_basis = [proj0, projx, projy, projz]

function generateEigenbasisOfSi()
end

