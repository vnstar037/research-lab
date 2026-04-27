include("moduleRandomDensityMatrix.jl")
include("moduleSimulateMeasurement.jl")
include("moduleMaximumLikelihoodEstimation.jl")
include("moduleLinearInversion.jl")
include("moduleSeeqst.jl")
include("StructureDensityMatrix.jl")
include("moduleSeeqstMLE.jl")
include("SeeqstGD.jl")


using LinearAlgebra
#import Pkg; Pkg.add("StatsBase")
using StatsBase
#Pkg.add("IterTools")
using IterTools 
#import Pkg; Pkg.add("MatrixFunctions")
#using MatrixFunctions
using QuantumInformation
using Plots
using Convex
using SCS
using Zygote                    # neu
using Optimisers 

using .RandomDensityMatrix
using .QuantumMLE
using .LineareInversion
using QuantumInformation
using .SEEQSTEigenstates
using. SeeqstMLE
using .SeeqstGD     

n=4
dim=2^n
nMeasurements=100000

#RhoTrue=GenerateRandomDensityMatrix(dim)

"
RhoTrue = ComplexF64[
  0.135   0.020+0.015im  0.018-0.010im  0.015+0.008im  0.017-0.006im  0.014+0.009im  0.013-0.007im  0.011+0.005im
  0.020-0.015im  0.130   0.019+0.012im  0.016-0.009im  0.014+0.007im  0.013-0.008im  0.012+0.006im  0.010-0.004im
  0.018+0.010im  0.019-0.012im  0.125   0.021+0.011im  0.018-0.008im  0.016+0.007im  0.014-0.006im  0.012+0.004im
  0.015-0.008im  0.016+0.009im  0.021-0.011im  0.120   0.019+0.009im  0.017-0.007im  0.015+0.006im  0.013-0.005im
  0.017+0.006im  0.014-0.007im  0.018+0.008im  0.019-0.009im  0.115   0.020+0.010im  0.017-0.008im  0.015+0.006im
  0.014-0.009im  0.013+0.008im  0.016-0.007im  0.017+0.007im  0.020-0.010im  0.110   0.018+0.009im  0.016-0.007im
  0.013+0.007im  0.012-0.006im  0.014+0.006im  0.015-0.006im  0.017+0.008im  0.018-0.009im  0.105   0.019+0.010im
  0.011-0.005im  0.010+0.004im  0.012-0.004im  0.013+0.005im  0.015-0.006im  0.016+0.007im  0.019-0.010im  0.060
]
"

function GenerateRandomDensityMatrixNoZeros(n::Int)#; epsilon::Float64 = 1e-3)
    d=2^n
    # Zufällige komplexe Matrix
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    
    # Positiv semidefinite Matrix
    rho = M * M'
    
    # Füge kleine Konstante hinzu, damit keine Null-Einträge
    #rho .+= epsilon
    
    # Spur normieren
    rho /= tr(rho)
    
    return rho
end

RhoTrue=GenerateRandomDensityMatrixNoZeros(n)

#=
RhoTrue=[
  0.0755259867501587+0.0im            -0.02821276437259833-0.01502626656153282im   0.025556762458025673-0.0014753267059642477im  0.0024135392417497744+0.01612673793578668im   0.024322573007706422+0.0013663360703109226im  -0.007353671308960119-0.05450402227778001im   0.0564870921009008+0.0010569758211795947im    0.02842922625243099-0.0013932185787066803im
 -0.02821276437259833+0.01502626656153282im   0.11457643585729763+0.0im            0.012813417959115074+0.016225804736725528im  -0.013048836654219992-0.008242151694217456im  -0.042115692575640795+0.015480557216397652im  -0.024566282152498387+0.008260672001901049im   0.026421967779530542+0.04300112002429882im   -0.01627505252612963+0.03947065216302618im
  0.025556762458025673+0.0014753267059642477im 0.012813417959115074-0.016225804736725528im  0.16193190002369293+0.0im           -0.04356165918172081+0.01488078353924184im   -0.008557423866264251+0.01088991923175203im    0.00022939623097796813-0.006460916408762293im  0.06443524160275278+0.031242431161250597im    0.01883469571026719+0.026303311306108475im
  0.0024135392417497744-0.01612673793578668im -0.013048836654219992+0.008242151694217456im -0.04356165918172081-0.01488078353924184im  0.0733165162875684+0.0im             0.005832455926489849+0.013150219463279473im   -0.017926771184819796-0.037582116879580005im   0.0010717759560033074-0.014890038488797105im  -0.024206422694796895-0.020427912164343323im
  0.024322573007706422-0.0013663360703109226im -0.042115692575640795-0.015480557216397652im -0.008557423866264251-0.01088991923175203im  0.005832455926489849-0.013150219463279473im  0.07617117270296164+0.0im            -0.03903170383303372-0.009656998281625511im    0.0005338555935992341-0.04315827320966698im   0.025290630125316852-0.04699161390020296im
 -0.007353671308960119+0.05450402227778001im   -0.024566282152498387-0.008260672001901049im  0.00022939623097796813+0.006460916408762293im -0.017926771184819796+0.037582116879580005im -0.03903170383303372+0.009656998281625511im   0.19305386798647944+0.0im           -0.025342142046903878+0.03400789036846214im    0.03536411077371565+0.01629471731997165im
  0.0564870921009008-0.0010569758211795947im   0.026421967779530542-0.04300112002429882im   0.06443524160275278-0.031242431161250597im   0.0010717759560033074+0.014890038488797105im  0.0005338555935992341+0.04315827320966698im  -0.025342142046903878-0.03400789036846214im   0.1783928958792616+0.0im             0.0673114897899542+0.04822240493754667im
  0.02842922625243099+0.0013932185787066803im  -0.01627505252612963-0.03947065216302618im    0.01883469571026719-0.026303311306108475im  -0.024206422694796895+0.020427912164343323im  0.025290630125316852+0.04699161390020296im   0.03536411077371565-0.01629471731997165im    0.0673114897899542-0.04822240493754667im    0.12703122451257978+0.0im
]

println(RhoTrue)
=#
RhoRecreatedMLE=RecreatingDensityMatrixWithMaximumLikelihoodEstimation(RhoTrue,nMeasurements)
RhoRecreatedLI=RecreatingDensityMatrixWithLineareInversion(RhoTrue,nMeasurements)
RhoRecreatedSeeqst=RecreatingDensityMatrixWithSeeqst(RhoTrue,nMeasurements)
RhoRecreatedSeeqstMLE=reconstructDensityMatrixWithSeeqstMLE(RhoTrue,nMeasurements)

RhoRecreatedSeeqstGD = reconstructDensityMatrixWithSeeqstGD(
    RhoTrue, n;
    n_shots    = nMeasurements,
    lr         = 0.1,
    decay      = 0.999,
    iterations = 100 * n * 3    # wie beim Autor: 100*N*3
)

StructMLE=ReconstructedDensityMatrix(RhoRecreatedMLE,RhoTrue)
StructLI=ReconstructedDensityMatrix(RhoRecreatedLI,RhoTrue)
StructSeeqst=ReconstructedDensityMatrix(RhoRecreatedSeeqst,RhoTrue)
StructSeeqstMLE=ReconstructedDensityMatrix(RhoRecreatedSeeqstMLE,RhoTrue)
StructSeeqstGD  = ReconstructedDensityMatrix(RhoRecreatedSeeqstGD,  RhoTrue)
#println("Matrixelement-Differenzen:")
#println(Δρ)
println("fidelity LI :", StructLI.fidelity)
println("fidelity MLE :", StructMLE.fidelity)
println("fidelity Seeqst :", StructSeeqst.fidelity)
println("fidelity SeeqstMLE :", StructSeeqstMLE.fidelity)
println("fidelity SeeqstGD:   ", StructSeeqstGD.fidelity)

println("purity LI : ", StructLI.purity)
println("purity MLE : ", StructMLE.purity)
println("purity Seeqst : ", StructSeeqst.purity)

#=
n_shots_list = 100:50:50000  # von 100 bis 10000 in 100er Schritten
fidelities = Float64[]


for n_shots in n_shots_list
    #rho_num2 = RecreatingDensityMatrixWithLineareInversion(RhoTrue, n_shots)
    #rho_num2 = RecreatingDensityMatrixWithMaximumLikelihoodEstimation(RhoTrue, n_shots)
    rho_num2 = reconstructDensityMatrixWithSeeqstMLE(RhoTrue, n_shots)    
    F = fidelity(RhoTrue, rho_num2)
    #F_adj = 1 - abs(F - 1)   # Betraglicher Abstand zu 1, dann von 1 abziehen
    #push!(fidelities, F_adj)
    push!(fidelities, F)
end


plt1 = plot(n_shots_list, fidelities,
    xlabel="Number of measurements (m)",
    ylabel="Fidelity",
    title="Fidelity of SEEQST reconstruction vs. number of measurements",
    legend=false,
    titlefontsize=11,)

display(plt1)
savefig(plt1, "fidelity_plot_seeqst_normal.png")

# Second plot: zoom on high fidelity
plt2 = plot(n_shots_list, fidelities,
    xlabel="Number of measurements (m)",
    ylabel="Fidelity",
    title="Fidelity of SEEQST reconstruction (Zoom 0.9-1.0)",
    legend=false,
    titlefontsize=12,
    ylim=(0.9, 1.0))

display(plt2)
savefig(plt2, "fidelity_plot_seeqst_zoom.png")

println(fidelities)
=#
