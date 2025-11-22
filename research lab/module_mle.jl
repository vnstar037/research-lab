module QuantumMLE

#using LinearAlgebra
#using StatsBase
#using Convex
#using Iterators
#using SCS

using LinearAlgebra
using StatsBase
using Distributions
using Convex
using IterTools
using SCS
using Plots

export rho_MLE   # einzige Funktion, die nach außen geht


# ---------------------------------------------------------
# 1) 1-Qubit Projektoren (intern)
# ---------------------------------------------------------

function _projektoren_basis_1qubit()
    # Z-Basis
    p0 = [1, 0]
    p1 = [0, 1]
    proj_z = [p0*p0', p1*p1']

    # X-Basis
    px1 = (1/sqrt(2)) * [1; 1]
    px2 = (1/sqrt(2)) * [1; -1]
    proj_x = [px1*px1', px2*px2']

    # Y-Basis
    py1 = (1/sqrt(2)) * [1; 1im]
    py2 = (1/sqrt(2)) * [1; -1im]
    proj_y = [py1*py1', py2*py2']

    return [proj_z, proj_x, proj_y, proj_z]
end


# ---------------------------------------------------------
# 2) Messung simulieren (intern)
# ---------------------------------------------------------

function _simulate_measurement(rho, projectors, n_shots)
    projectors = vec(projectors)

    probs = [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), n_shots)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]

    return counts
end


# ---------------------------------------------------------
# 3) Maximum Likelihood Tomographie (extern!)
# ---------------------------------------------------------

"""
    rho_MLE(rho_true, n_shots)

Führt vollständige Maximum-Likelihood-Quanten­tomographie
durch und gibt die rekonstruierte Dichtematrix zurück.

Verwendung:
    rho_est = rho_MLE(rho_true, 10_000)

Dabei wird automatisch:
- die benötigten Projektoren erzeugt
- simulate_measurement intern ausgeführt
- die MLE-Optimierung gelöst
"""
function rho_MLE(rho_true, n_shots)

    dim = size(rho_true, 1)
    n = Int(round(log2(dim)))

    projectors_basis = _projektoren_basis_1qubit()

    # --- alle Tensorprodukte ---
    combi_proj = collect(Iterators.product((projectors_basis for _ in 1:n)...))
    combi_proj = reshape(combi_proj, :)

    projector_list = Matrix{ComplexF64}[]
    counts = Float64[]

    # Für alle Basenkombinationen messen
    for tupel in combi_proj
        ts = Iterators.product(tupel...)
        tkron = [reduce(kron, k) for k in ts]

        c = _simulate_measurement(rho_true, tkron, n_shots)

        append!(projector_list, tkron)
        append!(counts, c)
    end

    # --- MLE-Optimierung ---
    rho = ComplexVariable(dim, dim)

    constraints = [
        rho == rho',
        rho ⪰ 0,
        tr(rho) == 1
    ]

    logL = sum(
        counts[i] * (-log(real(tr(rho * projector_list[i])) + 1e-12))
        for i in 1:length(projector_list)
    )

    problem = minimize(logL, constraints)
    solve!(problem, SCS.Optimizer)

    return evaluate(rho)
end


end # module
