module QuantumMLEQutrit

using LinearAlgebra
using StatsBase
using Convex
using IterTools
using SCS

export RecreatingDensityMatrixWithMLEQutrit

# ---------------------------------------------------------
# 1) Projektoren für jeden Gell-Mann Generator
# ---------------------------------------------------------

function _projektoren_basis_1qutrit()
    e0 = [1.0+0im, 0, 0]
    e1 = [0, 1.0+0im, 0]
    e2 = [0, 0, 1.0+0im]

    # ── Identität: Rechenbasis ────────────────────────────
    proj_I = [e0*e0', e1*e1', e2*e2']

    # ── λ₁: |0⟩↔|1⟩ X-artig ─────────────────────────────
    vp1 = (e0 + e1) / sqrt(2)
    vm1 = (e0 - e1) / sqrt(2)
    proj_1 = [vp1*vp1', vm1*vm1', e2*e2']

    # ── λ₂: |0⟩↔|1⟩ Y-artig ─────────────────────────────
    vp2 = (e0 + 1im*e1) / sqrt(2)
    vm2 = (e0 - 1im*e1) / sqrt(2)
    proj_2 = [vp2*vp2', vm2*vm2', e2*e2']

    # ── λ₃: diagonal ─────────────────────────────────────
    proj_3 = [e0*e0', e1*e1', e2*e2']

    # ── λ₄: |0⟩↔|2⟩ X-artig ─────────────────────────────
    vp4 = (e0 + e2) / sqrt(2)
    vm4 = (e0 - e2) / sqrt(2)
    proj_4 = [vp4*vp4', e1*e1', vm4*vm4']

    # ── λ₅: |0⟩↔|2⟩ Y-artig ─────────────────────────────
    vp5 = (e0 + 1im*e2) / sqrt(2)
    vm5 = (e0 - 1im*e2) / sqrt(2)
    proj_5 = [vp5*vp5', e1*e1', vm5*vm5']

    # ── λ₆: |1⟩↔|2⟩ X-artig ─────────────────────────────
    vp6 = (e1 + e2) / sqrt(2)
    vm6 = (e1 - e2) / sqrt(2)
    proj_6 = [e0*e0', vp6*vp6', vm6*vm6']

    # ── λ₇: |1⟩↔|2⟩ Y-artig ─────────────────────────────
    vp7 = (e1 + 1im*e2) / sqrt(2)
    vm7 = (e1 - 1im*e2) / sqrt(2)
    proj_7 = [e0*e0', vp7*vp7', vm7*vm7']

    # ── λ₈: diagonal ─────────────────────────────────────
    proj_8 = [e0*e0', e1*e1', e2*e2']

    return [proj_I, proj_1, proj_2, proj_3,
            proj_4, proj_5, proj_6, proj_7, proj_8]
end

# ---------------------------------------------------------
# 2) Messung simulieren
# ---------------------------------------------------------

function _simulate_measurement_qutrit(rho, projectors, n_shots)
    projectors = vec(projectors)
    probs = [real(tr(rho * P)) for P in projectors]
    probs = max.(probs, 0)
    probs = probs / sum(probs)
    outcomes = sample(1:length(projectors), Weights(probs), n_shots)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n_shots
end

# ---------------------------------------------------------
# 3) MLE Tomographie
# ---------------------------------------------------------

"""
    RecreatingDensityMatrixWithMLEQutrit(rho_true, n_shots)

Vollständige MLE Tomographie für N Qutrits:

1. Generiere alle 9^N Projektorkombinationen
2. Simuliere Messungen
3. Maximiere Log-Likelihood mit Convex.jl
4. Gib rekonstruierte Dichtematrix zurück
"""
function RecreatingDensityMatrixWithMLEQutrit(rho_true, n_shots)

    dim = size(rho_true, 1)
    N   = Int(round(log(3, dim)))

    println("N=$N Qutrits, dim=$dim")
    println("Anzahl Messbasen: 9^$N = $(9^N)")
    println("Shots pro Basis: $n_shots\n")

    projektoren_basis = _projektoren_basis_1qutrit()

    # ── Alle Tensorprodukt-Kombinationen ──────────────────
    combi_proj = collect(Iterators.product(
        (projektoren_basis for _ in 1:N)...))
    combi_proj = reshape(combi_proj, :)

    projector_list = Matrix{ComplexF64}[]
    counts         = Float64[]

    for tupel in combi_proj
        # Tensorprodukt der Projektoren
        ts    = Iterators.product(tupel...)
        tkron = [reduce(kron, k) for k in ts]

        # Messe
        c = _simulate_measurement_qutrit(rho_true, tkron, n_shots)

        append!(projector_list, tkron)
        append!(counts, c)
    end

    println("Anzahl Projektoren: ", length(projector_list))
    println("Anzahl Counts:      ", length(counts))

    # ── MLE Optimierung ───────────────────────────────────
    println("\n── MLE Optimierung ──")

    rho = ComplexVariable(dim, dim)

    constraints = [
        rho == rho',
        rho ⪰ 0,
        tr(rho) == 1
    ]

    logL = sum(
        counts[i] * log(real(tr(rho * projector_list[i])) + 1e-10)
        for i in 1:length(projector_list)
    )

    problem = maximize(logL, constraints)
    solve!(problem, SCS.Optimizer; silent_solver=true)

    rho_rec = Matrix{ComplexF64}(evaluate(rho))

    println("Spur ρ_rec:          ", round(real(tr(rho_rec)), digits=4))
    println("Kleinster Eigenwert: ", round(minimum(real(eigvals(rho_rec))), digits=6))

    return rho_rec
end

end # module QuantumMLEQutrit