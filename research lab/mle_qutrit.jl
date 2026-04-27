using LinearAlgebra
using StatsBase
using Convex
using Clarabel
using QuantumInformation
using Printf

# ══════════════════════════════════════════════════════════════════════════════
# 1. Zufällige Dichtematrix generieren
# ══════════════════════════════════════════════════════════════════════════════

function GenerateRandomDensityMatrixQutrit(n::Int)
    d = 3^n
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end

# ══════════════════════════════════════════════════════════════════════════════
# 2. Einzel-Qutrit Projektoren (analog zu _projektoren_basis_1qubit)
#
# Für Qutrits braucht man 9 Projektoren (3×3 Gell-Mann + Identität)
# aufgeteilt in 3 Messbasen mit je 3 Projektoren:
#
#   Basis 1: Rechenbasis         |0⟩,|1⟩,|2⟩
#   Basis 2: λ₁-Basis (|0⟩↔|1⟩) (|+⟩,|-⟩,|2⟩)
#   Basis 3: λ₂-Basis (|0⟩↔|1⟩) (|+i⟩,|-i⟩,|2⟩)
#   Basis 4: λ₄-Basis (|0⟩↔|2⟩) (|+⟩,|-⟩,|1⟩)
#   Basis 5: λ₅-Basis (|0⟩↔|2⟩) (|+i⟩,|-i⟩,|1⟩)
#   Basis 6: λ₆-Basis (|1⟩↔|2⟩) (|+⟩,|-⟩,|0⟩)
#   Basis 7: λ₇-Basis (|1⟩↔|2⟩) (|+i⟩,|-i⟩,|0⟩)
#
# 7 Basen × 3 Projektoren = 21 Messungen pro Qutrit
# Für N Qutrits: 7^N Basenkombinationen
# ══════════════════════════════════════════════════════════════════════════════

function projektoren_basis_1qutrit()
    # Rechenbasis
    e0 = [1.0+0im, 0, 0]
    e1 = [0.0+0im, 1, 0]
    e2 = [0.0+0im, 0, 1]

    s = 1/sqrt(2)

    # λ₁-Basis: Unterraum |0⟩↔|1⟩, reell
    p1_plus  = s * [1, 1, 0] .+ 0im
    p1_minus = s * [1, -1, 0] .+ 0im

    # λ₂-Basis: Unterraum |0⟩↔|1⟩, imaginär
    p2_plus  = s * ComplexF64[1, im, 0]
    p2_minus = s * ComplexF64[1, -im, 0]

    # λ₄-Basis: Unterraum |0⟩↔|2⟩, reell
    p4_plus  = s * [1, 0, 1] .+ 0im
    p4_minus = s * [1, 0, -1] .+ 0im

    # λ₅-Basis: Unterraum |0⟩↔|2⟩, imaginär
    p5_plus  = s * ComplexF64[1, 0, im]
    p5_minus = s * ComplexF64[1, 0, -im]

    # λ₆-Basis: Unterraum |1⟩↔|2⟩, reell
    p6_plus  = s * [0, 1, 1] .+ 0im
    p6_minus = s * [0, 1, -1] .+ 0im

    # λ₇-Basis: Unterraum |1⟩↔|2⟩, imaginär
    p7_plus  = s * ComplexF64[0, 1, im]
    p7_minus = s * ComplexF64[0, 1, -im]

    # Jede Basis hat 3 Projektoren (vollständige POVM)
    basis_1 = [e0*e0', e1*e1', e2*e2']                         # Rechenbasis
    basis_2 = [p1_plus*p1_plus', p1_minus*p1_minus', e2*e2']   # λ₁
    basis_3 = [p2_plus*p2_plus', p2_minus*p2_minus', e2*e2']   # λ₂
    basis_4 = [p4_plus*p4_plus', p4_minus*p4_minus', e1*e1']   # λ₄
    basis_5 = [p5_plus*p5_plus', p5_minus*p5_minus', e1*e1']   # λ₅
    basis_6 = [p6_plus*p6_plus', p6_minus*p6_minus', e0*e0']   # λ₆
    basis_7 = [p7_plus*p7_plus', p7_minus*p7_minus', e0*e0']   # λ₇

    return [basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, basis_7]
end

# ══════════════════════════════════════════════════════════════════════════════
# 3. Messung simulieren (analog zu _simulate_measurement)
# ══════════════════════════════════════════════════════════════════════════════

function simulate_measurement(rho, projectors, n_shots::Int)
    projectors = vec(projectors)
    probs = [real(tr(rho * P)) for P in projectors]
    probs = max.(probs, 0.0)
    probs ./= sum(probs)
    outcomes = sample(1:length(projectors), Weights(probs), n_shots)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n_shots
end

# ══════════════════════════════════════════════════════════════════════════════
# 4. MLE Rekonstruktion (analog zu RecreatingDensityMatrixWithMLE)
# ══════════════════════════════════════════════════════════════════════════════

"""
    mle_qutrit(rho_true, n_shots)

Maximum Likelihood Tomographie für N-Qutrit-Systeme.

Ablauf (analog zum Qubit-MLE-Modul):
  1. Einzel-Qutrit Projektoren in 7 Messbasen
  2. Alle 7^N Tensorprodukt-Kombinationen
  3. Messung simulieren → Häufigkeiten
  4. MLE: maximize Σᵢ fᵢ · log(Tr(ρ · Πᵢ))
     subject to: ρ = ρ†, ρ ≥ 0, Tr(ρ) = 1
"""
function mle_qutrit(rho_true::Matrix{ComplexF64}, n_shots::Int)

    d = size(rho_true, 1)
    n = round(Int, log(3, d))   # Anzahl Qutrits: d = 3^n

    projectors_basis = projektoren_basis_1qutrit()   # 7 Basen

    # Alle 7^N Basenkombinationen (analog zu Iterators.product im Qubit-Code)
    projector_list = Matrix{ComplexF64}[]
    counts         = Float64[]

    for combo in Iterators.product((projectors_basis for _ in 1:n)...)
        # combo ist ein Tupel von n Basen (jede Basis hat 3 Projektoren)
        # Tensorprodukt aller Kombinationen von Projektoren
        for proj_tuple in Iterators.product(combo...)
            P = reduce(kron, collect(proj_tuple))
            push!(projector_list, P)
        end

        # Messung für diese Basenkombination
        projs_combined = vec([reduce(kron, pt) for pt in Iterators.product(combo...)])
        c = simulate_measurement(rho_true, projs_combined, n_shots)
        append!(counts, c)
    end

    # MLE Optimierung
    ρ = ComplexVariable(d, d)
    constraints = [ρ == ρ', ρ ⪰ 0, tr(ρ) == 1]

    eps = 1e-9
    loglik = sum(
        counts[i] * log(real(tr(ρ * projector_list[i])) + eps)
        for i in eachindex(projector_list)
    )

    problem = maximize(loglik, constraints)
    solve!(problem, Clarabel.Optimizer; silent=true)

    return Matrix{ComplexF64}(evaluate(ρ))
end

# ══════════════════════════════════════════════════════════════════════════════
# 5. Fidelität
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

n       = 4          # Anzahl Qutrits
n_shots = 100_000     # Anzahl Messungen

println("N=$n Qutrits | d=$(3^n) | shots=$n_shots")
println("─"^50)

rho_true  = GenerateRandomDensityMatrixQutrit(n)
rho_recon = mle_qutrit(rho_true, n_shots)

F = fidelity(rho_true, rho_recon)
P = real(tr(rho_recon * rho_recon))

@printf("Fidelität: %.6f\n", F)
@printf("Reinheit:  %.6f\n", P)