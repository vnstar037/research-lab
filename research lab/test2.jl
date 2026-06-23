using LinearAlgebra
using QuantumInformation
using Printf

include("moduleSeeqstQutrit.jl")
include("test3.jl")

using .SeeqstHybridQutrit
using .SeeqstMixedQutrit

labels = ["00","01","02","10","11","12","20","21","22"]

# ══════════════════════════════════════════════════════════════
# Erstelle Matrix mit Elementen NUR aus einem gemischten Block
# Nicht notwendigerweise physikalisch!
# ══════════════════════════════════════════════════════════════
function create_block_matrix(t1::Int, t2::Int)
    levels = Dict(1=>(0,1), 2=>(0,2), 3=>(1,2))
    a0,b0  = levels[t1]
    a1,b1  = levels[t2]
    idx(q0,q1) = q0*3 + q1 + 1

    # 4 Eckzustände des gemischten Blocks
    corners = [idx(a0,a1), idx(a0,b1), idx(b0,a1), idx(b0,b1)]

    # Zufälliger Zustandsvektor NUR in den 4 Eckzuständen
    psi = zeros(ComplexF64, 9)
    v   = randn(ComplexF64, 4)
    v  /= norm(v)
    for (k,c) in enumerate(corners)
        psi[c] = v[k]
    end

    # ρ = |ψ⟩⟨ψ| → nur Elemente in corners×corners
    rho = psi * psi'
    return rho, corners
end

# ══════════════════════════════════════════════════════════════
# Zeige Matrix-Elemente
# ══════════════════════════════════════════════════════════════
function show_nonzero(rho, name)
    println("── $name ──")
    for i in 1:9, j in 1:9
        abs(rho[i,j]) < 1e-10 && continue
        println(@sprintf("  ρ[|%s⟩,|%s⟩] = %+.4f%+.4fi",
            labels[i], labels[j], real(rho[i,j]), imag(rho[i,j])))
    end
    println()
end

# ══════════════════════════════════════════════════════════════
# Test: Fidelity für alle 6 gemischten Blöcke
# ══════════════════════════════════════════════════════════════
mixed_blocks = [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)]
shots        = 5000

println("═"^70)
println("Test: Rekonstruktion von Matrizen mit nur gemischten Block-Elementen")
println("═"^70)
println()

println(@sprintf("  %-10s  %-14s  %-14s  %-14s",
    "Block", "Standard", "Hybrid", "Mixed (theor.)"))
println("  " * "─"^58)

for (t1,t2) in mixed_blocks
    rho_true, corners = create_block_matrix(t1, t2)

    println()
    println("Block ($t1,$t2): aktive Zustände = $(join(["|"*labels[c]*"⟩" for c in corners], ", "))")
    show_nonzero(rho_true, "Dichtematrix")

    # Standard SEEQST
    rho_std = RecreatingDensityMatrixWithNonentanglingQutrit(
        rho_true, shots; verbose=false)
    F_std = fidelity(rho_std, rho_true)

    # Hybrid SEEQST (CINC)
    rho_hyb = RecreatingDensityMatrixWithSeeqstQutrit(
        rho_true, shots; verbose=false)
    F_hyb = fidelity(rho_hyb, rho_true)

    # Mixed SEEQST (theoretische Gates)
    rho_mix = RecreatingDensityMatrixWithMixedSeeqstQutrit(
        rho_true, shots; verbose=false)
    F_mix = fidelity(rho_mix, rho_true)

    println(@sprintf("  Block(%d,%d)   F=%.4f        F=%.4f        F=%.4f",
        t1, t2, F_std, F_hyb, F_mix))
end

println()
println("═"^70)
println("Erwartung:")
println("  Mixed SEEQST ≥ Standard ≥ Hybrid")
println("  Hybrid ist schlechter weil CINC Gates gemischte Blöcke")
println("  NICHT selektiv messen können!")
println("═"^70)