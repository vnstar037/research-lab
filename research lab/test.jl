cd(@__DIR__)

include("moduleSeeqstQutrit.jl")
using .SeeqstHybridQutrit
using LinearAlgebra
using QuantumInformation
using Printf

println("═"^65)
println("Vergleich: Standard vs SEEQST vs tSEEQST (Qutrits)")
println("═"^65)
println()

shots = 5000
N     = 3

function embed_qubit_in_qutrit(rho_qubit::Matrix{ComplexF64}, N::Int)
    d_qubit  = 2^N   # = 8
    d_qutrit = 3^N   # = 27

    rho_qutrit = zeros(ComplexF64, d_qutrit, d_qutrit)

    # Kopiere die 8×8 Qubit-Matrix in die ersten 8×8 Einträge
    for i in 1:d_qubit, j in 1:d_qubit
        rho_qutrit[i, j] = rho_qubit[i, j]
    end

    return rho_qutrit
end

# ── Deine 3-Qubit Matrix ───────────────────────────────────────
rho_true = zeros(ComplexF64, 8, 8)
rho_true[1,1] =  0.5        # ρ[000,000]
rho_true[2,2] =  0.5        # ρ[001,001]
rho_true[1,2] = -0.5im      # ρ[000,001]
rho_true[2,1] =  0.5im      # ρ[001,000]

# ── Einbettung ─────────────────────────────────────────────────
rho_true = embed_qubit_in_qutrit(rho_true, 3)

#rho_true = GenerateRandomDensityMatrixNoZerosQutrits(N)

println(@sprintf("N = %d Qutrits, shots = %d", N, shots))
println()

# ── Circuit Anzahl ─────────────────────────────────────────────
blocks_all = collect(0:(4^N - 1))
non_circs  = BuildNonEntanglingCircuitsQutrit(blocks_all, N)
hyb_circs  = BuildHybridCircuitsQutrit(blocks_all, N)

n_standard = length(unique(String[
    c for g in non_circs for c in g if c != ""]))
n_seeqst   = length(unique(String[
    startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
    for g in hyb_circs for c in g if c != ""]))

println("── Circuit Anzahl ──")
println(@sprintf("  Standard SEEQST: %d Circuits", n_standard))
println(@sprintf("  SEEQST (Hybrid): %d Circuits", n_seeqst))
println()

# ── Standard SEEQST ────────────────────────────────────────────
println("── Standard SEEQST ──")
t0      = time()
rho_std = RecreatingDensityMatrixWithNonentanglingQutrit(
    rho_true, shots; verbose=false)
t_std   = time() - t0
F_std   = fidelity(rho_std, rho_true)
println(@sprintf("  Fidelity: %.4f  Runtime: %.2fs  Circuits: %d",
    F_std, t_std, n_standard))
println()

# ── SEEQST Hybrid ──────────────────────────────────────────────
println("── SEEQST (Hybrid) ──")
t0         = time()
rho_seeqst = RecreatingDensityMatrixWithSeeqstQutrit(
    rho_true, shots; verbose=false)
t_seeqst   = time() - t0
F_seeqst   = fidelity(rho_seeqst, rho_true)
println(@sprintf("  Fidelity: %.4f  Runtime: %.2fs  Circuits: %d",
    F_seeqst, t_seeqst, n_seeqst))
println()

# ── tSEEQST ────────────────────────────────────────────────────
println("── tSEEQST ──")
println()
println(@sprintf("  %-8s  %-10s  %-12s  %-12s  %-10s  %-10s",
    "t", "Circuits", "Fidelity", "F_bound", "Runtime", "Reduktion"))
println("  " * "─"^65)

rho_diag = real.(diag(rho_true))

for t in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    blocks_rel    = BlocksAboveThresholdQutrit(N, rho_diag, t)
    hybrid_circs_rel = BuildHybridCircuitsQutrit(blocks_rel, N)
    n_circ_t         = length(unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for g in hybrid_circs_rel for c in g if c != ""]))
    fb            = FidelityBoundQutrit(rho_true, rho_diag, t)
    reduktion     = (1 - n_circ_t / n_standard) * 100

    t0      = time()
    rho_rec = RecreatingDensityMatrixWithTSeeqstQutrit(
        rho_true, shots, t; verbose=false)
    t_t     = time() - t0
    F       = fidelity(rho_rec, rho_true)

    println(@sprintf("  t=%.2f   %-10d  %-12.4f  %-12.4f  %-10.2fs  %.1f%%",
        t, n_circ_t, F, fb, t_t, reduktion))
end
println()

# ── Zusammenfassung ────────────────────────────────────────────
println("═"^65)
println("Zusammenfassung")
println("═"^65)
println(@sprintf("  %-20s  %-10s  %-10s  %-10s",
    "Methode", "Fidelity", "Circuits", "Runtime"))
println("  " * "─"^55)
println(@sprintf("  %-20s  %-10.4f  %-10d  %.2fs",
    "Standard SEEQST", F_std, n_standard, t_std))
println(@sprintf("  %-20s  %-10.4f  %-10d  %.2fs",
    "SEEQST Hybrid",   F_seeqst, n_seeqst, t_seeqst))
println("  tSEEQST: siehe Tabelle oben")
println("═"^65)