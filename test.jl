cd(@__DIR__)

using Printf
using LinearAlgebra
using QuantumInformation
using Random

include("moduleSeeqstQutrit.jl")
include("moduleSeeqstQutrit2.jl")
include("moduleTSeeqstQutrit2.jl")

using .SeeqstHybridQutrit
using .SeeqstMixedQutrit
using .TSeeqstMixedQutrit

# ══════════════════════════════════════════════════════════════
# Zufällige Dichtematrix
# ══════════════════════════════════════════════════════════════
function GenerateRandomDensityMatrixQutrits(N::Int)
    d        = 3^N
    n_active = rand(1:d)
    active   = randperm(d)[1:n_active]
    rho      = zeros(ComplexF64, d, d)
    for _ in 1:n_active
        psi         = zeros(ComplexF64, d)
        psi[active] = randn(ComplexF64, n_active) .+
                      1im .* randn(ComplexF64, n_active)
        psi        /= norm(psi)
        rho        += psi * psi'
    end
    return rho / tr(rho)
end

# ══════════════════════════════════════════════════════════════
# Einstellungen
# ══════════════════════════════════════════════════════════════
N_values = [2, 3, 4, 5]
shots    = 5000
t_fixed  = 0.05

Random.seed!(42)

# ══════════════════════════════════════════════════════════════
# Messungen
# ══════════════════════════════════════════════════════════════
println("═"^55)
println("Fidelity Comparison  (shots=$shots, t=$t_fixed)")
println("═"^55)
@printf("%-4s │ %-10s │ %-10s │ %-10s\n",
    "N", "Standard", "SEEQST", "tSEEQST")
println("─"^45)

for N in N_values

    rho_true = GenerateRandomDensityMatrixQutrits(N)

    # ── Standard ────────────────────────────────────────────
    F_std = NaN
    if N <= 4
        rho_std = RecreatingDensityMatrixWithNonentanglingQutrit(
            rho_true, shots; verbose=false)
        F_std = fidelity(Matrix{ComplexF64}(rho_std), rho_true)
    end

    # ── SEEQST ──────────────────────────────────────────────
    rho_seeqst = RecreatingDensityMatrixWithMixedSeeqstQutrit(
        rho_true, shots; verbose=false)
    F_seeqst = fidelity(Matrix{ComplexF64}(rho_seeqst), rho_true)

    # ── tSEEQST ─────────────────────────────────────────────
    rho_t = RecreatingDensityMatrixWithTMixedSeeqstQutrit(
        rho_true, shots, t_fixed; verbose=false)
    F_t = fidelity(Matrix{ComplexF64}(rho_t), rho_true)

    # ── Ausgabe ─────────────────────────────────────────────
    std_str = isnan(F_std) ? "OOM" : @sprintf("%.6f", F_std)
    @printf("%-4d │ %-10s │ %-10.6f │ %-10.6f\n",
        N, std_str, F_seeqst, F_t)
end

println("═"^55)
println("N=5 Standard: übersprungen (Speicherlimit)")