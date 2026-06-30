cd(@__DIR__)

include("moduleSeeqstQutrit.jl")
include("moduleTSeeqstQutrit.jl")
include("moduleTSeeqstQutrit2.jl")

using .SeeqstHybridQutrit
using .TSeeqstQutrit
using .TSeeqstMixedQutrit
using QuantumInformation
using LinearAlgebra
using Printf

N    = 3   # ← nur das geändert!
d    = 3^N

rho_true   = SeeqstHybridQutrit.GenerateRandomDensityMatrixNoZerosQutrits(N)
rho_diag   = real.(diag(rho_true))
blocks_all = collect(0:4^N-1)

function count_tseeqst(blocks::Vector{Int}, N::Int)
    hyb = SeeqstHybridQutrit.BuildHybridCircuitsQutrit(blocks, N)
    return length(unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for g in hyb for c in g if c != ""]))
end

n_std1 = count_tseeqst(blocks_all, N)
n_std2 = TSeeqstMixedQutrit.count_circuits_for_blocks(blocks_all, N)

println("═"^75)
println("Vergleich Circuit-Anzahl: TSeeqstQutrit vs TSeeqstMixedQutrit")
println("═"^75)
println()
println(@sprintf("  N=%d, d=%d, d²-1=%d, min=%d",
    N, d, d^2-1, ceil(Int,(d^2-1)/d)))
println()
println(@sprintf("  Standard TSeeqstQutrit:      %d Circuits", n_std1))
println(@sprintf("  Standard TSeeqstMixedQutrit: %d Circuits", n_std2))
println()

println(@sprintf("  %-8s  %-20s  %-20s  %-12s",
    "t", "TSeeqstQutrit", "TSeeqstMixedQutrit", "Diff"))
println("  " * "─"^65)

for t in [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

    # ── TSeeqstQutrit ─────────────────────────────────────────
    blocks_t1 = TSeeqstQutrit.BlocksAboveThresholdQutrit(N, rho_diag, t)
    n_t1      = count_tseeqst(blocks_t1, N)

    blocks_r1 = copy(blocks_t1)
    n_f1      = n_t1
    if n_f1 * d < d^2 - 1
        missing = setdiff(blocks_all, blocks_r1)
        sorted  = sort(missing, rev=true,
            by = k -> begin
                k_digits = digits(k, base=4, pad=N) |> reverse
                max_b = 0.0
                for i in 0:d-1, j in 0:d-1
                    i==j && continue
                    di = digits(i, base=3, pad=N) |> reverse
                    dj = digits(j, base=3, pad=N) |> reverse
                    if all(TSeeqstQutrit.transition_type_qutrit(di[l],dj[l])==k_digits[l]
                           for l in 1:N)
                        max_b = max(max_b, sqrt(rho_diag[i+1]*rho_diag[j+1]))
                    end
                end
                max_b
            end)
        for k in sorted
            push!(blocks_r1, k); sort!(blocks_r1)
            n_f1 = count_tseeqst(blocks_r1, N)
            n_f1 * d ≥ d^2 - 1 && break
        end
    end

    # ── TSeeqstMixedQutrit ────────────────────────────────────
    blocks_t2 = TSeeqstMixedQutrit.BlocksAboveThresholdQutrit(N, rho_diag, t)
    n_t2      = TSeeqstMixedQutrit.count_circuits_for_blocks(blocks_t2, N)

    blocks_r2 = copy(blocks_t2)
    n_f2      = n_t2
    if n_f2 * d < d^2 - 1
        missing = setdiff(blocks_all, blocks_r2)
        sorted  = sort(missing,
            by=k->TSeeqstMixedQutrit.max_bound_for_block(k, N, rho_diag), rev=true)
        for k in sorted
            push!(blocks_r2, k); sort!(blocks_r2)
            n_f2 = TSeeqstMixedQutrit.count_circuits_for_blocks(blocks_r2, N)
            n_f2 * d ≥ d^2 - 1 && break
        end
    end

    s1 = n_f1 > n_t1 ? @sprintf("(%d→%d)", n_t1, n_f1) : @sprintf("(%d)", n_f1)
    s2 = n_f2 > n_t2 ? @sprintf("(%d→%d)", n_t2, n_f2) : @sprintf("(%d)", n_f2)

    println(@sprintf("  t=%.2f   %-20s  %-20s  %+d",
        t, s1, s2, n_f1 - n_f2))
end

println()
println("Legende: (X) = kein Fill-up, (X→Y) = Fill-up von X auf Y")
println("Diff = TSeeqstQutrit - TSeeqstMixedQutrit")
println("Positiv = Mixed braucht weniger Circuits ✓")
println("═"^75)