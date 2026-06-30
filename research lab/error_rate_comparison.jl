cd(@__DIR__)

using Printf
using LinearAlgebra
using QuantumInformation
using Plots
using Random

include("moduleSeeqstQutrit.jl")
include("moduleSeeqstQutrit2.jl")
include("moduleTSeeqstQutrit2.jl")

using .SeeqstHybridQutrit
using .SeeqstMixedQutrit
using .TSeeqstMixedQutrit

# ══════════════════════════════════════════════════════════════
# Einstellungen
# ══════════════════════════════════════════════════════════════
N_values = [2, 3, 4, 5]
n_runs   = 5
shots    = 1000
t_fixed  = 0.05

Random.seed!(42)

results = Dict(
    :standard => Dict(N => Tuple{Float64,Int}[] for N in N_values),
    :seeqst   => Dict(N => Tuple{Float64,Int}[] for N in N_values),
    :tseeqst  => Dict(N => Tuple{Float64,Int}[] for N in N_values),
)

# ── Circuit-Anzahl: Standard (SeeqstHybridQutrit) ──────────────
function get_n_circuits_standard(N::Int)
    blocks    = collect(0:(4^N - 1))
    non_circs = BuildNonEntanglingCircuitsQutrit(blocks, N)
    return length(unique(String[
        c for g in non_circs for c in g if c != ""]))
end

# ── Circuit-Anzahl: SEEQST (SeeqstMixedQutrit) ─────────────────
function get_n_circuits_seeqst(N::Int)
    blocks = collect(0:(4^N - 1))
    return TSeeqstMixedQutrit.count_circuits_for_blocks(blocks, N)
end

# ── Circuit-Anzahl: tSEEQST (TSeeqstMixedQutrit, mit Fill-up) ──
function get_n_circuits_tseeqst(N::Int, rho_diag::Vector{Float64}, t::Float64)
    d        = 3^N
    blocks_t = TSeeqstMixedQutrit.BlocksAboveThresholdQutrit(N, rho_diag, t)
    n_circ   = TSeeqstMixedQutrit.count_circuits_for_blocks(blocks_t, N)

    if n_circ * d < d^2 - 1
        blocks_all = collect(0:4^N-1)
        blocks_r   = copy(blocks_t)
        missing    = setdiff(blocks_all, blocks_r)
        sorted     = sort(missing,
            by=k->TSeeqstMixedQutrit.max_bound_for_block(k, N, rho_diag),
            rev=true)
        for k in sorted
            push!(blocks_r, k); sort!(blocks_r)
            n_circ = TSeeqstMixedQutrit.count_circuits_for_blocks(blocks_r, N)
            n_circ * d ≥ d^2 - 1 && break
        end
    end

    return n_circ
end

# ══════════════════════════════════════════════════════════════
# Messungen
# ══════════════════════════════════════════════════════════════
println("═"^65)
println("Error Rate Experiment")
println("═"^65)

for N in N_values
    println("── N = $N Qutrits ──")
    nc_std    = get_n_circuits_standard(N)
    nc_seeqst = get_n_circuits_seeqst(N)
    println(@sprintf("  Standard: %d Circuits", nc_std))
    println(@sprintf("  SEEQST:   %d Circuits", nc_seeqst))

    for run in 1:n_runs
        println(@sprintf("  Run %d/%d", run, n_runs))

        rho_true = SeeqstHybridQutrit.GenerateRandomDensityMatrixNoZerosQutrits(N)
        rho_diag = real.(diag(rho_true))
        nc_t     = get_n_circuits_tseeqst(N, rho_diag, t_fixed)

        # ── Standard ────────────────────────────────────────────
        rho_std = RecreatingDensityMatrixWithNonentanglingQutrit(
            rho_true, shots; verbose=false)
        F_std   = fidelity(Matrix{ComplexF64}(rho_std), rho_true)
        push!(results[:standard][N], (1 - F_std, nc_std))

        # ── SEEQST ──────────────────────────────────────────────
        rho_seeqst = RecreatingDensityMatrixWithMixedSeeqstQutrit(
            rho_true, shots; verbose=false)
        F_seeqst   = fidelity(Matrix{ComplexF64}(rho_seeqst), rho_true)
        push!(results[:seeqst][N], (1 - F_seeqst, nc_seeqst))

        # ── tSEEQST ─────────────────────────────────────────────
        rho_t = RecreatingDensityMatrixWithTMixedSeeqstQutrit(
            rho_true, shots, t_fixed; verbose=false)
        F_t   = fidelity(Matrix{ComplexF64}(rho_t), rho_true)
        push!(results[:tseeqst][N], (1 - F_t, nc_t))

        println(@sprintf("    Standard: Δerr=%.6f  nc=%d", 1-F_std, nc_std))
        println(@sprintf("    SEEQST:   Δerr=%.6f  nc=%d", 1-F_seeqst, nc_seeqst))
        println(@sprintf("    tSEEQST:  Δerr=%.6f  nc=%d", 1-F_t, nc_t))
    end
end

# ── Colorbar-Normierung ─────────────────────────────────────────
all_circuits = vcat(
    [last.(results[:standard][N]) for N in N_values]...,
    [last.(results[:seeqst][N])   for N in N_values]...,
    [last.(results[:tseeqst][N])  for N in N_values]...
)
c_min = minimum(all_circuits)
c_max = maximum(all_circuits)
norm_color(c) = (c - c_min) / max(c_max - c_min, 1)

# ══════════════════════════════════════════════════════════════
# Plot — Legende außerhalb, rechts neben dem Graphen
# ══════════════════════════════════════════════════════════════
pk = (
    size           = (1050, 600),
    left_margin    = 15Plots.mm,
    right_margin   = 5Plots.mm,
    top_margin     = 10Plots.mm,
    bottom_margin  = 12Plots.mm,
    guidefontsize  = 14,
    tickfontsize   = 12,
    legendfontsize = 11,
    titlefontsize  = 13,
    formatter      = :plain,
)

p = plot(;
    xlabel  = "Number of qutrits (N)",
    ylabel  = "Δerr",
    title   = @sprintf("Average reconstruction error (shots=%d, t=%.2f)",
                shots, t_fixed),
    yscale  = :log10,
    xticks  = (N_values, string.(N_values)),
    legend  = :outertopright,
    pk...)

for N in N_values
    for (run_idx, (Δerr, nc)) in enumerate(results[:standard][N])
        scatter!(p, [N - 0.15], [Δerr];
            marker            = :star5,
            markersize        = 11,
            color             = cgrad(:coolwarm)[norm_color(nc)],
            label             = (N==N_values[1] && run_idx==1) ?
                                "Standard" : "",
            markerstrokewidth = 0.5,
            markerstrokecolor = :black)
    end

    for (run_idx, (Δerr, nc)) in enumerate(results[:seeqst][N])
        scatter!(p, [N + 0.0], [Δerr];
            marker            = :circle,
            markersize        = 10,
            color             = cgrad(:coolwarm)[norm_color(nc)],
            label             = (N==N_values[1] && run_idx==1) ?
                                "SEEQST" : "",
            markerstrokewidth = 0.5,
            markerstrokecolor = :black)
    end

    for (run_idx, (Δerr, nc)) in enumerate(results[:tseeqst][N])
        scatter!(p, [N + 0.15], [Δerr];
            marker            = :square,
            markersize        = 10,
            color             = cgrad(:coolwarm)[norm_color(nc)],
            label             = (N==N_values[1] && run_idx==1) ?
                                "tSEEQST" : "",
            markerstrokewidth = 0.5,
            markerstrokecolor = :black)
    end
end

# ── Colorbar ──────────────────────────────────────────────────
scatter!(p, [NaN], [NaN];
    marker_z       = [0.0],
    color          = :coolwarm,
    colorbar       = true,
    clims          = (c_min, c_max),
    colorbar_title = "\nNumber of\ncircuits",
    label          = "")

savefig(p, "error_rate_comparison_v3.png")
println("✓ Plot gespeichert: error_rate_comparison_v3.png")