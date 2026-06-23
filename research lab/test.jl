cd(@__DIR__)

using Printf
using LinearAlgebra
using QuantumInformation
using Plots
using Random

include("moduleSeeqstQutrit.jl")
include("moduleTSeeqstQutrit.jl")

using .SeeqstHybridQutrit
using .TSeeqstQutrit

# ══════════════════════════════════════════════════════════════
# Einstellungen
# ══════════════════════════════════════════════════════════════
N_values = [2, 3, 4]
n_runs   = 5
shots    = 1000
t_fixed  = 0.05

Random.seed!(42)

log_file = open("results_error_rate.txt", "w")
function log(msg::String)
    println(msg)
    println(log_file, msg)
    flush(log_file)
end

log("═"^65)
log("Error Rate Experiment")
log("═"^65)
log(@sprintf("N_values=%s, shots=%d, t=%.2f, runs=%d",
    string(N_values), shots, t_fixed, n_runs))
log("")

# ══════════════════════════════════════════════════════════════
# Hilfsfunktionen: Circuit-Anzahl
# ══════════════════════════════════════════════════════════════
function get_n_circuits_standard(N::Int)
    blocks    = collect(0:(4^N - 1))
    non_circs = BuildNonEntanglingCircuitsQutrit(blocks, N)
    return length(unique(String[
        c for g in non_circs for c in g if c != ""]))
end

function get_n_circuits_seeqst(N::Int)
    blocks       = collect(0:(4^N - 1))
    hybrid_circs = BuildHybridCircuitsQutrit(blocks, N)
    return length(unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for g in hybrid_circs for c in g if c != ""]))
end

function get_n_circuits_tseeqst(N::Int, rho_diag::Vector{Float64}, t::Float64)
    blocks_t     = TSeeqstQutrit.BlocksAboveThresholdQutrit(N, rho_diag, t)
    hybrid_circs = BuildHybridCircuitsQutrit(blocks_t, N)
    return length(unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for g in hybrid_circs for c in g if c != ""]))
end

# ══════════════════════════════════════════════════════════════
# Ergebnis-Speicher: (Δerr, n_circuits) pro Run
# ══════════════════════════════════════════════════════════════
results = Dict(
    :standard => Dict(N => Tuple{Float64,Int}[] for N in N_values),
    :seeqst   => Dict(N => Tuple{Float64,Int}[] for N in N_values),
    :tseeqst  => Dict(N => Tuple{Float64,Int}[] for N in N_values),
)

# ══════════════════════════════════════════════════════════════
# Messungen
# ══════════════════════════════════════════════════════════════
for N in N_values
    log("━"^65)
    log(@sprintf("N = %d Qutrits", N))
    log("━"^65)

    nc_std    = get_n_circuits_standard(N)
    nc_seeqst = get_n_circuits_seeqst(N)
    log(@sprintf("  Standard: %d Circuits", nc_std))
    log(@sprintf("  SEEQST:   %d Circuits", nc_seeqst))
    log("")

    for run in 1:n_runs
        log(@sprintf("  Run %d/%d", run, n_runs))

        rho_true = GenerateRandomDensityMatrixQutrits(N)
        rho_diag = real.(diag(rho_true))
        nc_t     = get_n_circuits_tseeqst(N, rho_diag, t_fixed)

        # Standard SEEQST
        rho_std = RecreatingDensityMatrixWithNonentanglingQutrit(
            rho_true, shots; verbose=false)
        F_std   = fidelity(Matrix{ComplexF64}(rho_std), rho_true)
        push!(results[:standard][N], (1 - F_std, nc_std))

        # SEEQST Hybrid
        rho_hyb = RecreatingDensityMatrixWithSeeqstQutrit(
            rho_true, shots; verbose=false)
        F_hyb   = fidelity(Matrix{ComplexF64}(rho_hyb), rho_true)
        push!(results[:seeqst][N], (1 - F_hyb, nc_seeqst))

        # tSEEQST
        rho_t = RecreatingDensityMatrixWithTSeeqstQutrit(
            rho_true, shots, t_fixed; verbose=false)
        F_t   = fidelity(Matrix{ComplexF64}(rho_t), rho_true)
        push!(results[:tseeqst][N], (1 - F_t, nc_t))

        log(@sprintf("    Standard: Δerr=%.6f  nc=%d", 1-F_std, nc_std))
        log(@sprintf("    SEEQST:   Δerr=%.6f  nc=%d", 1-F_hyb, nc_seeqst))
        log(@sprintf("    tSEEQST:  Δerr=%.6f  nc=%d", 1-F_t,   nc_t))
    end
    log("")
end

# ══════════════════════════════════════════════════════════════
# Colorbar-Normierung
# ══════════════════════════════════════════════════════════════
all_circuits = vcat(
    [last.(results[:standard][N]) for N in N_values]...,
    [last.(results[:seeqst][N])   for N in N_values]...,
    [last.(results[:tseeqst][N])  for N in N_values]...
)
c_min = minimum(all_circuits)
c_max = maximum(all_circuits)

norm_color(c) = (c - c_min) / max(c_max - c_min, 1)

# ══════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════
pk = (
    size           = (900, 600),
    left_margin    = 15Plots.mm,
    right_margin   = 45Plots.mm,
    top_margin     = 12Plots.mm,
    bottom_margin  = 12Plots.mm,
    guidefontsize  = 14,
    tickfontsize   = 12,
    legendfontsize = 12,
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
    legend  = :topright,
    pk...)

for N in N_values
    # ── Standard QST: ★ versetzt nach links ──────────────────
    for (run_idx, (Δerr, nc)) in enumerate(results[:standard][N])
        scatter!(p, [N - 0.15], [Δerr];
            marker            = :star5,
            markersize        = 11,
            color             = cgrad(:coolwarm)[norm_color(nc)],
            label             = (N==N_values[1] && run_idx==1) ?
                                "Standard QST" : "",
            markerstrokewidth = 0.5,
            markerstrokecolor = :black)
    end

    # ── SEEQST: ● zentriert ───────────────────────────────────
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

    # ── tSEEQST: ■ versetzt nach rechts ──────────────────────
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

savefig(p, "error_rate_comparison_v2.png")
log("✓ Plot gespeichert: error_rate_comparison_v2.png")

# ══════════════════════════════════════════════════════════════
# Zusammenfassung
# ══════════════════════════════════════════════════════════════
log("")
log("═"^65)
log("Zusammenfassung (Mittelwerte)")
log("═"^65)
log("")

for N in N_values
    log(@sprintf("── N = %d ──", N))
    for (method, sym) in [("Standard", :standard),
                           ("SEEQST",   :seeqst),
                           ("tSEEQST",  :tseeqst)]
        errs = first.(results[sym][N])
        ncs  = last.(results[sym][N])
        log(@sprintf("  %-10s: Δerr=%.4f ± %.4f  nc=%d",
            method, mean(errs), std(errs), ncs[1]))
    end
    log("")
end

close(log_file)
println("✓ Log gespeichert: results_error_rate.txt")