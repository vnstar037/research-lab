cd(@__DIR__)

using Printf
using LinearAlgebra
using QuantumInformation
using Plots
using DelimitedFiles

include("StructureDensityMatrix.jl")
include("moduleSeeqstQutrit.jl")
include("moduleTSeeqstQutrit.jl")

using .SeeqstHybridQutrit
using .TSeeqstQutrit

log_file = open("results_fidelity_fixedt.txt", "w")
function log(msg::String)
    println(msg)
    println(log_file, msg)
    flush(log_file)
end

# ── Lade Dichtematrix ──────────────────────────────────────────
N = 3
RhoTrue_real = readdlm("RhoTrue_real.csv", ',', Float64)
RhoTrue_imag = readdlm("RhoTrue_imag.csv", ',', Float64)
RhoTrue      = complex.(RhoTrue_real, RhoTrue_imag)

log("Dichtematrix geladen:")
log(@sprintf("  Spur:          %.6f", real(tr(RhoTrue))))
log(@sprintf("  Min Eigenwert: %.6f", minimum(real(eigvals(RhoTrue)))))
log("")

# ── Fester Threshold ────────────────────────────────────────────
t_fixed = 0.05
log(@sprintf("Fester Threshold: t = %.3f", t_fixed))
log("")

# ── Circuit-Anzahl Standard und SEEQST ────────────────────────
blocks_all = collect(0:(4^N - 1))
non_circs  = BuildNonEntanglingCircuitsQutrit(blocks_all, N)
hyb_circs  = BuildHybridCircuitsQutrit(blocks_all, N)

n_standard = length(unique(String[
    c for g in non_circs for c in g if c != ""]))
n_seeqst   = length(unique(String[
    startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
    for g in hyb_circs for c in g if c != ""]))

log(@sprintf("Standard SEEQST: %d Circuits", n_standard))
log(@sprintf("SEEQST Hybrid:   %d Circuits", n_seeqst))
log("")

# ── Messungen ──────────────────────────────────────────────────
shots_list = collect(100:20:5000)

fidelities_Standard = Float64[]
fidelities_Hybrid   = Float64[]
fidelities_TSeeqst  = Float64[]

circuits_Standard   = Int[]
circuits_Hybrid     = Int[]
circuits_TSeeqst    = Int[]

times_Standard      = Float64[]
times_Hybrid        = Float64[]
times_TSeeqst       = Float64[]

log("═"^65)
log("Messungen starten...")
log("═"^65)
log("")

for (idx, shots) in enumerate(shots_list)
    log(@sprintf("[%d/%d] shots=%d  t=%.3f (fest)",
        idx, length(shots_list), shots, t_fixed))

    # ── Standard SEEQST ───────────────────────────────────────
    local t0 = time()
    local rho_std = RecreatingDensityMatrixWithNonentanglingQutrit(
        RhoTrue, shots; verbose=false)
    push!(times_Standard, time() - t0)
    push!(fidelities_Standard, fidelity(Matrix{ComplexF64}(rho_std), RhoTrue))
    push!(circuits_Standard, n_standard)

    # ── SEEQST Hybrid ─────────────────────────────────────────
    t0 = time()
    local rho_hyb = RecreatingDensityMatrixWithSeeqstQutrit(
        RhoTrue, shots; verbose=false)
    push!(times_Hybrid, time() - t0)
    push!(fidelities_Hybrid, fidelity(Matrix{ComplexF64}(rho_hyb), RhoTrue))
    push!(circuits_Hybrid, n_seeqst)

    # ── tSEEQST mit festem t ───────────────────────────────────
    local rho_diag = real.(diag(RhoTrue))
    local blocks_t = BlocksAboveThresholdQutrit(N, rho_diag, t_fixed)
    local hyb_t    = BuildHybridCircuitsQutrit(blocks_t, N)
    local n_circ_t = length(unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for g in hyb_t for c in g if c != ""]))

    t0 = time()
    local rho_t = RecreatingDensityMatrixWithTSeeqstQutrit(
        RhoTrue, shots, t_fixed; verbose=false)
    push!(times_TSeeqst, time() - t0)
    push!(fidelities_TSeeqst, fidelity(Matrix{ComplexF64}(rho_t), RhoTrue))
    push!(circuits_TSeeqst, n_circ_t)

    log(@sprintf("  Standard: F=%.4f  Circuits=%d  t=%.2fs",
        fidelities_Standard[end], circuits_Standard[end], times_Standard[end]))
    log(@sprintf("  SEEQST:   F=%.4f  Circuits=%d  t=%.2fs",
        fidelities_Hybrid[end],   circuits_Hybrid[end],   times_Hybrid[end]))
    log(@sprintf("  tSEEQST:  F=%.4f  Circuits=%d  t=%.2fs",
        fidelities_TSeeqst[end],  circuits_TSeeqst[end],  times_TSeeqst[end]))
    log("")
end

# ══════════════════════════════════════════════════════════════
# Plot Einstellungen
# ══════════════════════════════════════════════════════════════
pk = (
    size          = (1100, 700),
    left_margin   = 35Plots.mm,
    right_margin  = 35Plots.mm,
    top_margin    = 12Plots.mm,
    bottom_margin = 14Plots.mm,
    guidefontsize = 14,
    tickfontsize  = 12,
    legendfontsize= 11,
    titlefontsize = 13,
    linewidth     = 2,
    formatter     = :plain,
)

# ── Hilfsfunktion: Dual-Achsen Plot ───────────────────────────
function dual_plot(shots, fidelities, circuits,
                    method_name, color, ylim_f, filename, title_str)

    p = plot(shots, fidelities;
        label     = "Fidelity",
        xlabel    = "Number of measurements (m)",
        ylabel    = "Fidelity",
        color     = color,
        ylim      = ylim_f,
        legend    = :bottomright,
        title     = title_str,
        pk...)

    p2 = twinx(p)
    plot!(p2, shots, circuits;
        label     = "nc (Circuits)",
        color     = color,
        linestyle = :dash,
        alpha     = 0.7,
        ylabel    = "Number of circuits (nc)",
        legend    = :topright,
        ylim      = (0, maximum(circuits) * 1.2),
        pk...)

    savefig(p, filename)
    log("✓ Plot gespeichert: $filename")
    return p
end

# ══════════════════════════════════════════════════════════════
# Plots: Standard SEEQST
# ══════════════════════════════════════════════════════════════
log("")
log("── Plots Standard SEEQST ──")

dual_plot(shots_list, fidelities_Standard, circuits_Standard,
    "Standard", :purple, (0.0, 1.05),
    "fidelity_standard_full_fixedt.png",
    "Standard: Fidelity and Circuits vs. Measurements (N=$N)")

dual_plot(shots_list, fidelities_Standard, circuits_Standard,
    "Standard", :purple, (0.9, 1.01),
    "fidelity_standard_zoom_fixedt.png",
    "Standard: Fidelity and Circuits vs. Measurements (Zoom)")

# ══════════════════════════════════════════════════════════════
# Plots: SEEQST Hybrid
# ══════════════════════════════════════════════════════════════
log("── Plots SEEQST ──")

dual_plot(shots_list, fidelities_Hybrid, circuits_Hybrid,
    "SEEQST", :green, (0.0, 1.05),
    "fidelity_seeqst_full_fixedt.png",
    "SEEQST: Fidelity and Circuits vs. Measurements (N=$N)")

dual_plot(shots_list, fidelities_Hybrid, circuits_Hybrid,
    "SEEQST", :green, (0.9, 1.01),
    "fidelity_seeqst_zoom_fixedt.png",
    "SEEQST: Fidelity and Circuits vs. Measurements (Zoom)")

# ══════════════════════════════════════════════════════════════
# Plots: tSEEQST
# ══════════════════════════════════════════════════════════════
log("── Plots tSEEQST ──")

dual_plot(shots_list, fidelities_TSeeqst, circuits_TSeeqst,
    "tSEEQST", :orange, (0.0, 1.05),
    "fidelity_tseeqst_full_fixedt.png",
    @sprintf("tSEEQST: Fidelity and Circuits vs. Measurements (N=%d, t=%.2f)", N, t_fixed))

dual_plot(shots_list, fidelities_TSeeqst, circuits_TSeeqst,
    "tSEEQST", :orange, (0.9, 1.01),
    "fidelity_tseeqst_zoom_fixedt.png",
    "tSEEQST: Fidelity and Circuits vs. Measurements (Zoom)")

# ══════════════════════════════════════════════════════════════
# Plot: Runtime Comparison
# ══════════════════════════════════════════════════════════════
log("── Plot: Runtime Comparison ──")

p_runtime = plot(shots_list, times_Standard;
    label         = "Standard SEEQST",
    xlabel        = "Number of measurements (m)",
    ylabel        = "Runtime (s)",
    title         = "Runtime vs. Measurements (N=$N Qutrits)",
    color         = :purple,
    legend        = :topleft,
    pk...)
plot!(p_runtime, shots_list, times_Hybrid;
    label = "SEEQST",   color = :green)
plot!(p_runtime, shots_list, times_TSeeqst;
    label = "tSEEQST",  color = :orange)

savefig(p_runtime, "runtime_comparison_fixedt.png")
log("✓ Plot gespeichert: runtime_comparison_fixedt.png")

# ══════════════════════════════════════════════════════════════
# Zusammenfassung
# ══════════════════════════════════════════════════════════════
log("")
log("═"^65)
log("Zusammenfassung bei shots=$(shots_list[end]), t=$t_fixed (fest)")
log("═"^65)
log(@sprintf("  %-20s  %-10s  %-12s  %-10s  %-10s",
    "Methode", "Fidelity", "Runtime (s)", "Circuits", "Total t"))
log("  " * "─"^65)
log(@sprintf("  %-20s  %-10.4f  %-12.2f  %-10d  %-10.1fs",
    "Standard SEEQST",
    fidelities_Standard[end], times_Standard[end],
    circuits_Standard[end],   sum(times_Standard)))
log(@sprintf("  %-20s  %-10.4f  %-12.2f  %-10d  %-10.1fs",
    "SEEQST",
    fidelities_Hybrid[end],   times_Hybrid[end],
    circuits_Hybrid[end],     sum(times_Hybrid)))
log(@sprintf("  %-20s  %-10.4f  %-12.2f  %-10d  %-10.1fs",
    "tSEEQST (t=%.2f)" |> s -> @sprintf("tSEEQST (t=%.2f)", t_fixed),
    fidelities_TSeeqst[end],  times_TSeeqst[end],
    circuits_TSeeqst[end],    sum(times_TSeeqst)))
log("═"^65)

close(log_file)
println("✓ Log gespeichert: results_fidelity_fixedt.txt")