cd(@__DIR__)

using Printf
using LinearAlgebra
using QuantumInformation
using Plots
using Statistics
using DelimitedFiles

include("moduleSeeqstQutrit.jl")
include("moduleSeeqstQutrit2.jl")
include("moduleTSeeqstQutrit2.jl")

using .SeeqstHybridQutrit
using .SeeqstMixedQutrit
using .TSeeqstMixedQutrit

log_file = open("results_comparison.txt", "w")
function log(msg::String)
    println(msg)
    println(log_file, msg)
    flush(log_file)
end

# ══════════════════════════════════════════════════════════════
# Einstellungen
# ══════════════════════════════════════════════════════════════
N          = 3
shots_list = collect(100:25:5000)
t_values   = [0.02, 0.05, 0.10]

# ── Dichtematrix ───────────────────────────────────────────────
RhoTrue_real = readdlm("RhoTrue_real.csv", ',', Float64)
RhoTrue_imag = readdlm("RhoTrue_imag.csv", ',', Float64)
RhoTrue      = complex.(RhoTrue_real, RhoTrue_imag)
rho_diag     = real.(diag(RhoTrue))

log("═"^65)
log("Vergleich: Standard vs SEEQST vs tSEEQST")
log("═"^65)
log(@sprintf("  N=%d, shots=%d..%d", N, shots_list[1], shots_list[end]))
log(@sprintf("  t_values: %s  + adaptiv t=1/√shots", string(t_values)))
log(@sprintf("  Spur:          %.6f", real(tr(RhoTrue))))
log(@sprintf("  Min Eigenwert: %.6f", minimum(real(eigvals(RhoTrue)))))
log("")

# ── Circuit-Anzahl ─────────────────────────────────────────────
blocks_all = collect(0:(4^N - 1))
non_circs  = BuildNonEntanglingCircuitsQutrit(blocks_all, N)

n_standard = length(unique(String[
    c for g in non_circs for c in g if c != ""]))
n_seeqst   = TSeeqstMixedQutrit.count_circuits_for_blocks(blocks_all, N)

log(@sprintf("  Standard: %d Circuits", n_standard))
log(@sprintf("  SEEQST:   %d Circuits", n_seeqst))
for t in t_values
    blocks_t = TSeeqstMixedQutrit.BlocksAboveThresholdQutrit(N, rho_diag, t)
    n_t      = TSeeqstMixedQutrit.count_circuits_for_blocks(blocks_t, N)
    log(@sprintf("  tSEEQST t=%.2f: %d Circuits (vor Fill-up)", t, n_t))
end
log(@sprintf("  tSEEQST adaptiv: t=1/√m (variabel)"))
log("")

# ══════════════════════════════════════════════════════════════
# Hilfsfunktion: NC mit Fill-up berechnen
# ══════════════════════════════════════════════════════════════
function compute_nc_with_fillup(N::Int, rho_diag::Vector{Float64}, t::Float64)
    d        = 3^N
    blocks_t = TSeeqstMixedQutrit.BlocksAboveThresholdQutrit(N, rho_diag, t)
    n_circ   = TSeeqstMixedQutrit.count_circuits_for_blocks(blocks_t, N)

    if n_circ * d < d^2 - 1
        blocks_r = copy(blocks_t)
        missing  = setdiff(collect(0:4^N-1), blocks_r)
        sorted   = sort(missing,
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
log("Messungen starten...")
log("")

fid_standard  = Float64[]
fid_seeqst    = Float64[]
fid_tseeqst   = Dict(t => Float64[] for t in t_values)
nc_tseeqst    = Dict(t => Int[]     for t in t_values)
fid_adaptive  = Float64[]
nc_adaptive   = Int[]
t_adaptive    = Float64[]

time_standard = Float64[]
time_seeqst   = Float64[]
time_tseeqst  = Dict(t => Float64[] for t in t_values)
time_adaptive = Float64[]

for (idx, shots) in enumerate(shots_list)
    log(@sprintf("[%d/%d] shots=%d", idx, length(shots_list), shots))

    # ── Standard ──────────────────────────────────────────────
    t0 = time()
    rho_std = RecreatingDensityMatrixWithNonentanglingQutrit(
        RhoTrue, shots; verbose=false)
    push!(time_standard, time() - t0)
    push!(fid_standard, fidelity(Matrix{ComplexF64}(rho_std), RhoTrue))

    # ── SEEQST ────────────────────────────────────────────────
    t0 = time()
    rho_mix = RecreatingDensityMatrixWithMixedSeeqstQutrit(
        RhoTrue, shots; verbose=false)
    push!(time_seeqst, time() - t0)
    push!(fid_seeqst, fidelity(Matrix{ComplexF64}(rho_mix), RhoTrue))

    log(@sprintf("  Standard: F=%.4f  nc=%d  t=%.2fs",
        fid_standard[end], n_standard, time_standard[end]))
    log(@sprintf("  SEEQST:   F=%.4f  nc=%d  t=%.2fs",
        fid_seeqst[end], n_seeqst, time_seeqst[end]))

    # ── tSEEQST feste t-Werte ─────────────────────────────────
    for t in t_values
        t0 = time()
        rho_t = RecreatingDensityMatrixWithTMixedSeeqstQutrit(
            RhoTrue, shots, t; verbose=false)
        push!(time_tseeqst[t], time() - t0)
        push!(fid_tseeqst[t], fidelity(Matrix{ComplexF64}(rho_t), RhoTrue))

        n_t = compute_nc_with_fillup(N, rho_diag, t)
        push!(nc_tseeqst[t], n_t)

        log(@sprintf("  tSEEQST t=%.2f: F=%.4f  nc=%d  t=%.2fs",
            t, fid_tseeqst[t][end], n_t, time_tseeqst[t][end]))
    end

    # ── tSEEQST adaptiv t = 1/√m ──────────────────────────────
    t_adapt = 1.0 / sqrt(shots)
    push!(t_adaptive, t_adapt)
    t0 = time()
    rho_adapt = RecreatingDensityMatrixWithTMixedSeeqstQutrit(
        RhoTrue, shots, t_adapt; verbose=false)
    push!(time_adaptive, time() - t0)
    push!(fid_adaptive, fidelity(Matrix{ComplexF64}(rho_adapt), RhoTrue))

    n_adapt = compute_nc_with_fillup(N, rho_diag, t_adapt)
    push!(nc_adaptive, n_adapt)

    log(@sprintf("  tSEEQST t=1/√m=%.3f: F=%.4f  nc=%d  t=%.2fs",
        t_adapt, fid_adaptive[end], n_adapt, time_adaptive[end]))
    log("")
end

# ══════════════════════════════════════════════════════════════
# Plot Einstellungen
# ══════════════════════════════════════════════════════════════
pk = (
    size          = (1100, 700),
    left_margin   = 35Plots.mm,
    right_margin  = 20Plots.mm,
    top_margin    = 12Plots.mm,
    bottom_margin = 14Plots.mm,
    guidefontsize = 14,
    tickfontsize  = 12,
    legendfontsize= 11,
    titlefontsize = 13,
    linewidth     = 2,
    formatter     = :plain,
)

colors_t = [:orange, :red, :darkred]

# ── Hilfsfunktion: full + zoom ────────────────────────────────
function save_fidelity_plots(shots, fid, label_str, color,
                              title_str, filename_base;
                              ylim_zoom=(0.85, 1.01))
    p = plot(shots, fid;
        label   = label_str,
        xlabel  = "Number of measurements (m)",
        ylabel  = "Fidelity",
        title   = "$title_str (N=$N Qutrits)",
        color   = color,
        legend  = :bottomright,
        ylim    = (0.0, 1.05),
        pk...)
    hline!(p, [1.0]; color=:black, linestyle=:dot, label="", alpha=0.5)
    savefig(p, "$(filename_base)_full.png")
    log("✓ Plot: $(filename_base)_full.png")

    pz = plot(shots, fid;
        label   = label_str,
        xlabel  = "Number of measurements (m)",
        ylabel  = "Fidelity",
        title   = "$title_str (Zoom, N=$N)",
        color   = color,
        legend  = :bottomright,
        ylim    = ylim_zoom,
        pk...)
    hline!(pz, [1.0]; color=:black, linestyle=:dot, label="", alpha=0.5)
    savefig(pz, "$(filename_base)_zoom.png")
    log("✓ Plot: $(filename_base)_zoom.png")
end

# ══════════════════════════════════════════════════════════════
# Plot 1+2: Standard
# ══════════════════════════════════════════════════════════════
log("")
log("── Plots Standard ──")
save_fidelity_plots(shots_list, fid_standard,
    "Standard (nc=$n_standard)", :purple,
    "Standard: Fidelity vs. Measurements",
    "fidelity_standard")

# ══════════════════════════════════════════════════════════════
# Plot 3+4: SEEQST
# ══════════════════════════════════════════════════════════════
log("── Plots SEEQST ──")
save_fidelity_plots(shots_list, fid_seeqst,
    "SEEQST (nc=$n_seeqst)", :blue,
    "SEEQST: Fidelity vs. Measurements",
    "fidelity_seeqst")

# ══════════════════════════════════════════════════════════════
# Plot 5+6: tSEEQST (feste t-Werte)
# ══════════════════════════════════════════════════════════════
log("── Plots tSEEQST (feste t) ──")

for (full_zoom, ylim, suffix) in [
        (true,  (0.0,  1.05), "full"),
        (false, (0.85, 1.01), "zoom")]

    p = plot(;
        xlabel  = "Number of measurements (m)",
        ylabel  = "Fidelity",
        title   = full_zoom ?
            "tSEEQST: Fidelity vs. Measurements (N=$N Qutrits)" :
            "tSEEQST: Fidelity vs. Measurements (Zoom, N=$N)",
        legend  = :bottomright,
        ylim    = ylim,
        pk...)

    for (i, t) in enumerate(t_values)
        nc_avg = round(Int, mean(nc_tseeqst[t]))
        plot!(p, shots_list, fid_tseeqst[t];
            label = @sprintf("tSEEQST t=%.2f (nc≈%d)", t, nc_avg),
            color = colors_t[i])
    end

    hline!(p, [1.0]; color=:black, linestyle=:dot, label="", alpha=0.5)
    savefig(p, "fidelity_tseeqst_$suffix.png")
    log("✓ Plot: fidelity_tseeqst_$suffix.png")
end

# ══════════════════════════════════════════════════════════════
# Plot 7+8: tSEEQST t=1/√m
# ══════════════════════════════════════════════════════════════
log("── Plots tSEEQST t=1/√m ──")
save_fidelity_plots(shots_list, fid_adaptive,
    "tSEEQST t=1/√m", :green,
    "tSEEQST t=1/√m: Fidelity vs. Measurements",
    "fidelity_tseeqst_adaptive")

# ══════════════════════════════════════════════════════════════
# Plot 9: Runtime Comparison
# Standard + SEEQST + tSEEQST t=1/√m
# ══════════════════════════════════════════════════════════════
log("── Plot Runtime Comparison ──")

p9 = plot(shots_list, time_standard;
    label   = "Standard (nc=$n_standard)",
    xlabel  = "Number of measurements (m)",
    ylabel  = "Runtime (s)",
    title   = "Runtime vs. Measurements (N=$N Qutrits)",
    color   = :purple,
    legend  = :topleft,
    pk...)
plot!(p9, shots_list, time_seeqst;
    label     = "SEEQST (nc=$n_seeqst)",
    color     = :blue)
plot!(p9, shots_list, time_adaptive;
    label     = "tSEEQST t=1/√m",
    color     = :green,
    linewidth = 2.5)
savefig(p9, "runtime_comparison.png")
log("✓ Plot: runtime_comparison.png")

# ══════════════════════════════════════════════════════════════
# Plot 10: Circuits t=1/√m vs Shots
# ══════════════════════════════════════════════════════════════
p10 = plot(shots_list, nc_adaptive;
    label   = "tSEEQST t=1/√m",
    xlabel  = "Number of measurements (m)",
    ylabel  = "Number of circuits (nc)",
    title   = "Circuit Reduction: tSEEQST t=1/√m (N=$N)",
    color   = :green,
    legend  = :topright,
    pk...)
hline!(p10, [n_standard]; color=:purple, linestyle=:dash,
    label="Standard (nc=$n_standard)")
hline!(p10, [n_seeqst]; color=:blue, linestyle=:dash,
    label="SEEQST (nc=$n_seeqst)")
savefig(p10, "circuits_adaptive.png")
log("✓ Plot: circuits_adaptive.png")

# ══════════════════════════════════════════════════════════════
# Plot 11: Circuits feste t vs Shots
# ══════════════════════════════════════════════════════════════
p11 = plot(;
    xlabel  = "Number of measurements (m)",
    ylabel  = "Number of circuits (nc)",
    title   = "Circuit Reduction: Feste t-Werte (N=$N)",
    legend  = :topright,
    pk...)
hline!(p11, [n_standard]; color=:purple, linestyle=:dash,
    label="Standard (nc=$n_standard)")
hline!(p11, [n_seeqst]; color=:blue, linestyle=:dash,
    label="SEEQST (nc=$n_seeqst)")
for (i, t) in enumerate(t_values)
    plot!(p11, shots_list, nc_tseeqst[t];
        label = @sprintf("tSEEQST t=%.2f", t),
        color = colors_t[i])
end
savefig(p11, "circuits_fixed_t.png")
log("✓ Plot: circuits_fixed_t.png")

# ══════════════════════════════════════════════════════════════
# Plot 12: t-Wert vs Shots
# ══════════════════════════════════════════════════════════════
p12 = plot(shots_list, t_adaptive;
    label   = "t = 1/√m",
    xlabel  = "Number of measurements (m)",
    ylabel  = "Threshold t",
    title   = "Adaptiver Threshold t=1/√m vs. Messungen",
    color   = :green,
    legend  = :topright,
    pk...)
for t in t_values
    hline!(p12, [t]; linestyle=:dash,
        label=@sprintf("t=%.2f (fest)", t))
end
savefig(p12, "threshold_vs_shots.png")
log("✓ Plot: threshold_vs_shots.png")

# ══════════════════════════════════════════════════════════════
# Zusammenfassung
# ══════════════════════════════════════════════════════════════
log("")
log("═"^70)
log(@sprintf("Zusammenfassung bei shots=%d", shots_list[end]))
log("═"^70)
log(@sprintf("  %-22s  %-10s  %-10s  %-10s  %-12s",
    "Methode", "Fidelity", "Circuits", "Zeit(s)", "Reduktion"))
log("  " * "─"^68)

log(@sprintf("  %-22s  %-10.4f  %-10d  %-10.2f  0%%",
    "Standard", fid_standard[end], n_standard, time_standard[end]))

log(@sprintf("  %-22s  %-10.4f  %-10d  %-10.2f  %.1f%%",
    "SEEQST", fid_seeqst[end], n_seeqst, time_seeqst[end],
    (1 - n_seeqst/n_standard)*100))

for t in t_values
    nc_t = nc_tseeqst[t][end]
    log(@sprintf("  %-22s  %-10.4f  %-10d  %-10.2f  %.1f%%",
        @sprintf("tSEEQST t=%.2f", t),
        fid_tseeqst[t][end], nc_t, time_tseeqst[t][end],
        (1 - nc_t/n_standard)*100))
end

log(@sprintf("  %-22s  %-10.4f  %-10d  %-10.2f  %.1f%%",
    "tSEEQST t=1/√m",
    fid_adaptive[end], nc_adaptive[end], time_adaptive[end],
    (1 - nc_adaptive[end]/n_standard)*100))

log("═"^70)
log("")
log(@sprintf("  Gesamtzeit Standard:        %.1fs", sum(time_standard)))
log(@sprintf("  Gesamtzeit SEEQST:          %.1fs", sum(time_seeqst)))
for t in t_values
    log(@sprintf("  Gesamtzeit tSEEQST t=%.2f:  %.1fs",
        t, sum(time_tseeqst[t])))
end
log(@sprintf("  Gesamtzeit tSEEQST t=1/√m:  %.1fs", sum(time_adaptive)))
log("═"^70)

close(log_file)
println("✓ Fertig! Log: results_comparison.txt")