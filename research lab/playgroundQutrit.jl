cd(@__DIR__)

using Printf

# ── Logging ────────────────────────────────────────────────────
log_file = open("results_log.txt", "w")

function log(msg::String)
    println(msg)
    println(log_file, msg)
    flush(log_file)
end

# ── Imports ────────────────────────────────────────────────────
include("StructureDensityMatrix.jl")
include("moduleLinearInversionQutrit.jl")
include("moduleMaximumLikelihoodEstimationQutrit.jl")
include("moduleSeeqstQutrit.jl")

using .SeeqstHybridQutrit
using .LineareInversionQutrit
using .QuantumMLEQutrit
using LinearAlgebra
using QuantumInformation
using Plots
using DelimitedFiles

N = 3

RhoTrue_real = readdlm("RhoTrue_real.csv", ',', Float64)
RhoTrue_imag = readdlm("RhoTrue_imag.csv", ',', Float64)
RhoTrue = complex.(RhoTrue_real, RhoTrue_imag)

log("── Properties of True Density Matrix ──")
log("Dimension:        $(size(RhoTrue))")
log("Trace:            $(round(real(tr(RhoTrue)), digits=6))")
log("Hermitian:        $(RhoTrue ≈ RhoTrue')")
log("Min eigenvalue:   $(round(minimum(real(eigvals(RhoTrue))), digits=6))")
log("Zero elements:    $(count(x -> abs(x) < 1e-10, RhoTrue))")
log("")

# ── Shots List ─────────────────────────────────────────────────
shots_list = collect(100:25:15000)
log("Number of shot steps: $(length(shots_list))")
log("Range: $(shots_list[1]) to $(shots_list[end]) in steps of 25")
log("")

# ── Fidelity and Time Lists ────────────────────────────────────
fidelities_LI     = Float64[]
fidelities_MLE    = Float64[]
fidelities_Hybrid = Float64[]

times_LI     = Float64[]
times_MLE    = Float64[]
times_Hybrid = Float64[]

# ── Run Measurements ───────────────────────────────────────────
for (idx, shots) in enumerate(shots_list)
    log("[$idx/$(length(shots_list))] shots = $shots")

    t0 = time()
    rho_li = LineareInversionQutrit.RecreatingDensityMatrixWithLineareInversionQutrit(
        RhoTrue, shots)
    t_li = time() - t0
    push!(fidelities_LI,  fidelity(Matrix{ComplexF64}(rho_li), RhoTrue))
    push!(times_LI, t_li)

    t0 = time()
    rho_mle = QuantumMLEQutrit.RecreatingDensityMatrixWithMLEQutrit(
        RhoTrue, shots)
    t_mle = time() - t0
    push!(fidelities_MLE, fidelity(Matrix{ComplexF64}(rho_mle), RhoTrue))
    push!(times_MLE, t_mle)

    t0 = time()
    rho_hyb = RecreatingDensityMatrixWithSeeqstQutrit(RhoTrue, shots)
    t_hyb = time() - t0
    push!(fidelities_Hybrid, fidelity(Matrix{ComplexF64}(rho_hyb), RhoTrue))
    push!(times_Hybrid, t_hyb)

    log(@sprintf("  LI:     F=%.4f  t=%.2fs", fidelities_LI[end],     times_LI[end]))
    log(@sprintf("  MLE:    F=%.4f  t=%.2fs", fidelities_MLE[end],    times_MLE[end]))
    log(@sprintf("  Hybrid: F=%.4f  t=%.2fs", fidelities_Hybrid[end], times_Hybrid[end]))
    log("")
end

# ── Plot Settings ──────────────────────────────────────────────
pk_full = (
    linewidth      = 2,
    size           = (1000, 650),
    left_margin    = 30Plots.mm,
    right_margin   = 10Plots.mm,
    top_margin     = 12Plots.mm,
    bottom_margin  = 14Plots.mm,
    guidefontsize  = 14,
    tickfontsize   = 12,
    titlefontsize  = 13,
    legendfontsize = 12,
    formatter      = :plain,
)

pk_zoom = (
    linewidth      = 2,
    size           = (1000, 650),
    left_margin    = 40Plots.mm,
    right_margin   = 10Plots.mm,
    top_margin     = 12Plots.mm,
    bottom_margin  = 14Plots.mm,
    guidefontsize  = 14,
    tickfontsize   = 12,
    titlefontsize  = 13,
    legendfontsize = 12,
    formatter      = :plain,
)

# ── Plot 1: LI full ────────────────────────────────────────────
plt1 = plot(shots_list, fidelities_LI;
    xlabel = "Number of measurements (m)",
    ylabel = "Fidelity",
    title  = "Linear Inversion: Fidelity vs. Measurements",
    legend = false,
    color  = :blue,
    ylim   = (0.0, 1.0),
    pk_full...)
savefig(plt1, "fidelity_LI_full.png")
display(plt1)
log("✓ Plot 1 saved: fidelity_LI_full.png")

# ── Plot 2: MLE full ───────────────────────────────────────────
plt2 = plot(shots_list, fidelities_MLE;
    xlabel = "Number of measurements (m)",
    ylabel = "Fidelity",
    title  = "MLE: Fidelity vs. Measurements",
    legend = false,
    color  = :red,
    ylim   = (0.0, 1.0),
    pk_full...)
savefig(plt2, "fidelity_MLE_full.png")
display(plt2)
log("✓ Plot 2 saved: fidelity_MLE_full.png")

# ── Plot 3: SEEQST full ────────────────────────────────────────
plt3 = plot(shots_list, fidelities_Hybrid;
    xlabel = "Number of measurements (m)",
    ylabel = "Fidelity",
    title  = "SEEQST: Fidelity vs. Measurements",
    legend = false,
    color  = :green,
    ylim   = (0.0, 1.0),
    pk_full...)
savefig(plt3, "fidelity_Hybrid_full.png")
display(plt3)
log("✓ Plot 3 saved: fidelity_Hybrid_full.png")

# ── Plot 4: LI zoom ────────────────────────────────────────────
plt4 = plot(shots_list, fidelities_LI;
    xlabel = "Number of measurements (m)",
    ylabel = "Fidelity",
    title  = "Linear Inversion: Fidelity vs. Measurements (Zoom)",
    legend = false,
    color  = :blue,
    ylim   = (0.95, 1.0),
    pk_zoom...)
savefig(plt4, "fidelity_LI_zoom.png")
display(plt4)
log("✓ Plot 4 saved: fidelity_LI_zoom.png")

# ── Plot 5: MLE zoom ───────────────────────────────────────────
plt5 = plot(shots_list, fidelities_MLE;
    xlabel = "Number of measurements (m)",
    ylabel = "Fidelity",
    title  = "MLE: Fidelity vs. Measurements (Zoom)",
    legend = false,
    color  = :red,
    ylim   = (0.95, 1.0),
    pk_zoom...)
savefig(plt5, "fidelity_MLE_zoom.png")
display(plt5)
log("✓ Plot 5 saved: fidelity_MLE_zoom.png")

# ── Plot 6: SEEQST zoom ────────────────────────────────────────
plt6 = plot(shots_list, fidelities_Hybrid;
    xlabel = "Number of measurements (m)",
    ylabel = "Fidelity",
    title  = "SEEQST: Fidelity vs. Measurements (Zoom)",
    legend = false,
    color  = :green,
    ylim   = (0.95, 1.0),
    pk_zoom...)
savefig(plt6, "fidelity_Hybrid_zoom.png")
display(plt6)
log("✓ Plot 6 saved: fidelity_Hybrid_zoom.png")

# ── Plot 7: Comparison full ────────────────────────────────────
plt7 = plot(shots_list, fidelities_LI;
    label  = "Linear Inversion",
    xlabel = "Number of measurements (m)",
    ylabel = "Fidelity",
    title  = "Comparison of Methods (N=$N Qutrits)",
    color  = :blue,
    ylim   = (0.0, 1.0),
    legend = :bottomright,
    pk_full...)
plot!(plt7, shots_list, fidelities_MLE;
    label = "MLE",           color = :red)
plot!(plt7, shots_list, fidelities_Hybrid;
    label = "SEEQST Hybrid", color = :green)
savefig(plt7, "fidelity_comparison_full.png")
display(plt7)
log("✓ Plot 7 saved: fidelity_comparison_full.png")

# ── Plot 8: Runtime ────────────────────────────────────────────
plt8 = plot(shots_list, times_LI;
    label  = "Linear Inversion",
    xlabel = "Number of measurements (m)",
    ylabel = "Runtime (s)",
    title  = "Runtime vs. Measurements (N=$N Qutrits)",
    color  = :blue,
    legend = :outertopright, 
    pk_full...)
plot!(plt8, shots_list, times_MLE;
    label = "MLE",           color = :red)
plot!(plt8, shots_list, times_Hybrid;
    label = "SEEQST Hybrid", color = :green)
savefig(plt8, "runtime_comparison.png")
display(plt8)
log("✓ Plot 8 saved: runtime_comparison.png")

# ── Final Summary ──────────────────────────────────────────────
log("\n" * "━"^55)
log("Final Results at $(shots_list[end]) shots")
log("━"^55)
log(@sprintf("%-20s  %-10s  %-10s", "Method", "Fidelity", "Total time"))
log("─"^55)
log(@sprintf("%-20s  %-10.4f  %-8.1fs",
    "Linear Inversion", fidelities_LI[end],     sum(times_LI)))
log(@sprintf("%-20s  %-10.4f  %-8.1fs",
    "MLE",              fidelities_MLE[end],    sum(times_MLE)))
log(@sprintf("%-20s  %-10.4f  %-8.1fs",
    "SEEQST Hybrid",    fidelities_Hybrid[end], sum(times_Hybrid)))
log("━"^55)
log("✓ All plots saved")

close(log_file)
println("✓ Log saved: results_log.txt")