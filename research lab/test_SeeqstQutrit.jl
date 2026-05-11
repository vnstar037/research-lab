include("sqt2.jl")
using .SeeqstQutrit
using LinearAlgebra
using Printf
using QuantumInformation
println("═══ Tests BuildHybridCircuitsQutrit ═══\n")

type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]

# ── Test 1: Block 0 → leer ────────────────────────────────────
println("── Test 1: Block 0 ──")
seq = BuildHybridCircuitsQutrit([0], 2)[1]
@assert seq == [""]  "Fehler: Block 0 sollte leer sein"
println("✓ Block 0: leer\n")

# ── Test 2: Block 5 = 11₄ (beide |0⟩↔|1⟩) ────────────────────
println("── Test 2: Block 5 = 11₄ ──")
seq = BuildHybridCircuitsQutrit([5], 2)[1]
println("Circuits:")
for c in seq; println("  ", c); end
# E: RL2+RL2 (2 Y) oder RL1+RL1 (0 Y)
# O: RL2+RL1 (1 Y) oder RL1+RL2 (1 Y)
@assert any(startswith(c, "E:") && occursin("CINC", c) for c in seq)  "Fehler: E mit CINC fehlt"
@assert any(startswith(c, "O:") && occursin("CINC", c) for c in seq)  "Fehler: O mit CINC fehlt"
println("✓ Block 5: CINC + lokale Rotationen\n")

# ── Test 3: Block 1 = 01₄ (nur 1 aktiv) ──────────────────────
println("── Test 3: Block 1 = 01₄ (1 aktives Qutrit) ──")
seq = BuildHybridCircuitsQutrit([1], 2)[1]
println("Circuits:")
for c in seq; println("  ", c); end
@assert !any(occursin("CINC", c) for c in seq)  "Fehler: kein CINC bei 1 Qutrit"
println("✓ Block 1: kein CINC\n")

# ── Test 4: Block 3232₄ (dein Beispiel) ───────────────────────
println("── Test 4: Block 3232₄ (L67×L45×L67×L45) ──")
# Block 3232₄: 3*4³ + 2*4² + 3*4 + 2 = 192+32+12+2 = 238
block_3232 = 3*4^3 + 2*4^2 + 3*4 + 2
block_types = digits(block_3232, base=4, pad=4) |> reverse
println("Block ", block_3232, " = ", join(block_types), "₄")
println("  ", join([type_names[t+1] for t in block_types], " × "))

seq = BuildHybridCircuitsQutrit([block_3232], 4)[1]
println("Circuits:")
for c in seq; println("  ", c); end

n_e = count(c -> startswith(c, "E:"), seq)
n_o = count(c -> startswith(c, "O:"), seq)
println("E-Circuits: $n_e, O-Circuits: $n_o")
@assert any(occursin("CINC33", c) for c in seq)  "Fehler: CINC33 fehlt"
@assert any(occursin("CINC22", c) for c in seq)  "Fehler: CINC22 fehlt"
println("✓ Block 3232₄: CINC33 + CINC22 + lokale Rotationen\n")

# ── Test 5: Vergleich Anzahl Circuits ─────────────────────────
println("── Test 5: Vergleich Anzahl Circuits ──")
println(@sprintf("%-15s  %-12s  %-12s  %-12s",
    "Block", "Hybrid", "NonEnt", "Entangl"))
println("─"^55)

for block in [0, 1, 5, 15, block_3232]
    N_test = block > 255 ? 4 : 2
    hybrid   = BuildHybridCircuitsQutrit([block], N_test)[1]
    non_ent  = BuildNonEntanglingCircuitsQutrit([block], N_test)[1]
    entangl  = BuildEntanglerBlocksQutrit([block], N_test)[1]

    n_h = length(hybrid)
    n_n = length(non_ent)
    n_e = length(entangl)

    println(@sprintf("%-15s  %-12d  %-12d  %-12d",
        "Block $block", n_h, n_n, n_e))
end

println("\n═══ Alle Tests bestanden ✓ ═══")