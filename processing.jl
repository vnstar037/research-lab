using LinearAlgebra
using Zygote
using Optimisers
using Printf

include("/home/jesko/Documents/research lab/setup.jl") 

function GenerateRandomDensityMatrixNoZeros(n::Int)
    d = 2^n
    M   = randn(ComplexF64, d, d) + im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end



# --------------------------------------------------------------------
# CNOTGate
# --------------------------------------------------------------------
function CNOTGate(n::Int, control::Int, target::Int)
    dim = 2^n
    gate = zeros(ComplexF64, dim, dim)

    for state in 0:(dim-1)
        control_bit = (state >> (n - 1 - control)) & 1
        if control_bit == 1
            # flippe target bit
            new_state = xor(state, 1 << (n - 1 - target))
        else
            new_state = state
        end
        gate[new_state + 1, state + 1] = 1.0
    end

    return gate
end

function randomInitialT(N::Int)
    dim = 2^N
    return randn(dim, dim) + im * randn(dim, dim)
end

function densityMatrixFromT(T::Matrix{ComplexF64})
    AtA = T' * T
    return AtA / tr(AtA)
end

# --------------------------------------------------------------------
# ParseCircuitToMatrix
# --------------------------------------------------------------------
function ParseCircuitToMatrix(text_circuits::Vector{String}, n::Int)
    unitary_list = Matrix{ComplexF64}[]

    for circuit in text_circuits
        U = Matrix{ComplexF64}(I, 2^n, 2^n)

        # Operationen extrahieren
        operations = split(circuit, ")")
        operations = [strip(op, ['(', ')']) for op in operations if !isempty(strip(op))]

        for op in operations
            parts     = split(op, ":")
            length(parts) < 2 && continue
            gate_name = String(parts[1])
            indices   = parse.(Int, split(parts[2], ","))

            if gate_name == "RX90"
                rx = [cos(π/4) -im*sin(π/4); -im*sin(π/4) cos(π/4)]
                gate = Matrix{ComplexF64}(I, 2^n, 2^n)
                # Tensorprodukt
                full = ones(ComplexF64, 1, 1)
                for q in 0:(n-1)
                    full = kron(full, q == indices[1] ? rx : Matrix{ComplexF64}(I, 2, 2))
                end
                U = full * U

            elseif gate_name == "RY90"
                ry = [cos(π/4) -sin(π/4); sin(π/4) cos(π/4)]
                full = ones(ComplexF64, 1, 1)
                for q in 0:(n-1)
                    full = kron(full, q == indices[1] ? ry : Matrix{ComplexF64}(I, 2, 2))
                end
                U = full * U

            elseif gate_name == "CNOT"
                U = CNOTGate(n, indices[1], indices[2]) * U

            elseif gate_name == "H"
                h = [1 1; 1 -1] / sqrt(2)
                full = ones(ComplexF64, 1, 1)
                for q in 0:(n-1)
                    full = kron(full, q == indices[1] ? h : Matrix{ComplexF64}(I, 2, 2))
                end
                U = full * U
            else
                error("Unbekanntes Gate: $gate_name")
            end
        end

        push!(unitary_list, U)
    end

    return unitary_list
end

# --------------------------------------------------------------------
# PrepareState
# --------------------------------------------------------------------
function PrepareState(initial_text::String, N::Int)
    U_list = ParseCircuitToMatrix([initial_text], N)
    U      = U_list[1]

    # |00...0⟩
    psi0      = zeros(ComplexF64, 2^N)
    psi0[1]   = 1.0

    psi       = U * psi0
    rho_ideal = psi * psi'
    return rho_ideal
end

# --------------------------------------------------------------------
# FlattenList
# --------------------------------------------------------------------
# Richtig:
function FlattenList(nested::Vector{Vector{T}}) where {T}
    return vcat(nested...)
end
# --------------------------------------------------------------------
# BlockSeparator
# --------------------------------------------------------------------
function BlockSeparator(rho::Matrix{ComplexF64}, selective_blocks::Vector{Int}, N::Int)
    @assert all(0 .<= selective_blocks .< 2^N) "Alle Blöcke müssen < 2^N sein"

    result = zeros(ComplexF64, 2^N, 2^N)

    for block in selective_blocks
        block_bits = digits(block, base=2, pad=N) |> reverse

        for i in 0:(2^N - 1)
            i_bits = digits(i, base=2, pad=N) |> reverse
            for j in 0:(2^N - 1)
                j_bits = digits(j, base=2, pad=N) |> reverse

                # Prüfe ob (i,j) zu diesem Block gehört
                matches = all(
                    block_bits[k] == 1 ? i_bits[k] != j_bits[k] : i_bits[k] == j_bits[k]
                    for k in 1:N
                )

                if matches
                    result[i+1, j+1] += rho[i+1, j+1]
                end
            end
        end
    end

    return result
end

# --------------------------------------------------------------------
# DataPredictFromRho
# --------------------------------------------------------------------
function DataPredictFromRho(rho::Matrix{ComplexF64}, unitaries::Vector{Matrix{ComplexF64}}, shots::Int)
    predictions = Vector{Vector{Float64}}()

    for U in unitaries
        # pᵢ = diag(U · ρ · U†) · shots
        probs = real(diag(U * rho * U'))
        push!(predictions, probs .* shots)
    end

    return predictions
end

# --------------------------------------------------------------------
# DataPredictFromRhoSampled
# --------------------------------------------------------------------
function DataPredictFromRhoSampled(rho::Matrix{ComplexF64}, unitaries::Vector{Matrix{ComplexF64}}, shots::Int)
    samples = Vector{Vector{Int}}()

    for U in unitaries
        probs = real(diag(U * rho * U'))
        probs = max.(probs, 0)
        probs = probs / sum(probs)

        # Multinomial sampling
        counts = zeros(Int, length(probs))
        for _ in 1:shots
            r   = rand()
            cum = 0.0
            for (k, p) in enumerate(probs)
                cum += p
                if r <= cum
                    counts[k] += 1
                    break
                end
            end
        end

        push!(samples, counts)
    end

    return samples
end

# --------------------------------------------------------------------
# ProcessData
# --------------------------------------------------------------------
# Fix: kein conj
function ProcessData(
    data::Vector{Vector{Int}},
    unitaries::Vector{Matrix{ComplexF64}},
    selective_blocks::Vector{Int},
    shots::Int,
    N::Int;
    rho_ideal::Union{Matrix{ComplexF64}, Nothing} = nothing,
    lr::Float64      = 0.1,
    decay::Float64   = 0.999,
    iterations::Int  = 600,
    patience::Int    = 50,
    tol::Float64     = 1e-10
)
    measured_flat = Float64.(vcat(data...))

    function loss(T)
        rho       = T' * T
        rho       = rho / tr(rho)
        predicted = vcat([real(diag(U * rho * U')) for U in unitaries]...)
        return -sum(measured_flat .* log.(predicted .+ 1e-8)) / shots
    end

    T = randomInitialT(N)

    L0, _ = Zygote.withgradient(loss, T)
    println("=== DEBUG Julia ===")
    println("Loss initial: ", L0)

    m  = zeros(ComplexF64, size(T))
    v  = zeros(Float64,    size(T))
    β₁, β₂, ε_adam = 0.9, 0.999, 1e-8
    losses     = Float64[]
    no_improve = 0

    for i in 1:iterations
        L, grads = Zygote.withgradient(loss, T)
        g = grads[1]  # ← kein conj!

        η  = lr * decay^i
        m  = β₁ .* m + (1 - β₁) .* g
        v  = β₂ .* v + (1 - β₂) .* abs2.(g)
        m̂  = m ./ (1 - β₁^i)
        v̂  = v ./ (1 - β₂^i)
        T  = T - η .* m̂ ./ (sqrt.(v̂) .+ ε_adam)

        push!(losses, L)

        if i % 100 == 0
            println("Iteration $i | Loss: $(round(L, digits=6))")
        end

        if length(losses) > 1
            if abs(losses[end] - losses[end-1]) < tol
                no_improve += 1
            else
                no_improve = 0
            end
        end
        no_improve >= patience && break
    end

    rho_rec = T' * T
    rho_rec = rho_rec / tr(rho_rec)

    return BlockSeparator(rho_rec, selective_blocks, N)
end

#=
N = 2

println("── ProcessData N=4 ──")
rho_true = GenerateRandomDensityMatrixNoZeros(N)
println(rho_true)
# Automatisch generieren
blocks    = collect(0:(2^N-1))
ent_circs = BuildEntanglerBlocks(blocks, N)
circuits  = FlattenList(ent_circs)
println("Anzahl Schaltkreise: ", length(circuits))

Us_all = ParseCircuitToMatrix(circuits, N)
shots  = 1000 * 2^N
data   = DataPredictFromRhoSampled(rho_true, Us_all, shots)

rho_rec = ProcessData(
    data, Us_all, blocks, shots, N;
    iterations    = 3000,
    lr            = 0.1,
    decay         = 0.9999,  # ← langsamer sinkend
    patience      = 200,     # ← länger warten
    tol           = 1e-12    # ← strenger
)

F = real(tr(sqrt(sqrt(rho_true) * rho_rec * sqrt(rho_true))))^2
println("Fidelität: ", round(F, digits=4))
=#

# ── Hilfsfunktionen ────────────────────────────────────────────
function GenerateRandomDensityMatrixNoZeros(N::Int)
    d = 2^N
    M = randn(ComplexF64, d, d)
    rho = M * M'
    return rho / tr(rho)
end

# ── Diagonale messen (nutzt DataPredictFromRhoSampled) ─────────
function MeasureDiagonal(rho::Matrix{ComplexF64}, shots::Int)
    d = size(rho, 1)
    # U = Identität → misst in Standardbasis → gibt Diagonale!
    U_identity = [Matrix{ComplexF64}(I, d, d)]
    result = DataPredictFromRhoSampled(rho, U_identity, shots)
    return Float64.(result[1]) ./ shots
end

# ── Block Pattern für Element (i,j) ───────────────────────────
function element_block(i::Int, j::Int, N::Int)
    i_bits = digits(i, base=2, pad=N) |> reverse
    j_bits = digits(j, base=2, pad=N) |> reverse
    block_bits = [i_bits[k] != j_bits[k] ? 1 : 0 for k in 1:N]
    return sum(block_bits[k] * 2^(N-k) for k in 1:N)
end
function BlocksAboveThreshold(N::Int, rho_diag::Vector{Float64}, t::Float64)
    d   = 2^N
    tol = max(t, 1e-10)  # ← Minimum-Threshold verhindert 0==0 Problem
    relevant = Int[]

    for block in 0:(2^N - 1)
        if block == 0
            push!(relevant, block)
            continue
        end

        block_bits = digits(block, base=2, pad=N) |> reverse
        found = false

        for i in 0:d-1
            found && break
            for j in 0:d-1
                i == j && continue

                i_bits = digits(i, base=2, pad=N) |> reverse
                j_bits = digits(j, base=2, pad=N) |> reverse

                matches = all(
                    block_bits[k] == 1 ?
                        i_bits[k] != j_bits[k] :
                        i_bits[k] == j_bits[k]
                    for k in 1:N
                )

                if matches
                    bound = sqrt(rho_diag[i+1] * rho_diag[j+1])
                    if bound ≥ tol
                        found = true
                        break
                    end
                end
            end
        end

        found && push!(relevant, block)
    end

    return relevant
end

function FidelityBound(rho_true::Matrix{ComplexF64},
                        rho_diag::Vector{Float64}, t::Float64)
    d = size(rho_true, 1)
    r = rank(rho_true, atol=1e-10)
    sum_below = 0.0
    for i in 0:d-1, j in 0:d-1
        i == j && continue
        bound = sqrt(rho_diag[i+1] * rho_diag[j+1])
        if bound < t
            sum_below += rho_diag[i+1] * rho_diag[j+1]
        end
    end
    return max(1.0 - sqrt(r * sum_below), 0.0)^2
end

function CountCircuits(blocks::Vector{Int}, N::Int)
    return sum(b == 0 ? 1 : 2 for b in blocks)
end

function ReconstructDensityMatrixWithTSEEQST(
    rho_true::Matrix{ComplexF64},
    shots::Int,
    t::Float64,
    N::Int;
    iterations::Int  = 3000,
    lr::Float64      = 0.1,
    decay::Float64   = 0.9999,
    patience::Int    = 200,
    tol::Float64     = 1e-12
)
    # ── Schritt 1: Diagonale messen ────────────────────────────
    # U = Identität → misst in Standardbasis → gibt Diagonale
    d = 2^N
    U_identity = [Matrix{ComplexF64}(I, d, d)]
    diag_raw   = DataPredictFromRhoSampled(rho_true, U_identity, shots)
    rho_diag   = Float64.(diag_raw[1]) ./ shots

    # ── Schritt 2: Relevante Blöcke bestimmen ──────────────────
    blocks = BlocksAboveThreshold(N, rho_diag, t)
    n_circ = CountCircuits(blocks, N)
    fb     = FidelityBound(rho_true, rho_diag, t)

    # ── Schritt 3: Circuits bauen ──────────────────────────────
    ent_circs = BuildEntanglerBlocks(blocks, N)
    circuits  = FlattenList(ent_circs)
    Us_all    = ParseCircuitToMatrix(circuits, N)

    # ── Schritt 4: Messen ──────────────────────────────────────
    data = DataPredictFromRhoSampled(rho_true, Us_all, shots)

    # ── Schritt 5: Rekonstruieren ──────────────────────────────
    rho_rec = ProcessData(
        data, Us_all, blocks, shots, N;
        iterations = iterations,
        lr         = lr,
        decay      = decay,
        patience   = patience,
        tol        = tol
    )

    # ── Schritt 6: Fidelity ────────────────────────────────────
    sq     = sqrt(Hermitian(rho_true))
    M      = sq * rho_rec * sq
    M      = Hermitian((M + M') / 2)
    eigs   = max.(eigvals(M), 0.0)
    F      = (sum(sqrt.(eigs)))^2

    return rho_rec, F, fb, n_circ, blocks
end

println("═"^65)
println("Test: BlocksAboveThreshold - Matrixelemente pro Block")
println("═"^65)
println()

N = 2

# ── Hilfsfunktion: alle (i,j) Paare eines Blocks ──────────────
function elements_of_block(block::Int, N::Int)
    d = 2^N
    block_bits = digits(block, base=2, pad=N) |> reverse
    pairs = Tuple{Int,Int}[]
    for i in 0:d-1, j in 0:d-1
        i_bits = digits(i, base=2, pad=N) |> reverse
        j_bits = digits(j, base=2, pad=N) |> reverse
        matches = all(
            block_bits[k] == 1 ?
                i_bits[k] != j_bits[k] :
                i_bits[k] == j_bits[k]
            for k in 1:N
        )
        matches && push!(pairs, (i, j))
    end
    return pairs
end

# ── Zeige alle Blöcke mit ihren Elementen ─────────────────────
println("── Block-Struktur für N=2 ──")
println()
for block in 0:2^N-1
    pairs = elements_of_block(block, N)
    off_diag = filter(p -> p[1] != p[2], pairs)
    println(@sprintf("  Block %d:", block))
    println(@sprintf("    Alle Elemente:        %s", string(pairs)))
    println(@sprintf("    Off-diagonal:         %s", string(off_diag)))
    println()
end

# ── Test 1: Manuell konstruierte Dichtematrix ─────────────────
println("═"^65)
println("Test 1: Manuell konstruierte ρ")
println("═"^65)
println()

# Konstruiere ρ sodass wir wissen welche Blöcke relevant sein sollen
# ρ[0,0] = 0.6, ρ[1,1] = 0.4, alle anderen diagonal = 0
# → √(ρ[0,0]·ρ[1,1]) = √(0.6·0.4) = 0.49 (groß)
# → √(ρ[0,0]·ρ[2,2]) = √(0.6·0.0) = 0.0  (null)
# → √(ρ[1,1]·ρ[2,2]) = 0.0                (null)

d = 2^N
rho_manual = zeros(ComplexF64, d, d)
rho_manual[1,1] = 0.6   # ρ[0,0]
rho_manual[2,2] = 0.4   # ρ[1,1]
rho_manual[3,3] = 0.0   # ρ[2,2]
rho_manual[4,4] = 0.0   # ρ[3,3]
# off-diagonal Elemente die konsistent sind:
rho_manual[1,2] = 0.1; rho_manual[2,1] = 0.1  # ρ[0,1]

rho_diag_manual = real.(diag(rho_manual))
println("Diagonale: ", rho_diag_manual)
println()

# Cauchy-Schwarz Schranken für alle off-diagonalen Paare
println("Cauchy-Schwarz Schranken √(ρᵢᵢ·ρⱼⱼ):")
for i in 0:d-1, j in i+1:d-1
    bound = sqrt(rho_diag_manual[i+1] * rho_diag_manual[j+1])
    b     = element_block(i, j, N)
    println(@sprintf("  ρ[%d,%d] → Block %d: √(%.2f·%.2f) = %.4f",
        i, j, b,
        rho_diag_manual[i+1], rho_diag_manual[j+1], bound))
end
println()

println("Erwartung:")
println("  Block 0: immer relevant (Diagonale)")
println("  Block 1: enthält ρ[0,1] und ρ[2,3]")
println("           ρ[0,1]: √(0.6·0.4)=0.49 → relevant wenn t<0.49")
println("           ρ[2,3]: √(0.0·0.0)=0.0  → nie relevant")
println("           → Block 1 relevant wenn t<0.49")
println("  Block 2: enthält ρ[0,2] und ρ[1,3]")
println("           ρ[0,2]: √(0.6·0.0)=0.0  → nie relevant")
println("           ρ[1,3]: √(0.4·0.0)=0.0  → nie relevant")
println("           → Block 2 NIE relevant!")
println("  Block 3: enthält ρ[0,3] und ρ[1,2]")
println("           ρ[0,3]: √(0.6·0.0)=0.0  → nie relevant")
println("           ρ[1,2]: √(0.4·0.0)=0.0  → nie relevant")
println("           → Block 3 NIE relevant!")
println()

println("Ergebnis BlocksAboveThreshold:")
for t in [0.0, 0.1, 0.3, 0.5]
    blocks  = BlocksAboveThreshold(N, rho_diag_manual, t)
    correct = t < 0.49 ? [0, 1] : [0]
    ok      = sort(blocks) == sort(correct)
    println(@sprintf("  t=%.2f: Blöcke=%s  Erwartet=%s  %s",
        t, string(sort(blocks)), string(correct),
        ok ? "✓" : "✗ FEHLER!"))
end
println()

# ── Test 2: Zeige welche Elemente pro Block über Threshold ─────
println("═"^65)
println("Test 2: Welche Elemente sind über Threshold?")
println("═"^65)
println()

rho_diag_test = [0.5, 0.3, 0.15, 0.05]
println("Diagonale: ", rho_diag_test)
println()

t = 0.1
println(@sprintf("Threshold t = %.2f", t))
println()

println("Alle off-diagonalen Elemente mit ihrer Schranke:")
println(@sprintf("  %-12s  %-8s  %-12s  %-10s",
    "Element", "Block", "Schranke", "Relevant?"))
println("  " * "─"^48)

for i in 0:d-1, j in i+1:d-1
    bound    = sqrt(rho_diag_test[i+1] * rho_diag_test[j+1])
    b        = element_block(i, j, N)
    relevant = bound ≥ t
    println(@sprintf("  ρ[%d,%d]      Block %d   %.4f       %s",
        i, j, b, bound, relevant ? "✓ ja" : "✗ nein"))
end
println()

blocks = BlocksAboveThreshold(N, rho_diag_test, t)
println("Relevante Blöcke: ", sort(blocks))
println()

println("Verifikation pro Block:")
for block in 0:2^N-1
    pairs    = elements_of_block(block, N)
    off_diag = filter(p -> p[1] != p[2], pairs)
    max_bound = isempty(off_diag) ? 0.0 :
        maximum(sqrt(rho_diag_test[p[1]+1] * rho_diag_test[p[2]+1])
                for p in off_diag)
    relevant = block in blocks
    expected = block == 0 || max_bound ≥ t
    ok       = relevant == expected
    println(@sprintf("  Block %d: max_bound=%.4f, relevant=%s, erwartet=%s  %s",
        block, max_bound,
        relevant ? "ja" : "nein",
        expected ? "ja" : "nein",
        ok ? "✓" : "✗ FEHLER!"))
end
println()

# ── Test 3: Extremfälle ────────────────────────────────────────
println("═"^65)
println("Test 3: Extremfälle")
println("═"^65)
println()

# Fall A: t=0 → alle Blöcke relevant
println("Fall A: t=0 → alle Blöcke müssen relevant sein")
rho_diag_a = [0.25, 0.25, 0.25, 0.25]
blocks_a   = BlocksAboveThreshold(N, rho_diag_a, 0.0)
ok_a       = length(blocks_a) == 2^N
println(@sprintf("  Blöcke: %s  %s", string(sort(blocks_a)), ok_a ? "✓" : "✗"))
println()

# Fall B: sehr großes t → nur Block 0
println("Fall B: t=1.0 → nur Block 0 relevant")
blocks_b = BlocksAboveThreshold(N, rho_diag_a, 1.0)
ok_b     = blocks_b == [0]
println(@sprintf("  Blöcke: %s  %s", string(blocks_b), ok_b ? "✓" : "✗"))
println()

# Fall C: ein dominanter Zustand
println("Fall C: ρ[0,0]=1.0 (reiner |00⟩ Zustand)")
rho_diag_c = [1.0, 0.0, 0.0, 0.0]
println("  √(ρᵢᵢ·ρⱼⱼ) für alle (i,j):")
for i in 0:d-1, j in i+1:d-1
    bound = sqrt(rho_diag_c[i+1] * rho_diag_c[j+1])
    println(@sprintf("    ρ[%d,%d]: %.4f", i, j, bound))
end
blocks_c = BlocksAboveThreshold(N, rho_diag_c, 0.01)
ok_c     = blocks_c == [0]
println(@sprintf("  t=0.01: Blöcke=%s (erwartet [0]) %s",
    string(blocks_c), ok_c ? "✓" : "✗"))
println()

println("═"^65)
println("✓ Alle Tests abgeschlossen")
println("═"^65)