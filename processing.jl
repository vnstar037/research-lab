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

# ══════════════════════════════════════════════════════════════
# ReconstructDensityMatrixWithSEEQST
# Verwendet entangling Gates (CNOT-basiert)
# ══════════════════════════════════════════════════════════════
function ReconstructDensityMatrixWithSEEQST(
    rho_true::Matrix{ComplexF64},
    shots::Int;
    iterations::Int = 3000,
    lr::Float64     = 0.1,
    decay::Float64  = 0.9999,
    patience::Int   = 200,
    tol::Float64    = 1e-12
)
    N = Int(round(log2(size(rho_true, 1))))

    # Alle Blöcke
    blocks = collect(0:(2^N - 1))

    # Entangling Circuits
    ent_circs = BuildEntanglerBlocks(blocks, N)
    circuits  = FlattenList(ent_circs)
    Us_all    = ParseCircuitToMatrix(circuits, N)

    # Messen
    data = DataPredictFromRhoSampled(rho_true, Us_all, shots)

    # Rekonstruieren
    rho_rec = ProcessData(
        data, Us_all, blocks, shots, N;
        iterations = iterations,
        lr         = lr,
        decay      = decay,
        patience   = patience,
        tol        = tol
    )

    return Matrix{ComplexF64}(rho_rec)
end

# ══════════════════════════════════════════════════════════════
# ReconstructDensityMatrixWithStandard
# Verwendet nur lokale Messungen (kein CNOT)
# Standard QST: {I, Rx(π/2), Ry(π/2)}^⊗N
# ══════════════════════════════════════════════════════════════
function BuildStandardCircuits(N::Int)
    # Standard QST: alle Kombinationen von I, RX90, RY90
    # pro Qubit → 3^N Circuits
    gates_per_qubit = ["I", "RX90", "RY90"]
    circuits = String[]

    # Alle Kombinationen
    for combo in Iterators.product([gates_per_qubit for _ in 1:N]...)
        circuit = ""
        for (q, gate) in enumerate(combo)
            if gate != "I"
                circuit *= "($gate:$(q-1))"
            end
        end
        push!(circuits, circuit)
    end

    return circuits
end

function ReconstructDensityMatrixWithStandard(
    rho_true::Matrix{ComplexF64},
    shots::Int;
    iterations::Int = 3000,
    lr::Float64     = 0.1,
    decay::Float64  = 0.9999,
    patience::Int   = 200,
    tol::Float64    = 1e-12
)
    N = Int(round(log2(size(rho_true, 1))))

    # Alle Blöcke (für ProcessData)
    blocks = collect(0:(2^N - 1))

    # Standard lokale Circuits (3^N)
    circuits = BuildStandardCircuits(N)
    Us_all   = ParseCircuitToMatrix(circuits, N)

    # Messen
    data = DataPredictFromRhoSampled(rho_true, Us_all, shots)

    # Rekonstruieren
    rho_rec = ProcessData(
        data, Us_all, blocks, shots, N;
        iterations = iterations,
        lr         = lr,
        decay      = decay,
        patience   = patience,
        tol        = tol
    )

    return Matrix{ComplexF64}(rho_rec)
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

# ── Explizite 8×8 Matrix ───────────────────────────────────────
rho_test = zeros(ComplexF64, 8, 8)
rho_test[1,1] =  0.5        # ρ[000,000]
rho_test[2,2] =  0.5        # ρ[001,001]
rho_test[1,2] = -0.5im      # ρ[000,001]
rho_test[2,1] =  0.5im      # ρ[001,000]

N = 3

println("═"^60)
println("Test N=3: Explizite Matrix mit 6 inaktiven Blöcken")
println("═"^60)
println()

# ── Verifikation ──────────────────────────────────────────────
println("── Verifikation ──")
println(@sprintf("  Tr(ρ)          = %.4f  (soll 1.0)",
    real(tr(rho_test))))
println(@sprintf("  Hermitisch     = %s",
    rho_test ≈ rho_test'))
println(@sprintf("  Min Eigenwert  = %.6f  (soll ≥ 0)",
    minimum(real(eigvals(rho_test)))))
println()

# ── Nicht-Null Elemente ───────────────────────────────────────
println("── Nicht-Null Elemente und Blöcke ──")
for i in 0:7, j in 0:7
    if abs(rho_test[i+1, j+1]) > 1e-10
        b = element_block(i, j, N)
        println(@sprintf("  ρ[%d,%d] = %+.3f%+.3fi  → Block %d",
            i, j,
            real(rho_test[i+1,j+1]),
            imag(rho_test[i+1,j+1]),
            b))
    end
end
println()

# ── Threshold Test ────────────────────────────────────────────
println("── BlocksAboveThreshold ──")
rho_diag = real.(diag(rho_test))
println("  Diagonale: ", rho_diag)
println()

println(@sprintf("  %-8s  %-20s  %-20s",
    "t", "Aktiv", "Inaktiv"))
println("  " * "─"^50)
for t in [0.01, 0.1, 0.4, 0.6]
    blocks   = BlocksAboveThreshold(N, rho_diag, t)
    inactive = setdiff(0:2^N-1, blocks)
    println(@sprintf("  t=%.2f   %-20s  %-20s",
        t, string(blocks), string(collect(inactive))))
end
println()

println("Erwartung:")
println("  Block 0: aktiv (Diagonale immer)")
println("  Block 1: aktiv (ρ[0,1]≠0, √(0.5·0.5)=0.5)")
println("  Block 2-7: inaktiv (alle Elemente = 0)")
println()

# ── Rekonstruktion ────────────────────────────────────────────
println("── Rekonstruktion (shots=5000) ──")
println()
shots = 5000

# Standard
t0      = time()
rho_std = ReconstructDensityMatrixWithStandard(rho_test, shots)
t_std   = time() - t0
F_std   = fidelity(rho_std, rho_test)
println(@sprintf("  Standard QST:  F=%.4f  t=%.2fs  Circuits=%d",
    F_std, t_std, 3^N))

# SEEQST
t0         = time()
rho_seeqst = ReconstructDensityMatrixWithSEEQST(rho_test, shots)
t_seeqst   = time() - t0
F_seeqst   = fidelity(rho_seeqst, rho_test)
println(@sprintf("  SEEQST:        F=%.4f  t=%.2fs  Circuits=%d",
    F_seeqst, t_seeqst, 2^(N+1)-1))

# tSEEQST
println()
println(@sprintf("  %-8s  %-10s  %-10s  %-12s  %-8s",
    "t", "Circuits", "Fidelity", "F_bound", "Runtime"))
println("  " * "─"^52)
for t in [0.01, 0.1, 0.4, 0.6]
    t0      = time()
    rho_rec, F, fb, nc, _ = ReconstructDensityMatrixWithTSEEQST(
        rho_test, shots, t, N)
    t_tseeqst = time() - t0
    F_actual  = fidelity(Matrix{ComplexF64}(rho_rec), rho_test)
    println(@sprintf("  t=%.2f   %-10d  %-10.4f  %-12.4f  %.2fs",
        t, nc, F_actual, fb, t_tseeqst))
end
println()

println("═"^60)
println("Erwartete Circuit-Reduktion:")
println(@sprintf("  Standard QST:  %d Circuits", 3^N))
println(@sprintf("  SEEQST:        %d Circuits", 2^(N+1)-1))
println("  tSEEQST t=0.4: 3 Circuits (nur Block 0+1)")
println("  → 97% Reduktion gegenüber Standard QST!")
println("═"^60)