module TSeeqstMixedQutrit

using LinearAlgebra
using Zygote
using Printf
using QuantumInformation
using Random

export RecreatingDensityMatrixWithTMixedSeeqstQutrit,
       BlocksAboveThresholdQutrit,
       FidelityBoundQutrit,
       count_circuits_for_blocks,
       max_bound_for_block

const LEVELS = Dict(1 => (0,1), 2 => (0,2), 3 => (1,2))

# ══════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════
function transition_type_qutrit(x::Int, y::Int)
    x == y                   && return 0
    Set([x,y]) == Set([0,1]) && return 1
    Set([x,y]) == Set([0,2]) && return 2
    return 3
end

# ── Circuit-Zählung: IDENTISCH zu TSeeqstQutrit ───────────────
# Diagonal:       1 Circuit
# Gleicher Typ:   2 Circuits (E und O)
# Gemischter Typ: 2 Circuits (Real und Imag)
function count_circuits_for_block(block::Int, N::Int)
    block_types   = digits(block, base=4, pad=N) |> reverse
    nonzero_types = filter(!=(0), block_types)
    isempty(nonzero_types) && return 1
    return 2
end

function count_circuits_for_blocks(blocks::Vector{Int}, N::Int)
    return sum(count_circuits_for_block(b, N) for b in blocks)
end

# ══════════════════════════════════════════════════════════════
# Threshold Funktionen (IDENTISCH zu TSeeqstQutrit)
# ══════════════════════════════════════════════════════════════
function block_above_threshold_qutrit(k::Int, N::Int,
                                       rho_diag::Vector{Float64}, t::Float64)
    k == 0 && return true
    tol      = max(t, 1e-10)
    d        = 3^N
    k_digits = digits(k, base=4, pad=N) |> reverse
    for i in 0:d-1, j in 0:d-1
        i == j && continue
        di = digits(i, base=3, pad=N) |> reverse
        dj = digits(j, base=3, pad=N) |> reverse
        matches = all(
            transition_type_qutrit(di[l], dj[l]) == k_digits[l]
            for l in 1:N)
        if matches
            bound = sqrt(rho_diag[i+1] * rho_diag[j+1])
            bound ≥ tol && return true
        end
    end
    return false
end

function BlocksAboveThresholdQutrit(N::Int, rho_diag::Vector{Float64}, t::Float64)
    return [k for k in 0:4^N-1 if block_above_threshold_qutrit(k, N, rho_diag, t)]
end

function FidelityBoundQutrit(rho_true::Matrix{ComplexF64},
                              rho_diag::Vector{Float64}, t::Float64)
    d         = size(rho_true, 1)
    r         = rank(rho_true, atol=1e-10)
    sum_below = 0.0
    for i in 0:d-1, j in 0:d-1
        i == j && continue
        bound = sqrt(rho_diag[i+1] * rho_diag[j+1])
        bound < t && (sum_below += rho_diag[i+1] * rho_diag[j+1])
    end
    return max(1.0 - sqrt(r * sum_below), 0.0)^2
end

function max_bound_for_block(k::Int, N::Int, rho_diag::Vector{Float64})
    k_digits = digits(k, base=4, pad=N) |> reverse
    d        = 3^N
    max_b    = 0.0
    for i in 0:d-1, j in 0:d-1
        i == j && continue
        di = digits(i, base=3, pad=N) |> reverse
        dj = digits(j, base=3, pad=N) |> reverse
        matches = all(
            transition_type_qutrit(di[l], dj[l]) == k_digits[l]
            for l in 1:N)
        if matches
            bound = sqrt(rho_diag[i+1] * rho_diag[j+1])
            max_b = max(max_b, bound)
        end
    end
    return max_b
end

# ══════════════════════════════════════════════════════════════
# Gates
# ══════════════════════════════════════════════════════════════
function GellMannRotation(k::Int)
    λ = Dict(
        1 => [0 1 0; 1 0 0; 0 0 0] .+ 0im,
        2 => [0 -im 0; im 0 0; 0 0 0],
        4 => [0 0 1; 0 0 0; 1 0 0] .+ 0im,
        5 => [0 0 -im; 0 0 0; im 0 0],
        6 => [0 0 0; 0 0 1; 0 1 0] .+ 0im,
        7 => [0 0 0; 0 0 -im; 0 im 0],
    )
    return exp(-im * π/4 * λ[k])
end

function CINC11Gate(n::Int, control::Int, target::Int)
    dim  = 3^n
    gate = Matrix{ComplexF64}(I, dim, dim)
    for state in 0:(dim-1)
        ct = (state ÷ 3^(n-1-control)) % 3
        tt = (state ÷ 3^(n-1-target))  % 3
        if ct == 1 && tt in (0,1)
            nt = tt == 0 ? 1 : 0
            ns = state - tt*3^(n-1-target) + nt*3^(n-1-target)
            gate[:,state+1] .= 0
            gate[ns+1,state+1] = 1.0
        end
    end
    return gate
end

function CINC22Gate(n::Int, control::Int, target::Int)
    dim  = 3^n
    gate = Matrix{ComplexF64}(I, dim, dim)
    for state in 0:(dim-1)
        ct = (state ÷ 3^(n-1-control)) % 3
        tt = (state ÷ 3^(n-1-target))  % 3
        if ct == 2 && tt in (0,2)
            nt = tt == 0 ? 2 : 0
            ns = state - tt*3^(n-1-target) + nt*3^(n-1-target)
            gate[:,state+1] .= 0
            gate[ns+1,state+1] = 1.0
        end
    end
    return gate
end

function CINC33Gate(n::Int, control::Int, target::Int)
    dim  = 3^n
    gate = Matrix{ComplexF64}(I, dim, dim)
    for state in 0:(dim-1)
        ct = (state ÷ 3^(n-1-control)) % 3
        tt = (state ÷ 3^(n-1-target))  % 3
        if ct == 2 && tt in (1,2)
            nt = tt == 1 ? 2 : 1
            ns = state - tt*3^(n-1-target) + nt*3^(n-1-target)
            gate[:,state+1] .= 0
            gate[ns+1,state+1] = 1.0
        end
    end
    return gate
end

function build_mixed_block_gate_real(block::Int, N::Int)
    d  = 3^N
    U  = Matrix{ComplexF64}(I, d, d)
    t0 = (block ÷ 4) % 4
    t1 =  block      % 4
    a0,b0 = LEVELS[t0]; a1,b1 = LEVELS[t1]
    idx(q0,q1) = q0*3 + q1 + 1
    i1=idx(a0,a1); j1=idx(b0,b1)
    i2=idx(a0,b1); j2=idx(b0,a1)
    U[i1,i1] =  1/√2;  U[i1,j1] =  1/√2
    U[j1,i1] =  1/√2;  U[j1,j1] = -1/√2
    U[i2,i2] =  1/√2;  U[i2,j2] =  1/√2
    U[j2,i2] =  1/√2;  U[j2,j2] = -1/√2
    return U
end

function build_mixed_block_gate_imag(block::Int, N::Int)
    d  = 3^N
    U  = Matrix{ComplexF64}(I, d, d)
    t0 = (block ÷ 4) % 4
    t1 =  block      % 4
    a0,b0 = LEVELS[t0]; a1,b1 = LEVELS[t1]
    idx(q0,q1) = q0*3 + q1 + 1
    i1=idx(a0,a1); j1=idx(b0,b1)
    i2=idx(a0,b1); j2=idx(b0,a1)
    U[i1,i1] =  1/√2;  U[i1,j1] =  im/√2
    U[j1,i1] =  1/√2;  U[j1,j1] = -im/√2
    U[i2,i2] =  1/√2;  U[i2,j2] =  im/√2
    U[j2,i2] =  1/√2;  U[j2,j2] = -im/√2
    return U
end

function build_unitaries_for_block(block::Int, N::Int)
    block_types   = digits(block, base=4, pad=N) |> reverse
    nonzero_types = filter(!=(0), block_types)
    unique_types  = unique(nonzero_types)
    e_gate  = Dict(1=>2, 2=>5, 3=>7)
    o_gate  = Dict(1=>1, 2=>4, 3=>6)
    cinc_fn = Dict(1=>CINC11Gate, 2=>CINC22Gate, 3=>CINC33Gate)

    isempty(nonzero_types) && return [Matrix{ComplexF64}(I, 3^N, 3^N)]

    if length(unique_types) == 1
        t           = unique_types[1]
        active_quts = [i-1 for (i,bt) in enumerate(block_types) if bt != 0]
        entangling  = Matrix{ComplexF64}(I, 3^N, 3^N)
        if length(active_quts) > 1
            for k in 2:length(active_quts)
                entangling = cinc_fn[t](N, active_quts[1], active_quts[k]) * entangling
            end
        end
        rot_E  = GellMannRotation(e_gate[t])
        rot_O  = GellMannRotation(o_gate[t])
        full_E = ones(ComplexF64,1,1)
        full_O = ones(ComplexF64,1,1)
        for q in 0:(N-1)
            I3     = Matrix{ComplexF64}(I,3,3)
            full_E = kron(full_E, q==active_quts[1] ? rot_E : I3)
            full_O = kron(full_O, q==active_quts[1] ? rot_O : I3)
        end
        return [full_E * entangling, full_O * entangling]
    end

    if N == 2
        return [build_mixed_block_gate_real(block, N),
                build_mixed_block_gate_imag(block, N)]
    else
        active_quts = [(i-1, bt) for (i,bt) in enumerate(block_types) if bt != 0]
        options     = [[GellMannRotation(e_gate[bt]),
                        GellMannRotation(o_gate[bt])] for (_,bt) in active_quts]
        qutrit_idxs = [q for (q,_) in active_quts]
        circuits    = Matrix{ComplexF64}[]
        for choice in Iterators.product(options...)
            U = Matrix{ComplexF64}(I, 3^N, 3^N)
            for (rot, q) in zip(choice, qutrit_idxs)
                full = ones(ComplexF64,1,1)
                for qq in 0:(N-1)
                    full = kron(full, qq==q ? rot : Matrix{ComplexF64}(I,3,3))
                end
                U = full * U
            end
            push!(circuits, U)
        end
        return circuits
    end
end

# ══════════════════════════════════════════════════════════════
# Messung
# ══════════════════════════════════════════════════════════════
function DataPredictFromRhoSampledQutrit(
    rho::Matrix{ComplexF64},
    unitaries::Vector{Matrix{ComplexF64}},
    shots::Int
)
    all_counts = Vector{Vector{Int}}()
    for U in unitaries
        probs  = real(diag(U * rho * U'))
        probs  = max.(probs, 0)
        probs  = probs / sum(probs)
        counts = zeros(Int, length(probs))
        for _ in 1:shots
            r = rand(); cum = 0.0
            for (k,p) in enumerate(probs)
                cum += p
                if r <= cum; counts[k] += 1; break; end
            end
        end
        push!(all_counts, counts)
    end
    return all_counts
end

# ══════════════════════════════════════════════════════════════
# SGD + Cholesky
# ══════════════════════════════════════════════════════════════
function ProcessDataQutrit(
    data::Vector{Vector{Int}},
    unitaries::Vector{Matrix{ComplexF64}},
    shots::Int,
    N::Int;
    lr::Float64     = 0.1,
    decay::Float64  = 0.9999,
    iterations::Int = 1000,
    patience::Int   = 200,
    tol::Float64    = 1e-12,
    verbose::Bool   = true
)
    function loss(T)
        rho   = T' * T
        rho   = rho / tr(rho)
        total = 0.0
        for (U, counts) in zip(unitaries, data)
            predicted = real(diag(U * rho * U'))
            total    -= sum(Float64.(counts) .* log.(predicted .+ 1e-8))
        end
        return total / shots
    end

    T = randn(3^N, 3^N) + im*randn(3^N, 3^N)
    verbose && println("  Loss initial: ",
        round(Zygote.withgradient(loss,T)[1], digits=4))

    m = zeros(ComplexF64, size(T))
    v = zeros(Float64,    size(T))
    β₁, β₂, ε = 0.9, 0.999, 1e-8
    losses     = Float64[]
    no_improve = 0

    for i in 1:iterations
        L, grads = Zygote.withgradient(loss, T)
        g = grads[1]
        g === nothing && break
        η = lr * decay^i
        m = β₁ .* m + (1-β₁) .* g
        v = β₂ .* v + (1-β₂) .* abs2.(g)
        m̂ = m ./ (1-β₁^i)
        v̂ = v ./ (1-β₂^i)
        T = T - η .* m̂ ./ (sqrt.(v̂) .+ ε)
        push!(losses, L)
        verbose && i % 500 == 0 &&
            println("  Iteration $i | Loss: $(round(L, digits=6))")
        if length(losses) > 1
            rel = abs(losses[end]-losses[end-1]) / abs(losses[end-1])
            rel < tol ? no_improve += 1 : (no_improve = 0)
        end
        no_improve >= patience && break
    end

    rho_rec = T' * T
    return rho_rec / tr(rho_rec)
end

# ══════════════════════════════════════════════════════════════
# Hauptfunktion
# ══════════════════════════════════════════════════════════════
function RecreatingDensityMatrixWithTMixedSeeqstQutrit(
    rho_true::Matrix{ComplexF64},
    shots::Int,
    t::Float64;
    lr::Float64     = 0.1,
    decay::Float64  = 0.9999,
    iterations::Int = 1000,
    patience::Int   = 200,
    verbose::Bool   = true
)
    N = Int(round(log(3, size(rho_true,1))))
    d = 3^N

    verbose && println("═══ tSEEQST Mixed N=$N, t=$t ═══")
    verbose && println()

    # ── Schritt 1: Diagonale messen ────────────────────────────
    U_identity = [Matrix{ComplexF64}(I, d, d)]
    diag_raw   = DataPredictFromRhoSampledQutrit(rho_true, U_identity, shots)
    rho_diag   = Float64.(diag_raw[1]) ./ shots

    if verbose
        println("── Diagonale ──")
        for i in 0:d-1
            abs(rho_diag[i+1]) > 1e-6 &&
                @printf("  ρ[%d,%d] = %.4f\n", i, i, rho_diag[i+1])
        end
        println()
    end

    # ── Schritt 2: Threshold (IDENTISCH zu TSeeqstQutrit) ──────
    blocks_all       = collect(0:(4^N - 1))
    blocks_threshold = BlocksAboveThresholdQutrit(N, rho_diag, t)

    # ── Schritt 3: Minimum IDENTISCH zu TSeeqstQutrit ──────────
    # min_circ = ceil(Int, (d^2 - 1) / d)
    # Fill-up Bedingung: n_circ * d < d^2 - 1
    min_circ        = ceil(Int, (d^2 - 1) / d)
    blocks_relevant = copy(blocks_threshold)
    n_circ          = count_circuits_for_blocks(blocks_relevant, N)
    blocks_added    = Int[]

    if n_circ * d < d^2 - 1
        missing_blocks = setdiff(blocks_all, blocks_relevant)
        sorted_missing = sort(missing_blocks,
            by = k -> max_bound_for_block(k, N, rho_diag), rev=true)

        for k in sorted_missing
            push!(blocks_relevant, k)
            sort!(blocks_relevant)
            push!(blocks_added, k)
            n_circ = count_circuits_for_blocks(blocks_relevant, N)
            n_circ * d ≥ d^2 - 1 && break
        end
    end

    # ── Statistiken ────────────────────────────────────────────
    n_circ_all = count_circuits_for_blocks(blocks_all, N)
    n_circ_rel = count_circuits_for_blocks(blocks_relevant, N)
    fb         = FidelityBoundQutrit(rho_true, rho_diag, t)
    reduktion  = (1 - n_circ_rel / n_circ_all) * 100

    if verbose
        println("── Threshold-Entscheidung ──")
        @printf("  Alle Blöcke:          %d\n", length(blocks_all))
        @printf("  Threshold Blöcke:     %d\n", length(blocks_threshold))
        if !isempty(blocks_added)
            @printf("  Aufgefüllt (+%d):     %d Blöcke\n",
                length(blocks_added), length(blocks_relevant))
            @printf("  Aufgefüllte Blöcke:  %s\n", string(blocks_added))
        end
        @printf("  Circuits Standard:    %d\n", n_circ_all)
        @printf("  Circuits tSEEQST:     %d  (min: %d)\n", n_circ_rel, min_circ)
        @printf("  Reduktion:            %.1f%%\n", reduktion)
        @printf("  F_bound:              %.4f\n", fb)
        @printf("  Unterbestimmt:        %s\n",
            n_circ_rel * d < d^2 - 1 ? "✗ JA" : "✓ nein")
        println()
    end

    # ── Schritt 4: Circuits bauen ──────────────────────────────
    Us_all = vcat([build_unitaries_for_block(b, N) for b in blocks_relevant]...)

    verbose && @printf("── Messen ──\n  Circuits: %d\n\n", length(Us_all))

    data = DataPredictFromRhoSampledQutrit(rho_true, Us_all, shots)

    # ── Schritt 5: Rekonstruktion ──────────────────────────────
    verbose && println("── SGD + Cholesky ──")
    rho_rec = ProcessDataQutrit(data, Us_all, shots, N;
        lr=lr, decay=decay, iterations=iterations,
        patience=patience, verbose=verbose)

    if verbose
        F = fidelity(rho_rec, rho_true)
        println()
        println("── Ergebnisse ──")
        @printf("  Fidelität:           %.4f\n", F)
        @printf("  F_bound:             %.4f\n", fb)
        @printf("  Circuits:            %d / %d  (%.1f%% Reduktion)\n",
            n_circ_rel, n_circ_all, reduktion)
        @printf("  Spur ρ_rec:          %.6f\n", real(tr(rho_rec)))
        @printf("  Kleinster Eigenwert: %.6f\n", minimum(real(eigvals(rho_rec))))
        println("═"^45)
    end

    return Matrix{ComplexF64}(rho_rec)
end

end # module TSeeqstMixedQutrit