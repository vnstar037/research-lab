# ══════════════════════════════════════════════════════════════
# tSEEQST Block-Struktur & Threshold für N Qutrits (allgemein)
# ══════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Random

# ── Basis 4 Darstellung eines Blocks ───────────────────────────
# Jedes Pattern ist ein N-Tupel aus {0,1,2,3}
#   0 = diagonal (i_l == j_l)
#   1 = Übergang |0⟩↔|1⟩
#   2 = Übergang |0⟩↔|2⟩
#   3 = Übergang |1⟩↔|2⟩

function digits_base4(k::Int, N::Int)
    d = zeros(Int, N)
    for l in N:-1:1
        d[l] = k % 4
        k ÷= 4
    end
    return d  # d[1] = höchstwertige Stelle
end

function pattern_to_index(pattern::Vector{Int})
    k = 0
    for p in pattern
        k = k*4 + p
    end
    return k
end

# Übergangstyp zwischen Zuständen x,y ∈ {0,1,2}
function transition_type(x::Int, y::Int)
    if x == y; return 0
    elseif Set([x,y]) == Set([0,1]); return 1
    elseif Set([x,y]) == Set([0,2]); return 2
    else; return 3
    end
end

# Index (0..3^N-1) -> Basis-3 Digits
function digits_base3(idx::Int, N::Int)
    d = zeros(Int, N)
    for l in N:-1:1
        d[l] = idx % 3
        idx ÷= 3
    end
    return d
end

# Pattern für Element ρ[i,j] (i,j ∈ 0:3^N-1)
function block_pattern(i::Int, j::Int, N::Int)
    di = digits_base3(i, N)
    dj = digits_base3(j, N)
    return [transition_type(di[l], dj[l]) for l in 1:N]
end

n_blocks(N::Int) = 4^N

# ── Circuits pro Block ──────────────────────────────────────────
# - alle 0 (Diagonale): 1 Circuit
# - genau 1 nicht-null Eintrag: 2 Circuits (lokale Rotation oder CINC)
# - mehrere nicht-null Einträge, alle gleich: 2 Circuits (CINC)
# - mehrere nicht-null Einträge, verschieden: 2^M Circuits (M = Anzahl aktiver Qutrits)
function circuits_for_pattern(pattern::Vector{Int})
    nonzero = filter(!=(0), pattern)
    if isempty(nonzero)
        return 1
    elseif length(unique(nonzero)) == 1
        return 2
    else
        M = length(nonzero)
        return 2^M
    end
end

function total_circuits_seeqst(N::Int)
    total = 0
    for k in 0:n_blocks(N)-1
        pattern = digits_base4(k, N)
        total += circuits_for_pattern(pattern)
    end
    return total
end

# ── Elemente (i,j) für ein gegebenes Pattern (Index k) ─────────
function elements_in_block(k::Int, N::Int)
    target = digits_base4(k, N)
    d = 3^N
    pairs = Tuple{Int,Int}[]
    for i in 0:d-1, j in 0:d-1
        if block_pattern(i, j, N) == target
            push!(pairs, (i, j))
        end
    end
    return pairs
end

# ── Threshold-Logik ──────────────────────────────────────────────
function block_above_threshold(k::Int, N::Int,
                                 rho_diag::Vector{Float64}, t::Float64)
    if k == 0
        return true  # Diagonal-Block immer relevant
    end
    for (i,j) in elements_in_block(k, N)
        if i != j
            bound = sqrt(rho_diag[i+1] * rho_diag[j+1])
            if bound ≥ t
                return true
            end
        end
    end
    return false
end

function blocks_to_measure(N::Int, rho_diag::Vector{Float64}, t::Float64)
    return [k for k in 0:n_blocks(N)-1 if block_above_threshold(k, N, rho_diag, t)]
end

function circuits_with_threshold(N::Int, rho_diag::Vector{Float64}, t::Float64)
    blocks = blocks_to_measure(N, rho_diag, t)
    total = 0
    for k in blocks
        pattern = digits_base4(k, N)
        total += circuits_for_pattern(pattern)
    end
    return total, blocks
end

function fidelity_bound(rho_true::Matrix{ComplexF64},
                          rho_diag::Vector{Float64}, t::Float64)
    d = size(rho_true,1)
    r = rank(rho_true, atol=1e-10)
    sum_below = 0.0
    for i in 0:d-1, j in 0:d-1
        if i != j
            bound = sqrt(rho_diag[i+1] * rho_diag[j+1])
            if bound < t
                sum_below += rho_diag[i+1] * rho_diag[j+1]
            end
        end
    end
    inner = 1.0 - sqrt(r * sum_below)
    return max(inner, 0.0)^2
end

# ── Zustände ──────────────────────────────────────────────────────
function plus_y_state_qutrit(N::Int)
    plus_y = ComplexF64[1/√2, im/√2, 0.0]
    psi = plus_y
    for _ in 2:N
        psi = kron(psi, plus_y)
    end
    return psi * psi'
end

function sparse_state_qutrit(N::Int, sparsity::Float64=0.97)
    d = 3^N
    psi = zeros(ComplexF64, d)
    psi[1] = sqrt(sparsity)
    remaining = sqrt(1 - sparsity)
    rest = randn(ComplexF64, d-1)
    rest = rest / norm(rest) * remaining
    psi[2:end] = rest
    return psi * psi'
end

function sample_diagonal(rho::Matrix{ComplexF64}, shots::Int)
    p_true = real.(diag(rho))
    p_true = max.(p_true, 0.0)
    p_true = p_true / sum(p_true)
    d = length(p_true)
    counts = zeros(Int, d)
    cum = cumsum(p_true)
    for _ in 1:shots
        s = rand()
        idx = min(searchsortedfirst(cum, s), d)
        counts[idx] += 1
    end
    return counts ./ shots
end

# ══════════════════════════════════════════════════════════════
# Test für N=2 und N=3
# ══════════════════════════════════════════════════════════════
Random.seed!(123)

println("═"^65)
println("tSEEQST Block-Struktur: Circuit-Reduktion durch Threshold")
println("═"^65)
println()

shots = 5000

for N in [2, 3]
    println("── N = $N Qutrits ──")
    println("  Blöcke total: $(n_blocks(N))")
    println("  Standard SEEQST Circuits: $(total_circuits_seeqst(N))")
    println()

    # Dense state
    println("  Dense state |+y⟩^⊗N:")
    rho_dense = plus_y_state_qutrit(N)
    rho_diag = sample_diagonal(rho_dense, shots)
    for t in [0.0, 0.05, 0.1]
        circ, blocks = circuits_with_threshold(N, rho_diag, t)
        fb = fidelity_bound(rho_dense, rho_diag, t)
        println(@sprintf("    t=%.2f: %3d Circuits (%2d Blöcke), F_bound=%.4f",
            t, circ, length(blocks), fb))
    end
    println()

    # Sparse state
    println("  Sparse state (97% in |0...0⟩):")
    rho_sparse = sparse_state_qutrit(N, 0.97)
    rho_diag_s = sample_diagonal(rho_sparse, shots)
    for t in [0.0, 0.02, 0.05, 0.1, 0.2]
        circ, blocks = circuits_with_threshold(N, rho_diag_s, t)
        fb = fidelity_bound(rho_sparse, rho_diag_s, t)
        println(@sprintf("    t=%.2f: %3d Circuits (%2d Blöcke), F_bound=%.4f",
            t, circ, length(blocks), fb))
    end
    println()
end

println("═"^65)
println("✓ Test abgeschlossen")
println("═"^65)