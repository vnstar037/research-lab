# ══════════════════════════════════════════════════════════════
# tSEEQST für N-Qubit Systeme (allgemein)
# ══════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Random
using Statistics

# ── Bit-Operationen ────────────────────────────────────────────
function index_to_bits(idx::Int, N::Int)
    return [((idx >> (N-1-k)) & 1) for k in 0:N-1]
end

function bits_to_index(bits::Vector{Int})
    idx = 0
    for b in bits
        idx = idx*2 + b
    end
    return idx
end

# ── Zustände ────────────────────────────────────────────────────
function random_rho(d::Int)
    G = randn(ComplexF64, d, d)
    ρ = G * G'
    return ρ / tr(ρ)
end

function plus_y_state(N::Int)
    plus_y = ComplexF64[1/√2, im/√2]
    psi = plus_y
    for _ in 2:N
        psi = kron(psi, plus_y)
    end
    return psi * psi'
end

function sparse_state(N::Int, sparsity::Float64=0.9)
    d = 2^N
    psi = zeros(ComplexF64, d)
    psi[1] = sqrt(sparsity)
    remaining = sqrt(1 - sparsity)
    rest = randn(ComplexF64, d-1)
    rest = rest / norm(rest) * remaining
    psi[2:end] = rest
    return psi * psi'
end

# ══════════════════════════════════════════════════════════════
# SEEQST Block-Struktur für N Qubits
# ══════════════════════════════════════════════════════════════
function block_pattern(i::Int, j::Int, N::Int)
    bi = index_to_bits(i, N)
    bj = index_to_bits(j, N)
    pattern = [bi[k] == bj[k] ? 0 : 1 for k in 1:N]
    return bits_to_index(pattern)
end

n_blocks(N::Int) = 2^N

function block_M(k::Int, N::Int)
    return count(==(1), index_to_bits(k, N))
end

function elements_in_block(k::Int, N::Int)
    d = 2^N
    pairs = Tuple{Int,Int}[]
    for i in 0:d-1, j in 0:d-1
        if block_pattern(i, j, N) == k
            push!(pairs, (i, j))
        end
    end
    return pairs
end

function circuits_for_block(k::Int, N::Int)
    M = block_M(k, N)
    return M == 0 ? 1 : 2
end

function total_circuits_seeqst(N::Int)
    total = 0
    for k in 0:n_blocks(N)-1
        total += circuits_for_block(k, N)
    end
    return total
end

# ── Threshold-Logik ───────────────────────────────────────────────
function block_above_threshold(k::Int, N::Int,
                                 rho_diag::Vector{Float64}, t::Float64)
    if k == 0
        return true
    end
    pairs = elements_in_block(k, N)
    for (i, j) in pairs
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
    relevant = Int[]
    for k in 0:n_blocks(N)-1
        if block_above_threshold(k, N, rho_diag, t)
            push!(relevant, k)
        end
    end
    return relevant
end

function fidelity_bound(rho_true::Matrix{ComplexF64},
                          rho_diag::Vector{Float64}, t::Float64)
    d = size(rho_true, 1)
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
    inner = max(inner, 0.0)
    return inner^2
end

# ══════════════════════════════════════════════════════════════
# Gates: N-Qubit Basistransformationen für Block k
# ══════════════════════════════════════════════════════════════
I2()  = Matrix{ComplexF64}(I, 2, 2)
function Ry(theta)
    c, s = cos(theta/2), sin(theta/2)
    return ComplexF64[c -s; s c]
end
function Rx(theta)
    c, s = cos(theta/2), sin(theta/2)
    return ComplexF64[c -im*s; -im*s c]
end
CNOT_mat() = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

# Tensorprodukt einer Liste von 2x2 Matrizen
function tensor_product(mats::Vector{Matrix{ComplexF64}})
    result = mats[1]
    for m in mats[2:end]
        result = kron(result, m)
    end
    return result
end

# CNOT zwischen Qubit q1 (control) und q2 (target) in N-Qubit System
function cnot_gate(control::Int, target::Int, N::Int)
    d = 2^N
    G = zeros(ComplexF64, d, d)
    for idx in 0:d-1
        bits = index_to_bits(idx, N)
        if bits[control+1] == 1
            new_bits = copy(bits)
            new_bits[target+1] = 1 - new_bits[target+1]
            new_idx = bits_to_index(new_bits)
            G[new_idx+1, idx+1] = 1
        else
            G[idx+1, idx+1] = 1
        end
    end
    return G
end

# Basistransformation für Block k (Bitmuster pattern[1..N])
# pattern[l]==1 → Qubit l ist "off-diagonal" (M Qubits insgesamt)
# Erste off-diagonal Qubit bekommt Rotation, dann CNOTs zu allen
# anderen off-diagonal Qubits (GHZ-artig, Paper Fig.2)
function basis_transform(k::Int, N::Int, rotation::Symbol)
    pattern = index_to_bits(k, N)
    M = count(==(1), pattern)

    if M == 0
        return nothing  # Block S1, keine Transformation
    end

    # Finde Positionen der off-diagonal Qubits
    off_diag_qubits = findall(==(1), pattern) .- 1  # 0-basiert

    # Lokale Rotation auf erstem off-diagonal Qubit
    R = rotation == :E ? Ry(π/2) : Rx(π/2)

    local_gates = Vector{Matrix{ComplexF64}}(undef, N)
    for q in 0:N-1
        local_gates[q+1] = I2()
    end
    local_gates[off_diag_qubits[1]+1] = R

    U = tensor_product(local_gates)

    # CNOTs vom ersten off-diagonal Qubit zu allen weiteren
    for q in off_diag_qubits[2:end]
        U = cnot_gate(off_diag_qubits[1], q, N) * U
    end

    return U
end

# ══════════════════════════════════════════════════════════════
# Messung
# ══════════════════════════════════════════════════════════════
function sample_probs(rho::Matrix{ComplexF64}, U, shots::Int)
    if U === nothing
        p_true = real.(diag(rho))
    else
        p_true = real.(diag(U * rho * U'))
    end
    p_true = max.(p_true, 0.0)
    p_true = p_true / sum(p_true)

    d = length(p_true)
    counts = zeros(Int, d)
    cum = cumsum(p_true)
    for _ in 1:shots
        s = rand()
        idx = searchsortedfirst(cum, s)
        idx = min(idx, d)
        counts[idx] += 1
    end
    return counts ./ shots
end

# ══════════════════════════════════════════════════════════════
# Datensammlung + Rekonstruktion
# ══════════════════════════════════════════════════════════════
function collect_measurements(rho_true::Matrix{ComplexF64},
                                blocks::Vector{Int}, shots::Int, N::Int)
    data = []

    if 0 in blocks
        p1 = sample_probs(rho_true, nothing, shots)
        push!(data, (nothing, p1))
    end

    for k in blocks
        if k != 0
            U_E = basis_transform(k, N, :E)
            U_O = basis_transform(k, N, :O)
            pE = sample_probs(rho_true, U_E, shots)
            pO = sample_probs(rho_true, U_O, shots)
            push!(data, (U_E, pE))
            push!(data, (U_O, pO))
        end
    end

    return data
end

function loss_function(T_params::Vector{Float64}, data, d::Int)
    T = zeros(ComplexF64, d, d)
    idx = 1
    for i in 1:d, j in 1:i
        if i == j
            T[i,j] = T_params[idx]; idx += 1
        else
            T[i,j] = T_params[idx] + im*T_params[idx+1]; idx += 2
        end
    end
    rho = T * T'
    rho = rho / real(tr(rho))

    L = 0.0
    for (U, p_meas) in data
        if U === nothing
            p_pred = real.(diag(rho))
        else
            p_pred = real.(diag(U * rho * U'))
        end
        p_pred = max.(p_pred, 1e-12)
        L += sum((p_meas .- p_pred).^2)
    end
    return L
end

function reconstruct_rho(data, d::Int; iters=300, lr=0.05)
    n_params = d^2
    params = randn(n_params) * 0.1

    for it in 1:iters
        grad = zeros(n_params)
        L0 = loss_function(params, data, d)
        h = 1e-5
        for k in 1:n_params
            p2 = copy(params); p2[k] += h
            grad[k] = (loss_function(p2, data, d) - L0) / h
        end
        params .-= lr .* grad
    end

    T = zeros(ComplexF64, d, d)
    idx = 1
    for i in 1:d, j in 1:i
        if i == j
            T[i,j] = params[idx]; idx += 1
        else
            T[i,j] = params[idx] + im*params[idx+1]; idx += 2
        end
    end
    rho = T * T'
    return rho / real(tr(rho))
end

function fidelity(rho1, rho2)
    sq1 = sqrt(Hermitian(rho1))
    M = sq1 * rho2 * sq1
    M = Hermitian((M+M')/2)
    eigs = max.(eigvals(M), 0.0)
    return (sum(sqrt.(eigs)))^2
end

# ══════════════════════════════════════════════════════════════
# tSEEQST: vollständiger Ablauf für N Qubits
# ══════════════════════════════════════════════════════════════
function run_tseeqst(rho_true::Matrix{ComplexF64}, shots::Int, t::Float64, N::Int)
    rho_diag = sample_probs(rho_true, nothing, shots)
    blocks = blocks_to_measure(N, rho_diag, t)
    data = collect_measurements(rho_true, blocks, shots, N)
    d = 2^N
    rho_rec = reconstruct_rho(data, d)
    n_circuits = sum(circuits_for_block(k, N) for k in blocks)
    return rho_rec, n_circuits, blocks
end

# ══════════════════════════════════════════════════════════════
# Test
# ══════════════════════════════════════════════════════════════
Random.seed!(123)

println("═"^65)
println("tSEEQST Test: N-Qubit Systeme")
println("═"^65)
println()

shots = 5000

for N in [2, 3]
    println("── N = $N Qubits ──")
    println("  Standard SEEQST Circuits: $(total_circuits_seeqst(N))")
    println()

    # Dense state
    println("  Dense state |+y⟩^⊗N:")
    rho_dense = plus_y_state(N)
    for t in [0.0, 0.1]
        rho_rec, n_circ, blocks = run_tseeqst(rho_dense, shots, t, N)
        F = fidelity(rho_rec, rho_dense)
        println(@sprintf("    t=%.2f: %d Circuits, F=%.4f, Blocks=%s",
            t, n_circ, real(F), string(blocks)))
    end
    println()

    # Sparse state
    println("  Sparse state (97% in |0...0⟩):")
    rho_sparse = sparse_state(N, 0.97)
    for t in [0.0, 0.05, 0.1, 0.2]
        rho_rec, n_circ, blocks = run_tseeqst(rho_sparse, shots, t, N)
        F = fidelity(rho_rec, rho_sparse)
        println(@sprintf("    t=%.2f: %d Circuits, F=%.4f, Blocks=%s",
            t, n_circ, real(F), string(blocks)))
    end
    println()
end

println("═"^65)
println("✓ Test abgeschlossen")
println("═"^65)