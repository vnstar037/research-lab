module SeeqstGD

using LinearAlgebra
using StatsBase
using Zygote          # Automatische Differentiation (wie JAX in Python)
using Optimisers      # Adam-Optimizer (wie Optax in Python)
using Printf

export GenerateSGroups,
       generateEigenstatesE,
       generateEigenstatesO,
       ProjectorsFromEigenstates,
       simulateMeasurement,
       reconstructDensityMatrixWithSeeqstGD

# ══════════════════════════════════════════════════════════════════════════════
# Gruppen generieren
# ══════════════════════════════════════════════════════════════════════════════

function GenerateSGroups(N::Int)
    groups = Vector{Vector{String}}()
    for mask in 0:(2^N - 1)
        S = Vector{String}(undef, N)
        for qubit in 1:N
            bit = (mask >> (N - qubit)) & 1
            S[qubit] = bit == 0 ? "IZ" : "XY"
        end
        push!(groups, S)
    end
    return groups
end

# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen für Eigenbasen
# ══════════════════════════════════════════════════════════════════════════════

function basisVector(bits::Vector{Int})
    v = [1.0]
    for b in bits
        v = kron(v, b == 0 ? [1.0, 0.0] : [0.0, 1.0])
    end
    return v
end

bitFlip(bits::Vector{Int}) = [1 - b for b in bits]

function bitStrings(n::Int)
    n == 0 && return [Int[]]
    prev = bitStrings(n - 1)
    res = Vector{Vector{Int}}()
    for p in prev
        push!(res, vcat(p, 0))
        push!(res, vcat(p, 1))
    end
    return res
end

function permuteToPhysical(state::AbstractVector, offdiagPos::Vector{Int}, diagPos::Vector{Int})
    perm = vcat(offdiagPos, diagPos)
    N = length(perm)
    reshaped = reshape(state, ntuple(_ -> 2, N)...)
    invp = invperm(perm)
    return vec(permutedims(reshaped, invp))
end

function parsePaulis(paulis::Vector{String})
    offdiagPos = Int[]
    diagPos    = Int[]
    for (i, p) in enumerate(paulis)
        if p == "XY";  push!(offdiagPos, i)
        elseif p == "IZ"; push!(diagPos, i)
        else; error("Unbekannter Pauli-String: $p")
        end
    end
    return offdiagPos, diagPos
end

# ══════════════════════════════════════════════════════════════════════════════
# E-Eigenbasen
# ══════════════════════════════════════════════════════════════════════════════

function generateEigenstatesE(paulis::Vector{String})
    offdiagPos, diagPos = parsePaulis(paulis)
    M = length(offdiagPos); K = length(diagPos)
    states = Vector{Vector{ComplexF64}}()

    if M == 0
        for q in bitStrings(K)
            push!(states, permuteToPhysical(ComplexF64.(basisVector(q)), Int[], diagPos))
        end
        return states
    end

    p_bits = [p for p in bitStrings(M) if p[1] == 0]
    q_bits = K > 0 ? bitStrings(K) : [Int[]]

    for p in p_bits
        pbar = bitFlip(p)
        vp    = ComplexF64.(basisVector(p))
        vpbar = ComplexF64.(basisVector(pbar))
        plus  = (vp + vpbar) / sqrt(2)
        minus = (vp - vpbar) / sqrt(2)
        for q in q_bits
            if K > 0
                vq = ComplexF64.(basisVector(q))
                push!(states, permuteToPhysical(kron(plus,  vq), offdiagPos, diagPos))
                push!(states, permuteToPhysical(kron(minus, vq), offdiagPos, diagPos))
            else
                push!(states, permuteToPhysical(plus,  offdiagPos, Int[]))
                push!(states, permuteToPhysical(minus, offdiagPos, Int[]))
            end
        end
    end
    return states
end

# ══════════════════════════════════════════════════════════════════════════════
# O-Eigenbasen
# ══════════════════════════════════════════════════════════════════════════════

function generateEigenstatesO(paulis::Vector{String})
    offdiagPos = [i for (i,p) in enumerate(paulis) if p == "XY"]
    diagPos    = [i for (i,p) in enumerate(paulis) if p == "IZ"]
    M = length(offdiagPos); K = length(diagPos)
    M == 0 && return Vector{Vector{ComplexF64}}()

    states = Vector{Vector{ComplexF64}}()
    p_bits = [p for p in bitStrings(M) if p[1] == 0]
    q_bits = K > 0 ? bitStrings(K) : [Int[]]

    for p in p_bits
        pbar = bitFlip(p)
        vp    = ComplexF64.(basisVector(p))
        vpbar = ComplexF64.(basisVector(pbar))
        plus_O  = (vp + im*vpbar) / sqrt(2)
        minus_O = (vp - im*vpbar) / sqrt(2)
        for q in q_bits
            if K > 0
                vq = ComplexF64.(basisVector(q))
                push!(states, permuteToPhysical(kron(plus_O,  vq), offdiagPos, diagPos))
                push!(states, permuteToPhysical(kron(minus_O, vq), offdiagPos, diagPos))
            else
                push!(states, permuteToPhysical(plus_O,  offdiagPos, Int[]))
                push!(states, permuteToPhysical(minus_O, offdiagPos, Int[]))
            end
        end
    end
    return states
end

# ══════════════════════════════════════════════════════════════════════════════
# Projektoren
# ══════════════════════════════════════════════════════════════════════════════

function ProjectorsFromEigenstates(eigs)
    isempty(eigs) && return Matrix{ComplexF64}[]
    return [ψ * ψ' for ψ in eigs]
end

# ══════════════════════════════════════════════════════════════════════════════
# Messung simulieren
# ══════════════════════════════════════════════════════════════════════════════

function simulateMeasurement(rho, projectors, n_shots)
    isempty(projectors) && return Float64[]
    probs = [real(tr(rho * P)) for P in projectors]
    probs = max.(probs, 0.0)
    probs ./= sum(probs)
    outcomes = sample(1:length(projectors), Weights(probs), n_shots)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n_shots
end

# ══════════════════════════════════════════════════════════════════════════════
# GD-Cholesky: Kern-Funktionen
# (analog zu density_matrix, log_loss_function, gd_chol_rank in processing.py)
# ══════════════════════════════════════════════════════════════════════════════

"""
    density_matrix(T)

Cholesky-Parametrisierung: ρ = T†T / Tr(T†T)

Garantiert ρ ≥ 0 und Tr(ρ) = 1 für beliebiges T ∈ ℂ^{d×d}.
Analog zu `density_matrix(A)` in processing.py.
"""
function density_matrix(T::Matrix{ComplexF64})::Matrix{ComplexF64}
    AtA = T' * T
    return AtA / tr(AtA)
end

"""
    log_loss(T, data, projectors)

Negative Log-Likelihood:
  L(T) = -Σᵢ fᵢ · log(Tr(ρ(T) · Πᵢ) + ε)

  fᵢ       : gemessene Häufigkeit für Projektor i
  ρ(T)     : Cholesky-parametrisierte Dichtematrix
  Πᵢ = |vᵢ⟩⟨vᵢ| : Projektor

Analog zu `log_loss_function(A, data, circuits, shots)` in processing.py.
"""
function log_loss(T::Matrix{ComplexF64},
                  data::Vector{Float64},
                  projectors::Vector{Matrix{ComplexF64}})::Float64
    ρ   = density_matrix(T)
    eps = 1e-8
    return -sum(data[i] * log(real(tr(ρ * projectors[i])) + eps)
                for i in eachindex(projectors))
end

"""
    gd_cholesky(projectors, data, d; lr, decay, iterations, patience)

Gradient Descent mit Cholesky-Parametrisierung.

Analog zu `gd_chol_rank(...)` in processing.py:
  - Parametrisiert ρ = T†T / Tr(T†T)  (physikalität automatisch)
  - Minimiert negative Log-Likelihood via Adam + exponentiellem LR-Decay
  - Wirtinger-Gradient: conj(∂L/∂T*)
  - Early Stopping wenn Loss-Änderung < convergence_tol

Parameter:
  projectors  : alle Projektoren Πᵢ
  data        : gemessene Häufigkeiten fᵢ
  d           : Hilbertraum-Dimension
  lr          : initiale Lernrate (default 0.1)
  decay       : LR-Decay pro Iteration (default 0.999)
  iterations  : maximale Iterationen
  patience    : Early-Stopping-Geduld

Rückgabe: rekonstruierte Dichtematrix ρ
"""
function gd_cholesky(projectors::Vector{Matrix{ComplexF64}},
                     data::Vector{Float64},
                     d::Int;
                     lr::Float64       = 0.1,
                     decay::Float64    = 0.999,
                     iterations::Int   = 300,
                     patience::Int     = 50)::Matrix{ComplexF64}

    # Zufällige Initialisierung von T (analog zu random_matrix_jax in processing.py)
    T = randn(ComplexF64, d, d)

    # Adam-Optimizer (analog zu optax.chain(...scale_by_adam...) in processing.py)
    opt = Optimisers.setup(Optimisers.Adam(lr), T)

    convergence_tol   = 1e-10
    no_improve_counter = 0
    loss_prev          = Inf

    for iter in 1:iterations

        # Lernraten-Decay (analog zu optax.exponential_decay in processing.py)
        current_lr = lr * decay^iter
        Optimisers.adjust!(opt, current_lr)

        # Wirtinger-Gradient: conj(∂L/∂T*)
        # (analog zu grads = jnp.conj(grad_f) in processing.py)
        loss, grads = Zygote.withgradient(T -> log_loss(T, data, projectors), T)
        grad_T = conj(grads[1])

        # Adam Update (analog zu gradient_transform.update in processing.py)
        opt, T = Optimisers.update!(opt, T, grad_T)

        # Early Stopping (analog zu no_improve_counter in processing.py)
        if abs(loss - loss_prev) < convergence_tol
            no_improve_counter += 1
        else
            no_improve_counter = 0
        end
        loss_prev = loss

        no_improve_counter >= patience && break
    end

    # Finale Dichtematrix aus T rekonstruieren
    # (analog zu params1 = A†A / Tr(A†A) am Ende von gd_chol_rank)
    return density_matrix(T)
end

# ══════════════════════════════════════════════════════════════════════════════
# Hauptfunktion
# ══════════════════════════════════════════════════════════════════════════════

"""
    reconstructDensityMatrixWithSeeqstGD(rho_true, N; n_shots, lr, decay, iterations)

SEEQST Rekonstruktion mit GD-Cholesky statt Convex MLE.

Ablauf (analog zu process_data in processing.py):
  1. Alle 2^N Gruppen generieren
  2. E- und O-Eigenbasen + Projektoren berechnen
  3. Messung simulieren
  4. GD-Cholesky: minimiere -log L über T, ρ(T) = T†T/Tr(T†T)
  5. Rekonstruierte Dichtematrix zurückgeben
"""
function reconstructDensityMatrixWithSeeqstGD(
    rho_true::Matrix{ComplexF64},
    N::Int;
    n_shots::Int    = 1000,
    lr::Float64     = 0.1,
    decay::Float64  = 0.999,
    iterations::Int = -1)   # -1 → automatisch: 100*N*3 wie beim Autor

    d = size(rho_true, 1)

    # Iterationen analog zu `iterations = 100*N*3` in process_data
    iters = iterations > 0 ? iterations : 100 * N * 3

    # 1. Alle Gruppen
    S = GenerateSGroups(N)

    all_projectors = Matrix{ComplexF64}[]
    all_data       = Float64[]

    # 2. Für jede Gruppe: E und O Projektoren + Messung
    for si in S
        SE = generateEigenstatesE(si)
        SO = generateEigenstatesO(si)

        pSE = ProjectorsFromEigenstates(SE)
        pSO = ProjectorsFromEigenstates(SO)

        # Messung (analog zu data_predict_from_rho_sampled in processing.py)
        cSE = simulateMeasurement(rho_true, pSE, n_shots)
        cSO = simulateMeasurement(rho_true, pSO, n_shots)

        append!(all_projectors, pSE)
        append!(all_projectors, pSO)
        append!(all_data, cSE)
        append!(all_data, cSO)
    end

    # 3. GD-Cholesky Rekonstruktion (ersetzt Convex MLE)
    rho_recon = gd_cholesky(all_projectors, all_data, d;
                             lr=lr, decay=decay, iterations=iters)

    return rho_recon
end

end # module
