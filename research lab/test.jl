using LinearAlgebra
using StatsBase
using Convex
using Clarabel
using Printf
using QuantumInformation

# ══════════════════════════════════════════════════════════════════════════════
# Eigenbasen
# ══════════════════════════════════════════════════════════════════════════════

const GROUP_MAP = Dict(
    "12" => Dict(0 => 0, 1 => 1),
    "67" => Dict(0 => 1, 1 => 2),
    "45" => Dict(0 => 0, 1 => 2),
)

const NULL_STATE = Dict("12" => 2, "45" => 1, "67" => 0, "38" => -1)

function eigenbasen12(N::Int, experiment::Symbol)::Vector{Vector{ComplexF64}}
    N == 0 && return [ComplexF64[1.0+0im]]
    dim   = 3^N
    n_eig = 2^N
    norm  = 1.0 / sqrt(2.0^N)
    n_p   = 2^(N-1)
    states = Vector{Vector{ComplexF64}}(undef, n_eig)
    for combo in 0:(n_eig - 1)
        j_pk = Vector{ComplexF64}(undef, N)
        for k in 0:N-1
            bit = (combo >> (N-1-k)) & 1
            if k == 0 && experiment == :O
                j_pk[1] = bit == 0 ? 0.0+1im : 0.0-1im
            else
                j_pk[k+1] = bit == 0 ? 1.0+0im : -1.0+0im
            end
        end
        state = zeros(ComplexF64, dim)
        for p_bits_int in 0:(n_p - 1)
            p = Vector{Int}(undef, N)
            p[1] = 0
            for k in 2:N; p[k] = (p_bits_int >> (N-k)) & 1; end
            p_bar = 1 .- p
            j_p     = prod(p[k]     == 0 ? 1.0+0im : j_pk[k] for k in 1:N)
            j_p_bar = prod(p_bar[k] == 0 ? 1.0+0im : j_pk[k] for k in 1:N)
            idx_p     = sum(p[k]     * 3^(N-k) for k in 1:N)
            idx_p_bar = sum(p_bar[k] * 3^(N-k) for k in 1:N)
            state[idx_p     + 1] += norm * j_p
            state[idx_p_bar + 1] += norm * j_p_bar
        end
        states[combo + 1] = state
    end
    return states
end

function eigenbasen(groups::Vector{String}, experiment::Symbol)::Vector{Vector{ComplexF64}}
    N   = length(groups)
    dim = 3^N
    maps = [GROUP_MAP[g] for g in groups]
    base_states = eigenbasen12(N, experiment)
    shifted_states = Vector{Vector{ComplexF64}}(undef, length(base_states))
    for (s_idx, state) in enumerate(base_states)
        new_state = zeros(ComplexF64, dim)
        for (basis_idx, coeff) in enumerate(state)
            abs(coeff) < 1e-10 && continue
            old_idx = basis_idx - 1
            trits = Vector{Int}(undef, N); tmp = old_idx
            for k in N:-1:1; trits[k] = tmp % 3; tmp ÷= 3; end
            new_trits = [maps[k][trits[k]] for k in 1:N]
            new_idx   = sum(new_trits[k] * 3^(N-k) for k in 1:N)
            new_state[new_idx + 1] += coeff
        end
        shifted_states[s_idx] = new_state
    end
    return shifted_states
end

function eigenbasen_full(groups::Vector{String}, experiment::Symbol)::Vector{Vector{ComplexF64}}
    N   = length(groups)
    dim = 3^N
    off_idx  = [i for i in 1:N if groups[i] != "38"]
    M  = length(off_idx)
    NM = length(groups) - M
    if M == 0
        return [begin v = zeros(ComplexF64, dim); v[q+1] = 1.0; v end for q in 0:(dim-1)]
    end
    off_groups = [groups[i] for i in off_idx]
    off_states = eigenbasen(off_groups, experiment)
    NM == 0 && return off_states
    n_q = 3^NM
    q_configs = Vector{Vector{Int}}(undef, n_q)
    for q in 0:(n_q-1)
        trits = Vector{Int}(undef, NM); tmp = q
        for k in NM:-1:1; trits[k] = tmp % 3; tmp ÷= 3; end
        q_configs[q+1] = trits
    end
    all_states = Vector{Vector{ComplexF64}}()
    for off_state in off_states
        for q_trit in q_configs
            ns = zeros(ComplexF64, dim)
            for (obi, c) in enumerate(off_state)
                abs(c) < 1e-10 && continue
                off_trits = Vector{Int}(undef, M); tmp = obi - 1
                for k in M:-1:1; off_trits[k] = tmp % 3; tmp ÷= 3; end
                full_trits = Vector{Int}(undef, N)
                oc = 1; dc = 1
                for i in 1:N
                    if groups[i] == "38"; full_trits[i] = q_trit[dc]; dc += 1
                    else;                 full_trits[i] = off_trits[oc]; oc += 1
                    end
                end
                fi = sum(full_trits[k] * 3^(N-k) for k in 1:N)
                ns[fi + 1] += c
            end
            push!(all_states, ns)
        end
    end
    return all_states
end

# ══════════════════════════════════════════════════════════════════════════════
# Nullraum-Eigenbasen
# ══════════════════════════════════════════════════════════════════════════════

"""
    nullraum_basis(groups)

Berechnet die Nullraum-Rechenbasisvektoren für eine Gruppe.
Ein Basiszustand liegt im Nullraum wenn mindestens ein off-diagonales
Qutrit in seinem Nullzustand ist.
"""
function nullraum_basis(groups::Vector{String})::Vector{Vector{ComplexF64}}
    N   = length(groups)
    dim = 3^N
    null_vecs = Vector{Vector{ComplexF64}}()
    for basis_idx in 0:(dim-1)
        trits = Vector{Int}(undef, N); tmp = basis_idx
        for k in N:-1:1; trits[k] = tmp % 3; tmp ÷= 3; end
        is_null = any(NULL_STATE[groups[i]] == trits[i]
                      for i in 1:N if NULL_STATE[groups[i]] >= 0)
        if is_null
            v = zeros(ComplexF64, dim); v[basis_idx + 1] = 1.0
            push!(null_vecs, v)
        end
    end
    return null_vecs
end

# ══════════════════════════════════════════════════════════════════════════════
# Unitäre Matrizen
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_unitary(groups, experiment)

Baut die unitäre Matrix U für ein Experiment (E oder O).

Die Spalten von U sind:
  1. Die aktiven Eigenbasen (E oder O)
  2. Die Nullraum-Rechenbasisvektoren

Da E/O-Eigenbasen + Nullraum eine vollständige ONB bilden,
ist U unitär: U†U = I.

Messung: probs = diag(U† ρ U)
  → automatisch vollständige POVM ohne explizite Projektor-Normierung
"""
function build_unitary(groups::Vector{String}, experiment::Symbol)::Matrix{ComplexF64}
    active = eigenbasen_full(groups, experiment)
    null   = nullraum_basis(groups)
    # Spalten: aktive Eigenbasen zuerst, dann Nullraum
    return hcat(active..., null...)
end

# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════

function GenerateRandomDensityMatrixNoZerosQutrits(n::Int)
    d = 3^n
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end

function GenerateLambdaGroups(N::Int)
    groups = Vector{Vector{String}}()
    labels = ["12", "38", "45", "67"]
    for mask in 0:(4^N - 1)
        S = Vector{String}(undef, N)
        for qubit in 1:N
            digit = (mask ÷ 4^(N - qubit)) % 4
            S[qubit] = labels[digit + 1]
        end
        push!(groups, S)
    end
    return groups
end

# ══════════════════════════════════════════════════════════════════════════════
# Messung mit unitärer Matrix
# ══════════════════════════════════════════════════════════════════════════════

"""
    simulate_measurement_unitary(rho, U, n_shots)

Simuliert eine Messung in der durch U definierten Basis.

Wahrscheinlichkeiten: p = diag(U† ρ U)
  → p_i = ⟨vᵢ|ρ|vᵢ⟩  (Wahrscheinlichkeit für Eigenzustand i)

Da U unitär ist, gilt Σᵢ pᵢ = Tr(ρ) = 1 automatisch.

Rückgabe: Häufigkeitsvektor (counts / n_shots)
"""
function simulate_measurement_unitary(rho::Matrix{ComplexF64},
                                      U::Matrix{ComplexF64},
                                      n_shots::Int)::Vector{Float64}
    # p = diag(U† ρ U)
    probs = real(diag(U' * rho * U))

    # Numerische Korrektur für kleine negative Werte
    probs = max.(probs, 0.0)
    probs ./= sum(probs)

    counts = zeros(Int, length(probs))
    outcomes = sample(1:length(probs), Weights(probs), n_shots)
    for o in outcomes; counts[o] += 1; end

    return counts ./ n_shots
end

# ══════════════════════════════════════════════════════════════════════════════
# MLE Rekonstruktion
# ══════════════════════════════════════════════════════════════════════════════

"""
    MLE_reconstruct_unitary(d, unitaries, all_counts)

MLE mit unitären Matrizen statt Projektoren.

Likelihood:  L = Σ_{U,i} f_{U,i} · log(diag(U†ρU)_i)

  f_{U,i} : gemessene Häufigkeit für Ausgabe i in Basis U
  diag(U†ρU)_i : vorhergesagte Wahrscheinlichkeit

Parameter:
  unitaries  : Vector von unitären Matrizen (eine pro Experiment)
  all_counts : Vector von Häufigkeitsvektoren (eine pro Experiment)
"""
function MLE_reconstruct_unitary(d::Int,
                                  unitaries::Vector{Matrix{ComplexF64}},
                                  all_counts::Vector{Vector{Float64}})
    ρ = ComplexVariable(d, d)
    constraints = [
        ρ == ρ',
        ρ ⪰ 0,
        tr(ρ) == 1
    ]

    eps = 1e-9
    loglik = sum(
        sum(
            all_counts[k][i] * log(real(tr(ρ * (unitaries[k][:, i] * unitaries[k][:, i]'))) + eps)
            for i in 1:d
        )
        for k in eachindex(unitaries)
    )

    problem = maximize(loglik, constraints)
    solve!(problem, Clarabel.Optimizer; silent=true)

    return Matrix{ComplexF64}(evaluate(ρ))
end

# ══════════════════════════════════════════════════════════════════════════════
# Volle QST Pipeline
# ══════════════════════════════════════════════════════════════════════════════

"""
    full_QST(N; n_shots)

Vollständige Quantum State Tomography für N Qutrits.

Für jede der 4^N Gruppen:
  - Baue U_E und U_O (unitäre Matrizen aus Eigenbasen + Nullraum)
  - Messe: probs = diag(U† ρ U)
  - Übergebe Häufigkeiten und unitäre Matrizen an MLE
"""
function full_QST(N::Int; n_shots::Int = 10000)
    d = 3^N
    println("\n", "═"^60)
    println("  Volle QST  |  N=$N Qutrits  |  d=$d  |  shots=$n_shots")
    println("═"^60)

    # 1. Zufällige Dichtematrix
    rho_true = GenerateRandomDensityMatrixNoZerosQutrits(N)
    println("  Wahre Dichtematrix generiert.")

    # 2. Alle Gruppen
    all_groups = GenerateLambdaGroups(N)
    println("  $(length(all_groups)) Gruppen (4^$N = $(4^N)).")

    # 3. Für jede Gruppe: U_E und U_O bauen + messen
    unitaries  = Vector{Matrix{ComplexF64}}()
    all_counts = Vector{Vector{Float64}}()

    for group in all_groups
        for exp in [:E, :O]
            U = build_unitary(group, exp)
            freqs = simulate_measurement_unitary(rho_true, U, n_shots)
            push!(unitaries,  U)
            push!(all_counts, freqs)
        end
    end

    println("  $(length(unitaries)) Experimente durchgeführt ($(2 * length(all_groups)) = 2 × 4^$N).")

    # 4. MLE
    println("  Starte MLE Rekonstruktion...")
    rho_recon = MLE_reconstruct_unitary(d, unitaries, all_counts)

    # 5. Fidelität via QuantumInformation
    # fidelity erwartet Qobj-artige Objekte — konvertiere zu AbstractMatrix
    F = fidelity(rho_true, rho_recon)
    println(@sprintf("  Fidelität: %.6f", F))
    println("═"^60)

    return (rho_true=rho_true, rho_recon=rho_recon, fidelity=F)
end

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

#result = full_QST(2; n_shots=1000000)

function check_completeness(N::Int)
    d = 3^N
    all_groups = GenerateLambdaGroups(N)
    
    # Sammle alle Projektoren als Vektoren (vec(Π_i))
    rows = Vector{Vector{ComplexF64}}()
    for group in all_groups
        for exp in [:E, :O]
            U = build_unitary(group, exp)
            for i in 1:d
                v = U[:, i]
                P = v * v'
                push!(rows, vec(P))  # d²-dimensionaler Vektor
            end
        end
    end
    
    A = hcat(rows...)'  # n_measurements × d² Matrix
    r = rank(A)
    println("N=$N: Rang = $r / $(d^2)")
    return r
end

check_completeness(1)
check_completeness(2)

function find_missing_operators(N::Int)
    d = 3^N
    all_groups = GenerateLambdaGroups(N)
 
    # Gell-Mann-Matrizen (3×3)
    λ = Vector{Matrix{ComplexF64}}(undef, 9)
    λ[1] = [1 0 0; 0 1 0; 0 0 1]                              # λ0 = I
    λ[2] = [0 1 0; 1 0 0; 0 0 0]                              # λ1
    λ[3] = [0 -im 0; im 0 0; 0 0 0]                           # λ2
    λ[4] = [1 0 0; 0 -1 0; 0 0 0]                             # λ3
    λ[5] = [0 0 1; 0 0 0; 1 0 0]                              # λ4
    λ[6] = [0 0 -im; 0 0 0; im 0 0]                           # λ5
    λ[7] = [0 0 0; 0 0 1; 0 1 0]                              # λ6
    λ[8] = [0 0 0; 0 0 -im; 0 im 0]                           # λ7
    λ[9] = [1 0 0; 0 1 0; 0 0 -2] / sqrt(3)                  # λ8
 
    # Sensing matrix aufbauen
    rows = Vector{Vector{ComplexF64}}()
    for group in all_groups
        for exp in [:E, :O]
            U = build_unitary(group, exp)
            for i in 1:d
                v = U[:, i]
                push!(rows, vec(v * v'))
            end
        end
    end
    A = hcat(rows...)'
 
    # Prüfe welche Gell-Mann Tensorprodukte NICHT im Spaltenraum liegen
    println("\nFehlende Operatoren für N=$N:")
    n_missing = 0
    for a in 1:9, b in 1:9
        op = kron(λ[a], λ[b])
        v  = vec(op)
        r_before = rank(A)
        r_after  = rank(vcat(A, v'))
        if r_after > r_before
            println("  λ$(a-1) ⊗ λ$(b-1)")
            n_missing += 1
            # Füge zum Spaltenraum hinzu damit wir nicht redundant zählen
            A = vcat(A, v')
        end
    end
    println("  → $n_missing fehlende Operatoren (Rang = $(rank(hcat(rows...)'))/$(d^2))")
end
 

find_missing_operators(2)
