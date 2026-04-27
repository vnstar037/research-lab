using LinearAlgebra
using StatsBase
using Convex
using SCS
using Printf
using QuantumInformation

# ══════════════════════════════════════════════════════════════════════════════
# Eigenbasen (aus eigenbasen_final.jl)
# ══════════════════════════════════════════════════════════════════════════════

const GROUP_MAP = Dict(
    "12" => Dict(0 => 0, 1 => 1),
    "67" => Dict(0 => 1, 1 => 2),
    "45" => Dict(0 => 0, 1 => 2),
)

function eigenbasen12(N::Int, experiment::Symbol)::Vector{Vector{ComplexF64}}
    # Randfall: leeres System
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
    diag_idx = [i for i in 1:N if groups[i] == "38"]
    M  = length(off_idx)
    NM = length(diag_idx)

    # Randfall: keine off-diagonalen Qutrits → Rechenbasis
    if M == 0
        states = Vector{Vector{ComplexF64}}()
        for q in 0:(dim-1)
            v = zeros(ComplexF64, dim)
            v[q+1] = 1.0
            push!(states, v)
        end
        return states
    end

    off_groups = [groups[i] for i in off_idx]
    off_states = eigenbasen(off_groups, experiment)

    # Randfall: keine diagonalen Qutrits
    if NM == 0
        return off_states
    end

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

function simulateMeasurement(rho, projectors, n_shots)
    isempty(projectors) && return Float64[]
    probs = [real(tr(rho * P)) for P in projectors]
    # Numerische Korrektur: probs müssen positiv und normiert sein
    probs = max.(probs, 0.0)
    probs ./= sum(probs)
    outcomes = sample(1:length(projectors), Weights(probs), n_shots)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n_shots
end

# ══════════════════════════════════════════════════════════════════════════════
# Projektoren aus Eigenbasen
# ══════════════════════════════════════════════════════════════════════════════

"""
Berechnet alle Projektoren für eine Gruppe (E + O Experiment).
Rückgabe: (projectors, counts_placeholder)
  projectors: Vector von d×d Matrizen
"""
function GenerateProjectors(group::Vector{String})
    projs = Matrix{ComplexF64}[]

    for exp in [:E, :O]
        states = eigenbasen_full(group, exp)
        for v in states
            push!(projs, v * v')
        end
    end

    return projs
end

# ══════════════════════════════════════════════════════════════════════════════
# MLE Rekonstruktion
# ══════════════════════════════════════════════════════════════════════════════

function MLE_reconstruct(d::Int, all_projectors, all_counts)
    ρ = ComplexVariable(d, d)
    constraints = [
        ρ == ρ',
        ρ ⪰ 0,
        tr(ρ) == 1
    ]

    eps = 1e-9
    loglik = sum(
        all_counts[i] * log(real(tr(ρ * all_projectors[i])) + eps)
        for i in eachindex(all_projectors)
    )

    problem = maximize(loglik, constraints)
    solve!(problem, SCS.Optimizer; silent_solver=true)

    return evaluate(ρ)
end

# ══════════════════════════════════════════════════════════════════════════════
# Volle QST Pipeline
# ══════════════════════════════════════════════════════════════════════════════

"""
    full_QST(N, n_shots)

Vollständige Quantum State Tomography für N Qutrits.

1. Generiere zufällige Dichtematrix
2. Generiere alle 4^N Gruppen
3. Für jede Gruppe: berechne E+O Eigenbasen → Projektoren → messe
4. MLE Rekonstruktion
5. Berechne Fidelität
"""
function full_QST(N::Int; n_shots::Int)

    d = 3^N
    println("\n", "═"^60)
    println("  Volle QST  |  N=$N Qutrits  |  d=$d  |  shots=$n_shots")
    println("═"^60)

    # 1. Zufällige Dichtematrix
    rho_true = GenerateRandomDensityMatrixNoZerosQutrits(N)
    println("  Wahre Dichtematrix generiert.")

    # 2. Alle Gruppen
    all_groups = GenerateLambdaGroups(N)
    println("  $(length(all_groups)) Gruppen generiert (4^$N = $(4^N)).")

    # 3. Für jede Gruppe: Projektoren + Messung
    all_projectors = Matrix{ComplexF64}[]
    all_counts     = Float64[]

    for group in all_groups
        projs = GenerateProjectors(group)
        counts = simulateMeasurement(rho_true, projs, n_shots)
        for (P, c) in zip(projs, counts)
            push!(all_projectors, P)
            push!(all_counts, c)
        end
    end

    println("  $(length(all_projectors)) Projektoren gemessen.")

    # 4. MLE
    println("  Starte MLE Rekonstruktion...")
    rho_recon = MLE_reconstruct(d, all_projectors, all_counts)


    F = fidelity(rho_true, rho_recon)
    println(@sprintf("  Fidelität: %.6f", F))
    println("═"^60)

    return (rho_true=rho_true, rho_recon=rho_recon, fidelity=F)
end

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

result = full_QST(3; n_shots=9000000)