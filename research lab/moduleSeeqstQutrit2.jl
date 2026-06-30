module SeeqstMixedQutrit

# ══════════════════════════════════════════════════════════════
# SEEQST mit theoretischen Gates für gemischte Blöcke (t₁ ≠ t₂)
#
# Gleiche Blöcke (t₁ = t₂): CINC Gates (physikalisch realisierbar)
# Gemischte Blöcke (t₁ ≠ t₂): Hadamard-in-Subspace Gates
#                              (mathematisch exakt, physikalisch
#                               erfordert Ancilla oder off-resonante
#                               Kopplung → theoretisch!)
# ══════════════════════════════════════════════════════════════

using LinearAlgebra
using Zygote
using Printf
using QuantumInformation
using Random

export RecreatingDensityMatrixWithMixedSeeqstQutrit,
       BuildMixedCircuitsQutrit,
       GenerateRandomDensityMatrixNoZerosQutrits,
       GenerateRandomDensityMatrixQutrits

# ══════════════════════════════════════════════════════════════
# Dichtematrizen
# ══════════════════════════════════════════════════════════════
function GenerateRandomDensityMatrixNoZerosQutrits(N::Int)
    d = 3^N
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    return rho / tr(rho)
end

function GenerateRandomDensityMatrixQutrits(N::Int)
    d        = 3^N
    n_active = rand(1:d)
    active   = randperm(d)[1:n_active]
    rho      = zeros(ComplexF64, d, d)
    for _ in 1:n_active
        psi         = zeros(ComplexF64, d)
        psi[active] = randn(ComplexF64, n_active) + 1im*randn(ComplexF64, n_active)
        psi        /= norm(psi)
        rho        += psi * psi'
    end
    return rho / tr(rho)
end

# ══════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════
function transition_type_qutrit(x::Int, y::Int)
    x == y                       && return 0
    Set([x,y]) == Set([0,1])     && return 1
    Set([x,y]) == Set([0,2])     && return 2
    return 3
end

# Aktive Niveaus für Übergangstyp t
const LEVELS = Dict(1 => (0,1), 2 => (0,2), 3 => (1,2))

# ══════════════════════════════════════════════════════════════
# Gate für gemischten Block (N=2)
# Hadamard in den 2D Subspaces der aktiven Zustandspaare
# ══════════════════════════════════════════════════════════════
function build_mixed_block_gate_real(block::Int, N::Int)
    @assert N == 2 "Mixed block gates nur für N=2 implementiert"
    d  = 3^N
    U  = Matrix{ComplexF64}(I, d, d)

    t0 = (block ÷ 4) % 4
    t1 =  block      % 4

    a0, b0 = LEVELS[t0]
    a1, b1 = LEVELS[t1]

    # Index (1-basiert): state = q0*3 + q1 + 1
    idx(q0,q1) = q0*3 + q1 + 1

    # Paar 1: |a0,a1⟩ ↔ |b0,b1⟩ → misst Re(ρ)
    i1 = idx(a0,a1);  j1 = idx(b0,b1)
    U[i1,i1] =  1/√2;  U[i1,j1] =  1/√2
    U[j1,i1] =  1/√2;  U[j1,j1] = -1/√2

    # Paar 2: |a0,b1⟩ ↔ |b0,a1⟩ → misst Re(ρ)
    i2 = idx(a0,b1);  j2 = idx(b0,a1)
    U[i2,i2] =  1/√2;  U[i2,j2] =  1/√2
    U[j2,i2] =  1/√2;  U[j2,j2] = -1/√2

    return U
end

function build_mixed_block_gate_imag(block::Int, N::Int)
    @assert N == 2 "Mixed block gates nur für N=2 implementiert"
    d  = 3^N
    U  = Matrix{ComplexF64}(I, d, d)

    t0 = (block ÷ 4) % 4
    t1 =  block      % 4

    a0, b0 = LEVELS[t0]
    a1, b1 = LEVELS[t1]

    idx(q0,q1) = q0*3 + q1 + 1

    # Paar 1: Phase i → misst Im(ρ)
    i1 = idx(a0,a1);  j1 = idx(b0,b1)
    U[i1,i1] =  1/√2;  U[i1,j1] =  im/√2
    U[j1,i1] =  1/√2;  U[j1,j1] = -im/√2

    # Paar 2: Phase i → misst Im(ρ)
    i2 = idx(a0,b1);  j2 = idx(b0,a1)
    U[i2,i2] =  1/√2;  U[i2,j2] =  im/√2
    U[j2,i2] =  1/√2;  U[j2,j2] = -im/√2

    return U
end

# ══════════════════════════════════════════════════════════════
# CINC Gates (für gleiche Blöcke)
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# BuildMixedCircuitsQutrit
# Gibt für jeden Block die Unitary-Matrizen zurück
# ══════════════════════════════════════════════════════════════
function BuildMixedCircuitsQutrit(selective_blocks::Vector{Int}, N::Int)
    all_unitaries = Vector{Vector{Matrix{ComplexF64}}}()

    e_gate    = Dict(1 => 2, 2 => 5, 3 => 7)
    o_gate    = Dict(1 => 1, 2 => 4, 3 => 6)
    cinc_fn   = Dict(1 => CINC11Gate, 2 => CINC22Gate, 3 => CINC33Gate)
    y_like    = [2, 5, 7]

    for block in selective_blocks
        block_types    = digits(block, base=4, pad=N) |> reverse
        nonzero_types  = filter(!=(0), block_types)
        unique_types   = unique(nonzero_types)

        # ── Diagonal-Block: Identität ──────────────────────────
        if isempty(nonzero_types)
            push!(all_unitaries, [Matrix{ComplexF64}(I, 3^N, 3^N)])
            continue
        end

        # ── Gleicher Typ: CINC Gates (wie bisheriger Hybrid) ───
        if length(unique_types) == 1
            t           = unique_types[1]
            active_quts = [i-1 for (i,bt) in enumerate(block_types) if bt != 0]

            entangling = Matrix{ComplexF64}(I, 3^N, 3^N)
            if length(active_quts) > 1
                for idx in 2:length(active_quts)
                    entangling = cinc_fn[t](N, active_quts[1], active_quts[idx]) * entangling
                end
            end

            # E und O Circuit
            U_E = Matrix{ComplexF64}(I, 3^N, 3^N)
            U_O = Matrix{ComplexF64}(I, 3^N, 3^N)
            rot_E = GellMannRotation(e_gate[t])
            rot_O = GellMannRotation(o_gate[t])

            full_E = ones(ComplexF64,1,1)
            full_O = ones(ComplexF64,1,1)
            for q in 0:(N-1)
                I3 = Matrix{ComplexF64}(I,3,3)
                full_E = kron(full_E, q==active_quts[1] ? rot_E : I3)
                full_O = kron(full_O, q==active_quts[1] ? rot_O : I3)
            end
            U_E = full_E * entangling
            U_O = full_O * entangling

            push!(all_unitaries, [U_E, U_O])
            continue
        end

        # ── Gemischter Typ: theoretisches Hadamard Gate ────────
        if N == 2
            U_real = build_mixed_block_gate_real(block, N)
            U_imag = build_mixed_block_gate_imag(block, N)
            push!(all_unitaries, [U_real, U_imag])
        else
            # N>2: Fallback auf lokale Messungen
            # (gemischte Blöcke bei N>2 erfordern Ancilla)
            active_quts = [(i-1, bt) for (i,bt) in enumerate(block_types) if bt != 0]
            options = [[GellMannRotation(e_gate[bt]),
                        GellMannRotation(o_gate[bt])] for (_,bt) in active_quts]
            qutrit_idxs = [q for (q,_) in active_quts]

            circuits = Matrix{ComplexF64}[]
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
            push!(all_unitaries, circuits)
        end
    end

    return all_unitaries
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
# SGD + Cholesky Rekonstruktion
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
    verbose && println("  Loss initial: ", round(Zygote.withgradient(loss,T)[1], digits=4))

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
function RecreatingDensityMatrixWithMixedSeeqstQutrit(
    rho_true::Matrix{ComplexF64},
    shots::Int;
    lr::Float64     = 0.1,
    decay::Float64  = 0.9999,
    iterations::Int = 1000,
    patience::Int   = 200,
    verbose::Bool   = true
)
    N = Int(round(log(3, size(rho_true,1))))

    verbose && println("═══ Mixed SEEQST N=$N ═══")
    verbose && println()

    blocks        = collect(0:(4^N-1))
    all_unitaries = BuildMixedCircuitsQutrit(blocks, N)

    # Alle Unitaries flach zusammenfassen
    Us_flat   = vcat(all_unitaries...)
    Us_unique = unique(Us_flat)

    n_circ_mixed = length(Us_unique)

    if verbose
        # Zähle Circuits pro Typ
        n_same  = 0
        n_mixed = 0
        n_diag  = 0
        for (block, us) in zip(blocks, all_unitaries)
            types = filter(!=(0), digits(block, base=4, pad=N) |> reverse)
            if isempty(types);          n_diag  += length(us)
            elseif length(unique(types))==1; n_same  += length(us)
            else;                       n_mixed += length(us)
            end
        end
        println(@sprintf("  Circuits total:       %d", n_circ_mixed))
        println(@sprintf("    Diagonal:           %d", n_diag))
        println(@sprintf("    Gleiche Typen (CINC): %d", n_same))
        println(@sprintf("    Gemischte Typen:    %d", n_mixed))
        println()
    end

    data = DataPredictFromRhoSampledQutrit(rho_true, Us_unique, shots)

    verbose && println("── SGD + Cholesky ──")
    rho_rec = ProcessDataQutrit(data, Us_unique, shots, N;
        lr=lr, decay=decay, iterations=iterations,
        patience=patience, verbose=verbose)

    if verbose
        F = fidelity(rho_rec, rho_true)
        println()
        println("── Ergebnisse ──")
        println(@sprintf("  Fidelität:           %.4f", F))
        println(@sprintf("  Circuits:            %d", n_circ_mixed))
        println(@sprintf("  Spur ρ_rec:          %.6f", real(tr(rho_rec))))
        println(@sprintf("  Kleinster Eigenwert: %.6f", minimum(real(eigvals(rho_rec)))))
        println("═"^45)
    end

    return Matrix{ComplexF64}(rho_rec)
end

end # module SeeqstMixedQutrit
