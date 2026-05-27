module SeeqstHybridQutrit

using LinearAlgebra
using StatsBase
using Zygote
using Printf
using QuantumInformation


export GenerateRandomDensityMatrixNoZerosQutrits,
       GetSelectiveBlocksQutrit,
       GenerateSelectiveElementsQutrit,
       GenerateObservableSetsQutrit,
       BuildNonEntanglingCircuitsQutrit,
       BuildHybridCircuitsQutrit,
       CINC11Gate,
       CINC22Gate,
       CINC33Gate,
       GellMannRotation,
       ParseCircuitToMatrixQutrit,
       DataPredictFromRhoSampledQutrit,
       ProcessDataQutrit,
       RecreatingDensityMatrixWithNonentanglingQutrit,
       RecreatingDensityMatrixWithSeeqstQutrit

# --------------------------------------------------------------------
# Zufällige Dichtematrix
# --------------------------------------------------------------------
function GenerateRandomDensityMatrixNoZerosQutrits(n::Int)
    d = 3^n
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end

# --------------------------------------------------------------------
# Selective Blocks
# --------------------------------------------------------------------
function GetSelectiveBlocksQutrit(N::Int, wanted_indexes::Vector{Tuple{Int,Int}})
    max_index = 3^N
    selective_blocks = Vector{Int}()
    block_map = Dict{Int, Vector{Tuple{Int,Int}}}()

    for (i, j) in wanted_indexes
        @assert (0 <= i < max_index && 0 <= j < max_index) "Index ($i,$j) außerhalb des gültigen Bereichs"

        i_trits = digits(i, base=3, pad=N) |> reverse
        j_trits = digits(j, base=3, pad=N) |> reverse

        block_trits = Int[]
        for k in 1:N
            a, b = i_trits[k], j_trits[k]
            if a == b;                     push!(block_trits, 0)
            elseif (a,b) in [(0,1),(1,0)]; push!(block_trits, 1)
            elseif (a,b) in [(0,2),(2,0)]; push!(block_trits, 2)
            else;                          push!(block_trits, 3)
            end
        end

        block = sum(block_trits[k] * 4^(N-k) for k in 1:N)
        block ∉ selective_blocks && push!(selective_blocks, block)

        if !haskey(block_map, block)
            block_map[block] = Tuple{Int,Int}[]
        end
        push!(block_map[block], (i, j))
    end

    return selective_blocks, block_map
end

# --------------------------------------------------------------------
# Selective Elements
# --------------------------------------------------------------------
function GenerateSelectiveElementsQutrit(selective_blocks::Vector{Int}, N::Int)
    selective_elements = Vector{Vector{Tuple{Int,Int}}}()

    for block in selective_blocks
        block_types = digits(block, base=4, pad=N) |> reverse

        function allowed_i_trits(t::Int)
            t == 0 && return [0, 1, 2]
            t == 1 && return [0, 1]
            t == 2 && return [0, 2]
            t == 3 && return [1, 2]
        end

        function j_trit(a::Int, t::Int)
            t == 0 && return a
            t == 1 && return a == 0 ? 1 : 0
            t == 2 && return a == 0 ? 2 : 0
            t == 3 && return a == 1 ? 2 : 1
        end

        allowed      = [allowed_i_trits(t) for t in block_types]
        combinations = Iterators.product(allowed...)

        group = Tuple{Int,Int}[]
        for combo in combinations
            i_trits = collect(combo)
            j_trits = [j_trit(i_trits[k], block_types[k]) for k in 1:N]
            i = sum(i_trits[k] * 3^(N-k) for k in 1:N)
            j = sum(j_trits[k] * 3^(N-k) for k in 1:N)
            push!(group, (i, j))
        end

        push!(selective_elements, group)
    end

    return selective_elements
end

# --------------------------------------------------------------------
# Observable Sets
# --------------------------------------------------------------------
function GenerateObservableSetsQutrit(selective_blocks::Vector{Int}, N::Int)
    observable_sets = Dict{Int, Tuple{Vector{String}, Vector{String}}}()

    for block in selective_blocks
        block_types   = digits(block, base=4, pad=N) |> reverse
        pauli_options = Vector{Vector{String}}()

        for t in block_types
            if t == 0;     push!(pauli_options, ["I",  "Z"])
            elseif t == 1; push!(pauli_options, ["L1", "L2"])
            elseif t == 2; push!(pauli_options, ["L4", "L5"])
            elseif t == 3; push!(pauli_options, ["L6", "L7"])
            end
        end

        all_obs = [join(combo, "⊗") for combo in
                   Iterators.product(pauli_options...)] |> vec

        count_y(obs) = count(p -> p in ["L2","L5","L7"], split(obs,"⊗"))

        observable_sets[block] = (
            [o for o in all_obs if count_y(o) % 2 == 0],
            [o for o in all_obs if count_y(o) % 2 == 1]
        )
    end

    return observable_sets
end

# --------------------------------------------------------------------
# Non-Entangling Circuits
# --------------------------------------------------------------------
function BuildNonEntanglingCircuitsQutrit(selective_blocks::Vector{Int}, N::Int)
    all_block_circuits = Vector{Vector{String}}()

    gate_options = Dict(1 => ["RL2","RL1"], 2 => ["RL5","RL4"], 3 => ["RL7","RL6"])

    for block in selective_blocks
        block_types    = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i,t) in enumerate(block_types) if t != 0]

        if isempty(active_qutrits)
            push!(all_block_circuits, [""])
            continue
        end

        options      = [gate_options[t] for (q,t) in active_qutrits]
        qubits       = [q for (q,t) in active_qutrits]
        gate_choices = Iterators.product(options...)

        circuits = String[]
        for choice in gate_choices
            push!(circuits, join(["($gate:$q)" for (gate,q) in zip(choice,qubits)]))
        end

        push!(all_block_circuits, circuits)
    end

    return all_block_circuits
end

# --------------------------------------------------------------------
# Hybrid Circuits
# --------------------------------------------------------------------

function BuildHybridCircuitsQutrit(selective_blocks::Vector{Int}, N::Int)
    all_sequences = Vector{Vector{String}}()

    y_like    = ["RL2", "RL5", "RL7"]
    e_gate    = Dict(1 => "RL2", 2 => "RL5", 3 => "RL7")
    o_gate    = Dict(1 => "RL1", 2 => "RL4", 3 => "RL6")
    cinc_type = Dict(1 => "CINC11", 2 => "CINC22", 3 => "CINC33")

    for block in selective_blocks
        block_types    = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i,t) in enumerate(block_types) if t != 0]

        if isempty(active_qutrits)
            push!(all_sequences, [""])
            continue
        end

        # Gruppiere nach Übergangstyp
        type_groups = Dict{Int, Vector{Int}}()
        for (q, t) in active_qutrits
            haskey(type_groups, t) || (type_groups[t] = Int[])
            push!(type_groups[t], q)
        end

        entangling_gates = String[]
        local_qutrits    = Int[]

        for (t, qubits) in sort(collect(type_groups))
            if length(qubits) == 1
                # nur 1 Qutrit dieses Typs → kein CINC
                push!(local_qutrits, qubits[1])
            else
                # mehrere Qutrits gleichen Typs → CINC11/22/33
                head = [qubits[1]]
                tail = qubits[2:end]
                while !isempty(tail)
                    new_tail = Int[]
                    for h in head
                        isempty(tail) && break
                        tgt = popfirst!(tail)
                        push!(entangling_gates, "($(cinc_type[t]):$h,$tgt)")
                        push!(new_tail, tgt)
                    end
                    append!(head, new_tail)
                end
                # erstes Qutrit der Gruppe bekommt lokale Rotation
                push!(local_qutrits, qubits[1])
            end
        end

        local_qutrit_types = [t for (q,t) in active_qutrits if q in local_qutrits]
        local_qutrit_idxs  = [q for (q,t) in active_qutrits if q in local_qutrits]

        rot_options = [[e_gate[t], o_gate[t]] for t in local_qutrit_types]

        circuits = String[]
        for choice in Iterators.product(rot_options...)
            n_y         = count(r -> r in y_like, choice)
            local_gates = ["($gate:$q)" for (gate,q) in zip(choice, local_qutrit_idxs)]
            circuit_str = join(vcat(reverse(entangling_gates), local_gates))
            push!(circuits, (n_y % 2 == 0 ? "E:" : "O:") * circuit_str)
        end

        push!(all_sequences, circuits)
    end

    return all_sequences
end

# --------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------

function CINC11Gate(n::Int, control::Int, target::Int)
    # |0⟩↔|1⟩ Unterraum
    # Control=1: tauscht Target 0↔1
    # Control=0: unberührt
    dim  = 3^n
    gate = Matrix{ComplexF64}(I, dim, dim)

    for state in 0:(dim-1)
        ct = (state ÷ 3^(n-1-control)) % 3
        tt = (state ÷ 3^(n-1-target))  % 3

        # Nur aktiv wenn control=1 und target in {0,1}
        if ct == 1 && tt in (0, 1)
            nt = tt == 0 ? 1 : 0  # tausche 0↔1
            ns = state - tt * 3^(n-1-target) + nt * 3^(n-1-target)
            gate[:, state+1] .= 0
            gate[ns+1, state+1] = 1.0
        end
    end

    return gate
end

function CINC22Gate(n::Int, control::Int, target::Int)
    # |0⟩↔|2⟩ Unterraum
    # Control=2: tauscht Target 0↔2
    # Control=0: unberührt
    dim  = 3^n
    gate = Matrix{ComplexF64}(I, dim, dim)

    for state in 0:(dim-1)
        ct = (state ÷ 3^(n-1-control)) % 3
        tt = (state ÷ 3^(n-1-target))  % 3

        # Nur aktiv wenn control=2 und target in {0,2}
        if ct == 2 && tt in (0, 2)
            nt = tt == 0 ? 2 : 0  # tausche 0↔2
            ns = state - tt * 3^(n-1-target) + nt * 3^(n-1-target)
            gate[:, state+1] .= 0
            gate[ns+1, state+1] = 1.0
        end
    end

    return gate
end

function CINC33Gate(n::Int, control::Int, target::Int)
    # |1⟩↔|2⟩ Unterraum
    # Control=2: tauscht Target 1↔2
    # Control=1: unberührt
    dim  = 3^n
    gate = Matrix{ComplexF64}(I, dim, dim)

    for state in 0:(dim-1)
        ct = (state ÷ 3^(n-1-control)) % 3
        tt = (state ÷ 3^(n-1-target))  % 3

        # Nur aktiv wenn control=2 und target in {1,2}
        if ct == 2 && tt in (1, 2)
            nt = tt == 1 ? 2 : 1  # tausche 1↔2
            ns = state - tt * 3^(n-1-target) + nt * 3^(n-1-target)
            gate[:, state+1] .= 0
            gate[ns+1, state+1] = 1.0
        end
    end

    return gate
end


# --------------------------------------------------------------------
# Gell-Mann Rotation
# --------------------------------------------------------------------
function GellMannRotation(k::Int)
    λ = Dict(
        1 => [0 1 0; 1 0 0; 0 0 0] .+ 0im,
        2 => [0 -im 0; im 0 0; 0 0 0],
        3 => [1 0 0; 0 -1 0; 0 0 0] .+ 0im,
        4 => [0 0 1; 0 0 0; 1 0 0] .+ 0im,
        5 => [0 0 -im; 0 0 0; im 0 0],
        6 => [0 0 0; 0 0 1; 0 1 0] .+ 0im,
        7 => [0 0 0; 0 0 -im; 0 im 0],
        8 => [1 0 0; 0 1 0; 0 0 -2] ./ sqrt(3) .+ 0im,
    )
    return exp(-im * π/4 * λ[k])
end

# --------------------------------------------------------------------
# Parse Circuit to Matrix
# --------------------------------------------------------------------

function ParseCircuitToMatrixQutrit(text_circuits::Vector{String}, n::Int)
    unitary_list = Matrix{ComplexF64}[]

    for circuit in text_circuits
        U = Matrix{ComplexF64}(I, 3^n, 3^n)

        operations = [strip(op, ['(', ')']) for op in split(circuit, ")")
                      if !isempty(strip(op))]

        for op in operations
            parts     = split(op, ":")
            length(parts) < 2 && continue
            gate_name = String(parts[1])
            indices   = parse.(Int, split(parts[2], ","))

            if gate_name in ["RL1","RL2","RL4","RL5","RL6","RL7"]
                k    = parse(Int, gate_name[3:end])
                rot  = GellMannRotation(k)
                full = ones(ComplexF64, 1, 1)
                for q in 0:(n-1)
                    full = kron(full, q == indices[1] ? rot : Matrix{ComplexF64}(I,3,3))
                end
                U = full * U
            elseif gate_name == "CINC11"
                U = CINC11Gate(n, indices[1], indices[2]) * U
            elseif gate_name == "CINC22"
                U = CINC22Gate(n, indices[1], indices[2]) * U
            elseif gate_name == "CINC33"
                U = CINC33Gate(n, indices[1], indices[2]) * U
            else
                error("Unbekanntes Gate: $gate_name")
            end
        end

        push!(unitary_list, U)
    end

    return unitary_list
end

# --------------------------------------------------------------------
# Simulate Measurements
# --------------------------------------------------------------------
function DataPredictFromRhoSampledQutrit(
    rho::Matrix{ComplexF64},
    unitaries::Vector{Matrix{ComplexF64}},
    shots::Int
)
    all_counts = Vector{Vector{Int}}()

    for U in unitaries
        probs = real(diag(U * rho * U'))
        probs = max.(probs, 0)
        probs = probs / sum(probs)

        counts = zeros(Int, length(probs))
        for _ in 1:shots
            r = rand(); cum = 0.0
            for (k, p) in enumerate(probs)
                cum += p
                if r <= cum; counts[k] += 1; break; end
            end
        end

        push!(all_counts, counts)
    end

    return all_counts
end

# --------------------------------------------------------------------
# Process Data (SGD + Cholesky)
# --------------------------------------------------------------------
function ProcessDataQutrit(
    data::Vector{Vector{Int}},
    unitaries::Vector{Matrix{ComplexF64}},
    selective_blocks::Vector{Int},
    shots::Int,
    N::Int;
    lr::Float64     = 0.1,
    decay::Float64  = 0.9999,
    iterations::Int = 3000,
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

    T  = randn(3^N, 3^N) + im * randn(3^N, 3^N)
    L0, _ = Zygote.withgradient(loss, T)
    verbose && println("Loss initial: ", round(L0, digits=4))

    m  = zeros(ComplexF64, size(T))
    v  = zeros(Float64,    size(T))
    β₁, β₂, ε = 0.9, 0.999, 1e-8
    losses     = Float64[]
    no_improve = 0

    for i in 1:iterations
        L, grads = Zygote.withgradient(loss, T)
        g = grads[1]
        η = lr * decay^i
        m = β₁ .* m + (1-β₁) .* g
        v = β₂ .* v + (1-β₂) .* abs2.(g)
        m̂ = m ./ (1-β₁^i)
        v̂ = v ./ (1-β₂^i)
        T = T - η .* m̂ ./ (sqrt.(v̂) .+ ε)

        push!(losses, L)
        verbose && i % 500 == 0 && println("Iteration $i | Loss: $(round(L, digits=6))")

        if length(losses) > 1
            rel = abs(losses[end]-losses[end-1]) / abs(losses[end-1])
            rel < tol ? no_improve += 1 : (no_improve = 0)
        end
        no_improve >= patience && break
    end

    rho_rec = T' * T
    return rho_rec / tr(rho_rec)
end

# --------------------------------------------------------------------
# Volle Tomographie: NonEntangling
# --------------------------------------------------------------------
function RecreatingDensityMatrixWithNonentanglingQutrit(rho_true::Matrix{ComplexF64},
                                                         shots::Int;
                                                         lr::Float64     = 0.1,
                                                         decay::Float64  = 0.9999,
                                                         iterations::Int = 1000,
                                                         patience::Int   = 200)

    N   = Int(round(log(3, size(rho_true, 1))))
    dim = 3^N

    println("═══ Volle Tomographie (NonEntangling) N=$N ═══\n")

    blocks    = collect(0:(4^N - 1))
    non_circs = BuildNonEntanglingCircuitsQutrit(blocks, N)
    circuits  = unique(String[c for g in non_circs for c in g if c != ""])

    println("Anzahl Blöcke:   ", length(blocks))
    println("Anzahl Circuits: ", length(circuits))
    println("Shots/Circuit:   ", shots)

    Us_all  = ParseCircuitToMatrixQutrit(circuits, N)
    data    = DataPredictFromRhoSampledQutrit(rho_true, Us_all, shots)

    println("\n── SGD + Cholesky ──")
    rho_rec = ProcessDataQutrit(data, Us_all, blocks, shots, N;
        lr=lr, decay=decay, iterations=iterations,
        patience=patience, verbose=true)

    F = fidelity(rho_rec, rho_true)
    println("\n── Ergebnisse ──")
    println(@sprintf("Fidelität:           %.4f", F))
    println(@sprintf("Spur ρ_rec:          %.6f", real(tr(rho_rec))))
    println(@sprintf("Kleinster Eigenwert: %.6f", minimum(real(eigvals(rho_rec)))))
    println("═"^45)

    return rho_rec
end

# --------------------------------------------------------------------
# Volle Tomographie: Hybrid SEEQST
# --------------------------------------------------------------------
function RecreatingDensityMatrixWithSeeqstQutrit(rho_true::Matrix{ComplexF64},
                                                  shots::Int;
                                                  lr::Float64     = 0.1,
                                                  decay::Float64  = 0.9999,
                                                  iterations::Int = 1000,
                                                  patience::Int   = 200)

    N   = Int(round(log(3, size(rho_true, 1))))
    dim = 3^N

    println("═══ Volle Tomographie (Hybrid SEEQST) N=$N ═══\n")

    blocks       = collect(0:(4^N - 1))
    hybrid_circs = BuildHybridCircuitsQutrit(blocks, N)
    circuits     = unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for g in hybrid_circs for c in g if c != ""])

    println("Anzahl Blöcke:   ", length(blocks))
    println("Anzahl Circuits: ", length(circuits))
    println("Shots/Circuit:   ", shots)

    Us_all  = ParseCircuitToMatrixQutrit(circuits, N)
    data    = DataPredictFromRhoSampledQutrit(rho_true, Us_all, shots)

    println("\n── SGD + Cholesky ──")
    rho_rec = ProcessDataQutrit(data, Us_all, blocks, shots, N;
        lr=lr, decay=decay, iterations=iterations,
        patience=patience, verbose=true)

    F = fidelity(rho_rec, rho_true)
    println("\n── Ergebnisse ──")
    println(@sprintf("Fidelität:           %.4f", F))
    println(@sprintf("Spur ρ_rec:          %.6f", real(tr(rho_rec))))
    println(@sprintf("Kleinster Eigenwert: %.6f", minimum(real(eigvals(rho_rec)))))
    println("═"^45)

    return rho_rec
end

end # module SeeqstHybridQutrit
#=
include("StructureDensityMatrix.jl")


using .SeeqstHybridQutrit
using LinearAlgebra
using QuantumInformation
using Printf
function GenerateDensityMatrixOnlySameTransitions(N::Int)
    dim = 3^N

    function transition_type(a::Int, b::Int)
        s = Set([a, b])
        if s == Set([0, 1]); return 1
        elseif s == Set([0, 2]); return 2
        elseif s == Set([1, 2]); return 3
        else; return 0
        end
    end

    function allowed(i::Int, j::Int)
        trits_i = digits(i, base=3, pad=N) |> reverse
        trits_j = digits(j, base=3, pad=N) |> reverse
        types   = [transition_type(trits_i[k], trits_j[k]) for k in 1:N]
        active  = filter(t -> t != 0, types)
        return length(unique(active)) <= 1
    end

    # ── Erlaubte Indizes anzeigen ──────────────────────────────
    n_allowed = count(allowed(i,j) for i in 0:(dim-1), j in 0:(dim-1))
    println("N=$N: $n_allowed / $(dim^2) Elemente erlaubt")

    # ── Zufällige hermitesche Matrix nur in erlaubten Elementen
    # Methode: starte mit Null, fülle erlaubte Elemente zufällig
    # dann projiziere auf positiv semidefinit

    # Schritt 1: zufällige hermitesche Matrix in erlaubtem Unterraum
    H = zeros(ComplexF64, dim, dim)
    for i in 0:(dim-1), j in i:(dim-1)
        if allowed(i, j)
            if i == j
                H[i+1, i+1] = abs(randn()) + 0.1
            else
                v = randn(ComplexF64)
                H[i+1, j+1] = v
                H[j+1, i+1] = conj(v)
            end
        end
    end

    # Schritt 2: positiv semidefinit machen
    # via Eigenwertzerlegung
    eig    = eigen(H)
    vals   = max.(real(eig.values), 0.0) .+ 0.01
    vecs   = eig.vectors
    rho    = vecs * Diagonal(ComplexF64.(vals)) * vecs'

    # Schritt 3: erlaubte Struktur wiederherstellen
    # (Eigenwert-Operation kann verbotene Elemente einführen!)
    for i in 0:(dim-1), j in 0:(dim-1)
        if !allowed(i, j)
            rho[i+1, j+1] = 0.0
        end
    end

    # Schritt 4: Hermitizität sicherstellen
    rho = (rho + rho') / 2

    # Schritt 5: normieren
    rho = rho / tr(rho)

    # ── Prüfungen ──────────────────────────────────────────────
    n_nonzero    = count(abs(rho[i+1,j+1]) > 1e-10
                         for i in 0:(dim-1), j in 0:(dim-1))
    n_violations = count(!allowed(i,j) && abs(rho[i+1,j+1]) > 1e-10
                         for i in 0:(dim-1), j in 0:(dim-1))

    println("Nicht-Null Elemente:  $n_nonzero / $(dim^2)")
    println("Spur:                 $(round(real(tr(rho)), digits=6))")
    println("Hermitesch:           $(rho ≈ rho')")
    println("Min Eigenwert:        $(round(minimum(real(eigvals(rho))), digits=8))")
    println("Verletzungen:         $n_violations")

    return rho
end

println("═══ Test: Nur gleiche Übergänge ═══\n")

for N in [2, 3]
    println("━"^50)
    println("N = $N")
    println("━"^50)

    rho = GenerateDensityMatrixOnlySameTransitions(N)

    shots = 10000
    rho_non = RecreatingDensityMatrixWithNonentanglingQutrit(rho, shots)
    rho_hyb = RecreatingDensityMatrixWithSeeqstQutrit(rho, shots)

    F_non = fidelity(rho_non, rho)
    F_hyb = fidelity(rho_hyb, rho)

    println("NonEntangling: F = $(round(F_non, digits=4))")
    println("Hybrid:        F = $(round(F_hyb, digits=4))")
    println()
end
=#