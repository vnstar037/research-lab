

module SeeqstQutrit
using LinearAlgebra
using StatsBase
using Zygote
using Printf
using QuantumInformation


export GenerateRandomDensityMatrixNoZerosQutrits,
       GetSelectiveBlocksQutrit,
       GenerateSelectiveElementsQutrit,
       GenerateObservableSetsQutrit,
       BuildEntanglerBlocksQutrit,
       BuildNonEntanglingCircuitsQutrit,
       GenerateExperimentQutrit,
       CINCGate,
       GellMannRotation,
       ParseCircuitToMatrixQutrit,
       PrepareStateQutrit,
       randomInitialTQutrit,
       densityMatrixFromTQutrit,
       DataPredictFromRhoSampledQutrit,
       ProcessDataQutrit,
       BuildHybridCircuitsQutrit,
       FullTomographyHybrid,
       CINCDaggerGate, CINC33Gate,
       BuildHybridCircuitsNoRedundancyQutrit

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
        @assert (0 <= i < max_index && 0 <= j < max_index) "Index ($i,$j) außerhalb des gültigen Bereichs [0, $(max_index-1)]"

        i_trits = digits(i, base=3, pad=N) |> reverse
        j_trits = digits(j, base=3, pad=N) |> reverse

        block_trits = Int[]
        for k in 1:N
            a, b = i_trits[k], j_trits[k]
            if a == b
                push!(block_trits, 0)
            elseif (a,b) in [(0,1),(1,0)]
                push!(block_trits, 1)
            elseif (a,b) in [(0,2),(2,0)]
                push!(block_trits, 2)
            else
                push!(block_trits, 3)
            end
        end

        block = sum(block_trits[k] * 4^(N-k) for k in 1:N)

        if block ∉ selective_blocks
            push!(selective_blocks, block)
        end

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
            if t == 0; return [0, 1, 2]
            elseif t == 1; return [0, 1]
            elseif t == 2; return [0, 2]
            elseif t == 3; return [1, 2]
            end
        end

        function j_trit(a::Int, t::Int)
            if t == 0; return a
            elseif t == 1; return a == 0 ? 1 : 0
            elseif t == 2; return a == 0 ? 2 : 0
            elseif t == 3; return a == 1 ? 2 : 1
            end
        end

        allowed     = [allowed_i_trits(t) for t in block_types]
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

        all_observables = [join(combo, "⊗") for combo in Iterators.product(pauli_options...)]
        all_observables = vec(all_observables)

        function count_y_type(obs)
            parts = split(obs, "⊗")
            return count(p -> p in ["L2", "L5", "L7"], parts)
        end

        even_set = [obs for obs in all_observables if count_y_type(obs) % 2 == 0]
        odd_set  = [obs for obs in all_observables if count_y_type(obs) % 2 == 1]

        observable_sets[block] = (even_set, odd_set)
    end

    return observable_sets
end

# --------------------------------------------------------------------
# Entangler Circuits (mit CINC)
# --------------------------------------------------------------------
function BuildEntanglerBlocksQutrit(selective_blocks::Vector{Int}, N::Int)
    all_sequences = Vector{Vector{String}}()

    e_gate = Dict(1 => "RL2", 2 => "RL5", 3 => "RL7")
    o_gate = Dict(1 => "RL1", 2 => "RL4", 3 => "RL6")

    for block in selective_blocks
        block_types    = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i, t) in enumerate(block_types) if t != 0]

        if isempty(active_qutrits)
            push!(all_sequences, [""])
            continue
        end

        sequence_e = String[]
        sequence_o = String[]

        first_q, first_t = active_qutrits[1]
        push!(sequence_e, "($(e_gate[first_t]):$first_q)")
        push!(sequence_o, "($(o_gate[first_t]):$first_q)")

        head = [first_q]
        tail = active_qutrits[2:end]

        while !isempty(tail)
            new_tail = Int[]
            for h in head
                isempty(tail) && break
                tgt_q, tgt_t = popfirst!(tail)
                push!(sequence_e, "(CINC:$h,$tgt_q)")
                push!(sequence_o, "(CINC:$h,$tgt_q)")
                push!(new_tail, tgt_q)
            end
            append!(head, new_tail)
        end

        e_str = join(reverse(sequence_e))
        o_str = join(reverse(sequence_o))

        push!(all_sequences, [e_str, o_str])
    end

    return all_sequences
end

# --------------------------------------------------------------------
# Non-Entangling Circuits (nur lokale Rotationen)
# --------------------------------------------------------------------
function BuildNonEntanglingCircuitsQutrit(selective_blocks::Vector{Int}, N::Int)
    all_block_circuits = Vector{Vector{String}}()

    gate_options = Dict(
        1 => ["RL2", "RL1"],
        2 => ["RL5", "RL4"],
        3 => ["RL7", "RL6"],
    )

    for block in selective_blocks
        block_types    = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i, t) in enumerate(block_types) if t != 0]

        if isempty(active_qutrits)
            push!(all_block_circuits, [""])
            continue
        end

        options      = [gate_options[t] for (q, t) in active_qutrits]
        qubits       = [q for (q, t) in active_qutrits]
        gate_choices = Iterators.product(options...)

        circuits = String[]
        for choice in gate_choices
            circuit_str = join(["($gate:$q)" for (gate, q) in zip(choice, qubits)])
            push!(circuits, circuit_str)
        end

        push!(all_block_circuits, circuits)
    end

    return all_block_circuits
end

# --------------------------------------------------------------------
# Generate Experiment
# --------------------------------------------------------------------
function GenerateExperimentQutrit(wanted_indexes::Vector{Tuple{Int,Int}}, N::Int;
                                   verbose::Bool=true)
    selective_blocks, block_map    = GetSelectiveBlocksQutrit(N, wanted_indexes)
    selective_elements             = GenerateSelectiveElementsQutrit(selective_blocks, N)
    observable_sets                = GenerateObservableSetsQutrit(selective_blocks, N)
    entangler_circuits             = BuildEntanglerBlocksQutrit(selective_blocks, N)
    non_entangler_circuits         = BuildNonEntanglingCircuitsQutrit(selective_blocks, N)

    if verbose
        type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]
        println("Wanted Indexes $wanted_indexes → Selective Blocks: $selective_blocks\n")

        for (i, block) in enumerate(selective_blocks)
            block_types = digits(block, base=4, pad=N) |> reverse
            desc = join([type_names[t+1] for t in block_types], " × ")
            println("********************")
            println("Block $block = $(join(block_types))₄  ($desc)")
            println("  Wanted:    ", block_map[block])
            println("  Elements:  ", selective_elements[i])
            E, O = observable_sets[block]
            println("  E-Gruppe:  ", E)
            println("  O-Gruppe:  ", O)
            if entangler_circuits[i] == [""]
                println("  Circuit:   (leer)")
            else
                println("  Circuit E: ", entangler_circuits[i][1])
                println("  Circuit O: ", entangler_circuits[i][2])
            end
            println("  Non-Ent:   ", non_entangler_circuits[i])
            println()
        end
    end

    return selective_blocks, selective_elements, observable_sets,
           entangler_circuits, non_entangler_circuits, block_map
end

# --------------------------------------------------------------------
# CINC Gate
# --------------------------------------------------------------------
function CINCGate(n::Int, control::Int, target::Int)
    dim  = 3^n
    gate = zeros(ComplexF64, dim, dim)

    for state in 0:(dim-1)
        control_trit    = (state ÷ 3^(n - 1 - control)) % 3
        target_trit     = (state ÷ 3^(n - 1 - target))  % 3
        new_target_trit = (target_trit + control_trit) % 3
        new_state       = state - target_trit     * 3^(n - 1 - target) +
                                   new_target_trit * 3^(n - 1 - target)
        gate[new_state + 1, state + 1] = 1.0
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

        operations = split(circuit, ")")
        operations = [strip(op, ['(', ')']) for op in operations if !isempty(strip(op))]

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
                    full = kron(full, q == indices[1] ? rot : Matrix{ComplexF64}(I, 3, 3))
                end
                U = full * U

            elseif gate_name == "CINC"
                U = CINCGate(n, indices[1], indices[2]) * U

            elseif gate_name == "CINC22"
                U = CINCDaggerGate(n, indices[1], indices[2]) * U

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

# CINC† für |0⟩↔|2⟩: target -= control mod 3
function CINCDaggerGate(n::Int, control::Int, target::Int)
    dim  = 3^n
    gate = zeros(ComplexF64, dim, dim)

    for state in 0:(dim-1)
        control_trit    = (state ÷ 3^(n-1-control)) % 3
        target_trit     = (state ÷ 3^(n-1-target))  % 3
        new_target_trit = (target_trit - control_trit + 3) % 3  # ← minus!
        new_state       = state - target_trit     * 3^(n-1-target) +
                                   new_target_trit * 3^(n-1-target)
        gate[new_state+1, state+1] = 1.0
    end

    return gate
end

# CINC33 für |1⟩↔|2⟩: nur im {1,2} Unterraum
function CINC33Gate(n::Int, control::Int, target::Int)
    dim  = 3^n
    gate = zeros(ComplexF64, dim, dim)

    for state in 0:(dim-1)
        control_trit = (state ÷ 3^(n-1-control)) % 3
        target_trit  = (state ÷ 3^(n-1-target))  % 3

        new_target_trit = if control_trit == 0 || target_trit == 0
            target_trit              # |0⟩ immer unberührt
        else
            # {1,2} Unterraum: +1 mod 2, verschoben um 1
            (target_trit - 1 + 1) % 2 + 1
        end

        new_state = state - target_trit     * 3^(n-1-target) +
                            new_target_trit * 3^(n-1-target)
        gate[new_state+1, state+1] = 1.0
    end

    return gate
end

# --------------------------------------------------------------------
# Prepare State
# --------------------------------------------------------------------
function PrepareStateQutrit(initial_text::String, N::Int)
    U_list  = ParseCircuitToMatrixQutrit([initial_text], N)
    U       = U_list[1]
    psi0    = zeros(ComplexF64, 3^N)
    psi0[1] = 1.0
    psi     = U * psi0
    return psi * psi'
end

# --------------------------------------------------------------------
# Cholesky Parametrisierung
# --------------------------------------------------------------------
function randomInitialTQutrit(N::Int)
    dim = 3^N
    return randn(dim, dim) + im * randn(dim, dim)
end

function densityMatrixFromTQutrit(T::Matrix{ComplexF64})
    AtA = T' * T
    return AtA / tr(AtA)
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
    lr::Float64      = 0.1,
    decay::Float64   = 0.9999,
    iterations::Int  = 3000,
    patience::Int    = 200,
    tol::Float64     = 1e-12,
    verbose::Bool    = true
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

    T  = randomInitialTQutrit(N)
    L0, _ = Zygote.withgradient(loss, T)
    verbose && println("Loss initial: ", round(L0, digits=4))

    m  = zeros(ComplexF64, size(T))
    v  = zeros(Float64,    size(T))
    β₁, β₂, ε_adam = 0.9, 0.999, 1e-8
    losses     = Float64[]
    no_improve = 0

    for i in 1:iterations
        L, grads = Zygote.withgradient(loss, T)
        g = grads[1]

        η  = lr * decay^i
        m  = β₁ .* m + (1 - β₁) .* g
        v  = β₂ .* v + (1 - β₂) .* abs2.(g)
        m̂  = m ./ (1 - β₁^i)
        v̂  = v ./ (1 - β₂^i)
        T  = T - η .* m̂ ./ (sqrt.(v̂) .+ ε_adam)

        push!(losses, L)

        if verbose && i % 500 == 0
            println("Iteration $i | Loss: $(round(L, digits=6))")
        end

        if length(losses) > 1
            rel = abs(losses[end] - losses[end-1]) / abs(losses[end-1])
            rel < tol ? no_improve += 1 : (no_improve = 0)
        end
        no_improve >= patience && break
    end

    rho_rec = T' * T
    rho_rec = rho_rec / tr(rho_rec)
    return rho_rec
end


function BuildHybridCircuitsQutrit(selective_blocks::Vector{Int}, N::Int)
    all_sequences = Vector{Vector{String}}()

    # Y-artige Generatoren
    y_like = ["RL2", "RL5", "RL7"]

    # Rotationen pro Übergangstyp
    e_gate = Dict(1 => "RL2", 2 => "RL5", 3 => "RL7")
    o_gate = Dict(1 => "RL1", 2 => "RL4", 3 => "RL6")

    # CINC Typ pro Übergangstyp
    # t=1: |0⟩↔|1⟩ → Standard CINC
    # t=2: |0⟩↔|2⟩ → CINC† (CINC_22)
    # t=3: |1⟩↔|2⟩ → CINC_33
    cinc_type = Dict(1 => "CINC", 2 => "CINC22", 3 => "CINC33")

    for block in selective_blocks
        block_types    = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i, t) in enumerate(block_types) if t != 0]

        # Block 0: keine Gates nötig
        if isempty(active_qutrits)
            push!(all_sequences, [""])
            continue
        end

        # ── Schritt 1: Gruppiere nach Übergangstyp ──────────────
        type_groups = Dict{Int, Vector{Int}}()
        for (q, t) in active_qutrits
            if !haskey(type_groups, t)
                type_groups[t] = Int[]
            end
            push!(type_groups[t], q)
        end

        # ── Schritt 2: Entangling innerhalb jeder Gruppe ────────
        entangling_gates = String[]
        local_qutrits    = Int[]  # erstes Qutrit pro Gruppe

        for (t, qubits) in sort(collect(type_groups))
            if length(qubits) == 1
                # nur 1 Qutrit dieser Art → kein CINC
                push!(local_qutrits, (qubits[1], t)...)
            else
                # mehrere Qutrits → CINC zwischen ihnen
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

        # ── Schritt 3: Alle Kombinationen lokaler Rotationen ────
        # Pro lokalem Qutrit: E oder O Rotation
        local_qutrit_types = [t for (q, t) in active_qutrits
                               if q in local_qutrits]
        local_qutrit_idxs  = [q for (q, t) in active_qutrits
                               if q in local_qutrits]

        rot_options = [[e_gate[t], o_gate[t]] for t in local_qutrit_types]
        rot_choices = Iterators.product(rot_options...)

        circuits = String[]
        for choice in rot_choices
            # Zähle Y-artige Generatoren
            n_y = count(r -> r in y_like, choice)

            # Lokale Rotations-Gates
            local_gates = ["($gate:$q)" for (gate, q)
                           in zip(choice, local_qutrit_idxs)]

            # Vollständiger Circuit: lokale Rotation + CINC
            circuit_str = join(vcat(
                reverse(entangling_gates),
                local_gates
            ))

            push!(circuits, (n_y % 2 == 0 ? "E:" : "O:") * circuit_str)
        end

        push!(all_sequences, circuits)
    end

    return all_sequences
end

function FullTomographyHybrid(N::Int; shots_per_circuit::Int=1000, 
                               lr::Float64=0.1, decay::Float64=0.9999,
                               iterations::Int=3000, patience::Int=200)

    println("═══ Volle Tomographie (Hybrid) N=$N ═══\n")
    
    dim      = 3^N
    rho_true = GenerateRandomDensityMatrixNoZerosQutrits(N)
    blocks   = collect(0:(4^N - 1))

    # ── Schritt 1: Hybrid Circuits ────────────────────────────
    hybrid_circs = BuildHybridCircuitsQutrit(blocks, N)

    # Flatten und E/O Prefix entfernen
    circuits = String[]
    for circ_group in hybrid_circs
        for c in circ_group
            # Entferne "E:" oder "O:" Prefix
            circuit = startswith(c, "E:") || startswith(c, "O:") ? c[3:end] : c
            push!(circuits, circuit)
        end
    end
    # Duplikate entfernen
    circuits = unique(circuits)

    println("Anzahl Blöcke:       ", length(blocks))
    println("Anzahl Circuits:     ", length(circuits))
    println("Shots pro Circuit:   ", shots_per_circuit)
    println("Datenpunkte total:   ", length(circuits) * dim)

    # ── Schritt 2: Unitäre Matrizen ───────────────────────────
    Us_all = ParseCircuitToMatrixQutrit(circuits, N)

    # ── Schritt 3: Messungen simulieren ───────────────────────
    data = DataPredictFromRhoSampledQutrit(rho_true, Us_all, shots_per_circuit)

    # ── Schritt 4: Rekonstruktion ──────────────────────────────
    println("\n── SGD + Cholesky ──")
    rho_rec = ProcessDataQutrit(
        data, Us_all, blocks, shots_per_circuit, N;
        lr=lr, decay=decay, iterations=iterations,
        patience=patience, verbose=true
    )

    # ── Schritt 5: Ergebnisse ─────────────────────────────────
    F = fidelity(rho_rec, rho_true)
    println("\n── Ergebnisse ──")
    println(@sprintf("Fidelität:           %.4f", F))
    println(@sprintf("Spur ρ_rec:          %.6f", real(tr(rho_rec))))
    println("Hermitesch:          ", rho_rec ≈ rho_rec')
    println(@sprintf("Kleinster Eigenwert: %.6f",
        minimum(real(eigvals(rho_rec)))))

    @assert abs(real(tr(rho_rec)) - 1.0) < 1e-6     "Fehler: Spur ≠ 1"
    @assert minimum(real(eigvals(rho_rec))) >= -1e-6 "Fehler: neg. Eigenwerte"
    @assert F > 0.7                                   "Fehler: Fidelität zu niedrig"

    println("\n✓ Hybrid Tomographie N=$N bestanden")
    println("═"^45)

    return rho_rec, rho_true, F
end

function BuildHybridCircuitsNoRedundancyQutrit(selective_blocks::Vector{Int}, N::Int)
    
    y_like = ["RL2", "RL5", "RL7"]
    e_gate = Dict(1 => "RL2", 2 => "RL5", 3 => "RL7")
    o_gate = Dict(1 => "RL1", 2 => "RL4", 3 => "RL6")
    cinc_type = Dict(1 => "CINC", 2 => "CINC22", 3 => "CINC33")

    # Globales Set aller bereits gesehenen Circuits
    seen_circuits = Set{String}()
    all_sequences = Vector{Vector{String}}()

    for block in selective_blocks
        block_types    = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i, t) in enumerate(block_types) if t != 0]

        if isempty(active_qutrits)
            push!(all_sequences, [""])
            continue
        end

        # ── Schritt 1: Gruppiere nach Übergangstyp ──────────────
        type_groups = Dict{Int, Vector{Int}}()
        for (q, t) in active_qutrits
            if !haskey(type_groups, t)
                type_groups[t] = Int[]
            end
            push!(type_groups[t], q)
        end

        # ── Schritt 2: Entangling innerhalb jeder Gruppe ────────
        entangling_gates = String[]
        local_qutrits    = Int[]

        for (t, qubits) in sort(collect(type_groups))
            if length(qubits) == 1
                push!(local_qutrits, qubits[1])
            else
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
                push!(local_qutrits, qubits[1])
            end
        end

        # ── Schritt 3: Lokale Rotationen ohne Redundanz ─────────
        local_qutrit_types = [t for (q, t) in active_qutrits
                               if q in local_qutrits]
        local_qutrit_idxs  = [q for (q, t) in active_qutrits
                               if q in local_qutrits]

        rot_options  = [[e_gate[t], o_gate[t]] for t in local_qutrit_types]
        rot_choices  = Iterators.product(rot_options...)

        circuits = String[]
        for choice in rot_choices
            n_y = count(r -> r in y_like, choice)

            local_gates = ["($gate:$q)" for (gate, q)
                           in zip(choice, local_qutrit_idxs)]

            circuit_str = join(vcat(reverse(entangling_gates), local_gates))

            # ── Nur hinzufügen wenn noch nicht gesehen ──────────
            if circuit_str ∉ seen_circuits
                push!(seen_circuits, circuit_str)
                push!(circuits, (n_y % 2 == 0 ? "E:" : "O:") * circuit_str)
            end
        end

        # Wenn alle Circuits redundant waren → leeren Eintrag
        push!(all_sequences, isempty(circuits) ? [""] : circuits)
    end

    return all_sequences
end

end # module SeeqstQutrit
