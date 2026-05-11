using LinearAlgebra
using Zygote
using Optimisers
using LinearAlgebra
using StatsBase
using IterTools
using Convex
using SCS
using Printf
using QuantumInformation


function GenerateRandomDensityMatrixNoZerosQutrits(n::Int)
    d = 3^n
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end

function GetSelectiveBlocksQutrit(N::Int, wanted_indexes::Vector{Tuple{Int,Int}})
    max_index = 3^N
    selective_blocks = Vector{Int}()
    block_map = Dict{Int, Vector{Tuple{Int,Int}}}()

    for (i, j) in wanted_indexes
        @assert (0 <= i < max_index && 0 <= j < max_index) "Index ($i,$j) außerhalb des gültigen Bereichs [0, $(max_index-1)]"

        i_trits = digits(i, base=3, pad=N) |> reverse
        j_trits = digits(j, base=3, pad=N) |> reverse

        # Übergangstyp pro Trit-Paar
        block_trits = Int[]
        for k in 1:N
            a, b = i_trits[k], j_trits[k]
            if a == b
                push!(block_trits, 0)        # diagonal
            elseif (a,b) in [(0,1),(1,0)]
                push!(block_trits, 1)        # |0⟩↔|1⟩
            elseif (a,b) in [(0,2),(2,0)]
                push!(block_trits, 2)        # |0⟩↔|2⟩
            else                             # (1,2) oder (2,1)
                push!(block_trits, 3)        # |1⟩↔|2⟩
            end
        end

        # Block-Nummer in Basis 4
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

println("═══ Alle Blöcke und ihre Elemente für N=2 Qutrits ═══\n")

N = 2
dim = 3^N  # = 9

# Sammle alle Elemente pro Block
block_elements = Dict{Int, Vector{Tuple{Int,Int}}}()

for i in 0:(dim-1)
    for j in 0:(dim-1)
        blocks, _ = GetSelectiveBlocksQutrit(N, [(i,j)])
        block = blocks[1]

        if !haskey(block_elements, block)
            block_elements[block] = Tuple{Int,Int}[]
        end
        push!(block_elements[block], (i,j))
    end
end

#=
# Ausgabe pro Block
for block in sort(collect(keys(block_elements)))
    # Block-Nummer in Basis 4 dekodieren
    block_type = digits(block, base=4, pad=N) |> reverse
    
    println("━"^50)
    println("Block $block = $(join(block_type))₄")

    # Übergangstyp pro Qutrit
    type_names = ["diagonal", "|0⟩↔|1⟩ Übergang", "|0⟩↔|2⟩ Übergang", "|1⟩↔|2⟩ Übergang"]
    for (k, t) in enumerate(block_type)
        println("  Qutrit $k: $(type_names[t+1])")
    end
    println()

    println("  Elemente:")
    for (i,j) in block_elements[block]
        i_trits = digits(i, base=3, pad=N) |> reverse
        j_trits = digits(j, base=3, pad=N) |> reverse
        println(@sprintf("    ρ[%d,%d]  (%s, %s)", i+1, j+1, join(i_trits), join(j_trits)))
    end
    println("  Anzahl: ", length(block_elements[block]))
    println()
end
=#
function GenerateSelectiveElementsQutrit(selective_blocks::Vector{Int}, N::Int)
    selective_elements = Vector{Vector{Tuple{Int,Int}}}()

    for block in selective_blocks
        block_types = digits(block, base=4, pad=N) |> reverse

        # Erlaubte i-Trits pro Qutrit basierend auf Übergangstyp
        function allowed_i_trits(t::Int)
            if t == 0
                return [0, 1, 2]     # diagonal: alle Trits
            elseif t == 1
                return [0, 1]        # |0⟩↔|1⟩: nur 0 und 1
            elseif t == 2
                return [0, 2]        # |0⟩↔|2⟩: nur 0 und 2
            elseif t == 3
                return [1, 2]        # |1⟩↔|2⟩: nur 1 und 2
            end
        end

        # j-Trit aus i-Trit und Übergangstyp
        function j_trit(a::Int, t::Int)
            if t == 0
                return a             # diagonal: j = i
            elseif t == 1
                return a == 0 ? 1 : 0   # |0⟩↔|1⟩: 0↔1
            elseif t == 2
                return a == 0 ? 2 : 0   # |0⟩↔|2⟩: 0↔2
            elseif t == 3
                return a == 1 ? 2 : 1   # |1⟩↔|2⟩: 1↔2
            end
        end

        # Alle erlaubten i-Trit Kombinationen
        allowed = [allowed_i_trits(t) for t in block_types]
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

#=
println("═══ Test GenerateSelectiveElementsQutrit ═══\n")

N = 2

expected_counts = Dict(
    0  => 9,
    1  => 6,
    2  => 6,
    3  => 6,
    4  => 6,
    5  => 4,
    6  => 4,
    7  => 4,
    8  => 6,
    9  => 4,
    10 => 4,
    11 => 4,
    12 => 6,
    13 => 4,
    14 => 4,
    15 => 4,
)

type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]

all_elements = Set{Tuple{Int,Int}}()

for block in 0:15
    result      = GenerateSelectiveElementsQutrit([block], N)
    elems       = result[1]
    block_types = digits(block, base=4, pad=N) |> reverse
    desc        = join([type_names[t+1] for t in block_types], " × ")
    expected    = expected_counts[block]
    status      = length(elems) == expected ? "✓" : "✗"

    println("━"^60)
    println("$status Block $block = $(join(block_types))₄  ($desc)")
    println("   Anzahl: $(length(elems)) (erwartet $expected)")
    println("   Elemente:")

    for (i,j) in sort(elems)
        i_trits = digits(i, base=3, pad=N) |> reverse
        j_trits = digits(j, base=3, pad=N) |> reverse
        println(@sprintf("     ρ[%d,%d]  (%s, %s)", i+1, j+1, join(i_trits), join(j_trits)))
    end
    println()

    for e in elems
        push!(all_elements, e)
    end
end

println("━"^60)
println("Gesamtzahl einzigartiger Elemente: ", length(all_elements))
println("Erwartet: 81 (9×9 Matrix)")
@assert length(all_elements) == 81 "Fehler: nicht alle 81 Elemente abgedeckt!"
println("✓ Alle 81 Elemente abgedeckt ✓")
=#

function GenerateObservableSetsQutrit(selective_blocks::Vector{Int}, N::Int)
    observable_sets = Dict{Int, Tuple{Vector{String}, Vector{String}}}()

    for block in selective_blocks
        # Block in Basis 4 dekodieren
        block_types = digits(block, base=4, pad=N) |> reverse

        # Pro Qutrit: Übergangstyp → Observablen
        pauli_options = Vector{Vector{String}}()
        for t in block_types
            if t == 0
                push!(pauli_options, ["I", "Z"])   # diagonal
            elseif t == 1
                push!(pauli_options, ["L1", "L2"]) # |0⟩↔|1⟩
            elseif t == 2
                push!(pauli_options, ["L4", "L5"]) # |0⟩↔|2⟩
            elseif t == 3
                push!(pauli_options, ["L6", "L7"]) # |1⟩↔|2⟩
            end
        end

        # Alle Kombinationen
        all_observables = [join(combo, "⊗") for combo in Iterators.product(pauli_options...)]
        all_observables = vec(all_observables)

        # Y-artige Generatoren: λ₂, λ₅, λ₇
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

#=
println("═══ Tests GenerateObservableSetsQutrit ═══\n")

type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]

# ── N=2: Alle Blöcke ──────────────────────────────────────────
println("━"^60)
println("N=2: Alle 16 Blöcke")
println("━"^60)

N = 2
obs_all = GenerateObservableSetsQutrit(collect(0:15), N)

for block in 0:15
    block_types = digits(block, base=4, pad=N) |> reverse
    desc = join([type_names[t+1] for t in block_types], " × ")
    E, O = obs_all[block]
    println("\nBlock $block = $(join(block_types))₄  ($desc)")
    println("  E-Gruppe: ", E)
    println("  O-Gruppe: ", O)
end

# ── N=2: Assertions ───────────────────────────────────────────
println("\n── Assertions N=2 ──")

# Block 0: nur diagonal → O leer
E, O = obs_all[0]
@assert "I⊗I" in E  && "Z⊗Z" in E  "Fehler Block 0 E"
@assert isempty(O)                   "Fehler Block 0 O"
println("✓ Block 0 bestanden")

# Block 1: diag × |0⟩↔|1⟩
E, O = obs_all[1]
@assert "I⊗L1" in E && "Z⊗L1" in E  "Fehler Block 1 E"
@assert "I⊗L2" in O && "Z⊗L2" in O  "Fehler Block 1 O"
println("✓ Block 1 bestanden")

# Block 2: diag × |0⟩↔|2⟩
E, O = obs_all[2]
@assert "I⊗L4" in E && "Z⊗L4" in E  "Fehler Block 2 E"
@assert "I⊗L5" in O && "Z⊗L5" in O  "Fehler Block 2 O"
println("✓ Block 2 bestanden")

# Block 3: diag × |1⟩↔|2⟩
E, O = obs_all[3]
@assert "I⊗L6" in E && "Z⊗L6" in E  "Fehler Block 3 E"
@assert "I⊗L7" in O && "Z⊗L7" in O  "Fehler Block 3 O"
println("✓ Block 3 bestanden")

# Block 5: |0⟩↔|1⟩ × |0⟩↔|1⟩
E, O = obs_all[5]
@assert "L1⊗L1" in E && "L2⊗L2" in E  "Fehler Block 5 E"
@assert "L1⊗L2" in O && "L2⊗L1" in O  "Fehler Block 5 O"
println("✓ Block 5 bestanden")

# Block 15: |1⟩↔|2⟩ × |1⟩↔|2⟩
E, O = obs_all[15]
@assert "L6⊗L6" in E && "L7⊗L7" in E  "Fehler Block 15 E"
@assert "L6⊗L7" in O && "L7⊗L6" in O  "Fehler Block 15 O"
println("✓ Block 15 bestanden")

println("\n✓ Alle N=2 Assertions bestanden\n")

# ── N=3: Ausgewählte Blöcke ───────────────────────────────────
println("━"^60)
println("N=3: Ausgewählte Blöcke")
println("━"^60)

N = 3
selected_blocks = [
    0,    # 000₄: alle diagonal
    1,    # 001₄: Qutrit 3: |0⟩↔|1⟩
    5,    # 011₄: Qutrit 2,3: |0⟩↔|1⟩
    21,   # 111₄: alle |0⟩↔|1⟩
    42,   # 222₄: alle |0⟩↔|2⟩
    63,   # 333₄: alle |1⟩↔|2⟩
    6,    # 012₄: Qutrit 2: |0⟩↔|1⟩, Qutrit 3: |0⟩↔|2⟩
    27,   # 120₄: gemischt
]

obs_n3 = GenerateObservableSetsQutrit(selected_blocks, N)

for block in selected_blocks
    block_types = digits(block, base=4, pad=N) |> reverse
    desc = join([type_names[t+1] for t in block_types], " × ")
    E, O = obs_n3[block]
    println("\nBlock $block = $(join(block_types))₄  ($desc)")
    println("  E-Gruppe: ", E)
    println("  O-Gruppe: ", O)
end

# ── N=3: Assertions ───────────────────────────────────────────
println("\n── Assertions N=3 ──")

# Block 0: alle diagonal → O leer
E, O = obs_n3[0]
@assert "I⊗I⊗I" in E   "Fehler N=3 Block 0 E"
@assert isempty(O)       "Fehler N=3 Block 0 O"
println("✓ N=3 Block 0 bestanden")

# Block 21 = 111₄: alle |0⟩↔|1⟩
E, O = obs_n3[21]
@assert "L1⊗L1⊗L1" in E  "Fehler N=3 Block 21 E"
@assert "L2⊗L2⊗L2" in O  "Fehler N=3 Block 21 O"
println("✓ N=3 Block 21 bestanden")

# Block 63 = 333₄: alle |1⟩↔|2⟩
E, O = obs_n3[63]
@assert "L6⊗L6⊗L6" in E  "Fehler N=3 Block 63 E"
@assert "L7⊗L7⊗L7" in O  "Fehler N=3 Block 63 O"
println("✓ N=3 Block 63 bestanden")

println("\n✓ Alle N=3 Assertions bestanden")
println("\n═══ Alle Tests bestanden ✓ ═══")
=#
#=
function BuildEntanglerBlocksQutrit(selective_blocks::Vector{Int}, N::Int)
    all_sequences = Vector{Vector{String}}()

    # E-Rotation pro Übergangstyp
    e_gate = Dict(1 => "RL2", 2 => "RL5", 3 => "RL7")
    # O-Rotation pro Übergangstyp
    o_gate = Dict(1 => "RL1", 2 => "RL4", 3 => "RL6")

    for block in selective_blocks
        block_types   = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i, t) in enumerate(block_types) if t != 0]

        # Block 0: keine Gates nötig
        if isempty(active_qutrits)
            push!(all_sequences, [""])
            continue
        end

        sequence_e = String[]
        sequence_o = String[]

        # Schritt 1: Rotation auf erstem aktiven Qutrit
        first_q, first_t = active_qutrits[1]
        push!(sequence_e, "($(e_gate[first_t]):$first_q)")
        push!(sequence_o, "($(o_gate[first_t]):$first_q)")

        head = [first_q]
        tail = active_qutrits[2:end]

        # Schritt 2: CINC parallel aufbauen + Rotation auf jedem Qutrit
        while !isempty(tail)
            new_tail = Int[]
            for h in head
                isempty(tail) && break
                tgt_q, tgt_t = popfirst!(tail)

                # Rotation auf target Qutrit
                push!(sequence_e, "($(e_gate[tgt_t]):$tgt_q)")
                push!(sequence_o, "($(o_gate[tgt_t]):$tgt_q)")

                # CINC von head zu target
                push!(sequence_e, "(CINC:$h,$tgt_q)")
                push!(sequence_o, "(CINC:$h,$tgt_q)")

                push!(new_tail, tgt_q)
            end
            append!(head, new_tail)
        end

        # Schritt 3: Umgekehrte Reihenfolge
        e_str = join(reverse(sequence_e))
        o_str = join(reverse(sequence_o))

        push!(all_sequences, [e_str, o_str])
    end

    return all_sequences
end
=#
#=
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

        # Schritt 1: Rotation NUR auf erstem aktiven Qutrit
        first_q, first_t = active_qutrits[1]
        push!(sequence_e, "($(e_gate[first_t]):$first_q)")
        push!(sequence_o, "($(o_gate[first_t]):$first_q)")

        head = [first_q]
        tail = active_qutrits[2:end]

        # Schritt 2: CINC parallel aufbauen
        # KEIN weiteres Rotation-Gate auf target Qutrits!
        while !isempty(tail)
            new_tail = Int[]
            for h in head
                isempty(tail) && break
                tgt_q, tgt_t = popfirst!(tail)

                # Nur CINC - keine Rotation auf target!
                push!(sequence_e, "(CINC:$h,$tgt_q)")
                push!(sequence_o, "(CINC:$h,$tgt_q)")

                push!(new_tail, tgt_q)
            end
            append!(head, new_tail)
        end

        # Schritt 3: Umgekehrte Reihenfolge
        e_str = join(reverse(sequence_e))
        o_str = join(reverse(sequence_o))

        push!(all_sequences, [e_str, o_str])
    end

    return all_sequences
end
=#
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

        # Schritt 1: Rotation NUR auf erstem aktiven Qutrit
        first_q, first_t = active_qutrits[1]
        push!(sequence_e, "($(e_gate[first_t]):$first_q)")
        push!(sequence_o, "($(o_gate[first_t]):$first_q)")

        head = [first_q]
        tail = active_qutrits[2:end]

        # Schritt 2: GHZ parallel aufbauen mit CINC
        # KEINE Rotation auf target Qutrit!
        while !isempty(tail)
            new_tail = Int[]
            for h in head
                isempty(tail) && break
                tgt_q, tgt_t = popfirst!(tail)

                # Nur CINC - keine Rotation!
                push!(sequence_e, "(CINC:$h,$tgt_q)")
                push!(sequence_o, "(CINC:$h,$tgt_q)")

                push!(new_tail, tgt_q)
            end
            append!(head, new_tail)
        end

        # Schritt 3: Umgekehrte Reihenfolge
        e_str = join(reverse(sequence_e))
        o_str = join(reverse(sequence_o))

        push!(all_sequences, [e_str, o_str])
    end

    return all_sequences
end
println("═══ Tests BuildEntanglerBlocksQutrit ═══\n")

# ── N=2: Alle Blöcke ──────────────────────────────────────────
println("━"^60)
println("N=2: Alle 16 Blöcke")
println("━"^60)

N = 2
type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]

for block in 0:15
    block_types = digits(block, base=4, pad=N) |> reverse
    desc = join([type_names[t+1] for t in block_types], " × ")
    seq  = BuildEntanglerBlocksQutrit([block], N)[1]

    println("\nBlock $block = $(join(block_types))₄  ($desc)")
    if seq == [""]
        println("  Circuit: (leer)")
    else
        println("  Circuit E: ", seq[1])
        println("  Circuit O: ", seq[2])
    end
end

# ── Assertions N=2 ────────────────────────────────────────────
println("\n── Assertions N=2 ──")

# Block 0: leer
seq = BuildEntanglerBlocksQutrit([0], 2)[1]
@assert seq == [""]  "Fehler Block 0"
println("✓ Block 0 bestanden")

# Block 1 = 01₄: nur Rotation auf Qutrit 2
seq = BuildEntanglerBlocksQutrit([1], 2)[1]
@assert occursin("RL2", seq[1])  "Fehler Block 1 E"
@assert occursin("RL1", seq[2])  "Fehler Block 1 O"
@assert !occursin("CINC", seq[1]) "Fehler Block 1: kein CINC"
println("✓ Block 1 bestanden")

# Block 5 = 11₄: Rotation + CINC
seq = BuildEntanglerBlocksQutrit([5], 2)[1]
@assert occursin("RL2", seq[1]) && occursin("CINC", seq[1])  "Fehler Block 5 E"
@assert occursin("RL1", seq[2]) && occursin("CINC", seq[2])  "Fehler Block 5 O"
println("✓ Block 5 bestanden")

# Block 15 = 33₄: |1⟩↔|2⟩ auf beiden
seq = BuildEntanglerBlocksQutrit([15], 2)[1]
@assert occursin("RL7", seq[1]) && occursin("CINC", seq[1])  "Fehler Block 15 E"
@assert occursin("RL6", seq[2]) && occursin("CINC", seq[2])  "Fehler Block 15 O"
println("✓ Block 15 bestanden")

println("\n✓ Alle N=2 Assertions bestanden\n")

# ── N=3: Ausgewählte Blöcke ───────────────────────────────────
println("━"^60)
println("N=3: Ausgewählte Blöcke")
println("━"^60)

N = 3
selected = [0, 1, 21, 42, 63, 27]

for block in selected
    block_types = digits(block, base=4, pad=N) |> reverse
    desc = join([type_names[t+1] for t in block_types], " × ")
    seq  = BuildEntanglerBlocksQutrit([block], N)[1]

    println("\nBlock $block = $(join(block_types))₄  ($desc)")
    if seq == [""]
        println("  Circuit: (leer)")
    else
        println("  Circuit E: ", seq[1])
        println("  Circuit O: ", seq[2])
    end
end

println("\n═══ Alle Tests bestanden ✓ ═══")


function BuildNonEntanglingCircuitsQutrit(selective_blocks::Vector{Int}, N::Int)
    all_block_circuits = Vector{Vector{String}}()

    # Mögliche Rotationen pro Übergangstyp
    gate_options = Dict(
        1 => ["RL2", "RL1"],  # |0⟩↔|1⟩: E dann O
        2 => ["RL5", "RL4"],  # |0⟩↔|2⟩: E dann O
        3 => ["RL7", "RL6"],  # |1⟩↔|2⟩: E dann O
    )

    for block in selective_blocks
        block_types   = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i, t) in enumerate(block_types) if t != 0]

        # Block 0: keine Gates nötig
        if isempty(active_qutrits)
            push!(all_block_circuits, [""])
            continue
        end

        # Alle Kombinationen von Rotationen
        options = [gate_options[t] for (q, t) in active_qutrits]
        qubits  = [q for (q, t) in active_qutrits]

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
#=
println("═══ Tests BuildNonEntanglingCircuitsQutrit ═══\n")

type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]

# ── N=2: Alle Blöcke ──────────────────────────────────────────
println("━"^60)
println("N=2: Alle 16 Blöcke")
println("━"^60)

N = 2
for block in 0:15
    block_types = digits(block, base=4, pad=N) |> reverse
    desc    = join([type_names[t+1] for t in block_types], " × ")
    circuits = BuildNonEntanglingCircuitsQutrit([block], N)[1]
    n_active = count(t -> t != 0, block_types)

    println("\nBlock $block = $(join(block_types))₄  ($desc)")
    println("  Anzahl Circuits: $(length(circuits))  (erwartet: $(2^n_active))")
    for c in circuits
        println("    $c")
    end
end

# ── Assertions N=2 ────────────────────────────────────────────
println("\n── Assertions N=2 ──")

# Block 0: leer
c = BuildNonEntanglingCircuitsQutrit([0], 2)[1]
@assert c == [""]           "Fehler Block 0"
println("✓ Block 0 bestanden")

# Block 1 = 01₄: 1 aktives Qutrit → 2 Circuits
c = BuildNonEntanglingCircuitsQutrit([1], 2)[1]
@assert length(c) == 2      "Fehler Block 1: sollte 2 Circuits haben"
@assert any(occursin("RL2", x) for x in c)  "Fehler Block 1: RL2 fehlt"
@assert any(occursin("RL1", x) for x in c)  "Fehler Block 1: RL1 fehlt"
println("✓ Block 1 bestanden")

# Block 5 = 11₄: 2 aktive Qutrits → 4 Circuits
c = BuildNonEntanglingCircuitsQutrit([5], 2)[1]
@assert length(c) == 4      "Fehler Block 5: sollte 4 Circuits haben"
println("✓ Block 5 bestanden")

# Block 15 = 33₄: 2 aktive Qutrits → 4 Circuits
c = BuildNonEntanglingCircuitsQutrit([15], 2)[1]
@assert length(c) == 4      "Fehler Block 15: sollte 4 Circuits haben"
@assert any(occursin("RL7", x) for x in c)  "Fehler Block 15: RL7 fehlt"
@assert any(occursin("RL6", x) for x in c)  "Fehler Block 15: RL6 fehlt"
println("✓ Block 15 bestanden")

println("\n✓ Alle N=2 Assertions bestanden\n")

# ── N=3: Ausgewählte Blöcke ───────────────────────────────────
println("━"^60)
println("N=3: Ausgewählte Blöcke")
println("━"^60)

N = 3
selected = [0, 1, 21, 42, 63, 27]

for block in selected
    block_types = digits(block, base=4, pad=N) |> reverse
    desc     = join([type_names[t+1] for t in block_types], " × ")
    circuits = BuildNonEntanglingCircuitsQutrit([block], N)[1]
    n_active = count(t -> t != 0, block_types)

    println("\nBlock $block = $(join(block_types))₄  ($desc)")
    println("  Anzahl Circuits: $(length(circuits))  (erwartet: $(2^n_active))")
    for c in circuits
        println("    $c")
    end
end

# ── Assertions N=3 ────────────────────────────────────────────
println("\n── Assertions N=3 ──")

# Block 21 = 111₄: 3 aktive Qutrits → 8 Circuits
c = BuildNonEntanglingCircuitsQutrit([21], 3)[1]
@assert length(c) == 8  "Fehler N=3 Block 21: sollte 8 Circuits haben"
println("✓ N=3 Block 21 bestanden")

# Block 63 = 333₄: 3 aktive Qutrits → 8 Circuits
c = BuildNonEntanglingCircuitsQutrit([63], 3)[1]
@assert length(c) == 8  "Fehler N=3 Block 63: sollte 8 Circuits haben"
println("✓ N=3 Block 63 bestanden")

println("\n═══ Alle Tests bestanden ✓ ═══")
=#
#=
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

        # Schritt 1: Rotation auf erstem aktiven Qutrit
        first_q, first_t = active_qutrits[1]
        push!(sequence_e, "($(e_gate[first_t]):$first_q)")
        push!(sequence_o, "($(o_gate[first_t]):$first_q)")

        head = [first_q]
        tail = active_qutrits[2:end]

        # Schritt 2: GHZ parallel aufbauen mit CINC
        while !isempty(tail)
            new_tail = Int[]
            for h in head
                isempty(tail) && break
                tgt_q, tgt_t = popfirst!(tail)

                # Rotation auf target Qutrit
                push!(sequence_e, "($(e_gate[tgt_t]):$tgt_q)")
                push!(sequence_o, "($(o_gate[tgt_t]):$tgt_q)")

                # CINC von head zu target
                push!(sequence_e, "(CINC:$h,$tgt_q)")
                push!(sequence_o, "(CINC:$h,$tgt_q)")

                push!(new_tail, tgt_q)
            end
            append!(head, new_tail)
        end

        # Schritt 3: Umgekehrte Reihenfolge
        e_str = join(reverse(sequence_e))
        o_str = join(reverse(sequence_o))

        push!(all_sequences, [e_str, o_str])
    end

    return all_sequences
end
=#
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

        # Schritt 1: Rotation NUR auf erstem aktiven Qutrit
        first_q, first_t = active_qutrits[1]
        push!(sequence_e, "($(e_gate[first_t]):$first_q)")
        push!(sequence_o, "($(o_gate[first_t]):$first_q)")

        head = [first_q]
        tail = active_qutrits[2:end]

        # Schritt 2: GHZ parallel aufbauen mit CINC
        # KEINE Rotation auf target Qutrit!
        while !isempty(tail)
            new_tail = Int[]
            for h in head
                isempty(tail) && break
                tgt_q, tgt_t = popfirst!(tail)

                # Nur CINC - keine Rotation!
                push!(sequence_e, "(CINC:$h,$tgt_q)")
                push!(sequence_o, "(CINC:$h,$tgt_q)")

                push!(new_tail, tgt_q)
            end
            append!(head, new_tail)
        end

        # Schritt 3: Umgekehrte Reihenfolge
        e_str = join(reverse(sequence_e))
        o_str = join(reverse(sequence_o))

        push!(all_sequences, [e_str, o_str])
    end

    return all_sequences
end
#=
println("═══ Tests BuildEntanglerBlocksQutrit ═══\n")

type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]

# ── N=2: Alle Blöcke ──────────────────────────────────────────
println("━"^60)
println("N=2: Alle 16 Blöcke")
println("━"^60)

N = 2
for block in 0:15
    block_types = digits(block, base=4, pad=N) |> reverse
    desc = join([type_names[t+1] for t in block_types], " × ")
    seq  = BuildEntanglerBlocksQutrit([block], N)[1]

    println("\nBlock $block = $(join(block_types))₄  ($desc)")
    if seq == [""]
        println("  Circuit: (leer)")
    elseif length(seq) == 2
        println("  Circuit E: ", seq[1])
        println("  Circuit O: ", seq[2])
    end
end

# ── Assertions N=2 ────────────────────────────────────────────
println("\n── Assertions N=2 ──")

# Block 0: leer
seq = BuildEntanglerBlocksQutrit([0], 2)[1]
@assert seq == [""]                          "Fehler Block 0"
println("✓ Block 0 bestanden")

# Block 1 = 01₄: nur Rotation auf Qutrit 2, kein CINC
seq = BuildEntanglerBlocksQutrit([1], 2)[1]
@assert length(seq) == 2                     "Fehler Block 1: 2 Circuits"
@assert occursin("RL2", seq[1])              "Fehler Block 1 E: RL2 fehlt"
@assert occursin("RL1", seq[2])              "Fehler Block 1 O: RL1 fehlt"
@assert !occursin("CINC", seq[1])            "Fehler Block 1: kein CINC"
println("✓ Block 1 bestanden")

# Block 2 = 02₄: |0⟩↔|2⟩ auf Qutrit 2
seq = BuildEntanglerBlocksQutrit([2], 2)[1]
@assert occursin("RL5", seq[1])              "Fehler Block 2 E: RL5 fehlt"
@assert occursin("RL4", seq[2])              "Fehler Block 2 O: RL4 fehlt"
println("✓ Block 2 bestanden")

# Block 3 = 03₄: |1⟩↔|2⟩ auf Qutrit 2
seq = BuildEntanglerBlocksQutrit([3], 2)[1]
@assert occursin("RL7", seq[1])              "Fehler Block 3 E: RL7 fehlt"
@assert occursin("RL6", seq[2])              "Fehler Block 3 O: RL6 fehlt"
println("✓ Block 3 bestanden")

# Block 5 = 11₄: beide |0⟩↔|1⟩ → CINC
seq = BuildEntanglerBlocksQutrit([5], 2)[1]
@assert occursin("RL2", seq[1])              "Fehler Block 5 E: RL2 fehlt"
@assert occursin("CINC", seq[1])             "Fehler Block 5 E: CINC fehlt"
@assert occursin("RL1", seq[2])              "Fehler Block 5 O: RL1 fehlt"
@assert occursin("CINC", seq[2])             "Fehler Block 5 O: CINC fehlt"
println("✓ Block 5 bestanden")

# Block 10 = 22₄: beide |0⟩↔|2⟩ → CINC
seq = BuildEntanglerBlocksQutrit([10], 2)[1]
@assert occursin("RL5", seq[1])              "Fehler Block 10 E: RL5 fehlt"
@assert occursin("CINC", seq[1])             "Fehler Block 10 E: CINC fehlt"
println("✓ Block 10 bestanden")

# Block 15 = 33₄: beide |1⟩↔|2⟩ → CINC
seq = BuildEntanglerBlocksQutrit([15], 2)[1]
@assert occursin("RL7", seq[1])              "Fehler Block 15 E: RL7 fehlt"
@assert occursin("CINC", seq[1])             "Fehler Block 15 E: CINC fehlt"
@assert occursin("RL6", seq[2])              "Fehler Block 15 O: RL6 fehlt"
println("✓ Block 15 bestanden")

println("\n✓ Alle N=2 Assertions bestanden\n")

# ── N=3: Ausgewählte Blöcke ───────────────────────────────────
println("━"^60)
println("N=3: Ausgewählte Blöcke")
println("━"^60)

N = 3
selected = [0, 1, 21, 42, 63, 27]

for block in selected
    block_types = digits(block, base=4, pad=N) |> reverse
    desc = join([type_names[t+1] for t in block_types], " × ")
    seq  = BuildEntanglerBlocksQutrit([block], N)[1]

    println("\nBlock $block = $(join(block_types))₄  ($desc)")
    if seq == [""]
        println("  Circuit: (leer)")
    else
        println("  Circuit E: ", seq[1])
        println("  Circuit O: ", seq[2])
    end
end

# ── Assertions N=3 ────────────────────────────────────────────
println("\n── Assertions N=3 ──")

# Block 21 = 111₄: alle |0⟩↔|1⟩ → 2 CINCs
seq = BuildEntanglerBlocksQutrit([21], 3)[1]
cnot_count = count(x -> x == 'C', seq[1]) ÷ 4
println("Block 21 Circuit E: ", seq[1])
@assert occursin("RL2", seq[1])   "Fehler N=3 Block 21 E: RL2 fehlt"
@assert count(m -> m.match == "CINC", eachmatch(r"CINC", seq[1])) == 2  "Fehler: sollte 2 CINCs haben"
println("✓ N=3 Block 21 bestanden")

# Block 63 = 333₄: alle |1⟩↔|2⟩ → 2 CINCs
seq = BuildEntanglerBlocksQutrit([63], 3)[1]
println("Block 63 Circuit E: ", seq[1])
@assert occursin("RL7", seq[1])   "Fehler N=3 Block 63 E: RL7 fehlt"
@assert count(m -> m.match == "CINC", eachmatch(r"CINC", seq[1])) == 2  "Fehler: sollte 2 CINCs haben"
println("✓ N=3 Block 63 bestanden")

println("\n═══ Alle Tests bestanden ✓ ═══")
=#
function GenerateExperimentQutrit(wanted_indexes::Vector{Tuple{Int,Int}}, N::Int)

    # Schritt 1: Selective Blocks bestimmen
    selective_blocks, block_map = GetSelectiveBlocksQutrit(N, wanted_indexes)

    # Schritt 2: Selective Elements generieren
    selective_elements = GenerateSelectiveElementsQutrit(selective_blocks, N)

    # Schritt 3: Observable Sets generieren
    observable_sets = GenerateObservableSetsQutrit(selective_blocks, N)

    # Schritt 4: Schaltkreise generieren
    entangler_circuits     = BuildEntanglerBlocksQutrit(selective_blocks, N)
    non_entangler_circuits = BuildNonEntanglingCircuitsQutrit(selective_blocks, N)

    # Ausgabe
    type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]
    println("Wanted Indexes $wanted_indexes corresponds to these Selective Blocks: $selective_blocks\n")

    for (i, block) in enumerate(selective_blocks)
        block_types = digits(block, base=4, pad=N) |> reverse
        desc = join([type_names[t+1] for t in block_types], " × ")

        println("********************")
        println("Selective Block $block = $(join(block_types))₄  ($desc)")
        println("Wanted Elements:           ", block_map[block])
        println("Selective elements in block:", selective_elements[i])
        E, O = observable_sets[block]
        println("Pauli Observables:")
        println("  Even-Y set (E): ", E)
        println("  Odd-Y set  (O): ", O)
        println("SEEQST circuit text:")
        if entangler_circuits[i] == [""]
            println("  Circuit 1: (leer)")
        else
            println("  Circuit 1 (E): ", entangler_circuits[i][1])
            println("  Circuit 2 (O): ", entangler_circuits[i][2])
        end
        println("Non-entangling circuits:")
        for c in non_entangler_circuits[i]
            println("  ", c == "" ? "(leer)" : c)
        end
        println()
    end

    return selective_blocks, selective_elements, observable_sets,
           entangler_circuits, non_entangler_circuits, block_map
end

#=

println("═══ Tests GenerateExperimentQutrit ═══\n")

type_names = ["diag", "|0⟩↔|1⟩", "|0⟩↔|2⟩", "|1⟩↔|2⟩"]

# ── Test 1: N=2, diagonale Elemente ───────────────────────────
println("━"^60)
println("── Test 1: N=2, wanted=[(0,0)] ──\n")
blocks, elems, obs, ent, non_ent, bmap = GenerateExperimentQutrit([(0,0)], 2)
@assert blocks == [0]                    "Fehler: sollte Block 0 sein"
@assert (0,0) in elems[1]               "Fehler: (0,0) fehlt"
E, O = obs[0]
@assert "I⊗I" in E                      "Fehler: I⊗I fehlt"
@assert isempty(O)                       "Fehler: O sollte leer sein"
@assert ent[1] == [""]                   "Fehler: kein Circuit nötig"
println("✓ Test 1 bestanden\n")

# ── Test 2: N=2, |0⟩↔|1⟩ Übergang ────────────────────────────
println("━"^60)
println("── Test 2: N=2, wanted=[(0,1)] ──\n")
blocks, elems, obs, ent, non_ent, bmap = GenerateExperimentQutrit([(0,1)], 2)
@assert blocks == [1]                    "Fehler: sollte Block 1 sein"
@assert (0,1) in elems[1]               "Fehler: (0,1) fehlt"
E, O = obs[1]
@assert "I⊗L1" in E                     "Fehler: I⊗L1 fehlt"
@assert "I⊗L2" in O                     "Fehler: I⊗L2 fehlt"
@assert occursin("RL2", ent[1][1])       "Fehler: RL2 fehlt in Circuit 1"
@assert occursin("RL1", ent[1][2])       "Fehler: RL1 fehlt in Circuit 2"
@assert length(non_ent[1]) == 2          "Fehler: sollte 2 lokale Circuits haben"
println("✓ Test 2 bestanden\n")

# ── Test 3: N=2, beide off-diagonal |0⟩↔|1⟩ ──────────────────
println("━"^60)
println("── Test 3: N=2, wanted=[(0,4)] (Block 5=11₄) ──\n")
blocks, elems, obs, ent, non_ent, bmap = GenerateExperimentQutrit([(0,4)], 2)
@assert blocks == [5]                    "Fehler: sollte Block 5 sein"
E, O = obs[5]
@assert "L1⊗L1" in E                    "Fehler: L1⊗L1 fehlt"
@assert "L1⊗L2" in O                    "Fehler: L1⊗L2 fehlt"
@assert occursin("CINC", ent[1][1])      "Fehler: CINC fehlt"
@assert length(non_ent[1]) == 4          "Fehler: sollte 4 lokale Circuits haben"
println("✓ Test 3 bestanden\n")

# ── Test 4: N=2, volle erste Zeile ────────────────────────────
println("━"^60)
println("── Test 4: N=2, volle erste Zeile ──\n")
wanted = [(0,j) for j in 0:8]
blocks, elems, obs, ent, non_ent, bmap = GenerateExperimentQutrit(wanted, 2)
println("Selective blocks: ", sort(blocks))
println("Anzahl Blöcke:    ", length(blocks))
@assert length(blocks) == 9             "Fehler: sollte 9 Blöcke haben (erste Zeile)"
println("✓ Test 4 bestanden\n")

# ── Test 5: N=3, ausgewählte Elemente ─────────────────────────
println("━"^60)
println("── Test 5: N=3, wanted=[(0,4),(0,8),(0,13)] ──\n")
blocks, elems, obs, ent, non_ent, bmap = GenerateExperimentQutrit(
    [(0,4), (0,8), (0,13)], 3
)
println("Selective blocks: ", blocks)
println("✓ Test 5 bestanden\n")

println("═══ Alle Tests bestanden ✓ ═══")
=#

function CINCGate(n::Int, control::Int, target::Int)
    dim = 3^n
    gate = zeros(ComplexF64, dim, dim)

    for state in 0:(dim-1)
        # Extrahiere control trit
        control_trit = (state ÷ 3^(n - 1 - control)) % 3

        # Extrahiere target trit
        target_trit = (state ÷ 3^(n - 1 - target)) % 3

        # Neuer target trit: (target + control) mod 3
        new_target_trit = (target_trit + control_trit) % 3

        # Baue neuen Zustand
        new_state = state - target_trit  * 3^(n - 1 - target) +
                            new_target_trit * 3^(n - 1 - target)

        gate[new_state + 1, state + 1] = 1.0
    end

    return gate
end

function randomInitialTQutrit(N::Int)
    dim = 3^N
    return randn(dim, dim) + im * randn(dim, dim)
end

function densityMatrixFromTQutrit(T::Matrix{ComplexF64})
    AtA = T' * T
    return AtA / tr(AtA)
end

function GellMannRotation(k::Int)
    # Gell-Mann Matrizen
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
    # R_λk(π/2) = exp(-iπ/4 · λk)
    return exp(-im * π/4 * λ[k])
end

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

            else
                error("Unbekanntes Gate: $gate_name")
            end
        end

        push!(unitary_list, U)
    end

    return unitary_list
end


#=
println("═══ Tests ParseCircuitToMatrixQutrit ═══\n")

N = 2
dim = 3^N  # = 9

# ── Test 1: Leerer Schaltkreis → Einheitsmatrix ────────────────
println("── Test 1: Leerer Schaltkreis ──")
Us = ParseCircuitToMatrixQutrit([""], N)
@assert Us[1] ≈ Matrix{ComplexF64}(I, dim, dim)  "Fehler: sollte Einheitsmatrix sein"
println("✓ Test 1 bestanden\n")

# ── Test 2: Einzelne Rotation RL1 auf Qutrit 0 ────────────────
println("── Test 2: RL1 auf Qutrit 0 ──")
Us = ParseCircuitToMatrixQutrit(["(RL1:0)"], N)
U  = Us[1]
@assert size(U) == (dim, dim)    "Fehler: falsche Dimension"
@assert U * U' ≈ I               "Fehler: nicht unitär"
println("✓ Test 2 bestanden\n")

# ── Test 3: Alle Rotationen unitär ────────────────────────────
println("── Test 3: Alle Rotationen sind unitär ──")
for k in [1,2,4,5,6,7]
    Us = ParseCircuitToMatrixQutrit(["(RL$k:0)"], N)
    @assert Us[1] * Us[1]' ≈ I  "Fehler: RL$k nicht unitär"
    println("  RL$k ✓")
end
println("✓ Test 3 bestanden\n")

# ── Test 4: CINC Gate ──────────────────────────────────────────
println("── Test 4: CINC Gate ──")
Us = ParseCircuitToMatrixQutrit(["(CINC:0,1)"], N)
U  = Us[1]
@assert size(U) == (dim, dim)    "Fehler: falsche Dimension"
@assert U * U' ≈ I               "Fehler: CINC nicht unitär"
println("✓ Test 4 bestanden\n")

# ── Test 5: Kombination RL2 + CINC ────────────────────────────
println("── Test 5: (CINC:0,1)(RL2:0) ──")
Us = ParseCircuitToMatrixQutrit(["(CINC:0,1)(RL2:0)"], N)
U  = Us[1]
@assert size(U) == (dim, dim)    "Fehler: falsche Dimension"
@assert U * U' ≈ I               "Fehler: nicht unitär"
println("✓ Test 5 bestanden\n")

# ── Test 6: Mehrere Schaltkreise ──────────────────────────────
println("── Test 6: Mehrere Schaltkreise ──")
circuits = [
    "",
    "(RL2:0)",
    "(RL1:0)",
    "(CINC:0,1)(RL2:0)(RL2:1)",
    "(CINC:0,1)(RL1:0)(RL1:1)"
]
Us = ParseCircuitToMatrixQutrit(circuits, N)
@assert length(Us) == 5          "Fehler: sollte 5 Unitäre haben"
for (i, U) in enumerate(Us)
    @assert U * U' ≈ I           "Fehler: U[$i] nicht unitär"
end
println("✓ Test 6 bestanden\n")

# ── Test 7: CINC Funktionalität ──────────────────────────────
println("── Test 7: CINC Funktionalität ──")
# |10⟩ → |12⟩ (control=1, target=(0+1)mod3=1? Nein: target+control mod3)
# state |10⟩: control_trit=1, target_trit=0 → new_target=(0+1)mod3=1 → |11⟩
Us   = ParseCircuitToMatrixQutrit(["(CINC:0,1)"], N)
U    = Us[1]
# |10⟩ in Dezimal: 1*3 + 0 = 3 → Index 4
state_10 = zeros(ComplexF64, dim); state_10[4] = 1.0
result = U * state_10
# Erwartetes Ergebnis: |11⟩ = 1*3+1 = 4 → Index 5
@assert abs(result[5] - 1.0) < 1e-10  "Fehler: CINC |10⟩ → |11⟩"
println("  CINC: |10⟩ → |11⟩ ✓")

# |20⟩: control_trit=2, target_trit=0 → new_target=(0+2)mod3=2 → |22⟩
state_20 = zeros(ComplexF64, dim); state_20[7] = 1.0
result = U * state_20
# |22⟩ = 2*3+2 = 8 → Index 9
@assert abs(result[9] - 1.0) < 1e-10  "Fehler: CINC |20⟩ → |22⟩"
println("  CINC: |20⟩ → |22⟩ ✓")
println("✓ Test 7 bestanden\n")

println("═══ Alle Tests bestanden ✓ ═══")
=#

function PrepareStateQutrit(initial_text::String, N::Int)
    U_list = ParseCircuitToMatrixQutrit([initial_text], N)
    U      = U_list[1]

    # |00...0⟩ für Qutrits
    psi0    = zeros(ComplexF64, 3^N)
    psi0[1] = 1.0

    psi       = U * psi0
    rho_ideal = psi * psi'
    return rho_ideal
end
#=
println("═══ Erweiterte CINC Tests ═══\n")

N = 2
U_cinc = CINCGate(2, 0, 1)

# Hilfsfunktion: State-Index zu Trit-Darstellung
function state_to_trits(idx, N)
    digits(idx-1, base=3, pad=N) |> reverse |> join
end

# ── Test 1: CINC auf alle Basis-Zustände ──────────────────────
println("── Test 1: CINC(0→1) auf alle Basiszustände ──")
println("Input    Output   Erklärung")
println("─"^50)

state_names = ["|00⟩", "|01⟩", "|02⟩", "|10⟩", "|11⟩", "|12⟩", "|20⟩", "|21⟩", "|22⟩"]
expected    = [1,       2,       3,       5,       6,       4,       9,       7,       8]
# CINC: target = (target + control) mod 3
# |00⟩ → |00⟩, |01⟩ → |01⟩, |02⟩ → |02⟩
# |10⟩ → |11⟩, |11⟩ → |12⟩, |12⟩ → |10⟩
# |20⟩ → |22⟩, |21⟩ → |20⟩, |22⟩ → |21⟩

for i in 1:9
    state = zeros(ComplexF64, 9); state[i] = 1.0
    result = U_cinc * state
    out_idx = argmax(abs.(result))
    control = (i-1) ÷ 3
    target  = (i-1) % 3
    new_target = (target + control) % 3
    println("$(state_names[i]) → $(state_names[out_idx])   control=$control, target=$target → $(new_target)")
    @assert out_idx == expected[i]  "Fehler: $(state_names[i]) sollte zu $(state_names[expected[i]]) gehen"
end
println("✓ Test 1 bestanden\n")

# ── Test 2: CINC ist unitär ────────────────────────────────────
println("── Test 2: CINC ist unitär ──")
@assert U_cinc * U_cinc' ≈ I  "Fehler: nicht unitär"
@assert U_cinc' * U_cinc ≈ I  "Fehler: nicht unitär"
println("✓ Test 2 bestanden\n")

# ── Test 3: CINC³ = I ─────────────────────────────────────────
println("── Test 3: CINC³ = I ──")
# Dreimal CINC anwenden sollte Identität ergeben (mod 3)
U_cinc3 = U_cinc * U_cinc * U_cinc
@assert U_cinc3 ≈ Matrix{ComplexF64}(I, 9, 9)  "Fehler: CINC³ ≠ I"
println("✓ Test 3 bestanden\n")
# ── Test 4: CINC Debug ────────────────────────────────────────
println("── Test 4: CINC Debug ──")

# Was macht RL2 auf Qutrit 0?
U_rl2 = ParseCircuitToMatrixQutrit(["(RL2:0)"], 2)[1]
psi0 = zeros(ComplexF64, 9); psi0[1] = 1.0
psi = U_rl2 * psi0
println("RL2|00⟩ = ", round.(psi, digits=4))
println("Nicht-null: ", findall(x -> abs(x) > 0.01, psi))
println("Index 1 = |00⟩, Index 4 = |10⟩")
println()

# CINC(0→1) auf |10⟩
U_cinc = CINCGate(2, 0, 1)
state_10 = zeros(ComplexF64, 9); state_10[4] = 1.0
result = U_cinc * state_10
println("|10⟩ nach CINC(0→1): Index ", findall(x -> abs(x) > 0.01, result))

# CINC(1→0) auf |10⟩
U_cinc_10 = CINCGate(2, 1, 0)
result2 = U_cinc_10 * state_10
println("|10⟩ nach CINC(1→0): Index ", findall(x -> abs(x) > 0.01, result2))

println("✓ Test 4 bestanden\n")# ── Test 5: Voller Qutrit GHZ ─────────────────────────────────
println("── Test 5: Voller Qutrit GHZ (alle 3 Komponenten) ──")
# Für vollen GHZ braucht man RL2 + RL4 Kombination
# oder direkten Superpositions-Zustand
# Teste mit N=1 zuerst: RL2|0⟩ = (|0⟩ + |1⟩... nein
# Echter Qutrit GHZ: (|00⟩ + |11⟩ + |22⟩)/√3
# Braucht andere Vorbereitung - zeige was CINC mit |+⟩ macht

# Superposition |0⟩+|1⟩+|2⟩ auf Qutrit 0 via RL2 + RL5
rho1 = PrepareStateQutrit("(CINC:0,1)(RL5:0)(RL2:0)", N)
psi1 = ParseCircuitToMatrixQutrit(["(CINC:0,1)(RL5:0)(RL2:0)"], N)[1] * begin
    v = zeros(ComplexF64, 9); v[1] = 1.0; v
end
println("Zustand nach RL2+RL5+CINC:")
println("  ", round.(psi1, digits=3))
println("  Nicht-null: ", findall(x -> abs(x) > 0.01, psi1))
println("✓ Test 5 bestanden\n")

# ── Test 6: CINC(1→0) vs CINC(0→1) ───────────────────────────
println("── Test 6: CINC(0→1) vs CINC(1→0) ──")
U_cinc_01 = CINCGate(2, 0, 1)
U_cinc_10 = CINCGate(2, 1, 0)

# |10⟩: control=1, target=0
state = zeros(ComplexF64, 9); state[4] = 1.0

r01 = U_cinc_01 * state
r10 = U_cinc_10 * state

println("CINC(0→1)|10⟩ → $(state_names[argmax(abs.(r01))])")
println("CINC(1→0)|10⟩ → $(state_names[argmax(abs.(r10))])")
@assert argmax(abs.(r01)) != argmax(abs.(r10))  "Fehler: sollten verschieden sein"
println("✓ Test 6 bestanden\n")

# ── Test 7: N=3 CINC ──────────────────────────────────────────
println("── Test 7: N=3 CINC ──")
U_cinc_n3 = CINCGate(3, 0, 2)
@assert size(U_cinc_n3) == (27, 27)  "Fehler: falsche Dimension"
@assert U_cinc_n3 * U_cinc_n3' ≈ I  "Fehler: nicht unitär"

# |100⟩ → |102⟩? nein: target=2, control=1 → (0+1)%3=1 → |101⟩
# i=1*9+0*3+0=9 → Index 10
state_n3 = zeros(ComplexF64, 27); state_n3[10] = 1.0
result_n3 = U_cinc_n3 * state_n3
out_idx = argmax(abs.(result_n3))
i_trits = digits(out_idx-1, base=3, pad=3) |> reverse
println("|100⟩ → |$(join(i_trits))⟩ via CINC(0→2)")
println("✓ Test 7 bestanden\n")

println("═══ Alle Tests bestanden ✓ ═══")
=#

function DataPredictFromRhoSampledQutrit(
    rho::Matrix{ComplexF64},
    unitaries::Vector{Matrix{ComplexF64}},
    shots::Int
)
    all_counts = Vector{Vector{Int}}()

    for U in unitaries
        # pᵢ = diag(U · ρ · U†)
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

        push!(all_counts, counts)
    end

    return all_counts
end
#=
println("═══ Tests DataPredictFromRhoSampledQutrit ═══\n")

N    = 2
dim  = 3^N
shots = 10000

# ── Test 1: Reinzustand |00⟩ ──────────────────────────────────
println("── Test 1: Reinzustand |00⟩, Identität ──")
rho_00 = zeros(ComplexF64, dim, dim)
rho_00[1,1] = 1.0

Us = ParseCircuitToMatrixQutrit([""], N)
counts = DataPredictFromRhoSampledQutrit(rho_00, Us, shots)

println("Counts: ", counts[1])
println("Summe:  ", sum(counts[1]))
@assert sum(counts[1]) == shots       "Fehler: Summe ≠ shots"
@assert counts[1][1] == shots         "Fehler: alle Counts sollten bei Index 1 sein"
@assert all(counts[1][2:end] .== 0)  "Fehler: alle anderen sollten 0 sein"
println("✓ Test 1 bestanden\n")

# ── Test 2: Maximaler Mischzustand ────────────────────────────
println("── Test 2: Maximaler Mischzustand ──")
rho_mix = Matrix{ComplexF64}(I, dim, dim) / dim

Us = ParseCircuitToMatrixQutrit([""], N)
counts = DataPredictFromRhoSampledQutrit(rho_mix, Us, shots)

println("Counts: ", counts[1])
println("Summe:  ", sum(counts[1]))
@assert sum(counts[1]) == shots       "Fehler: Summe ≠ shots"
# Jeder Zustand sollte etwa shots/dim Counts haben
expected_per_state = shots / dim
for c in counts[1]
    @assert abs(c - expected_per_state) < 3*sqrt(expected_per_state)  "Fehler: Counts zu weit von Erwartung"
end
println("✓ Test 2 bestanden\n")

# ── Test 3: Summe immer = shots ────────────────────────────────
println("── Test 3: Summe immer = shots ──")
rho_rand = GenerateRandomDensityMatrixNoZerosQutrits(N)
circuits = [
    "",
    "(RL2:0)",
    "(RL1:0)",
    "(RL5:1)",
    "(CINC:0,1)(RL2:0)(RL2:1)",
    "(CINC:0,1)(RL1:0)(RL1:1)"
]
Us = ParseCircuitToMatrixQutrit(circuits, N)
counts = DataPredictFromRhoSampledQutrit(rho_rand, Us, shots)

for (i, c) in enumerate(counts)
    @assert sum(c) == shots  "Fehler: Circuit $i Summe ≠ shots"
    println("  Circuit $i: Summe = $(sum(c)) ✓")
end
println("✓ Test 3 bestanden\n")

# ── Test 4: Länge der Counts ───────────────────────────────────
println("── Test 4: Länge der Counts = dim ──")
for (i, c) in enumerate(counts)
    @assert length(c) == dim  "Fehler: Circuit $i Länge ≠ dim"
end
println("  Alle Counts haben Länge $dim ✓")
println("✓ Test 4 bestanden\n")

# ── Test 5: Keine negativen Counts ────────────────────────────
println("── Test 5: Keine negativen Counts ──")
for (i, c) in enumerate(counts)
    @assert all(c .>= 0)  "Fehler: negative Counts in Circuit $i"
end
println("✓ Test 5 bestanden\n")

# ── Test 6: Konsistenz mit exakten Wahrscheinlichkeiten ───────
println("── Test 6: Konsistenz mit exakten Wahrscheinlichkeiten ──")
rho_test = GenerateRandomDensityMatrixNoZerosQutrits(N)
U_test   = ParseCircuitToMatrixQutrit(["(RL2:0)"], N)
exact_probs = real(diag(U_test[1] * rho_test * U_test[1]'))

counts_test = DataPredictFromRhoSampledQutrit(rho_test, U_test, shots)
sampled_probs = counts_test[1] ./ shots

println("Exakte Wahrscheinlichkeiten: ", round.(exact_probs, digits=4))
println("Gesampelte Wahrscheinlichkeiten: ", round.(sampled_probs, digits=4))

# Statistische Abweichung sollte klein sein
for k in 1:dim
    diff = abs(exact_probs[k] - sampled_probs[k])
    @assert diff < 0.05  "Fehler: zu große Abweichung bei Index $k: $diff"
end
println("✓ Test 6 bestanden\n")

println("═══ Alle Tests bestanden ✓ ═══")
=#
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
    tol::Float64     = 1e-12
)
    # Verlustfunktion: pro Schaltkreis separat
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

    T = randomInitialTQutrit(N)
    L0, _ = Zygote.withgradient(loss, T)
    println("Loss initial: ", L0)

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

        if i % 500 == 0
            println("Iteration $i | Loss: $(round(L, digits=6))")
        end

        if length(losses) > 1
            rel = abs(losses[end] - losses[end-1]) / abs(losses[end-1])
            rel < 1e-8 ? no_improve += 1 : (no_improve = 0)
        end
        no_improve >= patience && break
    end

    rho_rec = T' * T
    rho_rec = rho_rec / tr(rho_rec)
    return rho_rec
end
println("═══ Fidelity Test N=2 Qutrits nach Fix ═══\n")

N   = 4
dim = 3^N

rho_true = GenerateRandomDensityMatrixNoZerosQutrits(N)
shots    = 10000 * dim

# ── SEEQST mit CINC ───────────────────────────────────────────
println("── SEEQST mit CINC ──")
blocks    = collect(0:(4^N - 1))
ent_circs = BuildEntanglerBlocksQutrit(blocks, N)

circuits_seeqst = String[]
for circ_group in ent_circs
    append!(circuits_seeqst, circ_group)
end

println("Anzahl Schaltkreise: ", length(circuits_seeqst))
println("Beispiel Block 5:    ", ent_circs[6])  # Block 5 = Index 6

Us_seeqst   = ParseCircuitToMatrixQutrit(circuits_seeqst, N)
data_seeqst = DataPredictFromRhoSampledQutrit(rho_true, Us_seeqst, shots)

rho_seeqst = ProcessDataQutrit(
    data_seeqst, Us_seeqst, blocks, shots, N;
    lr=0.1, decay=0.9999, iterations=5000, patience=200
)
F_seeqst = fidelity(rho_seeqst, rho_true)
println("Fidelität SEEQST:    ", round(F_seeqst, digits=4))

# ── Ohne CINC (lokal) ─────────────────────────────────────────
println("\n── Ohne CINC (lokal) ──")
non_ent_circs = BuildNonEntanglingCircuitsQutrit(blocks, N)

circuits_local = String[]
for circ_group in non_ent_circs
    append!(circuits_local, circ_group)
end

println("Anzahl Schaltkreise: ", length(circuits_local))

Us_local   = ParseCircuitToMatrixQutrit(circuits_local, N)
data_local = DataPredictFromRhoSampledQutrit(rho_true, Us_local, shots)

rho_local = ProcessDataQutrit(
    data_local, Us_local, blocks, shots, N;
    lr=0.1, decay=0.9999, iterations=5000, patience=200
)
F_local = fidelity(rho_local, rho_true)
println("Fidelität lokal:     ", round(F_local, digits=4))

# ── Vergleich ─────────────────────────────────────────────────
println("\n── Vergleich ──")
println(@sprintf("%-25s  %-15s  %-15s", "Methode", "Schaltkreise", "Fidelität"))
println("─"^55)
println(@sprintf("%-25s  %-15d  %-15.4f", "SEEQST (mit CINC)",  length(circuits_seeqst), F_seeqst))
println(@sprintf("%-25s  %-15d  %-15.4f", "Lokal (ohne CINC)",  length(circuits_local),  F_local))

@assert F_seeqst > 0.8  "Fehler: SEEQST Fidelität zu niedrig"
@assert F_local  > 0.8  "Fehler: Lokal Fidelität zu niedrig"
println("\n✓ Alle Tests bestanden")