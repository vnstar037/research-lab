using LinearAlgebra
using StatsBase
using IterTools
using Convex
using SCS
using QuantumInformation

function simulateMeasurement(rho, projectors, n)
    isempty(projectors) && return Float64[]
    probs = [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), n)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n
end

function generateEigenstatesE(paulis::Vector{String})
    offdiagPos, diagPos = parsePaulis(paulis)

    M = length(offdiagPos)
    K = length(diagPos)

    states = Vector{Vector{ComplexF64}}()

    # --------------------------------------------------------
    # Case 1: M = 0 → pure computational basis
    # --------------------------------------------------------
    if M == 0
        for q in bitStrings(K)
            ψ = ComplexF64.(basisVector(q))
            push!(states, permuteToPhysical(ψ, Int[], diagPos))
        end
        return states
    end

    # --------------------------------------------------------
    # Case 2: M > 0, K = 0
    # --------------------------------------------------------
    if K == 0
        p_bits = [p for p in bitStrings(M) if p[1] == 0]

        for p in p_bits
            pbar = bitFlip(p)

            vp    = ComplexF64.(basisVector(p))
            vpbar = ComplexF64.(basisVector(pbar))

            plus  = (vp + vpbar) / sqrt(2)
            minus = (vp - vpbar) / sqrt(2)

            push!(states, permuteToPhysical(plus,  offdiagPos, Int[]))
            push!(states, permuteToPhysical(minus, offdiagPos, Int[]))
        end
        return states
    end

    # --------------------------------------------------------
    # Case 3: M > 0, K > 0
    # --------------------------------------------------------
    p_bits = [p for p in bitStrings(M) if p[1] == 0]
    q_bits = bitStrings(K)

    for p in p_bits
        pbar = bitFlip(p)

        vp    = ComplexF64.(basisVector(p))
        vpbar = ComplexF64.(basisVector(pbar))

        plus  = (vp + vpbar) / sqrt(2)
        minus = (vp - vpbar) / sqrt(2)

        for q in q_bits
            vq = ComplexF64.(basisVector(q))

            ψ_plus  = kron(plus,  vq)
            ψ_minus = kron(minus, vq)

            push!(states, permuteToPhysical(ψ_plus,  offdiagPos, diagPos))
            push!(states, permuteToPhysical(ψ_minus, offdiagPos, diagPos))
        end
    end

    return states
end

# ============================================================
function generateEigenstatesO(paulis::Vector{String})
    offdiagPos = [i for (i,p) in enumerate(paulis) if p == "XY"]
    diagPos    = [i for (i,p) in enumerate(paulis) if p == "IZ"]

    M = length(offdiagPos)
    K = length(diagPos)

    if M == 0
        return Vector{Vector{ComplexF64}}()
    end

    states = Vector{Vector{ComplexF64}}()
    q_bits = bitStrings(K)

    p_bits = [p for p in bitStrings(M) if p[1] == 0]

    for p in p_bits
        pbar = bitFlip(p)

        vp    = ComplexF64.(basisVector(p))
        vpbar = ComplexF64.(basisVector(pbar))

        # O-Basis: ungerade Anzahl σy → beide Varianten
        plus_O  = (vp + im*vpbar)/sqrt(2)
        minus_O = (vp - im*vpbar)/sqrt(2)

        for q in q_bits
            vq = ComplexF64.(basisVector(q))
            ψ_plus  = kron(plus_O, vq)
            ψ_minus = kron(minus_O, vq)
            push!(states, permuteToPhysical(ψ_plus, offdiagPos, diagPos))
            push!(states, permuteToPhysical(ψ_minus, offdiagPos, diagPos))
        end
    end

    return states
end

function ProjectorsFromEigenstates(eigs)
    isempty(eigs) && return Matrix{ComplexF64}[]
    return [ψ * ψ' for ψ in eigs]
end

function GetSelectiveBlocks(N::Int, wanted_indexes::Vector{Tuple{Int,Int}})
    max_index = 2^N
    selective_blocks = Int[]
    block_map = Dict{Int, Vector{Tuple{Int,Int}}}()

    for (i, j) in wanted_indexes
        # Prüfe gültige Indizes
        @assert (0 <= i < max_index && 0 <= j < max_index) "Index ($i,$j) außerhalb des gültigen Bereichs [0, $(max_index-1)]"

        # XOR → Block-Nummer
        block = xor(i, j)

        # Block zur Liste hinzufügen falls noch nicht vorhanden
        if block ∉ selective_blocks
            push!(selective_blocks, block)
        end

        # block_map aktualisieren
        if !haskey(block_map, block)
            block_map[block] = Tuple{Int,Int}[]
        end
        push!(block_map[block], (i, j))
    end

    return selective_blocks, block_map
end

function GenerateSelectiveElements(selective_blocks::Vector{Int}, N::Int)
    """
    Generiert alle 2ᴺ Elemente eines Subsets durch {I,X}⊗N Transformationen.
    
    Beispiel: N=2, Block 3 = 11
      Startelement: (0, 3) = (00, 11)
      mask=00: (00,11) → (0,3)
      mask=01: (01,10) → (1,2)
      mask=10: (10,01) → (2,1)
      mask=11: (11,00) → (3,0)
    """
    selective_elements = Vector{Vector{Tuple{Int,Int}}}()

    for block in selective_blocks
        # Binärdarstellung des Blocks mit N Stellen
        block_bits = digits(block, base=2, pad=N) |> reverse

        # Startelement: (0, block)
        zero_bits  = zeros(Int, N)
        block_bits_copy = copy(block_bits)

        group = Tuple{Int,Int}[]

        # Wende alle 2ᴺ möglichen Masken an
        for mask in 0:(2^N - 1)
            mask_bits = digits(mask, base=2, pad=N) |> reverse

            # XOR: flippe Bits wo Maske = 1
            new_state = [xor(zero_bits[k],  mask_bits[k]) for k in 1:N]
            new_block = [xor(block_bits[k], mask_bits[k]) for k in 1:N]

            # Binär → Dezimal
            i = sum(new_state[k] * 2^(N-k) for k in 1:N)
            j = sum(new_block[k] * 2^(N-k) for k in 1:N)

            push!(group, (i, j))
        end

        push!(selective_elements, group)
    end

    return selective_elements
end

function GenerateObservableSets(selective_blocks::Vector{Int}, N::Int)
    observable_sets = Dict{Int, Tuple{Vector{String}, Vector{String}}}()

    for block in selective_blocks
        block_bits = digits(block, base=2, pad=N) |> reverse

        # Pro Qubit: Bit=1 → (X,Y), Bit=0 → (I,Z)
        pauli_options = [bit == 1 ? ["X", "Y"] : ["I", "Z"] for bit in block_bits]

        # Alle 2ᴺ Kombinationen
        all_observables = [join(combo) for combo in Iterators.product(pauli_options...)]
        all_observables = vec(all_observables)

        # E: gerade Anzahl Y, O: ungerade Anzahl Y
        even_set = [obs for obs in all_observables if count(==('Y'), obs) % 2 == 0]
        odd_set  = [obs for obs in all_observables if count(==('Y'), obs) % 2 == 1]

        observable_sets[block] = (even_set, odd_set)
    end

    return observable_sets
end


function BuildEntanglerBlocks(selective_blocks::Vector{Int}, N::Int)
    all_sequences = Vector{Vector{String}}()

    for block in selective_blocks
        block_bits   = digits(block, base=2, pad=N) |> reverse
        active_qubits = [i-1 for (i, bit) in enumerate(block_bits) if bit == 1]

        # Block 0: keine Gates nötig
        if isempty(active_qubits)
            push!(all_sequences, [""])
            continue
        end

        sequence = String[]

        # Schritt 1: RY90 auf erstem aktiven Qubit
        push!(sequence, "(RY90:$(active_qubits[1]))")

        head = [active_qubits[1]]
        tail = active_qubits[2:end]

        # Schritt 2: GHZ parallel aufbauen
        while !isempty(tail)
            new_tail = Int[]
            for h in head
                isempty(tail) && break
                tgt = popfirst!(tail)
                push!(sequence, "(CNOT:$h,$tgt)")
                push!(new_tail, tgt)
            end
            append!(head, new_tail)
        end

        # Schritt 3: RX90 Version
        rx_sequence = [replace(g, "RY90" => "RX90") for g in sequence]

        # Schritt 4: Umgekehrte Reihenfolge
        ry_str = join(reverse(sequence))
        rx_str = join(reverse(rx_sequence))

        push!(all_sequences, [ry_str, rx_str])
    end

    return all_sequences
end


function BuildNonEntanglingCircuits(selective_blocks::Vector{Int}, N::Int)
    all_block_circuits = Vector{Vector{String}}()

    for block in selective_blocks
        block_bits    = digits(block, base=2, pad=N) |> reverse
        active_qubits = [i-1 for (i, bit) in enumerate(block_bits) if bit == 1]

        # Alle 2ᴹ Kombinationen von RY90/RX90
        gate_choices = Iterators.product(fill(["RY90", "RX90"], length(active_qubits))...)

        circuits = String[]
        for choice in gate_choices
            circuit_str = join(["($gate:$q)" for (gate, q) in zip(choice, active_qubits)])
            push!(circuits, circuit_str)
        end

        push!(all_block_circuits, circuits)
    end

    return all_block_circuits
end

function GenerateExperiment(wanted_indexes::Vector{Tuple{Int,Int}}, N::Int)
    
    # Schritt 1: Selective Blocks bestimmen
    selective_blocks, block_map = GetSelectiveBlocks(N, wanted_indexes)
    
    # Schritt 2: Selective Elements generieren
    selective_elements = GenerateSelectiveElements(selective_blocks, N)
    
    # Schritt 3: Observable Sets generieren
    observable_sets = GenerateObservableSets(selective_blocks, N)
    
    # Schritt 4: Schaltkreise generieren
    entangler_circuits     = BuildEntanglerBlocks(selective_blocks, N)
    non_entangler_circuits = BuildNonEntanglingCircuits(selective_blocks, N)

    # Ausgabe
    println("Wanted Indexes $wanted_indexes corresponds to these Selective Blocks: $selective_blocks\n")

    for (i, block) in enumerate(selective_blocks)
        println("********************")
        println("Selective Block $block")
        println("Wanted Elements: ", block_map[block])
        println("Selective elements in block: ", selective_elements[i])
        E, O = observable_sets[block]
        println("Pauli Observables:")
        println("  Even-Y set (E): ", E)
        println("  Odd-Y set  (O): ", O)
        println("SEEQST circuit text:")
        if entangler_circuits[i] == [""]
            println("  Circuit 1: ")
        else
            println("  Circuit 1: ", entangler_circuits[i][1])
            println("  Circuit 2: ", entangler_circuits[i][2])
        end
        println("Non-entangling circuits:")
        println("  ", non_entangler_circuits[i])
        println()
    end

    return selective_blocks, selective_elements, observable_sets,
           entangler_circuits, non_entangler_circuits, block_map
end

#=
println("═══ Test GetSelectiveBlocks ═══\n")

# ── Test 1: Beispiel aus dem Paper (N=2) ──────────────────────
println("── Test 1: N=2, wanted=[(0,0), (0,3)] ──")
blocks, bmap = GetSelectiveBlocks(2, [(0,0), (0,3)])
println("Selective blocks: ", blocks)
println("Block map:        ", bmap)
@assert blocks == [0, 3]          "Fehler: falsche Blöcke"
@assert haskey(bmap, 0)           "Fehler: Block 0 fehlt"
@assert haskey(bmap, 3)           "Fehler: Block 3 fehlt"
@assert (0,0) in bmap[0]          "Fehler: (0,0) nicht in Block 0"
@assert (0,3) in bmap[3]          "Fehler: (0,3) nicht in Block 3"
println("✓ Test 1 bestanden\n")

# ── Test 2: Volle erste Zeile (N=2) ───────────────────────────
println("── Test 2: N=2, volle erste Zeile [(0,j) für j=0..3] ──")
wanted = [(0,j) for j in 0:3]
blocks, bmap = GetSelectiveBlocks(2, wanted)
println("Selective blocks: ", sort(blocks))
println("Block map:        ", bmap)
@assert sort(blocks) == [0, 1, 2, 3]  "Fehler: nicht alle 4 Blöcke"
println("✓ Test 2 bestanden\n")

# ── Test 3: XOR Logik prüfen ──────────────────────────────────
println("── Test 3: XOR Logik ──")
# (0,1): 00 XOR 01 = 01 = 1 → Block 1
# (2,3): 10 XOR 11 = 01 = 1 → auch Block 1!
blocks, bmap = GetSelectiveBlocks(2, [(0,1), (2,3)])
println("Selective blocks: ", blocks)
println("Block map:        ", bmap)
@assert blocks == [1]              "Fehler: beide sollten Block 1 sein"
@assert length(bmap[1]) == 2      "Fehler: Block 1 sollte 2 Einträge haben"
println("✓ Test 3 bestanden\n")

# ── Test 4: N=3 ───────────────────────────────────────────────
println("── Test 4: N=3, wanted=[(0,7)] ──")
# 0 = 000, 7 = 111 → XOR = 111 = 7 → Block 7
blocks, bmap = GetSelectiveBlocks(3, [(0,7)])
println("Selective blocks: ", blocks)
@assert blocks == [7]              "Fehler: sollte Block 7 sein"
println("✓ Test 4 bestanden\n")

# ── Test 5: Ungültiger Index ──────────────────────────────────
println("── Test 5: Ungültiger Index ──")
try
    GetSelectiveBlocks(2, [(0, 5)])  # 5 > 2^2-1 = 3
    println("✗ Fehler: Exception hätte ausgelöst werden sollen")
catch e
    println("✓ Test 5 bestanden: Exception korrekt ausgelöst")
    println("  Exception: ", e)
end

println("\n═══ Alle Tests bestanden ✓ ═══")


println("═══ Test GenerateSelectiveElements ═══\n")

# ── Test 1: Block 3 (N=2) ─────────────────────────────────────
println("── Test 1: N=2, Block 3 = 11 ──")
result = GenerateSelectiveElements([3], 2)
println("Elemente: ", result[1])
@assert Set(result[1]) == Set([(0,3), (1,2), (2,1), (3,0)]) "Fehler: falsche Elemente"
@assert length(result[1]) == 4                               "Fehler: sollte 4 Elemente haben"
println("✓ Test 1 bestanden\n")

# ── Test 2: Block 0 (N=2) ─────────────────────────────────────
println("── Test 2: N=2, Block 0 = 00 (diagonal) ──")
result = GenerateSelectiveElements([0], 2)
println("Elemente: ", result[1])
@assert Set(result[1]) == Set([(0,0), (1,1), (2,2), (3,3)]) "Fehler: sollte Diagonale sein"
println("✓ Test 2 bestanden\n")

# ── Test 3: Block 1 (N=2) ─────────────────────────────────────
println("── Test 3: N=2, Block 1 = 01 ──")
result = GenerateSelectiveElements([1], 2)
println("Elemente: ", result[1])
@assert Set(result[1]) == Set([(0,1), (1,0), (2,3), (3,2)]) "Fehler: falsche Elemente"
println("✓ Test 3 bestanden\n")

# ── Test 4: Mehrere Blöcke (N=2) ──────────────────────────────
println("── Test 4: N=2, alle Blöcke [0,1,2,3] ──")
result = GenerateSelectiveElements([0,1,2,3], 2)
println("Anzahl Blöcke: ", length(result))

# Zusammen müssen alle 16 Elemente der 4×4 Matrix abgedeckt sein
all_elements = Set(vcat(result...))
expected = Set([(i,j) for i in 0:3 for j in 0:3])
@assert all_elements == expected "Fehler: nicht alle Elemente abgedeckt"
@assert length(result) == 4     "Fehler: sollte 4 Blöcke haben"
println("✓ Test 4 bestanden: alle 16 Elemente abgedeckt\n")

# ── Test 5: N=3, Block 7 ──────────────────────────────────────
println("── Test 5: N=3, Block 7 = 111 ──")
result = GenerateSelectiveElements([7], 3)
println("Anzahl Elemente: ", length(result[1]))
@assert length(result[1]) == 8  "Fehler: sollte 8 Elemente haben"
@assert (0,7) in result[1]      "Fehler: (0,7) sollte enthalten sein"
@assert (7,0) in result[1]      "Fehler: (7,0) sollte enthalten sein"
println("✓ Test 5 bestanden\n")

println("═══ Alle Tests bestanden ✓ ═══")


obs = GenerateObservableSets([1], 2)
E, O = obs[1]
println("Block 1 - E: ", E)
println("Block 1 - O: ", O)

println("═══ Tests Julia ═══\n")

# ────────────────────────────────────────────────────────────
# GenerateObservableSets
# ────────────────────────────────────────────────────────────
println("── GenerateObservableSets ──")

# Test 1: Block 0
obs = GenerateObservableSets([0], 2)
E, O = obs[0]
println("Block 0 - E: ", E)
println("Block 0 - O: ", O)
@assert Set(E) == Set(["II", "IZ", "ZI", "ZZ"]) "Fehler Block 0 E"
@assert isempty(O)                                "Fehler Block 0 O"
println("✓ Block 0 bestanden\n")

# Test 2: Block 3
obs = GenerateObservableSets([3], 2)
E, O = obs[3]
println("Block 3 - E: ", E)
println("Block 3 - O: ", O)
@assert Set(E) == Set(["XX", "YY"]) "Fehler Block 3 E"
@assert Set(O) == Set(["XY", "YX"]) "Fehler Block 3 O"
println("✓ Block 3 bestanden\n")

# Test 3: Block 1
obs = GenerateObservableSets([1], 2)
E, O = obs[1]
println("Block 1 - E: ", E)
println("Block 1 - O: ", O)
@assert Set(E) == Set(["IX", "ZX"]) "Fehler Block 1 E"
@assert Set(O) == Set(["IY", "ZY"]) "Fehler Block 1 O"
println("✓ Block 1 bestanden\n")

# ────────────────────────────────────────────────────────────
# BuildEntanglerBlocks
# ────────────────────────────────────────────────────────────
println("── BuildEntanglerBlocks ──")

# Test 1: Block 0
seq = BuildEntanglerBlocks([0], 2)
println("Block 0: ", seq[1])
@assert seq[1] == [""] "Fehler: Block 0 sollte leer sein"
println("✓ Block 0 bestanden\n")

# Test 2: Block 3
seq = BuildEntanglerBlocks([3], 2)
println("Block 3 Circuit 1 (RY): ", seq[1][1])
println("Block 3 Circuit 2 (RX): ", seq[1][2])
@assert seq[1][1] == "(CNOT:0,1)(RY90:0)" "Fehler Circuit 1"
@assert seq[1][2] == "(CNOT:0,1)(RX90:0)" "Fehler Circuit 2"
println("✓ Block 3 bestanden\n")

# Test 3: N=3, Block 7
seq = BuildEntanglerBlocks([7], 3)
println("Block 7 Circuit 1: ", seq[1][1])
println("Block 7 Circuit 2: ", seq[1][2])
@assert seq[1][1] == "(CNOT:0,2)(CNOT:0,1)(RY90:0)" "Fehler Circuit 1"
@assert seq[1][2] == "(CNOT:0,2)(CNOT:0,1)(RX90:0)" "Fehler Circuit 2"
println("✓ Block 7 bestanden\n")

# ────────────────────────────────────────────────────────────
# BuildNonEntanglingCircuits
# ────────────────────────────────────────────────────────────
println("── BuildNonEntanglingCircuits ──")

# Test 1: Block 0
circ = BuildNonEntanglingCircuits([0], 2)
println("Block 0: ", circ[1])
@assert circ[1] == [""] "Fehler: Block 0 sollte leer sein"
println("✓ Block 0 bestanden\n")

# Test 2: Block 3
circ = BuildNonEntanglingCircuits([3], 2)
println("Block 3: ", circ[1])
@assert length(circ[1]) == 4                             "Fehler: sollte 4 Kombinationen haben"
@assert Set(circ[1]) == Set(["(RY90:0)(RY90:1)",
                              "(RY90:0)(RX90:1)",
                              "(RX90:0)(RY90:1)",
                              "(RX90:0)(RX90:1)"]) "Fehler: falsche Kombinationen"
println("✓ Block 3 bestanden\n")

# Test 3: Block 1
circ = BuildNonEntanglingCircuits([1], 2)
println("Block 1: ", circ[1])
@assert length(circ[1]) == 2                      "Fehler: sollte 2 Kombinationen haben"
@assert Set(circ[1]) == Set(["(RY90:1)", "(RX90:1)"]) "Fehler: falsche Kombinationen"
println("✓ Block 1 bestanden\n")

# Test 4: N=3, Block 7
circ = BuildNonEntanglingCircuits([7], 3)
println("Block 7 Anzahl Circuits: ", length(circ[1]))
@assert length(circ[1]) == 8 "Fehler: sollte 8 Kombinationen haben"
println("✓ Block 7 bestanden\n")

println("═══ Alle Julia Tests bestanden ✓ ═══")
=#

println("═══ Tests GenerateExperiment Julia ═══\n")

# ── Test 1: N=2, wanted=[(0,0),(0,3)] ─────────────────────────
println("── Test 1: N=2, wanted=[(0,0),(0,3)] ──\n")
blocks, elements, obs, ent, non_ent, bmap = GenerateExperiment([(0,0),(0,3)], 2)

@assert blocks == [0, 3]                                    "Fehler: falsche Blöcke"
@assert Set(elements[1]) == Set([(0,0),(1,1),(2,2),(3,3)]) "Fehler: Block 0 Elemente"
@assert Set(elements[2]) == Set([(0,3),(1,2),(2,1),(3,0)]) "Fehler: Block 3 Elemente"
E0, O0 = obs[0]
@assert Set(E0) == Set(["II","IZ","ZI","ZZ"])              "Fehler: Block 0 E"
@assert isempty(O0)                                        "Fehler: Block 0 O"
E3, O3 = obs[3]
@assert Set(E3) == Set(["XX","YY"])                        "Fehler: Block 3 E"
@assert Set(O3) == Set(["XY","YX"])                        "Fehler: Block 3 O"
@assert ent[1] == [""]                                     "Fehler: Block 0 Circuit"
@assert ent[2][1] == "(CNOT:0,1)(RY90:0)"                 "Fehler: Block 3 Circuit 1"
@assert ent[2][2] == "(CNOT:0,1)(RX90:0)"                 "Fehler: Block 3 Circuit 2"
@assert non_ent[1] == [""]                                 "Fehler: Block 0 Non-Ent"
@assert length(non_ent[2]) == 4                            "Fehler: Block 3 Non-Ent Anzahl"
println("✓ Test 1 bestanden\n")

# ── Test 2: N=2, volle Tomographie ────────────────────────────
println("── Test 2: N=2, volle erste Zeile ──\n")
wanted = [(0,j) for j in 0:3]
blocks, elements, obs, ent, non_ent, bmap = GenerateExperiment(wanted, 2)

@assert sort(blocks) == [0,1,2,3]  "Fehler: nicht alle 4 Blöcke"
@assert length(elements) == 4     "Fehler: sollte 4 Element-Gruppen haben"

# Alle 16 Elemente abgedeckt?
all_elements = Set(e for group in elements for e in group)
expected     = Set((i,j) for i in 0:3 for j in 0:3)
@assert all_elements == expected   "Fehler: nicht alle 16 Elemente abgedeckt"
println("✓ Test 2 bestanden\n")

# ── Test 3: N=3, wanted=[(0,7)] ───────────────────────────────
println("── Test 3: N=3, wanted=[(0,7)] ──\n")
blocks, elements, obs, ent, non_ent, bmap = GenerateExperiment([(0,7)], 3)

@assert blocks == [7]              "Fehler: sollte Block 7 sein"
@assert length(elements[1]) == 8  "Fehler: Block 7 sollte 8 Elemente haben"
@assert (0,7) in elements[1]      "Fehler: (0,7) sollte in Block 7 sein"
E7, O7 = obs[7]
@assert length(E7) + length(O7) == 8  "Fehler: sollte 8 Observablen haben"
@assert length(non_ent[1]) == 8       "Fehler: sollte 8 Non-Ent Circuits haben"
println("✓ Test 3 bestanden\n")

println("═══ Alle Julia Tests bestanden ✓ ═══")