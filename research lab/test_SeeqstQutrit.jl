include("sqt2.jl")
using .SeeqstQutrit
using LinearAlgebra
using Printf
using QuantumInformation

#=
println("═══ Hybrid Tomographie Tests ═══\n")

# ── N=2 ───────────────────────────────────────────────────────
rho_rec, rho_true, F2 = FullTomographyHybrid(2;
    shots_per_circuit = 10000,
    lr                = 0.1,
    decay             = 0.9999,
    iterations        = 3000,
    patience          = 200
)

# ── N=3 ───────────────────────────────────────────────────────
rho_rec, rho_true, F3 = FullTomographyHybrid(3;
    shots_per_circuit = 10000,
    lr                = 0.1,
    decay             = 0.9999,
    iterations        = 5000,
    patience          = 200
)

# ── Vergleich alle Methoden für N=2 ───────────────────────────
println("\n━"^50)
println("Vergleich alle Methoden für N=2")
println("━"^50)

N   = 4
dim = 3^N
rho_true = GenerateRandomDensityMatrixNoZerosQutrits(N)
shots    = 1000
blocks   = collect(0:(4^N-1))

# SEEQST mit CINC
ent_circs = BuildEntanglerBlocksQutrit(blocks, N)
circs_ent = unique(String[c for g in ent_circs for c in g])
Us_ent    = ParseCircuitToMatrixQutrit(circs_ent, N)
data_ent  = DataPredictFromRhoSampledQutrit(rho_true, Us_ent, shots)
rho_ent   = ProcessDataQutrit(data_ent, Us_ent, blocks, shots, N;
    iterations=3000, verbose=false)
F_ent = fidelity(rho_ent, rho_true)

# NonEntangling
non_circs = BuildNonEntanglingCircuitsQutrit(blocks, N)
circs_non = unique(String[c for g in non_circs for c in g])
Us_non    = ParseCircuitToMatrixQutrit(circs_non, N)
data_non  = DataPredictFromRhoSampledQutrit(rho_true, Us_non, shots)
rho_non   = ProcessDataQutrit(data_non, Us_non, blocks, shots, N;
    iterations=3000, verbose=false)
F_non = fidelity(rho_non, rho_true)

# Hybrid
hyb_circs = BuildHybridCircuitsQutrit(blocks, N)
circs_hyb = unique(String[
    startswith(c,"E:") || startswith(c,"O:") ? c[3:end] : c
    for g in hyb_circs for c in g
])
Us_hyb   = ParseCircuitToMatrixQutrit(circs_hyb, N)
data_hyb = DataPredictFromRhoSampledQutrit(rho_true, Us_hyb, shots)
rho_hyb  = ProcessDataQutrit(data_hyb, Us_hyb, blocks, shots, N;
    iterations=3000, verbose=false)
F_hyb = fidelity(rho_hyb, rho_true)

println()
println(@sprintf("%-25s  %-12s  %-10s",
    "Methode", "Circuits", "Fidelität"))
println("─"^50)
println(@sprintf("%-25s  %-12d  %-10.4f",
    "SEEQST (Entangling)",  length(circs_ent), F_ent))
println(@sprintf("%-25s  %-12d  %-10.4f",
    "Hybrid",               length(circs_hyb), F_hyb))
println(@sprintf("%-25s  %-12d  %-10.4f",
    "NonEntangling",        length(circs_non), F_non))
=#
function BuildOptimalCircuitsQutrit(N::Int)
    blocks  = collect(0:(4^N - 1))
    y_like  = ["RL2", "RL5", "RL7"]
    e_gate  = Dict(1 => "RL2", 2 => "RL5", 3 => "RL7")
    o_gate  = Dict(1 => "RL1", 2 => "RL4", 3 => "RL6")
    cinc_type = Dict(1 => "CINC", 2 => "CINC22", 3 => "CINC33")

    # Sortiere Blöcke nach Anzahl aktiver Qutrits (absteigend)
    # → höchste Blöcke zuerst
    sorted_blocks = sort(blocks, by = b -> begin
        bt = digits(b, base=4, pad=N) |> reverse
        -count(t -> t != 0, bt)  # negativ → absteigende Reihenfolge
    end)

    seen_rotations = Set{String}()  # gesehene Rotations-Kombinationen
    all_circuits   = String[]

    for block in sorted_blocks
        block_types    = digits(block, base=4, pad=N) |> reverse
        active_qutrits = [(i-1, t) for (i, t) in enumerate(block_types) if t != 0]

        isempty(active_qutrits) && continue

        # Gruppiere nach Übergangstyp
        type_groups = Dict{Int, Vector{Int}}()
        for (q, t) in active_qutrits
            if !haskey(type_groups, t)
                type_groups[t] = Int[]
            end
            push!(type_groups[t], q)
        end

        # Entangling Gates
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

        local_qutrit_types = [t for (q,t) in active_qutrits if q in local_qutrits]
        local_qutrit_idxs  = [q for (q,t) in active_qutrits if q in local_qutrits]

        rot_options = [[e_gate[t], o_gate[t]] for t in local_qutrit_types]
        rot_choices = Iterators.product(rot_options...)

        for choice in rot_choices
            # Rotations-Schlüssel: welche Rotation auf welchem Qutrit
            rot_key = join(["($gate:$q)" for (gate,q)
                            in zip(choice, local_qutrit_idxs)], "")

            # Wenn diese Rotation schon gesehen → redundant!
            rot_key ∈ seen_rotations && continue
            push!(seen_rotations, rot_key)

            n_y = count(r -> r in y_like, choice)
            local_gates = ["($gate:$q)" for (gate,q)
                           in zip(choice, local_qutrit_idxs)]
            circuit_str = join(vcat(reverse(entangling_gates), local_gates))
            push!(all_circuits, (n_y % 2 == 0 ? "E:" : "O:") * circuit_str)
        end
    end

    return all_circuits
end

println("═══ Test BuildOptimalCircuitsQutrit ═══\n")

for N in [2, 3]
    println("━"^50)
    println("N=$N")

    optimal  = BuildOptimalCircuitsQutrit(N)
    blocks   = collect(0:(4^N-1))

    hybrid_red = BuildHybridCircuitsQutrit(blocks, N)
    circs_red  = unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for g in hybrid_red for c in g if c != ""])

    non_ent   = BuildNonEntanglingCircuitsQutrit(blocks, N)
    circs_non = unique(String[c for g in non_ent for c in g if c != ""])

    ent       = BuildEntanglerBlocksQutrit(blocks, N)
    circs_ent = unique(String[c for g in ent for c in g if c != ""])

    optimal_clean = unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for c in optimal if c != ""])

    println(@sprintf("%-25s  %d", "Entangling (SEEQST):", length(circs_ent)))
    println(@sprintf("%-25s  %d", "Hybrid (mit Red.):",   length(circs_red)))
    println(@sprintf("%-25s  %d", "Optimal (ohne Red.):", length(optimal_clean)))
    println(@sprintf("%-25s  %d", "NonEntangling:",       length(circs_non)))
    println()
end

# Fidelity für N=2
println("━"^50)
println("Fidelity Vergleich N=2")
println("━"^50)

N        = 3
blocks   = collect(0:(4^N-1))
rho_true = GenerateRandomDensityMatrixNoZerosQutrits(N)
shots    = 5000

methods = Dict(
    "Entangling" => unique(String[c for g in
        BuildEntanglerBlocksQutrit(blocks,N) for c in g if c != ""]),
    "Hybrid"     => unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for g in BuildHybridCircuitsQutrit(blocks,N) for c in g if c != ""]),
    "Optimal"    => unique(String[
        startswith(c,"E:")||startswith(c,"O:") ? c[3:end] : c
        for c in BuildOptimalCircuitsQutrit(N) if c != ""]),
    "NonEnt"     => unique(String[c for g in
        BuildNonEntanglingCircuitsQutrit(blocks,N) for c in g if c != ""]),
)

println()
println(@sprintf("%-15s  %-10s  %-10s", "Methode", "Circuits", "Fidelität"))
println("─"^40)
for (name, circs) in sort(collect(methods), by=x->length(x[2]))
    Us   = ParseCircuitToMatrixQutrit(circs, N)
    data = DataPredictFromRhoSampledQutrit(rho_true, Us, shots)
    rho  = ProcessDataQutrit(data, Us, blocks, shots, N;
        iterations=3000, patience=200, verbose=false)
    F    = fidelity(rho, rho_true)
    println(@sprintf("%-15s  %-10d  %-10.4f", name, length(circs), F))
end