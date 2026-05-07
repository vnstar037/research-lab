using LinearAlgebra
using StatsBase
using IterTools
using Convex
using Printf
using SCS
using Zygote
using Optimisers

#eigenstates nochmal überprüfen, da sie nicht in der + dann - reihenfolge sind

include("setup.jl") 

function GenerateRandomDensityMatrixNoZeros(n::Int)
    d = 2^n
    M   = randn(ComplexF64, d, d) + im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end


function GenerateSGroups(N::Int)
    groups = Vector{Vector{String}}()

    for mask in 0:(2^N - 1)
        S = Vector{String}(undef, N)

        for qubit in 1:N
            # höchstwertiges Bit zuerst
            bit = (mask >> (N - qubit)) & 1
            S[qubit] = bit == 0 ? "IZ" : "XY"
        end

        push!(groups, S)
    end

    return groups
end

function basisVector(bits::Vector{Int})
    v = [1.0]
    for b in bits
        v = kron(v, b == 0 ? [1.0, 0.0] : [0.0, 1.0])
    end
    return v
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


function GenerateAllUnitaries(N::Int)
    groups = GenerateSGroups(N)
    all_unitaries = Vector{Matrix{ComplexF64}}()
    group_sizes   = Int[]

    for paulis in groups
        eigs_E = generateEigenstatesE(paulis)
        eigs_O = generateEigenstatesO(paulis)

        if isempty(eigs_O)
            # Eigenvektoren nach Index sortieren
            sorted_E = sort(eigs_E, by = e -> argmax(abs.(e)))
            U_E = EigenstatesToUnitary(sorted_E)
            push!(all_unitaries, U_E)
            push!(group_sizes, 1)
        else
            sorted_E = sort(eigs_E, by = e -> argmax(abs.(e)))
            sorted_O = sort(eigs_O, by = e -> argmax(abs.(e)))
            U_E = EigenstatesToUnitary(sorted_E)
            U_O = EigenstatesToUnitary(sorted_O)
            push!(all_unitaries, U_E)
            push!(all_unitaries, U_O)
            push!(group_sizes, 2)
        end
    end

    return all_unitaries, group_sizes
end

function EigenstatesToUnitary(eigenstates::Vector{Vector{ComplexF64}})
    U_dag = hcat(eigenstates...)  # Eigenvektoren als Spalten → das ist U†
    return Matrix{ComplexF64}(U_dag')  # U = (U†)†
end


bitFlip(bits::Vector{Int}) = [1 - b for b in bits]

# ============================================================
function bitStrings(n::Int)
    if n == 0
        return [Int[]]
    end
    prev = bitStrings(n - 1)
    res = Vector{Vector{Int}}()
    for p in prev
        push!(res, vcat(p, 0))
        push!(res, vcat(p, 1))
    end
    return res
end

# ============================================================
# Permutation: canonical [offdiag | diag] → physical
# ============================================================
function permuteToPhysical(state::AbstractVector, offdiagPos::Vector{Int}, diagPos::Vector{Int})
    perm = vcat(offdiagPos, diagPos)
    N = length(perm)
    reshaped = reshape(state, ntuple(_ -> 2, N)...)
    invp = invperm(perm)
    return vec(permutedims(reshaped, invp))
end

# ============================================================
# Parse Pauli strings
# ============================================================
function parsePaulis(paulis::Vector{String})
    offdiagPos = Int[]
    diagPos    = Int[]

    for (i, p) in enumerate(paulis)
        if p == "XY"
            push!(offdiagPos, i)
        elseif p == "IZ"
            push!(diagPos, i)
        else
            error("Unbekannter Pauli-String: $p")
        end
    end

    return offdiagPos, diagPos
end


function PrepareState(initial_text::String, N::Int)
    U_list = ParseCircuitToMatrix([initial_text], N)
    U      = U_list[1]

    # |00...0⟩
    psi0      = zeros(ComplexF64, 2^N)
    psi0[1]   = 1.0

    psi       = U * psi0
    rho_ideal = psi * psi'
    return rho_ideal
end

# --------------------------------------------------------------------
# FlattenList
# --------------------------------------------------------------------
# Richtig:
function FlattenList(nested::Vector{Vector{T}}) where {T}
    return vcat(nested...)
end
# --------------------------------------------------------------------
# BlockSeparator
# --------------------------------------------------------------------
function BlockSeparator(rho::Matrix{ComplexF64}, selective_blocks::Vector{Int}, N::Int)
    @assert all(0 .<= selective_blocks .< 2^N) "Alle Blöcke müssen < 2^N sein"

    result = zeros(ComplexF64, 2^N, 2^N)

    for block in selective_blocks
        block_bits = digits(block, base=2, pad=N) |> reverse

        for i in 0:(2^N - 1)
            i_bits = digits(i, base=2, pad=N) |> reverse
            for j in 0:(2^N - 1)
                j_bits = digits(j, base=2, pad=N) |> reverse

                # Prüfe ob (i,j) zu diesem Block gehört
                matches = all(
                    block_bits[k] == 1 ? i_bits[k] != j_bits[k] : i_bits[k] == j_bits[k]
                    for k in 1:N
                )

                if matches
                    result[i+1, j+1] += rho[i+1, j+1]
                end
            end
        end
    end

    return result
end

# --------------------------------------------------------------------
# DataPredictFromRho
# --------------------------------------------------------------------
function DataPredictFromRho(rho::Matrix{ComplexF64}, unitaries::Vector{Matrix{ComplexF64}}, shots::Int)
    predictions = Vector{Vector{Float64}}()

    for U in unitaries
        # pᵢ = diag(U · ρ · U†) · shots
        probs = real(diag(U * rho * U'))
        push!(predictions, probs .* shots)
    end

    return predictions
end

# --------------------------------------------------------------------
# DataPredictFromRhoSampled
# --------------------------------------------------------------------
function DataPredictFromRhoSampled(rho::Matrix{ComplexF64}, unitaries::Vector{Matrix{ComplexF64}}, shots::Int)
    samples = Vector{Vector{Int}}()

    for U in unitaries
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

        push!(samples, counts)
    end

    return samples
end

function ProcessData(
    data::Vector{Vector{Int}},
    unitaries::Vector{Matrix{ComplexF64}},
    selective_blocks::Vector{Int},
    shots::Int,
    N::Int;
    rho_ideal::Union{Matrix{ComplexF64}, Nothing} = nothing,
    lr::Float64      = 0.1,
    decay::Float64   = 0.999,
    iterations::Int  = 600,
    patience::Int    = 50,
    tol::Float64     = 1e-10
)
    measured_flat = Float64.(vcat(data...))

    function loss(T)
        rho       = T' * T
        rho       = rho / tr(rho)
        predicted = vcat([real(diag(U * rho * U')) for U in unitaries]...)
        return -sum(measured_flat .* log.(predicted .+ 1e-8)) / shots
    end

    T = randomInitialT(N)

    L0, _ = Zygote.withgradient(loss, T)
    println("=== DEBUG Julia ===")
    println("Loss initial: ", L0)

    m  = zeros(ComplexF64, size(T))
    v  = zeros(Float64,    size(T))
    β₁, β₂, ε_adam = 0.9, 0.999, 1e-8
    losses     = Float64[]
    no_improve = 0

    for i in 1:iterations
        L, grads = Zygote.withgradient(loss, T)
        g = grads[1]  # ← kein conj!

        η  = lr * decay^i
        m  = β₁ .* m + (1 - β₁) .* g
        v  = β₂ .* v + (1 - β₂) .* abs2.(g)
        m̂  = m ./ (1 - β₁^i)
        v̂  = v ./ (1 - β₂^i)
        T  = T - η .* m̂ ./ (sqrt.(v̂) .+ ε_adam)

        push!(losses, L)

        if i % 100 == 0
            println("Iteration $i | Loss: $(round(L, digits=6))")
        end

        if length(losses) > 1
            if abs(losses[end] - losses[end-1]) < tol
                no_improve += 1
            else
                no_improve = 0
            end
        end
        no_improve >= patience && break
    end

    rho_rec = T' * T
    rho_rec = rho_rec / tr(rho_rec)

    return BlockSeparator(rho_rec, selective_blocks, N)
end


function CNOTGate(n::Int, control::Int, target::Int)
    dim = 2^n
    gate = zeros(ComplexF64, dim, dim)

    for state in 0:(dim-1)
        control_bit = (state >> (n - 1 - control)) & 1
        if control_bit == 1
            # flippe target bit
            new_state = xor(state, 1 << (n - 1 - target))
        else
            new_state = state
        end
        gate[new_state + 1, state + 1] = 1.0
    end

    return gate
end

function randomInitialT(N::Int)
    dim = 2^N
    return randn(dim, dim) + im * randn(dim, dim)
end

function densityMatrixFromT(T::Matrix{ComplexF64})
    AtA = T' * T
    return AtA / tr(AtA)
end

# --------------------------------------------------------------------
# ParseCircuitToMatrix
# --------------------------------------------------------------------
function ParseCircuitToMatrix(text_circuits::Vector{String}, n::Int)
    unitary_list = Matrix{ComplexF64}[]

    for circuit in text_circuits
        U = Matrix{ComplexF64}(I, 2^n, 2^n)

        # Operationen extrahieren
        operations = split(circuit, ")")
        operations = [strip(op, ['(', ')']) for op in operations if !isempty(strip(op))]

        for op in operations
            parts     = split(op, ":")
            length(parts) < 2 && continue
            gate_name = String(parts[1])
            indices   = parse.(Int, split(parts[2], ","))

            if gate_name == "RX90"
                rx = [cos(π/4) -im*sin(π/4); -im*sin(π/4) cos(π/4)]
                gate = Matrix{ComplexF64}(I, 2^n, 2^n)
                # Tensorprodukt
                full = ones(ComplexF64, 1, 1)
                for q in 0:(n-1)
                    full = kron(full, q == indices[1] ? rx : Matrix{ComplexF64}(I, 2, 2))
                end
                U = full * U

            elseif gate_name == "RY90"
                ry = [cos(π/4) -sin(π/4); sin(π/4) cos(π/4)]
                full = ones(ComplexF64, 1, 1)
                for q in 0:(n-1)
                    full = kron(full, q == indices[1] ? ry : Matrix{ComplexF64}(I, 2, 2))
                end
                U = full * U

            elseif gate_name == "CNOT"
                U = CNOTGate(n, indices[1], indices[2]) * U

            elseif gate_name == "H"
                h = [1 1; 1 -1] / sqrt(2)
                full = ones(ComplexF64, 1, 1)
                for q in 0:(n-1)
                    full = kron(full, q == indices[1] ? h : Matrix{ComplexF64}(I, 2, 2))
                end
                U = full * U
            else
                error("Unbekanntes Gate: $gate_name")
            end
        end

        push!(unitary_list, U)
    end

    return unitary_list
end

#=
N = 2
rho_true = GenerateRandomDensityMatrixNoZeros(N)
shots    = 1000 * 2^N

# ── Methode 1: Text-Schaltkreise ──────────────────────────────
blocks    = collect(0:(2^N-1))
ent_circs = BuildEntanglerBlocks(blocks, N)
circuits  = FlattenList(ent_circs)
Us_text   = ParseCircuitToMatrix(circuits, N)

# ── Methode 2: Eigenvektoren ───────────────────────────────────
Us_eig, _ = GenerateAllUnitaries(N)

println("═══ Vergleich Unitäre Matrizen ═══\n")

for i in 1:length(Us_text)
    println("── U[$i] ──")
    println("Text-Schaltkreis:")
    display(round.(Us_text[i], digits=4))
    println("Eigenvektoren:")
    display(round.(Us_eig[i], digits=4))
    println("Differenz: ", round(norm(Us_text[i] - Us_eig[i]), digits=6))
    println()
end

N = 3

# Methode 1: Text-Schaltkreise
blocks    = collect(0:(2^N-1))
ent_circs = BuildEntanglerBlocks(blocks, N)
circuits  = FlattenList(ent_circs)
Us_text   = ParseCircuitToMatrix(circuits, N)

# Methode 2: Eigenvektoren
Us_eig, _ = GenerateAllUnitaries(N)

println("Anzahl Text-Unitäre: ", length(Us_text))
println("Anzahl Eig-Unitäre:  ", length(Us_eig))
println()

println("═══ Vergleich U† (Autor) vs Eigenvektoren als Spalten ═══\n")

for i in 1:length(Us_text)
    println("── U[$i] ──")
    
    println("U† Text-Schaltkreis:")
    display(round.(Us_text[i]', digits=4))
    
    println("U† Eigenvektoren (= hcat(eigenstates...)):")
    display(round.(Us_eig[i]', digits=4))
    
    diff = norm(Us_text[i] - Us_eig[i])
    println("Differenz U_text vs U_eig: ", round(diff, digits=6))
    println()
end

println(generateEigenstatesE(["IZ","IZ","XY"]))
=#


function displayEigenstatesE(paulis::Vector{String})
    eigs = generateEigenstatesE(paulis)
    N = length(paulis)
    
    println("E-Gruppe für Paulis: ", paulis)
    println()
    
    for (k, e) in enumerate(eigs)
        # Finde nicht-null Einträge
        terms = String[]
        for (i, coeff) in enumerate(e)
            if abs(coeff) > 1e-10
                # Basis-Zustand als Binärstring
                basis = "|" * join(digits(i-1, base=2, pad=N) |> reverse) * "⟩"
                
                # Koeffizient formatieren
                re = real(coeff)
                im_part = imag(coeff)
                
                if abs(im_part) < 1e-10
                    # Reell
                    if abs(re - 1/sqrt(2)) < 1e-10
                        c = "+1/√2"
                    elseif abs(re + 1/sqrt(2)) < 1e-10
                        c = "-1/√2"
                    elseif abs(re - 1.0) < 1e-10
                        c = "+1"
                    elseif abs(re + 1.0) < 1e-10
                        c = "-1"
                    else
                        c = @sprintf("%+.4f", re)
                    end
                else
                    # Komplex
                    if abs(im_part - 1/sqrt(2)) < 1e-10
                        c = "+i/√2"
                    elseif abs(im_part + 1/sqrt(2)) < 1e-10
                        c = "-i/√2"
                    else
                        c = @sprintf("%+.4f+%.4fi", re, im_part)
                    end
                end
                
                push!(terms, c * basis)
            end
        end
        println("|e$k⟩ = ", join(terms, " "))
    end
    println()
end



function displayEigenstatesO(paulis::Vector{String})
    eigs = generateEigenstatesO(paulis)
    N = length(paulis)
    
    println("O-Gruppe für Paulis: ", paulis)
    println()
    
    if isempty(eigs)
        println("(leer)")
        println()
        return
    end
    
    for (k, e) in enumerate(eigs)
        terms = String[]
        for (i, coeff) in enumerate(e)
            if abs(coeff) > 1e-10
                basis = "|" * join(digits(i-1, base=2, pad=N) |> reverse) * "⟩"
                
                re = real(coeff)
                im_part = imag(coeff)
                
                if abs(im_part) < 1e-10
                    if abs(re - 1/sqrt(2)) < 1e-10
                        c = "+1/√2"
                    elseif abs(re + 1/sqrt(2)) < 1e-10
                        c = "-1/√2"
                    elseif abs(re - 1.0) < 1e-10
                        c = "+1"
                    elseif abs(re + 1.0) < 1e-10
                        c = "-1"
                    else
                        c = @sprintf("%+.4f", re)
                    end
                else
                    if abs(im_part - 1/sqrt(2)) < 1e-10
                        c = "+i/√2"
                    elseif abs(im_part + 1/sqrt(2)) < 1e-10
                        c = "-i/√2"
                    else
                        c = @sprintf("%+.4f+%.4fi", re, im_part)
                    end
                end
                
                push!(terms, c * basis)
            end
        end
        println("|o$k⟩ = ", join(terms, " "))
    end
    println()
end

using Printf

groups = GenerateSGroups(2)

println("═══ Alle Eigenbasen für N=2 ═══\n")

for paulis in groups
    println("━"^40)
    displayEigenstatesE(paulis)
    displayEigenstatesO(paulis)
end


using Printf

function formatCoeff(coeff::ComplexF64)
    re = real(coeff)
    im_part = imag(coeff)
    
    if abs(coeff) < 1e-10
        return nothing
    end
    
    if abs(im_part) < 1e-10
        if abs(re - 1/sqrt(2)) < 1e-10
            return "+1/√2"
        elseif abs(re + 1/sqrt(2)) < 1e-10
            return "-1/√2"
        elseif abs(re - 1.0) < 1e-10
            return "+1"
        elseif abs(re + 1.0) < 1e-10
            return "-1"
        else
            return @sprintf("%+.4f", re)
        end
    else
        if abs(re) < 1e-10
            if abs(im_part - 1/sqrt(2)) < 1e-10
                return "+i/√2"
            elseif abs(im_part + 1/sqrt(2)) < 1e-10
                return "-i/√2"
            else
                return @sprintf("+%.4fi", im_part)
            end
        else
            return @sprintf("%+.4f%+.4fi", re, im_part)
        end
    end
end

function displayUnitaryInBraket(U::Matrix{ComplexF64}, N::Int, label::String)
    dim = 2^N
    println("── $label ──")
    println()
    
    for col in 1:dim
        basis_col = "|" * join(digits(col-1, base=2, pad=N) |> reverse) * "⟩"
        terms = String[]
        
        for row in 1:dim
            coeff = U[row, col]
            c = formatCoeff(coeff)
            if !isnothing(c)
                basis_row = "|" * join(digits(row-1, base=2, pad=N) |> reverse) * "⟩"
                push!(terms, c * basis_row)
            end
        end
        
        println("Spalte $col ($basis_col): ", join(terms, " "))
    end
    println()
end

# ── Hauptprogramm ──────────────────────────────────────────────
N = 2

blocks    = collect(0:(2^N-1))
ent_circs = BuildEntanglerBlocks(blocks, N)
circuits  = FlattenList(ent_circs)
Us_text   = ParseCircuitToMatrix(circuits, N)

println("═══ U und U† für N=2 in Braket-Notation ═══\n")

circuit_names = [
    "Block 0: Identität",
    "Block 1: (RY90:1) → E-Gruppe",
    "Block 1: (RX90:1) → O-Gruppe",
    "Block 2: (RY90:0) → E-Gruppe",
    "Block 2: (RX90:0) → O-Gruppe",
    "Block 3: (CNOT:0,1)(RY90:0) → E-Gruppe",
    "Block 3: (CNOT:0,1)(RX90:0) → O-Gruppe"
]

for (i, U) in enumerate(Us_text)
    println("━"^50)
    println("Schaltkreis: ", circuit_names[i])
    println()
    displayUnitaryInBraket(U,  N, "U")
    displayUnitaryInBraket(Matrix(U'), N, "U†")
end 