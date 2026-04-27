using LinearAlgebra
using Zygote
using Optimisers

include("/home/jesko/Documents/research lab/setup.jl") 

function GenerateRandomDensityMatrixNoZeros(n::Int)
    d = 2^n
    M   = randn(ComplexF64, d, d) + im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end



# --------------------------------------------------------------------
# CNOTGate
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# PrepareState
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# ProcessData
# --------------------------------------------------------------------
# Fix: kein conj
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
N = 4

println("── ProcessData N=4 ──")
rho_true = GenerateRandomDensityMatrixNoZeros(N)

# Automatisch generieren
blocks    = collect(0:(2^N-1))
ent_circs = BuildEntanglerBlocks(blocks, N)
circuits  = FlattenList(ent_circs)
println("Anzahl Schaltkreise: ", length(circuits))

Us_all = ParseCircuitToMatrix(circuits, N)
shots  = 1000 * 2^N
data   = DataPredictFromRhoSampled(rho_true, Us_all, shots)

rho_rec = ProcessData(
    data, Us_all, blocks, shots, N;
    iterations    = 3000,
    lr            = 0.1,
    decay         = 0.9999,  # ← langsamer sinkend
    patience      = 200,     # ← länger warten
    tol           = 1e-12    # ← strenger
)

F = real(tr(sqrt(sqrt(rho_true) * rho_rec * sqrt(rho_true))))^2
println("Fidelität: ", round(F, digits=4))