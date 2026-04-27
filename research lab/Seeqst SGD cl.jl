using LinearAlgebra
using StatsBase
using IterTools
using Zygote
using QuantumInformation
using Plots

# --------------------------------------------------------------------
# Generate all stabilizer groups
# --------------------------------------------------------------------
function GenerateSGroups(N::Int)
    groups = Vector{Vector{String}}()
    for mask in 0:(2^N - 1)
        S = Vector{String}(undef, N)
        for qubit in 1:N
            bit = (mask >> (N - qubit)) & 1
            S[qubit] = bit == 0 ? "IZ" : "XY"
        end
        push!(groups, S)
    end
    return groups
end

# --------------------------------------------------------------------
# Computational basis helpers
# --------------------------------------------------------------------
function basisVector(bits::Vector{Int})
    v = [1.0]
    for b in bits
        v = kron(v, b == 0 ? [1.0, 0.0] : [0.0, 1.0])
    end
    return v
end

bitFlip(bits::Vector{Int}) = [1 - b for b in bits]

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

# --------------------------------------------------------------------
# Permutation: canonical [offdiag | diag] → physical
# --------------------------------------------------------------------
function permuteToPhysical(state::AbstractVector, offdiagPos::Vector{Int}, diagPos::Vector{Int})
    perm = vcat(offdiagPos, diagPos)
    N = length(perm)
    reshaped = reshape(state, ntuple(_ -> 2, N)...)
    invp = invperm(perm)
    return vec(permutedims(reshaped, invp))
end

# --------------------------------------------------------------------
# Parse Pauli strings
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# Eigenstates E-Gruppe
# --------------------------------------------------------------------
function generateEigenstatesE(paulis::Vector{String})
    offdiagPos, diagPos = parsePaulis(paulis)
    M = length(offdiagPos)
    K = length(diagPos)
    states = Vector{Vector{ComplexF64}}()

    if M == 0
        for q in bitStrings(K)
            ψ = ComplexF64.(basisVector(q))
            push!(states, permuteToPhysical(ψ, Int[], diagPos))
        end
        return states
    end

    if K == 0
        p_bits = [p for p in bitStrings(M) if p[1] == 0]
        for p in p_bits
            pbar  = bitFlip(p)
            vp    = ComplexF64.(basisVector(p))
            vpbar = ComplexF64.(basisVector(pbar))
            plus  = (vp + vpbar) / sqrt(2)
            minus = (vp - vpbar) / sqrt(2)
            push!(states, permuteToPhysical(plus,  offdiagPos, Int[]))
            push!(states, permuteToPhysical(minus, offdiagPos, Int[]))
        end
        return states
    end

    p_bits = [p for p in bitStrings(M) if p[1] == 0]
    q_bits = bitStrings(K)
    for p in p_bits
        pbar  = bitFlip(p)
        vp    = ComplexF64.(basisVector(p))
        vpbar = ComplexF64.(basisVector(pbar))
        plus  = (vp + vpbar) / sqrt(2)
        minus = (vp - vpbar) / sqrt(2)
        for q in q_bits
            vq = ComplexF64.(basisVector(q))
            push!(states, permuteToPhysical(kron(plus,  vq), offdiagPos, diagPos))
            push!(states, permuteToPhysical(kron(minus, vq), offdiagPos, diagPos))
        end
    end
    return states
end

# --------------------------------------------------------------------
# Eigenstates O-Gruppe
# --------------------------------------------------------------------
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
        pbar    = bitFlip(p)
        vp      = ComplexF64.(basisVector(p))
        vpbar   = ComplexF64.(basisVector(pbar))
        plus_O  = (vp + im*vpbar) / sqrt(2)
        minus_O = (vp - im*vpbar) / sqrt(2)
        for q in q_bits
            vq = ComplexF64.(basisVector(q))
            push!(states, permuteToPhysical(kron(plus_O,  vq), offdiagPos, diagPos))
            push!(states, permuteToPhysical(kron(minus_O, vq), offdiagPos, diagPos))
        end
    end
    return states
end

# --------------------------------------------------------------------
# Projektoren
# --------------------------------------------------------------------
function ProjectorsFromEigenstates(eigs)
    isempty(eigs) && return Matrix{ComplexF64}[]
    return [ψ * ψ' for ψ in eigs]
end

# --------------------------------------------------------------------
# Messung simulieren
# --------------------------------------------------------------------
function simulateMeasurement(rho, projectors, n)
    isempty(projectors) && return Float64[]
    probs    = [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), n)
    counts   = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n
end

# --------------------------------------------------------------------
# Cholesky Parametrisierung
# --------------------------------------------------------------------
function randomInitialT(N::Int)
    dim = 2^N
    return randn(dim, dim) + im * randn(dim, dim)
end

function densityMatrixFromT(T::Matrix{ComplexF64})
    AtA = T' * T
    return AtA / tr(AtA)
end

# --------------------------------------------------------------------
# Verlustfunktion
# --------------------------------------------------------------------
function predictProbabilities(T, projectors)
    rho = T' * T
    rho = rho / tr(rho)
    return [real(tr(rho * P)) for P in projectors]
end

function logLoss(T, measured_probs, projectors)
    predicted = predictProbabilities(T, projectors)
    return -sum(measured_probs .* log.(predicted .+ 1e-8))
end

# --------------------------------------------------------------------
# SGD Loop mit Adam + Zygote
# --------------------------------------------------------------------
function sgdLoop(
    T_init, measured_probs, projectors;
    learning_rate = 0.1,    # ← neuer Default
    iterations    = 1000,
    decay         = 0.9999, # ← neuer Default
    patience      = 200,
    tol           = 1e-12
)
    T  = copy(T_init)
    losses = Float64[]
    no_improve_counter = 0

    # Adam Zustand
    m  = zeros(ComplexF64, size(T))
    v  = zeros(Float64,    size(T))
    β₁, β₂, ε_adam = 0.9, 0.999, 1e-8

    for i in 1:iterations

        # Exakter Gradient via Zygote
        grad = Zygote.gradient(
            T -> logLoss(T, measured_probs, projectors), T
        )[1]

        # Lernraten-Decay
        η = learning_rate * decay^i

        # Adam Update
        m  = β₁ .* m + (1 - β₁) .* grad
        v  = β₂ .* v + (1 - β₂) .* abs2.(grad)
        m̂  = m ./ (1 - β₁^i)
        v̂  = v ./ (1 - β₂^i)
        T  = T - η .* m̂ ./ (sqrt.(v̂) .+ ε_adam)

        # Loss tracken
        L = logLoss(T, measured_probs, projectors)
        push!(losses, L)

        # Fortschritt ausgeben
        if i % 500 == 0
            println("Iteration $i | Loss: $(round(L, digits=6))")
        end

        # Early Stopping
        if length(losses) > 1
            if abs(losses[end] - losses[end-1]) < tol
                no_improve_counter += 1
            else
                no_improve_counter = 0
            end
        end

        if no_improve_counter >= patience
            println("Early stopping bei Iteration $i | Loss: $(round(L, digits=6))")
            break
        end
    end

    return densityMatrixFromT(T), losses
end

# --------------------------------------------------------------------
# Zufällige Dichtematrix
# --------------------------------------------------------------------
function GenerateRandomDensityMatrixNoZeros(n::Int)
    d = 2^n
    M   = randn(ComplexF64, d, d) + im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end

# --------------------------------------------------------------------
# Volle Tomographie
# --------------------------------------------------------------------
function fullTomography(
    rho_true::Matrix{ComplexF64}, N::Int;
    shots         = 1000,
    iterations    = 1000,
    learning_rate = 0.1,
    decay         = 0.9999,  # ← das fehlte
    patience      = 200,
    tol           = 1e-12
)
    groups = GenerateSGroups(N)

    all_projectors = Vector{Vector{Matrix{ComplexF64}}}()
    all_measured   = Vector{Vector{Float64}}()

    for paulis in groups
        proj_E   = ProjectorsFromEigenstates(generateEigenstatesE(paulis))
        proj_O   = ProjectorsFromEigenstates(generateEigenstatesO(paulis))
        proj_all = vcat(proj_E, proj_O)
        measured = simulateMeasurement(rho_true, proj_all, shots)
        push!(all_projectors, proj_all)
        push!(all_measured,   measured)
    end

    projectors_flat = vcat(all_projectors...)
    measured_flat   = vcat(all_measured...)

    T0 = randomInitialT(N)
    rho_rec, losses = sgdLoop(
        T0, measured_flat, projectors_flat;
        learning_rate = learning_rate,
        iterations    = iterations,
        decay         = decay,          # ← weitergeben
        patience      = patience,
        tol           = tol
    )

    return rho_rec, losses, projectors_flat, measured_flat
end
# ====================================================================
# Beispiel-Aufruf
# ====================================================================
N        = 3
rho_true = GenerateRandomDensityMatrixNoZeros(N)

rho_rec, losses, proj_flat, meas_flat = fullTomography(
    rho_true, N;
    shots         = 100000,
    iterations    = 30000,
    learning_rate = 0.1,    # war 0.01 → 10x größer
    decay         = 0.9999, # war 0.999 → viel langsamer sinkend
    patience      = 200,
    tol           = 1e-12
)
# Fidelität
F = fidelity(rho_rec, rho_true)
println("\n── Fidelität: $(round(F, digits=4)) ──")

# Gradient Test
grad = Zygote.gradient(
    T -> logLoss(T, meas_flat, proj_flat), randomInitialT(N)
)[1]
println("Gradient Norm: $(round(norm(grad), digits=6))")
println("Gradient ist Nothing: $(isnothing(grad))")

# Loss Plot
plot(losses,
    xlabel = "Iteration",
    ylabel = "Loss",
    title  = "Konvergenz SGD + Cholesky",
    yscale = :log10,
    legend = false)

# Debug in process_data – direkt vor dem SGD Loop
print("=== DEBUG Python ===")
print("data shape:          ", data.shape)
print("data sum:            ", float(data.sum()))
print("data min:            ", float(data.min()))
print("data max:            ", float(data.max()))
print("Anzahl Unitaries:    ", len(unitaries_jnp))
print("Anzahl Datenpunkte:  ", data.size)
A = random_matrix_jax(N)
L0 = log_loss_function(A, data.flatten(), unitaries_jnp, shots)
print("Loss initial:        ", float(L0))