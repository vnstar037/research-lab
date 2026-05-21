module LineareInversionQutrit

using LinearAlgebra
using StatsBase

export RecreatingDensityMatrixWithLineareInversionQutrit

# ---------------------------------------------------------
# 1) Gell-Mann Matrizen + Identität
# ---------------------------------------------------------

function gell_mann_matrices()
    λ = Dict{Int, Matrix{ComplexF64}}()

    λ[0] = [1 0 0; 0 1 0; 0 0 1] .+ 0im  # I

    λ[1] = [0 1 0; 1 0 0; 0 0 0] .+ 0im
    λ[2] = [0 -1im 0; 1im 0 0; 0 0 0]
    λ[3] = [1 0 0; 0 -1 0; 0 0 0] .+ 0im
    λ[4] = [0 0 1; 0 0 0; 1 0 0] .+ 0im
    λ[5] = [0 0 -1im; 0 0 0; 1im 0 0]
    λ[6] = [0 0 0; 0 0 1; 0 1 0] .+ 0im
    λ[7] = [0 0 0; 0 0 -1im; 0 1im 0]
    λ[8] = [1 0 0; 0 1 0; 0 0 -2] ./ sqrt(3) .+ 0im

    return λ
end

# ---------------------------------------------------------
# 2) Messung simulieren
# ---------------------------------------------------------

function simulate_measurement(rho::Matrix{ComplexF64},
                               U::Matrix{ComplexF64},
                               shots::Int)
    """
    Simuliert eine Messung in der Rechenbasis nach Rotation U.
    Gibt Wahrscheinlichkeiten zurück.
    """
    dim  = size(rho, 1)
    rho_rot = U * rho * U'
    probs = real(diag(rho_rot))
    probs = max.(probs, 0)
    probs = probs / sum(probs)

    counts = zeros(Int, dim)
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

    return counts ./ shots
end

# ---------------------------------------------------------
# 3) Rotationsmatrix für jeden Generator
# ---------------------------------------------------------

function rotation_matrix(k::Int)
    """
    R_λk(π/2) = exp(-iπ/4 · λk)
    Rotiert zur Eigenbasis von λk
    """
    λ = gell_mann_matrices()

    if k == 0
        return Matrix{ComplexF64}(I, 3, 3)  # Identität
    else
        return exp(-1im * π/4 * λ[k])
    end
end

# ---------------------------------------------------------
# 4) Alle Rotationen für N Qutrits
# ---------------------------------------------------------

function all_rotations(N::Int)
    """
    Alle 9^N Tensorprodukt-Rotationen für N Qutrits.
    Gibt (U, generator_indices) zurück.
    """
    # Einzelne Rotationen: 0=I, 1=λ₁, ..., 8=λ₈
    single_rots = [(k, rotation_matrix(k)) for k in 0:8]

    # Alle Kombinationen
    rotations = Tuple{Vector{Int}, Matrix{ComplexF64}}[]

    for combo in Iterators.product(fill(single_rots, N)...)
        indices = [c[1] for c in combo]
        U = reduce(kron, [c[2] for c in combo])
        push!(rotations, (indices, U))
    end

    return rotations
end

# ---------------------------------------------------------
# 5) Erwartungswert aus Messung berechnen
# ---------------------------------------------------------
function expectation_from_measurement(probs::Vector{Float64},
                                       U::Matrix{ComplexF64},
                                       O::Matrix{ComplexF64})
    # Tr(O·ρ) = Σᵢ ⟨i|U·O·U†|i⟩ · pᵢ
    # = Σᵢ diag(U·O·U†)ᵢ · pᵢ
    O_rot = U * O * U'
    eigenvalues = real(diag(O_rot))
    return sum(eigenvalues .* probs)
end

# ---------------------------------------------------------
# 6) Lineare Inversion
# ---------------------------------------------------------

"""
    LinearInversionQutrit(rho_true, N, shots)

Vollständige Tomographie für N Qutrits via Lineare Inversion:

1. Generiere alle 9^N Rotationen
2. Messe in jeder Basis (shots Messungen)
3. Berechne Erwartungswerte Tr(λᵢ⊗...⊗λⱼ · ρ)
4. Rekonstruiere ρ = (1/3^N) Σ cᵢ · (λᵢ⊗...⊗λⱼ)
5. Korrigiere auf physikalische Dichtematrix
"""
function RecreatingDensityMatrixWithLineareInversionQutrit(rho_true::Matrix{ComplexF64},
                                                      shots::Int)
    N   = Int(round(log(3, size(rho_true, 1))))
    dim = 3^N
    λ   = gell_mann_matrices()

    println("N=$N Qutrits, dim=$dim")
    println("Anzahl Messungen: 9^$N = $(9^N)")
    println("Shots pro Messung: $shots\n")

    # ── Alle Gell-Mann Tensorprodukte ─────────────────────
    gm_tensors = Dict{Vector{Int}, Matrix{ComplexF64}}()
    for combo in Iterators.product(fill(0:8, N)...)
        indices = collect(combo)
        O = reduce(kron, [λ[k] for k in indices])
        gm_tensors[indices] = O
    end

    # ── Erwartungswerte aus Messungen ─────────────────────
    expectations = Dict{Vector{Int}, Float64}()

    for (indices, O) in gm_tensors
        if all(k == 0 for k in indices)
            expectations[indices] = 1.0
            continue
        end

        eig         = eigen(Hermitian(O))
        eigenvalues = real(eig.values)
        U           = Matrix{ComplexF64}(eig.vectors')
        probs       = simulate_measurement(rho_true, U, shots)
        expectations[indices] = sum(eigenvalues .* probs)
    end

    # ── Rekonstruktion ────────────────────────────────────
    rho_rec = zeros(ComplexF64, dim, dim)
    for (indices, O) in gm_tensors
        norm_factor = real(tr(O * O))
        rho_rec    += (expectations[indices] / norm_factor) * O
    end

    println("Rekonstruktion fertig.")
    println("Spur ρ_rec: ", round(real(tr(rho_rec)), digits=4))

    # ── PSD Korrektur ─────────────────────────────────────
    eig     = eigen(Hermitian(rho_rec))
    D_new   = Diagonal(max.(real(eig.values), 0))
    rho_psd = eig.vectors * D_new * eig.vectors'
    rho_psd = rho_psd / tr(rho_psd)

    return rho_psd
end

end # module LineareInversionQutrit