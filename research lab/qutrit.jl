using LinearAlgebra
using StatsBase
using IterTools
using Convex
using SCS
using QuantumInformation

function simulate_measurement(rho, projectors, n)
    # 🔴 FALL: keine Projektoren → leeres Ergebnis
    isempty(projectors) && return Float64[]

    # Wahrscheinlichkeiten: p_i = Tr(rho * P_i)
    probs = [real(tr(rho * P)) for P in projectors]

    outcomes = sample(1:length(projectors), Weights(probs), n)

    counts = [sum(outcomes .== i) for i in 1:length(projectors)]

    return counts ./ n
end

function GenerateRandomDensityMatrixNoZerosQutrits(n::Int)

    d = 3^n

    # Zufällige komplexe Matrix
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)

    # Positiv semidefinite Matrix
    rho = M * M'

    # Spur normieren
    rho /= tr(rho)

    return rho
end

n=2
shots=10000
Rhotrue= GenerateRandomDensityMatrixNoZerosQutrits(2)

# =========================
# Gell-Mann-Matrizen
# =========================

λ0 = Matrix{ComplexF64}(I, 3, 3)

λ1 = [0 1 0;
      1 0 0;
      0 0 0]

λ2 = [0 -im 0;
      im  0 0;
      0   0 0]

λ3 = [1  0  0;
      0 -1  0;
      0  0  0]

λ4 = [0 0 1;
      0 0 0;
      1 0 0]

λ5 = [0 0 -im;
      0 0  0;
      im 0  0]

λ6 = [0 0 0;
      0 0 1;
      0 1 0]

λ7 = [0 0 0;
      0 0 -im;
      0 im 0]

λ8 = (1/sqrt(3)) * [1 0 0;
                    0 1 0;
                    0 0 -2]

omega = [λ0, λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8]

# =========================
# Basiszustände
# =========================

e1 = [1; 0; 0]
e2 = [0; 1; 0]
e3 = [0; 0; 1]

# =========================
# λ1
# =========================

p1λ1 = (1/sqrt(2)) * [1; 1; 0]
p2λ1 = (1/sqrt(2)) * [1; -1; 0]
p3λ1 = [0; 0; 1]

basisλ1 = [p1λ1, p2λ1, p3λ1]

projλ11 = p1λ1 * p1λ1'
projλ12 = p2λ1 * p2λ1'
projλ13 = p3λ1 * p3λ1'

projλ1 = [projλ11, projλ12, projλ13]

# =========================
# λ2
# =========================

p1λ2 = (1/sqrt(2)) * [1; im; 0]
p2λ2 = (1/sqrt(2)) * [1; -im; 0]
p3λ2 = [0; 0; 1]

basisλ2 = [p1λ2, p2λ2, p3λ2]

projλ21 = p1λ2 * p1λ2'
projλ22 = p2λ2 * p2λ2'
projλ23 = p3λ2 * p3λ2'

projλ2 = [projλ21, projλ22, projλ23]

# =========================
# λ3
# =========================

p1λ3 = [1; 0; 0]
p2λ3 = [0; 1; 0]
p3λ3 = [0; 0; 1]

basisλ3 = [p1λ3, p2λ3, p3λ3]

projλ31 = p1λ3 * p1λ3'
projλ32 = p2λ3 * p2λ3'
projλ33 = p3λ3 * p3λ3'

projλ3 = [projλ31, projλ32, projλ33]

# =========================
# λ4
# =========================

p1λ4 = (1/sqrt(2)) * [1; 0; 1]
p2λ4 = (1/sqrt(2)) * [1; 0; -1]
p3λ4 = [0; 1; 0]

basisλ4 = [p1λ4, p2λ4, p3λ4]

projλ41 = p1λ4 * p1λ4'
projλ42 = p2λ4 * p2λ4'
projλ43 = p3λ4 * p3λ4'

projλ4 = [projλ41, projλ42, projλ43]

# =========================
# λ5
# =========================

p1λ5 = (1/sqrt(2)) * [1; 0; im]
p2λ5 = (1/sqrt(2)) * [1; 0; -im]
p3λ5 = [0; 1; 0]

basisλ5 = [p1λ5, p2λ5, p3λ5]

projλ51 = p1λ5 * p1λ5'
projλ52 = p2λ5 * p2λ5'
projλ53 = p3λ5 * p3λ5'

projλ5 = [projλ51, projλ52, projλ53]

# =========================
# λ6
# =========================

p1λ6 = (1/sqrt(2)) * [0; 1; 1]
p2λ6 = (1/sqrt(2)) * [0; 1; -1]
p3λ6 = [1; 0; 0]

basisλ6 = [p1λ6, p2λ6, p3λ6]

projλ61 = p1λ6 * p1λ6'
projλ62 = p2λ6 * p2λ6'
projλ63 = p3λ6 * p3λ6'

projλ6 = [projλ61, projλ62, projλ63]

# =========================
# λ7
# =========================

p1λ7 = (1/sqrt(2)) * [0; 1; im]
p2λ7 = (1/sqrt(2)) * [0; 1; -im]
p3λ7 = [1; 0; 0]

basisλ7 = [p1λ7, p2λ7, p3λ7]

projλ71 = p1λ7 * p1λ7'
projλ72 = p2λ7 * p2λ7'
projλ73 = p3λ7 * p3λ7'

projλ7 = [projλ71, projλ72, projλ73]

# =========================
# λ8
# =========================

p1λ8 = [1; 0; 0]
p2λ8 = [0; 1; 0]
p3λ8 = [0; 0; 1]

basisλ8 = [p1λ8, p2λ8, p3λ8]

projλ81 = p1λ8 * p1λ8'
projλ82 = p2λ8 * p2λ8'
projλ83 = p3λ8 * p3λ8'

projλ8 = [projλ81, projλ82, projλ83]

# =========================
# Sammlungen
# =========================

basisAlle = [
    basisλ1,
    basisλ2,
    basisλ3,
    basisλ4,
    basisλ5,
    basisλ6,
    basisλ7,
    basisλ8
]

projektorenBasis = [
    projλ1,
    projλ2,
    projλ3,
    projλ4,
    projλ5,
    projλ6,
    projλ7,
    projλ8
]

function tensorBasis(basisA, basisB)

    tensorVecs = []

    for v in basisA
        for w in basisB
            push!(tensorVecs, kron(v, w))
        end
    end

    return tensorVecs
end

lambda12=tensorBasis(basisλ1,basisλ5)


function GenerateLambdaGroups(N::Int)
    groups = Vector{Vector{String}}()

    labels = ["12", "38", "45", "67"]

    for mask in 0:(4^N - 1)
        S = Vector{String}(undef, N)

        for qubit in 1:N
            # base-4 "digit" extrahieren
            digit = (mask ÷ 4^(N - qubit)) % 4
            S[qubit] = labels[digit + 1]
        end

        push!(groups, S)
    end

    return groups
end


function MatrixElementsForGroupQutrit(S)

    N = length(S)
    dim = 3^N

    trits = [reverse(digits(i, base=3, pad=N)) for i in 0:dim-1]

    pairs = []

    for i in 1:dim, j in 1:dim

        valid = true

        for k in 1:N
            a = trits[i][k]
            b = trits[j][k]

            if S[k] == "38"
                valid &= (a == b)

            elseif S[k] == "12"
                valid &= ((a == 0 && b == 1) || (a == 1 && b == 0))

            elseif S[k] == "45"
                valid &= ((a == 0 && b == 2) || (a == 2 && b == 0))

            elseif S[k] == "67"
                valid &= ((a == 1 && b == 2) || (a == 2 && b == 1))

            else
                error("Unbekanntes Label: $(S[k])")
            end
        end

        if valid
            push!(pairs, (i,j))
        end
    end

    return pairs
end

function GenerateComputationalBasisQutrit(N::Int)
    basis = Vector{Vector{Float64}}()
    dim = 3^N

    for i in 0:(dim-1)
        trits = reverse(digits(i, base=3, pad=N))
        ket = [1.0]

        for t in trits
            if t == 0
                ket = kron(ket, [1.0; 0.0; 0.0])
            elseif t == 1
                ket = kron(ket, [0.0; 1.0; 0.0])
            else
                ket = kron(ket, [0.0; 0.0; 1.0])
            end
        end

        push!(basis, ket)
    end

    return basis
end

function GenerateEigenstates(group)

    # Mapping von Label → Basis
    basisMap = Dict(
        "12" => basisλ1,
        "38" => basisλ8,
        "45" => basisλ4,
        "67" => basisλ6
    )

    # Starte mit erster Basis
    resultBasis = basisMap[group[1]]

    # iterativ Tensorprodukte aufbauen
    for i in 2:length(group)
        nextBasis = basisMap[group[i]]
        resultBasis = tensorBasis(resultBasis, nextBasis)
    end

    return resultBasis
end

function GenerateProjectors(group)

    basis = GenerateEigenstates(group)
    projs = []

    for v in basis
        push!(projs, v * v')
    end

    return projs
end

lambdagroups=GenerateLambdaGroups(2)

println(lambdagroups)
println(length(GenerateEigenstates(lambdagroups[2])))




#matrixelement distribution
#formalism for real eigenbasis and imaginary eigenbasis