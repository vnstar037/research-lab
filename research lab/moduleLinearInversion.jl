module LineareInversion

using LinearAlgebra
using StatsBase
using IterTools


export RecreatingDensityMatrixWithLineareInversion  # einzige Funktion, die du von außen brauchst


# ---------------------------------------------------------
# 1) Pauli-Matrizen
# ---------------------------------------------------------

const omega_0 = [1 0; 0 1]
const omega_x = [0 1; 1 0]
const omega_y = [0 -1im; 1im 0]
const omega_z = [1 0; 0 -1]

const omega = [omega_0,omega_x,omega_y,omega_z]


# ---------------------------------------------------------
# 2) Projektoren für Z, X, Y, Z
# ---------------------------------------------------------

function _projektoren_basis()
    p_0= [1 ;0]
    p_1= [0;1]
    proj_z1=p_0*p_0'
    proj_z2=p_1*p_1'
    #projektor x-basis
    p_2 = (1/sqrt(2)) * [1; 1]
    p_3 = (1/sqrt(2)) * [1; -1]
    proj_x1=p_2*p_2'
    proj_x2=p_3*p_3'

    #projektor y-basis
    p_4= (1/sqrt(2)) * [1; 1im]
    p_5= (1/sqrt(2)) * [1; -1im]
    proj_y1=p_4*p_4'
    proj_y2=p_5*p_5'

    proj_0 = [proj_z1, proj_z2]
    proj_z = [proj_z1, -proj_z2]
    proj_x = [proj_x1, -proj_x2]
    proj_y = [proj_y1, -proj_y2]

    # Reihenfolge laut deinem Code
    return [proj_0, proj_x, proj_y, proj_z]
end

const projektoren_basis = _projektoren_basis()


# ---------------------------------------------------------
# 3) Simulierte Messung (falscher sample-Aufruf gefixt!)
# ---------------------------------------------------------

function simulate_measurement(rho, projectors, n)
    probs = [real(tr(rho * P)) for P in projectors]

    # RICHTIG:
    outcomes = sample(1:length(projectors), Weights(probs), n)

    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n
end


# ---------------------------------------------------------
# 4) Simulierte "signed" Messung
# ---------------------------------------------------------

function simulate_measurement_signed1(rho, projectors, n)
    counts_signed = 0.0

    probs = Float64[]
    signs = Int[]

    for P in projectors
        p = real(tr(rho * P))
        sign = p ≥ 0 ? 1 : -1
        push!(probs, abs(p))
        push!(signs, sign)
    end

    norm_probs = probs ./ sum(probs)
    outcomes = sample(1:length(projectors), Weights(norm_probs), n)

    for o in outcomes
        counts_signed += signs[o]
    end

    return counts_signed / n
end


# ---------------------------------------------------------
# 5) Reconstruction-Schritt
# ---------------------------------------------------------

function num_p(S, og, n)
    dim = 2^n
    rho = zeros(ComplexF64, dim, dim)

    for i in 1:length(S)
        rho += S[i] * og[i]
    end

    return (1/2^n) * rho
end


# ---------------------------------------------------------
# 6) TOMOGRAPHY (Hauptfunktion, einziger Export!)
# ---------------------------------------------------------

"""
    rho_rec(rho, n_shots)

Führt deine vollständige Tomographie aus:

- generiert alle Projektoren
- sammelt signed measurement data
- generiert Pauli-Tensorprodukte
- rekonstruiert ρ
- repariert PSD (negative Eigenwerte → 0)

"""
function RecreatingDensityMatrixWithLineareInversion(rho, n_shots)

    dim = size(rho,1)
    n = Int(round(log2(dim)))

    # Alle Projektorkombinationen
    combi_proj = collect(Iterators.product((projektoren_basis for _ in 1:n)...))
#combi_proj1 = reshape(permutedims(combi_proj), 1, :)
    combi_proj1=reshape(combi_proj, :)

    S=zeros(1,dim^2)

    for i in 1:length(combi_proj1)
        tupel=combi_proj1[i]
        ts=product(tupel...)
        ts3=[reduce(kron, k) for k in ts]
       ts4=reshape((ts3), 1, :)
        S[i]=simulate_measurement_signed1(rho,ts4,n_shots)
    end

    og=Matrix{Matrix{ComplexF64}}(undef, 1, dim^2)
    combi_omega=collect(Iterators.product((omega for _ in 1:n)...))
    combi_omega1 = reshape((combi_omega), 1, :)

    for i in 1:length(combi_omega1)
        oTupel1=combi_omega1[i]
        omega_ts=reduce(kron,oTupel1)
        og[i]=omega_ts
    end

    rho_num1=num_p(S,og,n)
    # Eingebaute Version der "Reparatur"
    eig = eigen(rho_num1)
    Dnew = Diagonal(max.(eig.values, 0))  # Negative Eigenwerte -> 0
    rho_num2 = eig.vectors * Dnew * eig.vectors'  # Rekonstruktion
    return rho_num2
end


end # module
