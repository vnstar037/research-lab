
function generate_combinations(liste::Vector{String})
    # Wir kehren die Liste um (reverse), damit die Zeichen des 
    # ersten Strings im Produkt am schnellsten rotieren.
    kombinationen = Iterators.product(reverse(liste)...)
    
    # Beim Zusammenfügen (join) kehren wir die Tupel wieder um, 
    # damit die ursprüngliche Zeichenfolge (1. String, 2. String) erhalten bleibt.
    return vec([join(reverse(k)) for k in kombinationen])
end


function pauli_eigenvalues(p::Char)
    if p == 'I'
        return [1.0, 1.0]
    elseif p in ('X','Y','Z')
        return [1.0, -1.0]
    else
        error("Unknown Pauli operator: $p")
    end
end

# --------------------------------------------------------------------
# Eigenwerte eines Pauli-Strings (Kronecker-artiges Produkt)
# Die Reihenfolge der Bits ist so gewählt, dass die erste Position im String
# der langsam rotierende Faktor ist
# --------------------------------------------------------------------
function pauli_string_eigenvalues(s::String)
    ev = [1.0]

    # reverse(s) → erster Buchstabe rotiert langsam
    for p in reverse(s)
        local_eigs = pauli_eigenvalues(p)
        ev = vec([a*b for a in ev, b in local_eigs])
    end

    return ev
end

a = generate_combinations(["XY", "XY"])

#println(a)      # ["IX", "IY", "ZX", "ZY"]
#println(a[2])   # "IY"


for i in a
    b=pauli_string_eigenvalues(i)
    #println(b,i)
end

function GenerateEigenstatesE(S::Vector{String})
    ket0 = [1.0, 0.0]
    ket1 = [0.0, 1.0]
    ketp = (ket0 .+ ket1) ./ sqrt(2)
    ketm = (ket0 .- ket1) ./ sqrt(2)

    # lokale Basen
    local_bases = Vector{Vector{Vector{Float64}}}()
    for s in S
        if s == "IZ"
            push!(local_bases, [ket0, ket1])
        elseif s == "XY"
            push!(local_bases, [ketp, ketm])
        else
            error("Unknown stabilizer type $s")
        end
    end

    eigenstates = Vector{Vector{Float64}}()

    # ⬇️ WICHTIG: product über reversed(local_bases)
    for combo_rev in Base.Iterators.product(reverse(local_bases)...)
        combo = reverse(combo_rev)  # zurück zur physikalischen Reihenfolge

        ψ = combo[1]
        for k in combo[2:end]
            ψ = kron(ψ, k)
        end

        push!(eigenstates, ψ)
    end

    return eigenstates
end

function GenerateEigenstatesO(S::Vector{String})
    # 🔴 nur IZ → keine O-Eigenzustände
    if all(x -> x == "IZ", S)
        return Vector{Vector{ComplexF64}}()
    end

    # Basiszustände
    ket0 = ComplexF64[1, 0]
    ket1 = ComplexF64[0, 1]
    ket_ip = (ket0 .+ im .* ket1) ./ sqrt(2)   # |+i⟩
    ket_im = (ket0 .- im .* ket1) ./ sqrt(2)   # |-i⟩

    # lokale Basen pro Qubit
    local_bases = Vector{Vector{Vector{ComplexF64}}}()
    for s in S
        if s == "IZ"
            push!(local_bases, [ket0, ket1])
        elseif s == "XY"
            push!(local_bases, [ket_ip, ket_im])
        else
            error("Unknown stabilizer type $s")
        end
    end

    eigenstates = Vector{Vector{ComplexF64}}()

    # rechte Qubits laufen am schnellsten
    for combo_rev in Base.Iterators.product(reverse(local_bases)...)
        combo = reverse(combo_rev)

        ψ = combo[1]
        for k in combo[2:end]
            ψ = kron(ψ, k)
        end

        push!(eigenstates, ψ)
    end

    return eigenstates
end

function ProjectorsFromEigenstates(eigs)
    isempty(eigs) && return Matrix{ComplexF64}[]
    return [ψ * ψ' for ψ in eigs]
end

d=GenerateEigenstatesO(["XY","XY"])
d2=ProjectorsFromEigenstates(d)

#println(d)
#println(d2)