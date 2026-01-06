
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

println(a)      # ["IX", "IY", "ZX", "ZY"]
println(a[2])   # "IY"


for i in a
    b=pauli_string_eigenvalues(i)
    println(b,i)
end

