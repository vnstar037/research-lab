using LinearAlgebra

# =============================================================
#  Basiszustände eines einzelnen Qutrits
# =============================================================

const e0 = ComplexF64[1; 0; 0]   # |0⟩
const e1 = ComplexF64[0; 1; 0]   # |1⟩
const e2 = ComplexF64[0; 0; 1]   # |2⟩

const ω = exp(2π * im / 3)       # primitive dritte Einheitswurzel

# =============================================================
#
#  EIGENZUSTÄNDE FÜR QUTRIT-SEEQST
#
#  Für ein einzelnes Qutrit-Paar (il, jl) gibt es 4 Typen:
#    "38"  : il == jl              → diagonal
#    "12"  : Unterraum {|0⟩, |1⟩} → off-diagonal
#    "45"  : Unterraum {|0⟩, |2⟩} → off-diagonal
#    "67"  : Unterraum {|1⟩, |2⟩} → off-diagonal
#
#  Für off-diagonale Typen gibt es 3 Messbasen s ∈ {0, 1, 2},
#  analog zu E und O bei Qubits, mit ω = exp(2πi/3):
#
#    |e^(s)_p⟩ = (1/√2) ( |p⟩ + ω^s |p̄⟩ )
#
#  wobei |p⟩ und |p̄⟩ die beiden Basis­zustände des Unterraums sind.
#  Der dritte Zustand (außerhalb des Unterraums) bleibt unvermischt.
#
# =============================================================

"""
    single_qutrit_eigenstates(label, s) -> Vector{Vector{ComplexF64}}

Gibt die 3 Eigenzustände eines einzelnen Qutrits für Messbasis s ∈ {0,1,2}
zurück, abhängig vom Paar-Typ (label ∈ {"38","12","45","67"}).

Für "38" (diagonal): s wird ignoriert, Basis ist immer {|0⟩, |1⟩, |2⟩}.

Für off-diagonale Typen:
  s = 0 : reale Superposition  (1/√2)(|p⟩ + |p̄⟩)        — analog E bei Qubits
  s = 1 : ω-gewichtet          (1/√2)(|p⟩ + ω  |p̄⟩)     — analog O bei Qubits
  s = 2 : ω²-gewichtet         (1/√2)(|p⟩ + ω² |p̄⟩)     — neue dritte Basis
"""
function single_qutrit_eigenstates(label::String, s::Int)

    @assert s in (0, 1, 2) "s muss 0, 1 oder 2 sein"

    if label == "38"
        # Diagonaler Block: Standard-Rechenbasiszustände, unabhängig von s
        return [copy(e0), copy(e1), copy(e2)]

    elseif label == "12"
        # Off-diagonal: Unterraum {|0⟩, |1⟩}
        v1 = (1/sqrt(2)) * (e0 + ω^s       * e1)
        v2 = (1/sqrt(2)) * (e0 + ω^(s + 1) * e1)
        v3 = copy(e2)
        return [v1, v2, v3]

    elseif label == "45"
        # Off-diagonal: Unterraum {|0⟩, |2⟩}
        v1 = (1/sqrt(2)) * (e0 + ω^s       * e2)
        v2 = (1/sqrt(2)) * (e0 + ω^(s + 1) * e2)
        v3 = copy(e1)
        return [v1, v2, v3]

    elseif label == "67"
        # Off-diagonal: Unterraum {|1⟩, |2⟩}
        v1 = (1/sqrt(2)) * (e1 + ω^s       * e2)
        v2 = (1/sqrt(2)) * (e1 + ω^(s + 1) * e2)
        v3 = copy(e0)
        return [v1, v2, v3]

    else
        error("Unbekanntes Label: \"$label\". Erlaubt: \"38\", \"12\", \"45\", \"67\"")
    end
end
