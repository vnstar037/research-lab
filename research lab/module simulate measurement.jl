module SimulateMeasurement

using LinearAlgebra
using StatsBase

export simulate_measurement

"""
    simulate_measurement(rho, projectors, n)

Simuliert eine Messung mit Projektionen `projectors` an einem Zustand `rho`
und liefert die relativen Häufigkeiten der Ergebnisse (counts / n).
"""
#function simulate_measurement(rho, projectors, n)
#    # Wahrscheinlichkeiten: p_i = Tr(rho * P_i)
#    probs = [real(tr(rho * P)) for P in projectors]

    # richtig: 1:length(projectors) (ohne eckige Klammer außen)
#    outcomes = sample(1:length(projectors), Weights(probs), n)

    # Zähle Outcomes
#    counts = [sum(outcomes .== i) for i in 1:length(projectors)]

#    return counts ./ n
#end

function simulate_measurement(rho, projectors, n)
    # 🔴 FALL: keine Projektoren → leeres Ergebnis
    isempty(projectors) && return Float64[]

    # Wahrscheinlichkeiten: p_i = Tr(rho * P_i)
    probs = [real(tr(rho * P)) for P in projectors]

    outcomes = sample(1:length(projectors), Weights(probs), n)

    counts = [sum(outcomes .== i) for i in 1:length(projectors)]

    return counts ./ n
end

end # module
