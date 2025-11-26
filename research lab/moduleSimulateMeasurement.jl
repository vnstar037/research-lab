module SimulateMeasurement

using LinearAlgebra
using StatsBase

export SimulateMeasurement

"""
    simulate_measurement(rho, projectors, n)

Simuliert eine Messung mit Projektionen `projectors` an einem Zustand `rho`
und liefert die relativen Häufigkeiten der Ergebnisse (counts / n).
"""
function simulateMeasurement(rho, projectors, n)
    # Wahrscheinlichkeiten: p_i = Tr(rho * P_i)
    probs = [real(tr(rho * P)) for P in projectors]

    # richtig: 1:length(projectors) (ohne eckige Klammer außen)
    outcomes = sample(1:length(projectors), Weights(probs), n)

    # Zähle Outcomes
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]

    return counts ./ n
end

end # module
