module SimulateMeasurement

using LinearAlgebra
using StatsBase

export simulateMeasurement

"""
    simulate_measurement(rho, projectors, n)

Simuliert eine Messung mit Projektionen `projectors` an einem Zustand `rho`
und liefert die relativen Häufigkeiten der Ergebnisse (counts / n).
"""
function simulateMeasurement(rho, projectors, n)
    isempty(projectors) && return Float64[]
    probs = [real(tr(rho * P)) for P in projectors]
    outcomes = sample(1:length(projectors), Weights(probs), n)
    counts = [sum(outcomes .== i) for i in 1:length(projectors)]
    return counts ./ n
end


end # module
