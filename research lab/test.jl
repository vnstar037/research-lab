using Plots

# --------------------------------------
# 1. Vollständige 2-Qubit Pauli-Tomographie
# --------------------------------------
paulis = ["X", "Y", "Z"]
qubits = ["Q1", "Q2"]

full_projectors = []

for b1 in paulis
    for b2 in paulis
        push!(full_projectors, "$b1⊗$b2")
    end
end

# --------------------------------------
# 2. Patel-Gruppenansatz (E/O Basen)
# --------------------------------------
# Angenommen Gruppe deckt nur diese Basen ab:
patel_projectors = ["Z⊗Z", "X⊗X", "E-Basis", "O-Basis"]  # symbolisch
# 4 Projektoren für E + 4 für O → insgesamt 8

# --------------------------------------
# 3. Plot vorbereiten
# --------------------------------------
n_full = length(full_projectors)
n_pat = length(patel_projectors)

# Farbe: blau = full, rot = Patel
colors = [fill(:blue, n_full); fill(:red, n_pat)]
labels = vcat(full_projectors, patel_projectors)

# Balkendiagramm
bar(
    1:length(labels), 
    ones(length(labels)), 
    color = colors, 
    legend = false,
    xticks = (1:length(labels), labels),
    xrotation = 45,
    title = "Vollständige Projektoren vs Patel-Gruppen",
    ylabel = "Vorhanden",
    xlabel = "Projektor"
)