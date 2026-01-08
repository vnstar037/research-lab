using LinearAlgebra
using QuantumInformation 
# ---------------------------
# struct reconstructed density matrix
# ---------------------------
struct ReconstructedDensityMatrix
    rho::Matrix{ComplexF64}      
    purity::Float64               # Tr(rho^2)
    fidelity::Float64
    DeltaRho::Matrix{ComplexF64}            
end

# ---------------------------
# Konstruktor-Funktion
# ---------------------------
function ReconstructedDensityMatrix(rho_reconstructed::Matrix{ComplexF64}, rho_true::Matrix{ComplexF64})
    ρ = rho_reconstructed
    PurityVal = real(tr(ρ * ρ))               # Purity Tr(rho^2)
    FidelityVal = fidelity(ρ, rho_true)       # Fidelity 
    DeltaVal = rho_true - rho_reconstructed
    return ReconstructedDensityMatrix(ρ, PurityVal, FidelityVal, DeltaVal)
end
