module RandomDensityMatrix

export random_density_matrix

using LinearAlgebra

"""
    random_density_matrix(d::Int)

Erzeugt eine zufällige Dichte­matrix der Dimension `d × d`,
normalisiert auf Spur 1.
"""
function random_density_matrix(d::Int)
    M = randn(ComplexF64, d, d) + 1im * randn(ComplexF64, d, d)
    rho = M * M'
    rho /= tr(rho)
    return rho
end

end # module
