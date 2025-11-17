#import Pkg; Pkg.add("Optim")
#import Pkg; Pkg.add("Plots")
using Random, Optim, Distributions, Plots


# Wahre Koeffizienten des Polynoms
true_coeffs = [1.0, -2.0, 0.5, 3.0]  # a0 + a1*x + a2*x^2 + a3*x^3

# Funktion zur Polynom-Auswertung
function poly(x, coeffs)
    return sum(coeffs[i] * x^(i - 1) for i in 1:length(coeffs))
end

# Simuliere verrauschte Datenpunkte
function generate_data(true_coeffs, n_points=100, noise_std=0.5)
    x_data = range(-2, 2, length=n_points)
    y_true = [poly(x, true_coeffs) for x in x_data]
    y_obs = y_true .+ rand(Normal(0, noise_std), n_points)
    return x_data, y_obs
end

# Negative Log-Likelihood-Funktion (Gaussian noise)
function neg_log_likelihood(coeffs, x_data, y_obs, σ=0.5)
    y_pred = [poly(x, coeffs) for x in x_data]
    return sum(((y_obs[i] - y_pred[i]) / σ)^2 for i in eachindex(y_obs)) / 2
end

# Daten erzeugen
x_data, y_obs = generate_data(true_coeffs)

# Startwerte (z. B. alles 0)
initial_guess = zeros(length(true_coeffs))

# Optimierung
result = optimize(c -> neg_log_likelihood(c, x_data, y_obs), initial_guess, BFGS())
estimated_coeffs = Optim.minimizer(result)

println("Wahre Koeffizienten: ", true_coeffs)
println("Geschätzte Koeffizienten (MLE): ", estimated_coeffs)

# Optional: Visualisierung
plot(x_data, [poly(x, true_coeffs) for x in x_data], label="True", lw=2)
plot!(x_data, [poly(x, estimated_coeffs) for x in x_data], label="Estimated", lw=2, ls=:dash)
scatter!(x_data, y_obs, label="Noisy Data", color=:gray)
