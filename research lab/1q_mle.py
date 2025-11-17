import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm

# Pauli-Matrizen
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Projektoren in X-, Y- und Z-Basis
projectors = {
    "Z": [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])],
    "X": [0.5*np.array([[1, 1], [1, 1]]), 0.5*np.array([[1, -1], [-1, 1]])],
    "Y": [0.5*np.array([[1, -1j], [1j, 1]]), 0.5*np.array([[1, 1j], [-1j, 1]])]
}

# Funktion: Dichtematrix aus Bloch-Vektor
def rho_from_bloch(r):
    return 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

# Wahre Dichtematrix (z. B. Bloch-Vektor (0.3, 0.5, 0.7))
r_true = np.array([0.3, 0.5, 0.7])
rho_true = rho_from_bloch(r_true)

# Simuliere Messungen (n Samples pro Richtung)
n_shots = 1000
counts = {}
for basis, P in projectors.items():
    probs = [np.real(np.trace(rho_true @ p)) for p in P]
    outcomes = np.random.choice([0, 1], size=n_shots, p=probs)
    counts[basis] = [np.sum(outcomes == 0), np.sum(outcomes == 1)]

# Negative log-likelihood
def neg_log_likelihood(r):
    if np.linalg.norm(r) >= 1.0:
        return np.inf
    rho = rho_from_bloch(r)
    ll = 0.0
    for basis, P in projectors.items():
        for i in [0, 1]:
            p = np.real(np.trace(rho @ P[i]))
            p = max(p, 1e-10)
            ll -= counts[basis][i] * np.log(p)
    return ll

# MLE: Optimierung
result = minimize(neg_log_likelihood, x0=np.array([0.0, 0.0, 0.0]),
                  bounds=[(-1, 1), (-1, 1), (-1, 1)],
                  method='L-BFGS-B')

r_est = result.x
rho_est = rho_from_bloch(r_est)

# Fidelity-Funktion
def fidelity(rho, sigma):
    sqrt_rho = sqrtm(rho)
    inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return np.real(np.trace(inner))**2

f = fidelity(rho_true, rho_est)

# Ergebnisse
print("Wahrer Bloch-Vektor:     ", np.round(r_true, 3))
print("Geschätzter Bloch-Vektor:", np.round(r_est, 3))
print("\nWahre Dichtematrix:\n", np.round(rho_true, 3))
print("\nGeschätzte Dichtematrix:\n", np.round(rho_est, 3))
print("\nFidelity: {:.5f}".format(f))
