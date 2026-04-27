import qutip as qt
import numpy as np
import jax.numpy as jnp
from setup import generate_experiment
from processing import (
    process_data,
    flatten_list,
    parse_circuit_to_qobj,
    data_predict_from_rho_sampled,
    random_matrix_jax,
    log_loss_function,
    density_matrix
)

def full_tomography(N: int, rho_true: qt.Qobj, shots: int = 1000):
    """
    Volle Quantenzustandstomographie mit SEEQST.
    
    Args:
        N:        Anzahl der Qubits
        rho_true: Wahrer Quantenzustand (als Qobj)
        shots:    Anzahl der Messungen pro Schaltkreis
    
    Returns:
        rho_reconstructed: Rekonstruierte Dichtematrix (als Qobj)
        fidelity:          Fidelität zwischen wahrem und rekonstruiertem Zustand
    """

    # ── Schritt 1: Alle Subsets für volle Tomographie ───────────────
    # Für volle Tomographie braucht man die gesamte erste Zeile
    # d.h. alle Elemente (0,0), (0,1), ..., (0, 2^N - 1)
    wanted_index = [(0, j) for j in range(2**N)]
    print(f"Wanted indices: {wanted_index}")

    # ── Schritt 2: Experimente generieren ───────────────────────────
    options = {
        "density matrix plot": False,
        "wanted elements": False,
        "selective elements": False,
        "circuits text": False,
        "non entangling circuits text": False,
        "observable": False
    }

    selective_blocks, sel_circ_text, sel_circ_text_non_entangle, \
    sel_circ_qiskit, non_e_circ_qiskit, plot_rho = generate_experiment(
        wanted_index, N, options
    )
    print(f"Selective blocks: {selective_blocks}")

    # ── Schritt 3: Schaltkreise zu unitären Matrizen ─────────────────
    circuits = flatten_list(sel_circ_text)
    print(f"Anzahl Schaltkreise: {len(circuits)}")

    unitaries     = parse_circuit_to_qobj(circuits, N)
    unitaries_jnp = jnp.array([uni.full() for uni in unitaries])

    # ── Schritt 4: Fake-Daten generieren ────────────────────────────
    rho_true_jnp = jnp.array(rho_true.full())
    data = data_predict_from_rho_sampled(rho_true_jnp, unitaries_jnp, shots)
    print(f"Daten shape: {data.shape}")

    # ── Schritt 5: Rekonstruktion via SGD + Cholesky ─────────────────
    rho_reconstructed = process_data(
        data             = data,
        unitaries_jnp    = unitaries_jnp,
        selective_blocks = selective_blocks,
        shots            = shots,
        N                = N,
        rho_ideal        = rho_true
    )

    # ── Schritt 6: Fidelität berechnen ──────────────────────────────────
    # rho_reconstructed hat andere dims -> konvertieren
    rho_reconstructed = qt.Qobj(
        np.array(rho_reconstructed.full()),
        dims=[[2]*N, [2]*N]
    )
    rho_true_qobj = qt.Qobj(
        np.array(rho_true.full()),
        dims=[[2]*N, [2]*N]
    )

    fidelity = qt.fidelity(rho_true_qobj, rho_reconstructed)**2
    print(f"Fidelität: {fidelity:.4f}")

    return rho_reconstructed, fidelity


# ── Beispiel-Aufruf ──────────────────────────────────────────────────
if __name__ == "__main__":

    N = 3

    # Zufälligen wahren Zustand generieren
    rho_true = qt.rand_dm(2**N)

    # Volle Tomographie
    rho_rec, fidelity = full_tomography(
        N        = N,
        rho_true = rho_true,
        shots    = 5000 * 2**N
    )

    # Ergebnisse ausgeben
    print("\n── Wahrer Zustand ──")
    print(np.round(rho_true.full(), 3))

    print("\n── Rekonstruierter Zustand ──")
    print(np.round(np.array(rho_rec.full()), 3))

    print(f"\n── Fidelität: {fidelity:.4f} ──")