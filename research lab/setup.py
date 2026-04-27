

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def get_selective_blocks(num_qubits, wanted_indexes):
    """
    Given a number of qubits and a list of wanted index pairs,
    this function returns:whereis python
    1. List of selective blocks (bitwise XOR of index pairs)
    2. Dictionary mapping each block to contributing index pairs.

    Parameters:
        num_qubits (int): Number of qubits (N)
        wanted_indexes (list of tuples): Pairs of indexes (i, j) to process.

    Returns:
        selective_blocks (list of int)
        block_map (dict): {block_value: [index_pairs]}
    """
    max_index = 2 ** num_qubits
    selective_blocks = []
    block_map = {}

    for index_pair in wanted_indexes:
        i, j = index_pair

        # Assert that indices are within valid range
        assert 0 <= i < max_index and 0 <= j < max_index, \
            f"Index pair {index_pair} exceeds valid range [0, {max_index - 1}] for {num_qubits} qubits."

        # XOR operation
        xor_result = i ^ j  # Faster and simpler than using binary strings

        # Add to list if not already there
        if xor_result not in selective_blocks:
            selective_blocks.append(xor_result)

        # Update block_map
        if xor_result not in block_map:
            block_map[xor_result] = []
        block_map[xor_result].append((i, j))

    return selective_blocks, block_map


def generate_selective_elements(selective_blocks, wanted_indexes, num_qubits):
    """
    Generate selective elements from selective_blocks using
    bit-flip (X) and identity (I) operators on binary tuples.

    Parameters:
        selective_blocks (list): List of integers (e.g., [3,2]).
        wanted_indexes (list): List of wanted index pairs (not used directly here, kept for structure).
        num_qubits (int): Number of qubits (N), determines binary string length.

    Returns:
        List of lists: Each sublist contains 2^N decimal indices for one selective_block.
    """
    selective_elements = []

    for block in selective_blocks:
        # Convert to N-bit binary
        block_bin = format(block, f'0{num_qubits}b')

        # Create pair: (zero_state, block_state)
        zero_state = '0' * num_qubits
        pair = (zero_state, block_bin)

        # Apply (I,X)^⊗N: For each bit, apply I or X (flip bit or keep same)
        group = []
        for i in range(2 ** num_qubits):
            mask = format(i, f'0{num_qubits}b')
            # Apply X where mask has 1, else I
            new_state = ''.join(
                str(int(pair[0][j]) ^ int(mask[j])) for j in range(num_qubits)
            )
            new_block = ''.join(
                str(int(pair[1][j]) ^ int(mask[j])) for j in range(num_qubits)
            )
            # Convert both to decimal and store as a tuple
            index_pair = (int(new_state, 2), int(new_block, 2))
            group.append(index_pair)

        selective_elements.append(group)

    return selective_elements


def generate_observable_sets(selective_blocks, num_qubits):
    """
    For each selective block, generate two sets of Pauli observables [E, O],
    where E = observables with even number of 'Y', O = with odd number of 'Y'.

    Parameters:
        selective_blocks (list): List of integers like [3, 4]
        num_qubits (int): Number of qubits (N)

    Returns:
        dict: {block: [E_set, O_set]} where E_set and O_set are lists of Pauli strings
    """
    observable_sets = {}

    for block in selective_blocks:
        bin_block = format(block, f'0{num_qubits}b')

        # Prepare options for each qubit position
        pauli_options = []
        for bit in bin_block:
            if bit == '1':
                pauli_options.append(('X', 'Y'))  # bit flip positions
            else:
                pauli_options.append(('I', 'Z'))  # identity/control positions

        # Generate all combinations (2^N strings)
        all_observables = [''.join(p) for p in product(*pauli_options)]

        # Split into even-Y and odd-Y sets
        even_set = [obs for obs in all_observables if obs.count('Y') % 2 == 0]
        odd_set  = [obs for obs in all_observables if obs.count('Y') % 2 == 1]

        # Store result
        observable_sets[block] = [even_set, odd_set]

    return observable_sets



def plot_density_matrix_highlight(wanted_indexes, result, selective_blocks, num_qubits):
    """
    Plot N-qubit density matrix with:
      - Wanted indexes in red with legend
      - Selective block indexes in distinct colors with legend
      - Binary axis ticks
      - Grid aligned to cell centers

    Parameters:
        wanted_indexes (list of tuples): List of (i,j) pairs (highlighted in red)
        result (list of lists): List of [(i,j), ...] for each selective block
        selective_blocks (list): List of block integers corresponding to result
        num_qubits (int): Number of qubits (N)
    """
    dim = 2 ** num_qubits
    fig, ax = plt.subplots(figsize=(6, 6))

    # Move X ticks to top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(dim) + 0.5)
    ax.set_xticklabels([format(i, f'0{num_qubits}b') for i in range(dim)], rotation=90)

    # Set Y ticks
    ax.set_yticks(np.arange(dim) + 0.5)
    ax.set_yticklabels([format(i, f'0{num_qubits}b') for i in range(dim)])
    plt.setp(ax.get_xticklabels(), rotation=45)
    # Set limits to match cell centers
    ax.set_xlim(0, dim)
    ax.set_ylim(0,dim)
    ax.invert_yaxis()


    # Draw grid centered between ticks
    ax.set_xticks(np.arange(dim), minor=True)
    ax.set_yticks(np.arange(dim), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1)
    ax.tick_params(which='major', length=0)

    # Draw outer box
    ax.set_frame_on(True)

    # Color palette
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

    legend_elements = []

    # Plot selective block indexes
    for idx, block in enumerate(selective_blocks):
        color = color_list[idx % len(color_list)]
        for (i, j) in result[idx]:
            ax.add_patch(Rectangle((j,  i ), 1, 1, color=color, alpha=0.5))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', label=f'Selective Block {block}',
                                      markerfacecolor=color, markersize=12, alpha=0.6))

    # Plot wanted indexes (on top of everything)
    for i, j in wanted_indexes:
        ax.add_patch(Rectangle((j, i), 1, 1, color='red', alpha=0.9))
    legend_elements.insert(0, Line2D([0], [0], marker='s', color='w', label='Wanted Index',
                                     markerfacecolor='red', markersize=12, alpha=0.9))

    # Add legend
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    # Title and aspect
    ax.set_title(f'{num_qubits}-Qubit Density Matrix', y=1.08)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def build_parallel_entangler_blocks(selective_blocks, num_qubits):
    """
    Build parallel GHZ-style entangling gate sequences for each selective block.
    Returns two gate sequences per block: one with RY90 and one with RX90 (in reverse order).
    """
    all_sequences = []

    for block in selective_blocks:
        bin_str = format(block, f'0{num_qubits}b')
        active_qubits = [i for i, bit in enumerate(bin_str[::1]) if bit == '1']  # LSB = qubit 0

        if not active_qubits:
            all_sequences.append([''])  # No gates needed
            continue

        sequence = []

        # Step 1: Initial rotation on first qubit (arbitrary choice)
        sequence.append(f'(RY90:{active_qubits[0]})')

        head = [active_qubits[0]]
        tail = active_qubits[1:]

        # Step 2: GHZ layering: use ALL head qubits as controls
        while tail:
            new_tail = []
            for h in head:
                if not tail:
                    break
                # Assign one tail target to this control
                tgt = tail.pop(0)
                sequence.append(f'(CNOT:{h},{tgt})')
                new_tail.append(tgt)
            head.extend(new_tail)

        # Step 3: Create RX90 version of same circuit
        rx_sequence = [gate.replace('RY90', 'RX90') for gate in sequence]

        # Step 4: Return both sequences in reverse order
        all_sequences.append([''.join(sequence[::-1]), ''.join(rx_sequence[::-1])])

    return all_sequences




def build_non_entangling_circuits(selective_blocks, num_qubits):
    """
    For each selective block, generate all RX/RY tensor product combinations
    over active qubits (where the bit is 1 in the block binary).
    
    Returns list of list of circuits (each inner list is all combinations for a block).
    """
    all_block_circuits = []

    for block in selective_blocks:
        bin_str = format(block, f'0{num_qubits}b')
        active_qubits = [i for i, bit in enumerate(bin_str[::]) if bit == '1']  # LSB = qubit 0

        # Generate all 2^m combinations of RX/RY for m active qubits
        gate_choices = list(product(['RY90', 'RX90'], repeat=len(active_qubits)))

        circuits = []
        for choice in gate_choices:
            circuit_str = ''.join([f'({gate}:{q})' for gate, q in zip(choice, active_qubits)])
            circuits.append(circuit_str)

        all_block_circuits.append(circuits)

    return all_block_circuits


def parse_circuit(text_circuits, n_qubits,initial_text=""):
    """
    Convert a text-based circuit description into Qiskit circuits.
    
    Args:
        text_circuits (list of str): List of circuit descriptions in text format.
        n_qubits (int): Number of qubits and classical bits in each circuit.

    Returns:
        list of QuantumCircuit: List of Qiskit QuantumCircuit objects.
    """
    circuits = []

    for circuit_text in text_circuits:
        circuit_text=initial_text+circuit_text
        # Initialize quantum and classical registers
        qreg = QuantumRegister(n_qubits, 'q')
        creg = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qreg, creg)

        # Split operations while handling concatenated gates
        operations = circuit_text.split(')')

        for op in operations:
            if not op.strip():  # Skip empty parts
                continue
            op = op.strip().strip('(')  # Remove leading (
            gate_info = op.split(':')

            if len(gate_info) < 2:
                continue  # Skip invalid formats

            gate_name = gate_info[0]
            qubit_indices = list(map(int, gate_info[1].split(',')))  # Extract qubit indices

            # Map gate names to Qiskit gates
            if gate_name == "RX90":
                qc.rx(3.14159/2, qubit_indices[0])
            elif gate_name == "RY90":
                qc.ry(3.14159/2, qubit_indices[0])
            elif gate_name == "CNOT":
                qc.cx(qubit_indices[0], qubit_indices[1])
            elif gate_name == "H":
                qc.h(qubit_indices[0])
            elif gate_name == "MEAS":
                qc.measure(qubit_indices[0], qubit_indices[0])
            else:
                raise ValueError(f"Unsupported gate: {gate_name}")
            
        qc.measure_all()

        circuits.append(qc)

    return circuits



# Helper function to format complex numbers for display
def format_complex(val):
    real_part = val.real
    imag_part = val.imag
    if abs(real_part) < 1e-9: real_part = 0
    if abs(imag_part) < 1e-9: imag_part = 0
    if real_part == 0 and imag_part == 0: return "0.00"
    if imag_part == 0: return f"{real_part:.3f}"
    if real_part == 0: return f"{imag_part:.3f}j"
    sign = '+' if imag_part >= 0 else '-'
    return f"{real_part:.3f}\n{sign} {abs(imag_part):.3f}j"

# Helper function to pick black or white text for readability
def get_text_color(bg_color):
    rgb = mcolors.to_rgb(bg_color)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return 'white' if luminance < 0.5 else 'black'


def create_plotter(wanted_indexes, result, selective_blocks, num_qubits):
    """
    Creates a plotting function with a pre-defined style.
    
    Returns:
        A function that only needs 'rho' to make a plot.
    """
    
    def plot(rho):
        dim = 2 ** num_qubits
        fig, ax = plt.subplots(figsize=(6, 6))

        # --- Axes and Ticks (matched to original style) ---
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(dim) + 0.5)
        ax.set_xticklabels([format(i, f'0{num_qubits}b') for i in range(dim)])
        ax.set_yticks(np.arange(dim) + 0.5)
        ax.set_yticklabels([format(i, f'0{num_qubits}b') for i in range(dim)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        
        ax.set_xlim(0, dim)
        ax.set_ylim(0, dim)
        ax.invert_yaxis()

        # --- Grid (matched to original style) ---
        ax.set_xticks(np.arange(dim), minor=True)
        ax.set_yticks(np.arange(dim), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1)
        ax.tick_params(which='major', length=0)
        
        ax.set_frame_on(True)
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        legend_items = []

        # --- Plot colored blocks ---
        for i, label in enumerate(selective_blocks):
            color = colors[i % len(colors)]
            text_color = get_text_color(color)
            for (r, c) in result[i]: # r for row, c for column
                ax.add_patch(Rectangle((c, r), 1, 1, color=color, alpha=0.6))
                ax.text(c + 0.5, r + 0.5, format_complex(rho[r, c]), ha='center', va='center', color=text_color, fontsize=10)
            legend_items.append(Line2D([0], [0], marker='s', color='w', label=f'Block {label}',
                                       markerfacecolor=color, markersize=12, alpha=0.6))

        # --- Plot wanted indexes (red cells) ---
        text_color = get_text_color('red')
        for (r, c) in wanted_indexes:
            ax.add_patch(Rectangle((c, r), 1, 1, color='red', alpha=0.9))
            ax.text(c + 0.5, r + 0.5, format_complex(rho[r, c]), ha='center', va='center', color=text_color, fontsize=10)
        legend_items.insert(0, Line2D([0], [0], marker='s', color='w', label='Wanted Index',
                                     markerfacecolor='red', markersize=12, alpha=0.9))

        # --- Final touches (matched to original style) ---
        ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        ax.set_title(f'{num_qubits}-Qubit Density Matrix', y=1.08)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    return plot



def generate_experiment(wanted_indexes,N,
                        options={"density matrix plot":True, "wanted elements":True ,"selective elements":True,'circuits text':True,'non entangling circuits text':True,'observable':True}
                        ):
    selective_blocks, block_dict = get_selective_blocks(N, wanted_indexes)
    
    print('')
    print(f"Wanted Indexes {wanted_indexes} corresponds to these Selective Blocks:", selective_blocks)
    print('')
    
    
    result = generate_selective_elements(selective_blocks, wanted_indexes, N)
            
    observable_dict = generate_observable_sets(selective_blocks, N)

        
    
    sel_circ_text = build_parallel_entangler_blocks(selective_blocks, N)
            
    sel_circ_text_non_entangle = build_non_entangling_circuits(selective_blocks, N)
          
            
    for i, block in enumerate(selective_blocks):
        print('********************')
        print(f"Selective Block {block}")
        if options["wanted elements"]:
            print('')
            print(f"Wanted Elements {block_dict[block]}")
        if options["selective elements"]:
            print('')
            print(f"Selective elements in block",result[i])
        
        if options['observable']: 
            (E, O) = observable_dict[block]
            print('')
            print(f"Pauli Observables:")
            print("  Even-Y set (E):", E)
            print("  Odd-Y set  (O):", O)
            
        if options['circuits text']: 
            if len(sel_circ_text[i])==2:
                (ry_seq, rx_seq) =sel_circ_text[i]
                print('')
                print(f"SEEQST circuit text:")
                print("  Circuit 1:", ry_seq)
                print("  Circuit 2:", rx_seq) 
            else: 
                print('')
                print(f"SEEQST circuit text:")
                print("  Circuit 1:", sel_circ_text[i])
           
            
        if options['non entangling circuits text']:
            print('')
            print("Non-entangling cirucits:")
            print(sel_circ_text_non_entangle[i])
            

    if options["density matrix plot"]:
        plot_density_matrix_highlight(wanted_indexes, result, selective_blocks, N)  
        
    plot_rho = create_plotter(wanted_indexes, result, selective_blocks, N)      
            
    sel_circ_qiskit=[ parse_circuit(i,N) for i in sel_circ_text] 
    non_e_circ_qiskit=[parse_circuit(i,N) for i in sel_circ_text_non_entangle]       
    
    return selective_blocks, sel_circ_text, sel_circ_text_non_entangle, sel_circ_qiskit, non_e_circ_qiskit, plot_rho
    
    
    
"""    
from setup import get_selective_blocks

print("═══ Test get_selective_blocks ═══\n")

# ── Test 1: Beispiel aus dem Paper (N=2) ──────────────────────
print("── Test 1: N=2, wanted=[(0,0), (0,3)] ──")
blocks, bmap = get_selective_blocks(2, [(0,0), (0,3)])
print("Selective blocks:", blocks)
print("Block map:       ", bmap)
assert blocks == [0, 3],        "Fehler: falsche Blöcke"
assert 0 in bmap,               "Fehler: Block 0 fehlt"
assert 3 in bmap,               "Fehler: Block 3 fehlt"
assert (0,0) in bmap[0],        "Fehler: (0,0) nicht in Block 0"
assert (0,3) in bmap[3],        "Fehler: (0,3) nicht in Block 3"
print("✓ Test 1 bestanden\n")

# ── Test 2: Volle erste Zeile (N=2) ───────────────────────────
print("── Test 2: N=2, volle erste Zeile [(0,j) für j=0..3] ──")
wanted = [(0,j) for j in range(4)]
blocks, bmap = get_selective_blocks(2, wanted)
print("Selective blocks:", sorted(blocks))
print("Block map:       ", bmap)
assert sorted(blocks) == [0, 1, 2, 3], "Fehler: nicht alle 4 Blöcke"
print("✓ Test 2 bestanden\n")

# ── Test 3: XOR Logik prüfen ──────────────────────────────────
print("── Test 3: XOR Logik ──")
# (0,1): 00 XOR 01 = 01 = 1 → Block 1
# (2,3): 10 XOR 11 = 01 = 1 → auch Block 1!
blocks, bmap = get_selective_blocks(2, [(0,1), (2,3)])
print("Selective blocks:", blocks)
print("Block map:       ", bmap)
assert blocks == [1],           "Fehler: beide sollten Block 1 sein"
assert len(bmap[1]) == 2,       "Fehler: Block 1 sollte 2 Einträge haben"
print("✓ Test 3 bestanden\n")

# ── Test 4: N=3 ───────────────────────────────────────────────
print("── Test 4: N=3, wanted=[(0,7)] ──")
# 0 = 000, 7 = 111 → XOR = 111 = 7 → Block 7
blocks, bmap = get_selective_blocks(3, [(0,7)])
print("Selective blocks:", blocks)
assert blocks == [7],           "Fehler: sollte Block 7 sein"
print("✓ Test 4 bestanden\n")

# ── Test 5: Ungültiger Index ──────────────────────────────────
print("── Test 5: Ungültiger Index ──")
try:
    get_selective_blocks(2, [(0, 5)])  # 5 > 2^2-1 = 3
    print("✗ Fehler: Exception hätte ausgelöst werden sollen")
except AssertionError as e:
    print("✓ Test 5 bestanden: Exception korrekt ausgelöst")
    print("  Exception:", e)

print("\n═══ Alle Tests bestanden ✓ ═══")


from setup import generate_selective_elements

print("═══ Test generate_selective_elements ═══\n")

# ── Test 1: Block 3 (N=2) ─────────────────────────────────────
print("── Test 1: N=2, Block 3 = 11 ──")
result = generate_selective_elements([3], [(0,3)], 2)
print("Elemente:", result[0])
assert set(result[0]) == {(0,3),(1,2),(2,1),(3,0)}, "Fehler: falsche Elemente"
assert len(result[0]) == 4,                          "Fehler: sollte 4 Elemente haben"
print("✓ Test 1 bestanden\n")

# ── Test 2: Block 0 (N=2) ─────────────────────────────────────
print("── Test 2: N=2, Block 0 = 00 (diagonal) ──")
result = generate_selective_elements([0], [(0,0)], 2)
print("Elemente:", result[0])
assert set(result[0]) == {(0,0),(1,1),(2,2),(3,3)}, "Fehler: sollte Diagonale sein"
print("✓ Test 2 bestanden\n")

# ── Test 3: Block 1 (N=2) ─────────────────────────────────────
print("── Test 3: N=2, Block 1 = 01 ──")
result = generate_selective_elements([1], [(0,1)], 2)
print("Elemente:", result[0])
assert set(result[0]) == {(0,1),(1,0),(2,3),(3,2)}, "Fehler: falsche Elemente"
print("✓ Test 3 bestanden\n")

# ── Test 4: Mehrere Blöcke (N=2) ──────────────────────────────
print("── Test 4: N=2, alle Blöcke [0,1,2,3] ──")
wanted = [(0,j) for j in range(4)]
result = generate_selective_elements([0,1,2,3], wanted, 2)
all_elements = set(e for block in result for e in block)
expected     = {(i,j) for i in range(4) for j in range(4)}
assert all_elements == expected, "Fehler: nicht alle Elemente abgedeckt"
assert len(result) == 4,         "Fehler: sollte 4 Blöcke haben"
print("✓ Test 4 bestanden: alle 16 Elemente abgedeckt\n")

# ── Test 5: N=3, Block 7 ──────────────────────────────────────
print("── Test 5: N=3, Block 7 = 111 ──")
result = generate_selective_elements([7], [(0,7)], 3)
assert len(result[0]) == 8,  "Fehler: sollte 8 Elemente haben"
assert (0,7) in result[0],   "Fehler: (0,7) sollte enthalten sein"
assert (7,0) in result[0],   "Fehler: (7,0) sollte enthalten sein"
print("✓ Test 5 bestanden\n")

print("═══ Alle Tests bestanden ✓ ═══")


from setup import generate_observable_sets, build_parallel_entangler_blocks, build_non_entangling_circuits

print("═══ Tests Python ═══\n")

# ────────────────────────────────────────────────────────────
# generate_observable_sets
# ────────────────────────────────────────────────────────────
print("── generate_observable_sets ──")

# Test 1: Block 0
obs = generate_observable_sets([0], 2)
E, O = obs[0]
print("Block 0 - E:", E)
print("Block 0 - O:", O)
assert set(E) == {"II", "IZ", "ZI", "ZZ"}, "Fehler Block 0 E"
assert O == [],                              "Fehler Block 0 O"
print("✓ Block 0 bestanden\n")

# Test 2: Block 3
obs = generate_observable_sets([3], 2)
E, O = obs[3]
print("Block 3 - E:", E)
print("Block 3 - O:", O)
assert set(E) == {"XX", "YY"}, "Fehler Block 3 E"
assert set(O) == {"XY", "YX"}, "Fehler Block 3 O"
print("✓ Block 3 bestanden\n")

# Test 3: Block 1
obs = generate_observable_sets([1], 2)
E, O = obs[1]
print("Block 1 - E:", E)
print("Block 1 - O:", O)
print("✓ Block 1 bestanden\n")

# ────────────────────────────────────────────────────────────
# build_parallel_entangler_blocks
# ────────────────────────────────────────────────────────────
print("── build_parallel_entangler_blocks ──")

# Test 1: Block 0 → keine Gates
seq = build_parallel_entangler_blocks([0], 2)
print("Block 0:", seq[0])
assert seq[0] == [''],  "Fehler: Block 0 sollte leer sein"
print("✓ Block 0 bestanden\n")

# Test 2: Block 3
seq = build_parallel_entangler_blocks([3], 2)
print("Block 3 Circuit 1 (RY):", seq[0][0])
print("Block 3 Circuit 2 (RX):", seq[0][1])
assert "CNOT" in seq[0][0], "Fehler: CNOT fehlt in Circuit 1"
assert "RY90" in seq[0][0], "Fehler: RY90 fehlt in Circuit 1"
assert "CNOT" in seq[0][1], "Fehler: CNOT fehlt in Circuit 2"
assert "RX90" in seq[0][1], "Fehler: RX90 fehlt in Circuit 2"
print("✓ Block 3 bestanden\n")

# Test 3: N=3, Block 7
seq = build_parallel_entangler_blocks([7], 3)
print("Block 7 Circuit 1:", seq[0][0])
print("Block 7 Circuit 2:", seq[0][1])
assert "RY90" in seq[0][0], "Fehler: RY90 fehlt"
assert "RX90" in seq[0][1], "Fehler: RX90 fehlt"
print("✓ Block 7 bestanden\n")

# ────────────────────────────────────────────────────────────
# build_non_entangling_circuits
# ────────────────────────────────────────────────────────────
print("── build_non_entangling_circuits ──")

# Test 1: Block 0
circ = build_non_entangling_circuits([0], 2)
print("Block 0:", circ[0])
assert circ[0] == [''], "Fehler: Block 0 sollte leer sein"
print("✓ Block 0 bestanden\n")

# Test 2: Block 3
circ = build_non_entangling_circuits([3], 2)
print("Block 3:", circ[0])
assert len(circ[0]) == 4,                                       "Fehler: sollte 4 Kombinationen haben"
assert any("RY90:0" in c and "RY90:1" in c for c in circ[0]), "Fehler: RY90+RY90 fehlt"
assert any("RX90:0" in c and "RX90:1" in c for c in circ[0]), "Fehler: RX90+RX90 fehlt"
print("✓ Block 3 bestanden\n")

# Test 3: Block 1
circ = build_non_entangling_circuits([1], 2)
print("Block 1:", circ[0])
assert len(circ[0]) == 2, "Fehler: sollte 2 Kombinationen haben"
print("✓ Block 1 bestanden\n")

# Test 4: N=3, Block 7
circ = build_non_entangling_circuits([7], 3)
print("Block 7 Anzahl Circuits:", len(circ[0]))
assert len(circ[0]) == 8, "Fehler: sollte 8 Kombinationen haben"
print("✓ Block 7 bestanden\n")

print("═══ Alle Python Tests bestanden ✓ ═══")
"""

from setup import generate_experiment

print("═══ Tests GenerateExperiment Python ═══\n")

options = {
    "density matrix plot": False,
    "wanted elements": True,
    "selective elements": True,
    "circuits text": True,
    "non entangling circuits text": True,
    "observable": True
}

# ── Test 1: N=2, wanted=[(0,0),(0,3)] ─────────────────────────
print("── Test 1: N=2, wanted=[(0,0),(0,3)] ──\n")
blocks, ent, non_ent, ent_qiskit, non_ent_qiskit, plot = generate_experiment(
    [(0,0),(0,3)], 2, options
)

assert blocks == [0, 3],                        "Fehler: falsche Blöcke"
assert ent[0] == [''],                          "Fehler: Block 0 Circuit"
assert ent[1][0] == "(CNOT:0,1)(RY90:0)",      "Fehler: Block 3 Circuit 1"
assert ent[1][1] == "(CNOT:0,1)(RX90:0)",      "Fehler: Block 3 Circuit 2"
assert len(non_ent[0]) == 1,                   "Fehler: Block 0 Non-Ent"
assert len(non_ent[1]) == 4,                   "Fehler: Block 3 Non-Ent Anzahl"
print("✓ Test 1 bestanden\n")

# ── Test 2: N=2, volle Tomographie ────────────────────────────
print("── Test 2: N=2, volle erste Zeile ──\n")
wanted = [(0,j) for j in range(4)]
blocks, ent, non_ent, ent_qiskit, non_ent_qiskit, plot = generate_experiment(
    wanted, 2, options
)

assert sorted(blocks) == [0,1,2,3],  "Fehler: nicht alle 4 Blöcke"
assert len(ent) == 4,                "Fehler: sollte 4 Circuit-Gruppen haben"
print("✓ Test 2 bestanden\n")

# ── Test 3: N=3, wanted=[(0,7)] ───────────────────────────────
print("── Test 3: N=3, wanted=[(0,7)] ──\n")
blocks, ent, non_ent, ent_qiskit, non_ent_qiskit, plot = generate_experiment(
    [(0,7)], 3, options
)

assert blocks == [7],           "Fehler: sollte Block 7 sein"
assert len(non_ent[0]) == 8,   "Fehler: sollte 8 Non-Ent Circuits haben"
print("✓ Test 3 bestanden\n")

print("═══ Alle Python Tests bestanden ✓ ═══")