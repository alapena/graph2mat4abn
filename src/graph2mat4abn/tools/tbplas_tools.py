import math
import numpy as np
import tbplas as tb
import sisl
from pathlib import Path

def add_orbitals(cell: tb.PrimitiveCell, positions, onsites, labels) -> None:
    """
    Add orbitals to the model.

    There are n_atoms atoms, with n_orbs orbitals each in that same position. We will extract those orbitals from the atom info.
    """
    for i in range(positions.shape[0]):
        n_orbs = len(labels[i])
        for j in range(n_orbs):
            cell.add_orbital_cart(positions[i], unit=tb.ANG, energy=onsites[i*n_orbs+j], label=labels[i][j])


def add_hopping_terms(cell: tb.PrimitiveCell, iscs, orbs_in, orbs_out, hoppings) -> None:
    n_hops = len(iscs)
    for i in range(n_hops):
        cell.add_hopping(rn=iscs[i], orb_i=orbs_in[i], orb_j=orbs_out[i], energy=hoppings[i])

def extract_onsites_from_coo(hamiltonian_coo):
    rows = hamiltonian_coo.row
    cols = hamiltonian_coo.col
    data = hamiltonian_coo.data

    # Main diagonal length:
    n_diag = min(hamiltonian_coo.shape[0], hamiltonian_coo.shape[1])

    # Loop through all diagonal elements
    onsites = np.zeros(n_diag, dtype=data.dtype)
    for i in range(n_diag):
        # Find where both row and col equal i
        mask = (rows == i) & (cols == i)
        vals = data[mask]
        if len(vals) > 0:
            onsites[i] = vals[0]  # In COO, there could be duplicates, but take the first
        else:
            onsites[i] = 0  # Or np.nan if you prefer

    return onsites

def extract_hoppings_from_coo(h_mat, n_atoms, geometry):
    rows = h_mat.row
    cols = h_mat.col
    data = h_mat.data

    nnz = len(data)
    n_orbs = h_mat.shape[0] // n_atoms # Assuming all atoms have the same nr of orbitals
    iscs = []
    orbs_in = []
    orbs_out = []
    hoppings = []
    for k in range(nnz):
        row = rows[k]
        col = cols[k]
        if row != col:  # Only add hopping terms for off-diagonal elements
            iscs.append(geometry.o2isc(col))
            orbs_in.append(col % (n_atoms*n_orbs))
            orbs_out.append(row)
            hoppings.append(data[k])

    return iscs, orbs_in, orbs_out, hoppings


def compute_k_len(k_path):
    """
    Compute cumulative k-path length manually.
    k_points: ndarray of shape (N, d), where N = number of k-points, d = dimension (2 or 3).
    Returns: ndarray of shape (N,), cumulative path length.
    """
    # Differences between consecutive k-points
    diffs = np.diff(k_path, axis=0)
    # Euclidean distances
    distances = np.linalg.norm(diffs, axis=1)
    # Cumulative sum, prepend 0
    k_len = np.concatenate(([0], np.cumsum(distances)))
    return k_len


def select_kpath(n_atoms, cell, n_kpoints=160, structure=None):
    b1, b2, b3 = cell.get_reciprocal_vectors()/10 # Angstrom^-1
    B = np.vstack([b1, b2, b3])  # shape (3,3)
    match n_atoms:
        case 2:
            k_cart = np.array([[0.0, 0.0, 0.0], b1, b2, b3])
            k_label = ['Γ', "X", "Y", "Z"]

        case 8:
            if structure=="hBN":
                frac_kpts = np.array([
                    [0.0,         0.0,         0.0],  # Γ
                    [0.5,         0.0,         0.0],  # M
                    [1/3,         1/3,         0.0],  # K
                    [0.0,         0.0,         0.0],  # Γ
                    [0.0,         0.0,         0.5],  # A
                    [0.5,         0.0,         0.5],  # L
                    [1/3,         1/3,         0.5],  # H
                    [1/3,         1/3,         0.0],  # K
                    [1/3,         1/3,        -0.5],  # H2
                ])
                k_cart = frac_kpts @ np.array([b1, b2, b3])
                k_label = ['Γ', 'M', 'K', 'Γ', 'A', 'L', 'H', 'K', 'H2']
            elif structure=="cBN":
                frac_kpts = np.array([
                    [0.0, 0.0, 0.0],  # Γ
                    [0.0, 0.5, 0.0],  # X
                    [0.5, 0.5, 0.0],  # M
                    [0.0, 0.0, 0.0],  # Γ
                    [0.5, 0.5, 0.5],  # R
                    [0.0, 0.5, 0.0],  # X
                    [0.5, 0.5, 0.0],  # M
                    [0.5, 0.0, 0.0],  # X1
                ])
                k_cart = frac_kpts @ np.array([b1, b2, b3])
                k_label = ['Γ', 'X', 'M', 'Γ', 'R', 'X', 'M', 'X1']

            else:
                frac_kpts = np.array([
                    [0.0, 0.0, 0.0],  # Γ
                    [1.0, 0.0, 0.0],  # X
                    [1.0, 1.0, 0.0],  # XY
                    [1.0, 1.0, 1.0],  # XYZ
                    [0.0, 0.0, 0.0],  # Γ
                    [0.0, 1.0, 0.0],  # Y
                    [0.0, 1.0, 1.0],  # YZ
                    [1.0, 1.0, 1.0],  # XYZ
                ])
                k_cart = frac_kpts @ np.array([b1, b2, b3])
                k_label = ['Γ', 'X', 'XY', 'XYZ', 'Γ', 'Y', 'YZ', 'XYZ']

        case _:
            frac_kpts = np.array([
                [0.0,         0.0,         0.0],  # Γ
                [1.0,         1.0,         1.0],  # Γ
            ])
            k_cart = frac_kpts @ np.array([b1, b2, b3])
            k_label = ['Γ', 'XYZ']

    k_frac = np.array([np.linalg.solve(B.T, k) for k in k_cart])

    k_path, k_idx = tb.gen_kpath(k_frac, [n_kpoints for _ in range(len(k_frac) -1)])
            
    return k_path, k_idx, k_label