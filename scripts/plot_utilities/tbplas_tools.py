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

def get_onsites(h_mat):
    rows = h_mat.row
    cols = h_mat.col
    data = h_mat.data

    # Main diagonal length:
    n_diag = min(h_mat.shape[0], h_mat.shape[1])

    # Loop through all diagonal elements
    onsites_true = np.zeros(n_diag, dtype=data.dtype)
    for i in range(n_diag):
        # Find where both row and col equal i
        mask = (rows == i) & (cols == i)
        vals = data[mask]
        if len(vals) > 0:
            onsites_true[i] = vals[0]  # In COO, there could be duplicates, but take the first
        else:
            onsites_true[i] = 0  # Or np.nan if you prefer

    return onsites_true

def get_hoppings(h_mat, n_atoms, geometry):
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