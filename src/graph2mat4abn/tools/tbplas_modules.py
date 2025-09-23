import tbplas as tb
import numpy as np

from .tbplas_tools import get_onsites, add_orbitals, get_hoppings, add_hopping_terms

def get_tbplas_cell(geometry, hamiltonian, overlap=None, units=tb.ANG):
    '''
    From SISL geometry

    Inputs:
        geometry, sisl.Geometry
        hamiltonian, COO matrix
        overlap, COO matrix
    '''
    lattice_vectors = geometry.cell
    orbital_pos = geometry.xyz #Angstrom
    orbital_labels = [[orb.name() for orb in atom] for atom in geometry.atoms]
    n_atoms = len(geometry.atoms.Z)

    ham_cell = tb.PrimitiveCell(lattice_vectors, unit=units)

    onsites = get_onsites(hamiltonian)
    add_orbitals(ham_cell, orbital_pos, onsites, orbital_labels)

    iscs, orbs_in, orbs_out, hoppings = get_hoppings(hamiltonian, n_atoms, geometry)
    add_hopping_terms(ham_cell, iscs, orbs_in, orbs_out, hoppings)

    # Now add overlap
    if overlap is not None:
        overlap_cell = tb.PrimitiveCell(ham_cell.lat_vec, ham_cell.origin, 1.0)

        onsites = get_onsites(overlap)
        for j in range(ham_cell.num_orb):
            orbital = ham_cell.orbitals[j]
            overlap_cell.add_orbital(orbital.position, onsites[j])

        iscs, orbs_in, orbs_out, hoppings = get_hoppings(overlap, n_atoms, geometry)
        add_hopping_terms(overlap_cell, iscs, orbs_in, orbs_out, hoppings)

        return ham_cell, overlap_cell
    else:
        return ham_cell
