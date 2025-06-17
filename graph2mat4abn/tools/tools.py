import sisl
import torch

from graph2mat import PointBasis
from tqdm import tqdm


def z_one_hot(z, orbitals, nr_bits):
    """
    Generate one-hot encodings from a list of single-value tensors.

    Args:
        z (list of torch.Tensor): A list of single-value tensors, e.g., [[2], [3], [4], [2], [2], ...].
        orbitals (dict): A dictionary mapping numbers to their corresponding values.
        nr_bits (int): The number of bits for one-hot encoding.

    Returns:
        torch.Tensor: A tensor containing the one-hot encodings.
    """

    # Extract values from the list of single-value tensors
    node_map={}
    k=0
    for key in orbitals.keys():
        node_map[key]=k
        k+=1

    indices = [tensor.item() for tensor in z]

    # Create an empty tensor for one-hot encoding
    one_hot = torch.zeros(len(indices), nr_bits)

    # Fill in the one-hot encoding based on the indices
    for i, idx in enumerate(indices):
        if idx in orbitals:  # Ensure the index exists in orbitals
            one_hot[i, int(node_map[idx])] = 1  # Set the corresponding bit to 1
        else:
            raise ValueError(f"Index {idx} not found in orbitals.")

    return one_hot

def flatten(xss):
    return [x for xs in xss for x in xs]



def get_basis_from_structures_paths(paths, verbose=False, num_unique_z=None):
    """_summary_

    Args:
        paths (list[str]): _description_
        verbose (bool, optional): _description_. Defaults to False.
        num_unique_z (int, optional): If known, input the total number of different atom types here for faster performance.

    Returns:
        list[PointBasis]: _description_
    """
    if verbose:
        print("="*60)
        print("Basis computation.")
        print(f"Number of structures to look on: {len(paths)}")
        print("Looking for unique atoms in each structure...")

    unique_atom_types = []
    unique_atom_types_path_idx = []

    # Look for all atom types in your list of structures
    iterator = tqdm(enumerate(paths)) if verbose else enumerate(paths)
    for i, path in iterator:
        geometry = sisl.get_sile(path / "aiida.fdf").read_geometry()
        for z in geometry.atoms.Z:
            if z not in unique_atom_types:
                unique_atom_types.append(z)
                unique_atom_types_path_idx.append(i)
        if num_unique_z is not None and len(unique_atom_types) == num_unique_z:
            break

    if verbose:
        print(f"Found the following atomic numbers: {unique_atom_types}")
        print(f"Corresponding path indices: {unique_atom_types_path_idx}")

    # Build the basis
    basis = []
    unique_atom_types_basis = []
    for path_idx in unique_atom_types_path_idx:
        geometry = sisl.get_sile(paths[path_idx] / "aiida.fdf").read_geometry()
        for atom in geometry.atoms:
            if atom.Z not in unique_atom_types_basis:
                basis.append(PointBasis.from_sisl_atom(atom))
                unique_atom_types_basis.append(atom.Z)

    basis.sort(key=lambda x: x.type)
    unique_atom_types_basis.sort()

    if verbose:
        print(f"Basis with {len(basis)} elements built!")
        [print(f"\nBasis for atom {i}.\n\tAtom type: {basis[i].type}\n\tBasis: {basis[i].basis}\n\tBasis convention: {basis[i].basis_convention}\n\tR: {basis[i].R}") for i in range(len(basis))]

    return basis