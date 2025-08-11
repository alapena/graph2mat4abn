# # === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
# import sys
# from pathlib import Path
# # # Add the root directory to Python path
# # root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
# # sys.path.append(str(root_dir))

import numpy as np
# from tools.scripts_utils import real_space_to_kspace
import warnings
import sisl
import tbplas as tb
from plot_utilities.tbplas_tools import add_hopping_terms, add_orbitals, get_hoppings, get_onsites
from pathlib import Path



def main():
    # *********************************** #
    # * VARIABLES TO CHANGE BY THE USER * #
    # *********************************** #
    path = Path("/home/alapena/GitHub/graph2mat4abn/dataset/SHARE_OUTPUTS_64_ATOMS/9410-b52a-4124-9c9c-210304f661a1")
    savedir = Path("results_convergence_test_64atm")
    nks_dos = range(2,25)
    # *********************************** #

    # Define orbital labels (for now we will assume that all atoms have the same orbitals). Use the same order as appearance in the hamiltonian.
    orbitals = {
        0: "s1",
        1: "s2",
        2: "py1",
        3: "pz1",
        4: "px1",
        5: "py2",
        6: "pz2",
        7: "px2",
        8: "Pdxy",
        9: "Pdyz",
        10: "Pdz2",
        11: "Pdxz",
        12: "Pdx2-y2",
    }
    n_orbs = len(orbitals)

    # Hide some warnings
    warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
    warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")

    savedir.mkdir(exist_ok=True, parents=True)

    n_atoms = int(path.parts[-2].split('_')[2])
    structure = path.parts[-1]

    # 1. Plot structure
    file = sisl.get_sile(path / "aiida.fdf")
    fig = file.plot.geometry(axes="xyz")
    filepath = savedir / f"{n_atoms}atm_{structure}.png"
    fig.write_image(str(filepath))
    filepath = savedir / f"{n_atoms}atm_{structure}.html"
    fig.write_html(str(filepath))
    print("Saved structure plot at", filepath)


    # 2. TBPLaS
    file = sisl.get_sile(path / "aiida.HSX")
    geometry = file.read_geometry()

    vectors = geometry.cell
    cell = tb.PrimitiveCell(vectors, unit=tb.ANG)

    # Add orbitals
    positions = geometry.xyz #Angstrom
    labels = [[orb.name() for orb in atom] for atom in geometry.atoms]

    # To add the orbitals we need the onsite energies.
    h = file.read_hamiltonian()
    h_mat = h.tocsr().tocoo()

    rows = h_mat.row
    cols = h_mat.col
    data = h_mat.data

    # Main diagonal length:
    n_diag = min(h_mat.shape[0], h_mat.shape[1])

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

    # onsites = 
    add_orbitals(cell, positions, onsites, labels)


    # Add hopping terms.
    # We need to iterate though each nnz element of h and get the isc in a tuple, the orb_in, the orb_out and the hopping value.
    nnz = len(data)
    n_orbs = len(labels[0]) # Assuming all atoms have the same nr of orbitals
    n_atoms = int(path.parts[-2].split("_")[-2])
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
        

    add_hopping_terms(cell, iscs, orbs_in, orbs_out, hoppings)



    # Get overlap

    # To add the orbitals we need the onsite energies.
    o = file.read_overlap()
    o_mat = o.tocsr().tocoo()

    rows = o_mat.row
    cols = o_mat.col
    data = o_mat.data

    # Main diagonal length:
    n_diag = min(o_mat.shape[0], o_mat.shape[1])

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

    # Add onsites to overlap
    overlap = tb.PrimitiveCell(cell.lat_vec, cell.origin, 1.0)
    for i in range(cell.num_orb):
        orbital = cell.orbitals[i]
        overlap.add_orbital(orbital.position, onsites[i])


    # Add hopping terms to overlap

    # We need to iterate though each nnz element of h and get the isc in a tuple, the orb_in, the orb_out and the hopping value.
    nnz = len(data)
    n_orbs = len(labels[0]) # Assuming all atoms have the same nr of orbitals
    n_atoms = int(path.parts[-2].split("_")[-2])
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
            

    add_hopping_terms(overlap, iscs, orbs_in, orbs_out, hoppings)


    # Define a path in k-space

    b1, b2, b3 = cell.get_reciprocal_vectors()/10 # Angstrom^-1
    k_pos_frac, k_pos_cart = real_space_to_kspace(positions, b1, b2, b3)

    # Compute k path
    B = np.vstack([b1, b2, b3])  # shape (3,3)
    k_cart = np.array([
        [0.0, 0.0, 0.0],          # Γ
        b1/2,                     # X
        (b1 + b2)/2,              # M
        [0.0, 0.0, 0.0],          # Γ
        (b1 + b2 + b3)/2,         # R
        b1/2,                     # X
        (b1 + b2)/2,              # M (optional segment)
        (b1 + b2 + b3)/2          # R (optional segment end)
    ])

    k_label = ['Γ', 'X', 'M', 'Γ', 'R', 'X', 'M', 'R']

    k_frac = np.array([np.linalg.solve(B.T, k) for k in k_cart])

    n_ks = 200
    k_path, k_idx = tb.gen_kpath(k_frac, [n_ks for _ in range(len(k_frac) -1)])


    # Compute bands
    solver = tb.DiagSolver(cell, overlap)
    solver.config.k_points = k_path
    solver.config.prefix = "Bands"

    timer = tb.Timer()
    timer.tic("bands")
    k_len, bands = solver.calc_bands()
    timer.toc("bands")

    filepath = savedir / f"{structure}_bands.npz"
    np.savez(filepath, path=str(path), k_idx=k_idx, k_label=k_label, k_len=k_len, bands=bands,)


    # Compute dos
    for nk in nks_dos:
        print(f"Computing DOS for {nk} mesh...")

        k_mesh = tb.gen_kmesh((nk, nk, nk))  # Uniform meshgrid
        e_min = float(np.min(bands))
        e_max = float(np.max(bands))

        solver = tb.DiagSolver(cell, overlap)
        solver.config.k_points = k_mesh
        solver.config.prefix = "dos"
        solver.config.e_min = e_min
        solver.config.e_max = e_max
        timer = tb.Timer()
        timer.tic("dos")
        energies, dos = solver.calc_dos()
        timer.toc("dos")

        filepath = savedir / f"{structure}_dos_mesh{nk}.npz"
        np.savez(filepath, path=str(path), energies=energies, dos=dos)


def real_space_to_kspace(positions, b1, b2, b3):
    """
    Map real-space positions into reciprocal space (fractional and cartesian).
    Returns:
        k_frac: (N, 3) positions in fractional reciprocal coordinates
        k_cart: (N, 3) positions in cartesian k-space (nm^-1)
    """
    B = np.vstack([b1, b2, b3])  # reciprocal lattice vectors (3,3), Ang^-1
    # Fractional reciprocal coordinates
    k_frac = np.linalg.solve(B.T, positions.T).T  # shape (N,3)
    # Cartesian k-vectors
    k_cart = k_frac @ B
    return k_frac, k_cart


if __name__ == "__main__":
    main()