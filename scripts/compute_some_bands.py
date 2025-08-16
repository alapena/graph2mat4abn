# import sys
# sys.path.insert(0, '/home/alapena/GitHub/graph2mat4abn')
# import os
# os.chdir('/home/ICN2/alapena/GitHub/graph2mat4abn') # Change to the root directory of the project

from graph2mat4abn.tools.import_utils import load_config, get_object_from_module, read_fermi_level
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, get_kwargs, load_model
from graph2mat4abn.tools.scripts_utils import get_model_dataset, init_mace_g2m_model
from graph2mat4abn.tools.script_plots import update_loss_plots, plot_grad_norms
from torch_geometric.data import DataLoader
from graph2mat4abn.tools.scripts_utils import generate_g2m_dataset_from_paths
from pathlib import Path
from mace.modules import MACE, RealAgnosticResidualInteractionBlock
from graph2mat.models import MatrixMACE
from graph2mat.bindings.e3nn import E3nnGraph2Mat
import torch
import warnings
from graph2mat import BasisTableWithEdges

warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")

from joblib import dump, load
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sisl
import tbplas as tb

def main():
    paths = [
        # h_noc_2:
        # Training:
        # Path("dataset/SHARE_OUTPUTS_64_ATOMS/99a9-5416-41e9-940d-70653c6897f9")
        # Val:
        # Path("dataset/SHARE_OUTPUTS_2_ATOMS/a4a5-71a5-463a-a02e-acd977e1dcda"),
        # Path("dataset/SHARE_OUTPUTS_2_ATOMS/504a-71cd-4d25-a04a-b7fa45b92200"),
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/72f5-effe-42c4-bc67-12314ba36f5e"),
        # Path("dataset/SHARE_OUTPUTS_64_ATOMS/16eb-54f8-42cb-bdb1-7b16f24a650c"),

        # h_c_8
        # train
        # Path("dataset/SHARE_OUTPUTS_2_ATOMS/9b13-4a57-4de9-b863-1b35209370c4"),
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/4ed6-914e-4aa3-923a-53c873f0cc31"),

        #val
        # Path("dataset/SHARE_OUTPUTS_2_ATOMS/fc1c-6ab6-4c0e-921e-99710e6fe41b"),
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/7b57-1410-4da3-8535-5183ac1f2f61")
        # Path("dataset/SHARE_OUTPUTS_64_ATOMS/16eb-54f8-42cb-bdb1-7b16f24a650c"),

        # h_c_15
        #train 
        Path("dataset/SHARE_OUTPUTS_8_ATOMS/e1df-2940-4ada-b9c0-d210a6bb2a19"),

        # val
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/ffb8-6623-4c32-9320-49b359a920b6")
        
    ]
    # The current model:
    model_dir = Path("results/h_crystalls_15")
    filename = "train_best_model.tar"
    savedir = Path('results_dos/h_crystalls_15_train/bands')
    only_pred = False
    only_true = False

    config = load_config(model_dir / "config.yaml")
    print(paths)

    # Basis generation (needed to initialize the model)
    train_paths, val_paths = get_model_dataset(model_dir, verbose=False)
    # paths = train_paths + val_paths
    basis = get_basis_from_structures_paths(train_paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)

    print("Initializing model...")
    model, optimizer, lr_scheduler, loss_fn = init_mace_g2m_model(config, table)

    # Load the model
    model_path = model_dir / filename
    model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, model_path, lr_scheduler=None, initial_lr=None, device='cpu')
    history = checkpoint["history"]
    print(f"Loaded model {model_dir} in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")

    savedir.mkdir(exist_ok=True, parents=True)
    print("Created savedir", savedir)

    for i, path in enumerate(paths):
        print("\nComputing ", path)
        # === Inference ===
        dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, [path], verbose=False)
        dataloader = DataLoader(dataset, 1)
        model.eval()

        data = next(iter(dataloader))

        with torch.no_grad():
            model_predictions = model(data=data)

            h_pred = processor.matrix_from_data(data, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})[0]
            h_true = processor.matrix_from_data(data)[0]

        n_atoms = int(path.parts[-2].split('_')[2])
        structure = path.parts[-1]

        print(f"Computing structure {n_atoms}atm_{structure}...")
        
        # Plot structure
        file = sisl.get_sile(path / "aiida.fdf")
        fig = file.plot.geometry(axes="xyz")
        filepath = savedir / f"{n_atoms}atm_{structure}.png"
        fig.write_image(str(filepath))
        filepath = savedir / f"{n_atoms}atm_{structure}.html"
        fig.write_html(str(filepath))
        print("Saved structure plot at", filepath)

        file = sisl.get_sile(path / "aiida.HSX")
        geometry = file.read_geometry()

        vectors = geometry.cell

        # Add orbitals
        positions = geometry.xyz #Angstrom
        labels = [[orb.name() for orb in atom] for atom in geometry.atoms]

        # To add the orbitals we need the onsite energies.
        h = file.read_hamiltonian()
        for ham_idx in range(2):
            cell = tb.PrimitiveCell(vectors, unit=tb.ANG)
            if ham_idx == 0:
                h_mat = h.tocsr().tocoo()
                if only_pred:
                    continue
            if ham_idx == 1:
                if only_true:
                    break
                h_mat = h_pred.tocsr().tocoo()

            rows = h_mat.row
            cols = h_mat.col
            data = h_mat.data

            # Main diagonal length:
            n_diag = min(h_mat.shape[0], h_mat.shape[1])

            # Loop through all diagonal elements
            onsites = np.zeros(n_diag, dtype=data.dtype)
            for j in range(n_diag):
                # Find where both row and col equal i
                mask = (rows == j) & (cols == j)
                vals = data[mask]
                if len(vals) > 0:
                    onsites[j] = vals[0]  # In COO, there could be duplicates, but take the first
                else:
                    onsites[j] = 0  # Or np.nan if you prefer

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
            for j in range(n_diag):
                # Find where both row and col equal i
                mask = (rows == j) & (cols == j)
                vals = data[mask]
                if len(vals) > 0:
                    onsites[j] = vals[0]  # In COO, there could be duplicates, but take the first
                else:
                    onsites[j] = 0  # Or np.nan if you prefer

            # Add onsites to overlap
            overlap = tb.PrimitiveCell(cell.lat_vec, cell.origin, 1.0)
            for j in range(cell.num_orb):
                orbital = cell.orbitals[j]
                overlap.add_orbital(orbital.position, onsites[j])


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
            # k_pos_frac, k_pos_cart = real_space_to_kspace(positions, b1, b2, b3)

            # Compute k path (not definitive to use in the report)
            B = np.vstack([b1, b2, b3])  # shape (3,3)

            # # For Molecules:
            # k_cart = np.array([[0.0, 0.0, 0.0], b1, b2, b3])
            # k_label = ['Γ', "X", "Y", "Z"]

            # # For cubic:
            # # High-symmetry points in fractional coords (relative to b1, b2, b3)
            # frac_kpts = np.array([
            #     [0.0, 0.0, 0.0],  # Γ
            #     [0.0, 0.5, 0.0],  # X
            #     [0.5, 0.5, 0.0],  # M
            #     [0.0, 0.0, 0.0],  # Γ
            #     [0.5, 0.5, 0.5],  # R
            #     [0.0, 0.5, 0.0],  # X
            #     [0.5, 0.5, 0.0],  # M
            #     [0.5, 0.0, 0.0],  # X1
            # ])
            # k_cart = frac_kpts @ np.array([b1, b2, b3])
            # k_label = ['Γ', 'X', 'M', 'Γ', 'R', 'X', 'M', 'X1']

            # Physical hBN
            # High-symmetry points in fractional coords (relative to b1, b2, b3)
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

            # # amorphous
            # frac_kpts = np.array([
            #     [0.0,         0.0,         0.0],  # Γ
            #     [1.0,         1.0,         1.0],  # Γ
            # ])
            # k_cart = frac_kpts @ np.array([b1, b2, b3])
            # k_label = ['Γ', 'XYZ']


            k_frac = np.array([np.linalg.solve(B.T, k) for k in k_cart])

            n_ks = 80
            k_path, k_idx = tb.gen_kpath(k_frac, [n_ks for _ in range(len(k_frac) -1)])
            len(k_path)

            solver = tb.DiagSolver(cell, overlap)
            solver.config.k_points = k_path
            solver.config.prefix = "bands"

            timer = tb.Timer()
            timer.tic("bands")
            k_len, bands = solver.calc_bands()
            timer.toc("bands")
            timer.report_total_time()


            if ham_idx == 0:
                filepath = savedir / f"{n_atoms}atm_{structure}_bands_true.npz"
            if ham_idx == 1:
                filepath = savedir / f"{n_atoms}atm_{structure}_bands_pred.npz"
            np.savez(filepath, path=str(path), k_idx=k_idx, k_label=k_label, k_len=k_len, bands=bands,)











import numpy as np
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