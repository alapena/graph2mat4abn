import sys
sys.path.insert(0, '/home/alapena/GitHub/graph2mat4abn')
import os
os.chdir('/home/ICN2/alapena/GitHub/graph2mat4abn') # Change to the root directory of the project

from graph2mat4abn.tools.import_utils import load_config, get_object_from_module
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, get_kwargs, load_model
from graph2mat4abn.tools.scripts_utils import get_model_dataset, init_mace_g2m_model
from graph2mat4abn.tools.script_plots import update_loss_plots, plot_grad_norms
from torch_geometric.data import DataLoader
from graph2mat4abn.tools.scripts_utils import generate_g2m_dataset_from_paths
from pathlib import Path
from e3nn import o3
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

def main():
    paths = [
        Path("dataset/SHARE_OUTPUTS_64_ATOMS/9410-b52a-4124-9c9c-210304f661a1"), #64
        Path('dataset/SHARE_OUTPUTS_8_ATOMS/d4f5-6b48-494f-b1de-c7e944c09f38'), #Hexagonal normal
        Path('dataset/SHARE_OUTPUTS_8_ATOMS/11ad-ba95-4a26-8f92-5267f5787553'), # Cubic
        Path('dataset/SHARE_OUTPUTS_2_ATOMS/fc1c-6ab6-4c0e-921e-99710e6fe41b'), # N-N
        Path('dataset/SHARE_OUTPUTS_2_ATOMS/7bbb-6d51-41eb-9de4-329298202ebf'), #B-B
    ]
    # The current model:
    model_dir = Path("results/h_noc_2")
    filename = "train_best_model.tar"
    savedir = Path('results_dos')

    config = load_config(model_dir / "config.yaml")

    # Basis generation (needed to initialize the model)
    train_paths, val_paths = get_model_dataset(model_dir, verbose=False)
    paths = train_paths + val_paths
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)

    print("Initializing model...")
    model, optimizer, lr_scheduler, loss_fn = init_mace_g2m_model(config, table)

    # Load the model
    model_path = model_dir / filename
    model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, model_path, lr_scheduler=None, initial_lr=None, device='cpu')
    history = checkpoint["history"]
    print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")

    for i, path in enumerate(paths):
        # === Inference ===
        dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, [path], verbose=False)
        dataloader = DataLoader(dataset, 1)
        model.eval()

        data = next(iter(dataloader))

        with torch.no_grad():
            model_predictions = model(data=data)

            h_pred = processor.matrix_from_data(data, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})[0]
            h_true = processor.matrix_from_data(data)[0]

        savedir.mkdir(exist_ok=True, parents=True)

        n_atoms = int(path.parts[-2].split('_')[2])
        structure = path.parts[-1]
        
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
        cell = tb.PrimitiveCell(vectors, unit=tb.ANG)

        # Add orbitals
        positions = geometry.xyz #Angstrom
        labels = [[orb.name() for orb in atom] for atom in geometry.atoms]

        # To add the orbitals we need the onsite energies.
        h = file.read_hamiltonian()
        for ham_idx in range(2):
            if ham_idx == 0:
                h_mat = h.tocsr().tocoo()
            if ham_idx == 1:
                h_mat == h_pred.tocsr().tocoo()

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

            # Compute k path (not definitive to use in the report)
            B = np.vstack([b1, b2, b3])  # shape (3,3)
            k_cart = np.array([[0.0, 0.0, 0.0], b1, b2, b3])
            k_label = ['Γ', "X", "Y", "Z"]

            k_frac = np.array([np.linalg.solve(B.T, k) for k in k_cart])

            n_ks = 200
            k_path, k_idx = tb.gen_kpath(k_frac, [n_ks for _ in range(len(k_frac) -1)])
            len(k_path)

            solver = tb.DiagSolver(cell, overlap)
            solver.config.k_points = k_path
            solver.config.prefix = "Test"

            timer = tb.Timer()
            timer.tic("bands")
            k_len, bands = solver.calc_bands()
            timer.toc("bands")
            timer.report_total_time()


            if ham_idx == 0:
                filepath = savedir / f"{structure}_bands_true.npz"
            if ham_idx == 1:
                filepath = savedir / f"{structure}_bands_pred.npz"
            np.savez(filepath, path=str(path), k_idx=k_idx, k_label=k_label, k_len=k_len, bands=bands,)


            n_ks=17
            k_mesh = tb.gen_kmesh((n_ks, n_ks, n_ks))  # Uniform meshgrid
            e_min = float(np.min(bands))
            e_max = float(np.max(bands))

            solver = tb.DiagSolver(cell, overlap)
            solver.config.k_points = k_mesh
            # solver.config.prefix = "graphene"
            solver.config.e_min = e_min
            solver.config.e_max = e_max
            timer = tb.Timer()
            timer.tic("dos")
            energies, dos = solver.calc_dos()
            timer.toc("dos")
            timer.report_total_time()

            if ham_idx == 0:
                filepath = savedir / f"{structure}_dos_mesh{n_ks}_true.npz"
            if ham_idx == 1:
                filepath = savedir / f"{structure}_dos_mesh{n_ks}_pred.npz"
            np.savez(filepath, path=str(path), energies=energies, dos=dos)
















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