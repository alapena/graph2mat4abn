# # === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
# import sys
# from pathlib import Path
# # # Add the root directory to Python path
# # root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
# # sys.path.append(str(root_dir))

import numpy as np
import torch
from tools.import_utils import load_config
from tools.scripts_utils import generate_g2m_dataset_from_paths, get_model_dataset, init_mace_g2m_model
from tools.tools import get_basis_from_structures_paths, load_model, reconstruct_tim_from_coo, reduced_coord
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
import sisl
from tools.debug import create_sparse_matrix
from joblib import dump, load
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import scipy
from plot_utilities.plot_for_scripts import plot_diagonal, plot_hamiltonian, plot_diagonal_rows, plot_energy_bands
from tools.plot import plot_error_matrices_big
import tbplas as tb
from plot_utilities.tbplas_tools import add_hopping_terms, add_orbitals, get_hoppings, get_onsites
from pathlib import Path

from graph2mat import (
    BasisTableWithEdges,
)


def main():
    debug_mode = False
    # *********************************** #
    # * VARIABLES TO CHANGE BY THE USER * #
    # *********************************** #
    compute_bands_and_dos_tbplas = True
    compute_nnzdiag = False
    n_k_bands = 80
    n_k_dos = 50
    paths = [


    ]
    model_dir = Path("results/h_crystalls_9") # Results directory
    savedir = Path("results/h_crystalls_9/results")
    filename = "train_best_model.tar" # Model name (or relative path to the results directory)

    # compute_matrices_calculations = True # Save or Load calculations.
    # compute_eigenvalues_calculations = True
    # plot_nnzvalues_onsites_hops = True
    # plot_eigenvalues_and_energybands = True
    # plot_energybands = True

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
    if debug_mode:
        print("**************************************************")
        print("*                                                *")
        print("*              DEBUG MODE ACTIVATED              *")
        print("*                                                *")
        print("**************************************************")

    # Load the config of the model
    config = load_config(model_dir / "config.yaml")
    device = torch.device("cpu")

    # Results directory
    # if not debug_mode:
    paths = [Path(path) for path in paths]
    # else:
    #     paths = [Path(paths[0])]

    # Generate the G2M basis
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)

    # Generate dataset
    dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, paths, device=device, verbose=True)

    # Init the model, optimizer and others
    print("Initializing model...")
    if not debug_mode:
        model, optimizer, lr_scheduler, loss_fn = init_mace_g2m_model(config, table)
    else:
        model = create_sparse_matrix

    # Load the model
    if not debug_mode:
        print("Loading model...")
        model_path = model_dir / filename
        initial_lr = float(config["optimizer"].get("initial_lr", None))

        model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, model_path, lr_scheduler=lr_scheduler, initial_lr=initial_lr, device=device)
        print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")

    for i, data in enumerate(dataset):

        path = Path(data.metadata["path"])
        n_atoms = int(path.parts[-2].split('_')[2])
        structure = path.parts[-1]

        # Create a different saving dir for each structure
        savedir_struct = savedir / f"stats_{n_atoms}_ATOMS_{structure}"
        print("Computing", savedir_struct.parts[-1])
        savedir_struct.mkdir(exist_ok=True, parents=True)

        # Generate prediction. Create dataloaders (needed to generate a prediction)
        data = DataLoader([data], 1)
        data = next(iter(data))

        if not debug_mode:
            model_predictions = model(data=data)

            # Reconstruct matrices
            h_pred = processor.matrix_from_data(data, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})[0].tocsr().tocoo()
            h_true = processor.matrix_from_data(data)[0].tocsr().tocoo()
        else:
            h_pred = model(n_atoms=2, n_orbs=13, n_shifts=10, size=100).tocsr().tocoo()
            h_true = model(n_atoms=2, n_orbs=13, n_shifts=10, size=100).tocsr().tocoo()

        # 1. Plot structure
        print("Plotting structure...")
        file = sisl.get_sile(path / "aiida.fdf")
        fig = file.plot.geometry(axes="xyz")
        filepath = savedir_struct / f"{n_atoms}atm_{structure}.png"
        fig.write_image(str(filepath))
        filepath = savedir_struct / f"{n_atoms}atm_{structure}.html"
        fig.write_html(str(filepath))
        print("Saved structure plot at", filepath)

        # 2. Plot hamiltonian
        print("Plotting hamiltonian...")
        title = f"Hamiltonian of structure {n_atoms}_ATOMS/{structure}"
        if int(n_atoms) <= 32:
            filepath = savedir_struct / f"{n_atoms}atm_{structure}_hamiltonian.html"
            plot_hamiltonian(
                h_true.todense(), h_pred.todense(),
                matrix_label="Hamiltonian",
                figure_title=title,
                predicted_matrix_text=None,
                filepath=filepath
            )
        else:
            filepath = savedir_struct / f"{n_atoms}atm_{structure}_hamiltonian.png"
            plot_hamiltonian_matplotlib(
                h_true.todense(), h_pred.todense(),
                matrix_label="Hamiltonian",
                figure_title=title,
                predicted_matrix_text=None,
                filepath=filepath
            )
        print("Saved hamiltonian plot at", filepath)

        # 3. Plot nnz diagonal plot
        if compute_nnzdiag:
            plot_nonzero_elements(
                path, h_pred, h_true, orbitals, n_orbs, n_atoms,
                structure, model_dir, savedir_struct, plot_diagonal
            )


        # 4. Energy bands using TBPLaS 
        if compute_bands_and_dos_tbplas:    
            print("Computing bands and DOS with TBPLaS...")
            file = sisl.get_sile(path / "aiida.HSX")
            geometry = file.read_geometry()

            # Empty cell
            vectors = geometry.cell
            cell_true = tb.PrimitiveCell(vectors, unit=tb.ANG)
            cell_pred = tb.PrimitiveCell(vectors, unit=tb.ANG)

            # Add orbitals
            positions = geometry.xyz
            labels = [[orb.name() for orb in atom] for atom in geometry.atoms]
            onsites_true = get_onsites(h_true.tocsr().tocoo())
            onsites_pred = get_onsites(h_pred.tocsr().tocoo())

            add_orbitals(cell_true, positions, onsites_true, labels)
            add_orbitals(cell_pred, positions, onsites_pred, labels)

            # Add hoppings
            iscs_true, orbs_in_true, orbs_out_true, hoppings_true = get_hoppings(h_true.tocsr().tocoo(), n_atoms, geometry)
            iscs_pred, orbs_in_pred, orbs_out_pred, hoppings_pred = get_hoppings(h_pred.tocsr().tocoo(), n_atoms, geometry)

            add_hopping_terms(cell_true, iscs_true, orbs_in_true, orbs_out_true, hoppings_true)
            add_hopping_terms(cell_pred, iscs_pred, orbs_in_pred, orbs_out_pred, hoppings_pred)

            # Create overlap matrix
            overlap_true = tb.PrimitiveCell(cell_true.lat_vec, cell_true.origin, 1.0)

            o_true = file.read_overlap()
            onsites_overlap_true = get_onsites(o_true.tocsr().tocoo())
            for k in range(cell_true.num_orb):
                orbital = cell_true.orbitals[k]
                overlap_true.add_orbital(orbital.position, onsites_overlap_true[k])

            iscs_overlap_true, orbs_in_overlap_true, orbs_out_overlap_true, hoppings_overlap_true = get_hoppings(o_true.tocsr().tocoo(), n_atoms, geometry)
            add_hopping_terms(overlap_true, iscs_overlap_true, orbs_in_overlap_true, orbs_out_overlap_true, hoppings_overlap_true)
            

            # Magnitudes computation

            # Define a path in k-space

            # 2 atoms: Molecules. Path along the z axis.
            if i==0 or i==1:
                b1, b2, b3 = cell_true.get_reciprocal_vectors()/10 # Each shape (3), Angstrom^-1
                B = np.vstack([b1, b2, b3])  # shape (3,3)
                k_cart = np.array([[0.0, 0.0, 0.0], b3])
                k_label = [r"$\Gamma$", "Z"]

                k_frac = np.array([np.linalg.solve(B.T, k) for k in k_cart])
                k_path, k_idx = tb.gen_kpath(k_frac, [n_k_bands for _ in range(len(k_frac) -1)])
            # elif i==2:  # 8 atoms: Cubic.
            #     # Compute k path
            #     b1, b2, b3 = cell_true.get_reciprocal_vectors() # Each shape (3)
            #     B = np.vstack([b1, b2, b3])  # shape (3,3)
            #     k_cart = np.array([
            #         b1/2 + b2/2 + b3/2,    # Gamma
            #         b1 + b2/2 + b3/2,    # X
            #         b1 + b2 + b3/2,     # M
            #         b1/2 + b2/2 + b3/2,    # Gamma
            #         b1 + b2 + b3,   # R
            #         b1 + b2/2 + b3/2,    # X
            #     ])
            #     # Convert each to fractional coordinates
            #     k_frac = np.array([np.linalg.solve(B.T, k) for k in k_cart])
            #     k_label = [r"$\Gamma$", "X", "M", r"$\Gamma$", "R", "X"]

            #     n_ks = 25
            #     k_path, k_idx = tb.gen_kpath(k_frac, [n_ks for _ in range(len(k_frac) -1)])

            elif i == 3:  # 8 atoms: Hexagonal.
                print("Todavía no tengo este xD")

            

            # Bands
            solver = tb.DiagSolver(cell_true, overlap_true)
            solver.config.k_points = k_path
            solver.config.prefix = "bands_true"
            timer = tb.Timer()
            timer.tic("bands_true")
            k_len_true, bands_true = solver.calc_bands()
            timer.toc("bands_true")
            timer.report_total_time()

            solver = tb.DiagSolver(cell_pred, overlap_true)
            solver.config.k_points = k_path
            solver.config.prefix = "bands_pred"
            timer.tic("bands_pred")
            k_len_pred, bands_pred = solver.calc_bands()
            timer.toc("bands_pred")
            timer.report_total_time()

            filepath = savedir / f"{n_atoms}atm_{structure}_tbplasbands.npz"
            np.savez(filepath, path=str(path), k_len_true=k_len_true, k_idx=k_idx, k_label=k_label, bands_true=bands_true, k_len_pred=k_len_pred, bands_pred=bands_pred)
            print("Saved bands data at", filepath)


            # 5. DOS
            k_mesh = tb.gen_kmesh((n_k_dos, n_k_dos, n_k_dos))  # Uniform meshgrid
            e_min = float(np.min(bands_true))
            e_max = float(np.max(bands_true))

            solver = tb.DiagSolver(cell_true, overlap_true)
            solver.config.k_points = k_mesh
            solver.config.e_min = e_min
            solver.config.e_max = e_max
            solver.config.prefix = "dos_true"
            print("k_mesh.shape=", k_mesh.shape)
            print("e_min=", e_min, "e_max=", e_max)
            timer.tic("dos_true")
            energies_true, dos_true = solver.calc_dos()
            timer.toc("dos_true")
            timer.report_total_time()
            filepath = savedir / f"{n_atoms}atm_{structure}_tbplasdostrue.npz"
            np.savez(filepath, path=str(path), energies=energies_true, dos=dos_true)

            # e_min = float(np.min(bands_pred))
            # e_max = float(np.max(bands_pred))
            solver = tb.DiagSolver(cell_pred, overlap_true)
            solver.config.k_points = k_mesh
            solver.config.e_min = e_min
            solver.config.e_max = e_max
            solver.config.prefix = "dos_pred"
            print("e_min=", e_min, "e_max=", e_max)
            timer.tic("dos_pred")
            energies_pred, dos_pred = solver.calc_dos()
            timer.toc("dos_pred")
            timer.report_total_time()

            filepath = savedir / f"{n_atoms}atm_{structure}_tbplasdospred.npz"
            np.savez(filepath, path=str(path), energies=energies_pred, dos=dos_pred)
            print("Saved DOS data at", filepath)


    # Load data and plot.
    
    if plot_eigenvalues:
        energybands_paths = list(savedir.glob('*eigenvals.npz'))

        print("Plotting eigenvalues and energy bands.")

        # Read all files in data directory.    

        # For each file
        for energybands_path in tqdm(energybands_paths):
            energybands_path = Path(energybands_path)
            n_atoms = energybands_path.parts[-1][0]
            structure = energybands_path.stem.split("_")[1]
            savedir_struct = savedir / f"stats_{n_atoms}_ATOMS_{structure}"

            # Read it
            energyband_data = np.load(energybands_path)

            # Plot it
            k_path = energyband_data['k_path']
            energy_bands_true = energyband_data['energy_bands_true_array']
            energy_bands_pred = energyband_data['energy_bands_pred_array']
            path = Path(str(energyband_data['path']))

            titles_series = [f"k=({"{:.2f}".format(k_point[0]) if k_point[0] != 0 else 0}, {"{:.2f}".format(k_point[1]) if k_point[1] != 0 else 0}, {"{:.2f}".format(k_point[2]) if k_point[2] != 0 else 0})" for k_point in k_path]
            filepath = savedir_struct / f"{n_atoms}atm_{structure}_eigenvals.html"
            title = f"Eigenvalues comparison (eV).<br>Used model {model_dir.parts[-1]}. Using SIESTA overlap matrix."
            plot_diagonal_rows(
                predictions=energy_bands_pred.T,
                truths=energy_bands_true.T,
                series_names=titles_series,
                # x_error_perc=None,
                # y_error_perc=5,
                title=title,
                xaxis_title='True energy',
                yaxis_title='Predicted energy',
                legend_title='k points',
                show_diagonal=True,
                show_points_by_default=True,
                showlegend=True,
                filepath=filepath
            )
            print("Finished plotting eigenvalues!")


    if plot_energybands:
        energybands_paths = list(savedir.glob('*eigenvals.npz'))
        # For each file
        for energybands_path in tqdm(energybands_paths):
            energybands_path = Path(energybands_path)

            # Read it
            energyband_data = np.load(energybands_path)

            # Plot it
            k_path = energyband_data['k_path']
            energy_bands_true = energyband_data['energy_bands_true_array']
            energy_bands_pred = energyband_data['energy_bands_pred_array']
            path = Path(str(energyband_data['path']))

            title = f"Energy bands of structure {n_atoms}_ATOMS/{structure}.<br>Used model {model_dir.parts[-1]}. Using SIESTA overlap matrix."
            filepath = savedir_struct / f"{n_atoms}atm_{structure}_energybands.html"
            # x_axis = [k_path]*energy_bands_pred.shape[1]
            n_series = energy_bands_pred.shape[0]
            titles_pred = [f"Predicted band {i}" for i in range(n_series)]
            titles_true = [f"True band {i}" for i in range(n_series)]
            plot_energy_bands(
                list(range(len(k_path))),
                energy_bands_true,
                energy_bands_pred,
                xlabel = "k_path index",
                ylabel = "Energy (eV)",
                title = title,
                titles_pred=titles_pred,
                titles_true=titles_true,
                filepath = filepath
            )
        print("Finished plotting energybands!")

    print(f"Finished! Results saved at {savedir_struct}")


import numpy as np
import matplotlib.pyplot as plt

def plot_hamiltonian_matplotlib(true_matrix, predicted_matrix, matrix_label=None, figure_title=None, predicted_matrix_text=None, filepath=None, force_max_colorbar_abs_error=None):
    """Matplotlib visualization of error matrices."""

    # Error matrices
    absolute_error_matrix = true_matrix - predicted_matrix
    threshold = 0.001
    mask = true_matrix >= threshold
    relative_error_matrix = np.where(mask, absolute_error_matrix / (true_matrix + threshold) * 100, 0)

    # Colorbar limits
    vmin = np.min([np.min(true_matrix), np.min(predicted_matrix)])
    vmax = np.max([np.max(true_matrix), np.max(predicted_matrix)])
    lim_data = max(np.abs(vmin), np.abs(vmax))

    if force_max_colorbar_abs_error is None:
        lim_abs = np.max(np.abs(absolute_error_matrix))
    else:
        lim_abs = force_max_colorbar_abs_error

    lim_rel = 100.0  # %

    cbar_limits = [lim_data, lim_data, lim_abs, lim_rel]

    # Titles
    if matrix_label is None:
        matrix_label = ''
    titles = [
        "True " + matrix_label,
        "Predicted " + matrix_label,
        "Absolute error (T-P)",
        f"Relative error (T-P)/(T) (masked where T is above {threshold})"
    ]
    cbar_titles = ["eV", "eV", "eV", "%"]

    # Matrices to plot
    matrices = [true_matrix, predicted_matrix, absolute_error_matrix, relative_error_matrix]

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True, gridspec_kw={'hspace': 0.15})
    fig.suptitle(figure_title if figure_title else "Matrix Comparison and Errors", fontsize=16)

    for i, (matrix, ax) in enumerate(zip(matrices, axes)):
        im = ax.imshow(matrix, cmap='RdBu', vmin=-cbar_limits[i], vmax=cbar_limits[i])
        ax.set_title(titles[i])
        ax.set_ylabel("Row")
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.045, pad=0.02)
        cbar.set_label(cbar_titles[i])
        ax.set_xlabel("Col")
        ax.set_aspect('auto')

    # Text under predicted matrix (subplot 2)
    if predicted_matrix_text is not None:
        axes[1].text(1.0, -0.25, predicted_matrix_text, ha='right', va='center', fontsize=10, transform=axes[1].transAxes)

    # Absolute error stats (exclude zeros for stats)
    abs_err_nonzero = np.abs(absolute_error_matrix)[absolute_error_matrix != 0]
    if abs_err_nonzero.size > 0:
        mean_abs = np.mean(abs_err_nonzero)
        std_abs = np.std(abs_err_nonzero)
    else:
        mean_abs = std_abs = 0.0
    max_absolute_error = np.max(absolute_error_matrix)
    min_absolute_error = np.min(absolute_error_matrix)
    max_abs = np.max(np.abs([max_absolute_error, min_absolute_error]))

    fig.text(
        0.5, 0.245,
        f"mean_nnz(|T-P|) = {mean_abs:.3f} eV, std_nnz(|T-P|) = {std_abs:.3f} eV, |max| = {max_abs:.3f} eV",
        ha='center', va='center', fontsize=11,
    )

    # Relative error stats (exclude zeros for stats)
    rel_err_nonzero = np.abs(relative_error_matrix)[relative_error_matrix != 0]
    if rel_err_nonzero.size > 0:
        mean_rel = np.mean(rel_err_nonzero)
        std_rel = np.std(rel_err_nonzero)
    else:
        mean_rel = std_rel = 0.0
    max_relative_error = np.max(relative_error_matrix)
    min_relative_error = np.min(relative_error_matrix)
    max_abs_rel = np.max(np.abs([max_relative_error, min_relative_error]))

    fig.text(
        0.5, 0.001,
        f"mean_nnz(|T-P|) = {mean_rel:.3f} %, std_nnz(|T-P|) = {std_rel:.3f} %, |max| = {max_abs_rel:.3f} %",
        ha='center', va='center', fontsize=11,
    )
    axes[-1].set_xlabel(" ")

    # Output
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_nonzero_elements(
    path,
    h_pred,
    h_true,
    orbitals,
    n_orbs,
    n_atoms,
    structure,
    model_dir,
    savedir_struct,
    plot_diagonal
):
    """
    Plot non-zero elements for a given structure.

    Parameters
    ----------
    path : Path
        Path to the directory containing `aiida.fdf`.
    h_pred : object
        Predicted Hamiltonian object with `data`, `row`, and `col` attributes.
    h_true : object
        True Hamiltonian object with `data` attribute.
    orbitals : list
        List of orbital identifiers.
    n_orbs : int
        Number of orbitals per atom.
    n_atoms : int
        Number of atoms in the structure.
    structure : str
        Structure name or identifier.
    model_dir : Path
        Path to the model directory.
    savedir_struct : Path
        Path to save the plots.
    plot_diagonal : callable
        Function to create the plots.
    """


    geometry = sisl.get_sile(path / "aiida.fdf").read_geometry()
    nnz_el = len(h_pred.data)
    matrix_labels = []
    info = []

    for k in range(nnz_el):
        row = h_pred.row[k]
        col = h_pred.col[k]
        orb_in = orbitals[col % n_orbs]
        orb_out = orbitals[row % n_orbs]
        isc = str(geometry.o2isc(col))
        atom_in = (col // n_orbs) % n_atoms + 1
        atom_out = (row // n_orbs) % n_atoms + 1

        # Join altogether
        label = f"{orb_in} -> {orb_out} {isc} {atom_in} -> {atom_out}"

        # Store the labels
        matrix_labels.append(label)
        info.append((orb_in, orb_out, isc, atom_in, atom_out))

    filepath = savedir_struct / f"{n_atoms}atm_{structure}_nnzelements_shifts.html"
    title = f"Non-zero elements of structure {n_atoms}_ATOMS/{structure}.<br>Used model {model_dir.parts[-1]}"
    title_x = "True matrix elements (eV)"
    title_y = "Predicted matrix elements (eV)"
    true_values = h_true.data
    pred_values = h_pred.data
    orb_in_list, orb_out_list, iscs, atom_in_list, atom_out_list = map(list, zip(*info))

    plot_diagonal(
        true_values, pred_values, orb_in_list, orb_out_list, iscs, atom_in_list, atom_out_list,
        title=title, title_x=title_x, title_y=title_y, colors=None, filepath=filepath,
        group_by="shift", legendtitle="Shift indices",
    )

    filepath = savedir_struct / f"{n_atoms}atm_{structure}_nnzelements_orbs.html"

    plot_diagonal(
        true_values, pred_values, orb_in_list, orb_out_list, iscs, atom_in_list, atom_out_list,
        title=title, title_x=title_x, title_y=title_y, colors=None, filepath=filepath,
        group_by="orbs", legendtitle="Orbitals",
    )

    print("Nnz elements plotted!")


if __name__ == "__main__":
    main()