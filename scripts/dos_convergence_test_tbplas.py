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
    n_k_bands = 80
    path = "dataset/SHARE_OUTPUTS_2_ATOMS/2e65-1feb-4df2-8836-e5513b9bade0", # B-B Overlapped
    model_dir = Path("results/h_crystalls_9") # Results directory
    savedir = Path("results/h_crystalls_9") / "results"
    filename = "train_best_model.tar" # Model name (or relative path to the results directory)
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


    n_atoms = int(path.parts[-2].split('_')[2])
    structure = path.parts[-1]

    # 1. Plot structure
    file = sisl.get_sile(path / "aiida.fdf")
    fig = file.plot.geometry(axes="xyz")
    filepath = savedir_struct / f"{n_atoms}atm_{structure}.png"
    fig.write_image(str(filepath))
    filepath = savedir_struct / f"{n_atoms}atm_{structure}.html"
    fig.write_html(str(filepath))
    print("Saved structure plot at", filepath)

    # 2. Plot hamiltonian
    print("Saved hamiltonian plot at", filepath)

    # 3. Plot nnz diagonal plot
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
        label = f"{str(orb_in)} -> {str(orb_out)} {str(isc)} {str(atom_in)} -> {str(atom_out)}"


        # Store the labels 
        matrix_labels.append(label)
        info.append((orb_in, orb_out, isc, atom_in, atom_out))

    filepath= savedir_struct / f"{n_atoms}atm_{structure}_nnzelements_shifts.html"
    title = f"Non-zero elements of structure {n_atoms}_ATOMS/{structure}.<br>Used model {model_dir.parts[-1]}"
    title_x = "True matrix elements (eV)"
    title_y = "Predicted matrix elements (eV)"
    true_values = h_true.data
    pred_values = h_pred.data
    orb_in_list, orb_out_list, iscs, atom_in_list, atom_out_list = map(list, zip(*info))
    plot_diagonal(
        true_values, pred_values, orb_in_list, orb_out_list, iscs, atom_in_list, atom_out_list,# 1D array of elements.
        title=title, title_x=title_x, title_y=title_y, colors=None, filepath=filepath,
        group_by="shift", legendtitle="Shift indices", #SEGUIR CON ESTO
    )

    filepath= savedir_struct / f"{n_atoms}atm_{structure}_nnzelements_orbs.html"
    
    plot_diagonal(
        true_values, pred_values, orb_in_list, orb_out_list, iscs, atom_in_list, atom_out_list, # 1D array of elements.
        title=title, title_x=title_x, title_y=title_y, colors=None, filepath=filepath,
        group_by="orbs", legendtitle="Orbitals", #SEGUIR CON ESTO
    )

    print("Nnz elements plotted!")


    # 4. Energy bands
    if compute_eigenvals_scipy:
        print("=== COMPUTING EIGENVALUES ===")
        file = sisl.get_sile(path / "aiida.fdf")
        geometry = file.read_geometry()
        cell = geometry.cell

        # Define a path in k-space
        if not debug_mode:
            ks = 80
            kxs = np.linspace(0,1, num=ks)
            kys = np.linspace(0,1, num=ks)
            kzs = np.linspace(0,1, num=ks) # * Change the resolution here
            k_dir_x = geometry.rcell[:,0]
            k_dir_y = geometry.rcell[:,1]
            k_dir_z = geometry.rcell[:,2]
            k_path_x=np.array([kx*k_dir_x for kx in kxs])
            k_path_y=np.array([ky*k_dir_y for ky in kys])
            k_path_z=np.array([kz*k_dir_z for kz in kzs])
            k_path=np.concatenate([k_path_x, k_path_y, k_path_z])
        else:
            k_path = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1]]) 

        # TIM reconstruction
        h_uc = file.read_hamiltonian()
        s_uc = file.read_overlap()

        energy_bands_pred = []
        energy_bands_true = []
        for k_point in tqdm(k_path):
            # Ground truth:
            Hk_true = h_uc.Hk(reduced_coord(k_point, cell), gauge='cell').toarray()
            Sk_true = s_uc.Sk(reduced_coord(k_point, cell), gauge='cell').toarray()

            Ek_true = scipy.linalg.eigh(Hk_true, Sk_true, eigvals_only=True)
            energy_bands_true.append(Ek_true)

            # Prediction:
            Hk_pred = reconstruct_tim_from_coo(k_point, h_pred.tocsr().tocoo(), geometry, cell)
            # Sk = reconstruct_tim(k_point, s_pred, orb_i, orb_j, isc, cell)

            Ek = scipy.linalg.eigh(Hk_pred, Sk_true, eigvals_only=True)

            energy_bands_pred.append(Ek)

        # Save results
        energy_bands_true_array = np.stack(energy_bands_true, axis=0).T
        energy_bands_pred_array = np.stack(energy_bands_pred, axis=0).T

        filepath = savedir / f"{n_atoms}atm_{structure}_eigenvals.npz"
        np.savez(filepath, energy_bands_true_array=energy_bands_true_array, energy_bands_pred_array=energy_bands_pred_array, k_path=k_path, path=str(path))



        # 4. EQOS data at", filepath)


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



if __name__ == "__main__":
    main()