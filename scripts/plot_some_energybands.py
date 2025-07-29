# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
import sys
from pathlib import Path
# # Add the root directory to Python path
# root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
# sys.path.append(str(root_dir))

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
from plot_utilities.plot_for_scripts import combine_band_and_dos, plot_bands, plot_dos, plot_diagonal, plot_hamiltonian, plot_diagonal_rows, plot_energy_bands
from tools.plot import plot_error_matrices_big

from graph2mat import (
    BasisTableWithEdges,
)


def main():
    debug_mode = False
    # *********************************** #
    # * VARIABLES TO CHANGE BY THE USER * #
    # *********************************** #
    plot_eigenvalues = False
    plot_energybands = False
    plot_bands_and_dos = True
    paths = [
        # "./dataset/SHARE_OUTPUTS_2_ATOMS/c924-ac64-4837-a960-ff786d6c6836",
        # "./dataset/SHARE_OUTPUTS_8_ATOMS/bca3-f473-4c5e-8407-cbdc2d7c68a1",
        # "./dataset/SHARE_OUTPUTS_64_ATOMS/dcd8-ab99-4e8b-81ba-401f6739412e",

        "dataset/SHARE_OUTPUTS_8_ATOMS/39cf-a27b-42dd-a62e-62556132a798",
        "dataset/SHARE_OUTPUTS_2_ATOMS/c8ce-475a-431c-b659-39b166ea3959",

    ]
    model_dir = Path("results/h_crystalls_6") # Model directory
    savedir = model_dir / "results" / "train" # Results directory

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


    # Load data and plot.
    
    # if plot_eigenvalues:
    #     energybands_paths = list(savedir.glob('*eigenvals.npz'))

    #     print("Plotting eigenvalues and energy bands.")

    #     # Read all files in data directory.    

    #     # For each file
    #     for energybands_path in tqdm(energybands_paths):
    #         energybands_path = Path(energybands_path)
    #         n_atoms = energybands_path.parts[-1][0]
    #         structure = energybands_path.stem.split("_")[1]
    #         savedir_struct = savedir / f"stats_{n_atoms}_ATOMS_{structure}"

    #         # Read it
    #         energyband_data = np.load(energybands_path)

    #         # Plot it
    #         k_path = energyband_data['k_path']
    #         energy_bands_true = energyband_data['energy_bands_true_array']
    #         energy_bands_pred = energyband_data['energy_bands_pred_array']
    #         path = Path(str(energyband_data['path']))

    #         titles_series = [f"k=({"{:.2f}".format(k_point[0]) if k_point[0] != 0 else 0}, {"{:.2f}".format(k_point[1]) if k_point[1] != 0 else 0}, {"{:.2f}".format(k_point[2]) if k_point[2] != 0 else 0})" for k_point in k_path]
    #         filepath = savedir_struct / f"{n_atoms}atm_{structure}_eigenvals.html"
    #         title = f"Eigenvalues comparison (eV).<br>Used model {model_dir.parts[-1]}. Using SIESTA overlap matrix."
    #         plot_diagonal_rows(
    #             predictions=energy_bands_pred.T,
    #             truths=energy_bands_true.T,
    #             series_names=titles_series,
    #             # x_error_perc=None,
    #             # y_error_perc=5,
    #             title=title,
    #             xaxis_title='True energy',
    #             yaxis_title='Predicted energy',
    #             legend_title='k points',
    #             show_diagonal=True,
    #             show_points_by_default=True,
    #             showlegend=True,
    #             filepath=filepath
    #         )
    #         print("Finished plotting eigenvalues!")


    # if plot_energybands:
    #     energybands_paths = list(savedir.glob('*eigenvals.npz'))
    #     # For each file
    #     for energybands_path in tqdm(energybands_paths):
    #         energybands_path = Path(energybands_path)
    #         n_atoms = energybands_path.parts[-1][0]
    #         structure = energybands_path.stem.split("_")[1]
    #         savedir_struct = savedir / f"stats_{n_atoms}_ATOMS_{structure}"

    #         # Read it
    #         energyband_data = np.load(energybands_path)

    #         # Plot it
    #         k_path = energyband_data['k_path']
    #         energy_bands_true = energyband_data['energy_bands_true_array']
    #         energy_bands_pred = energyband_data['energy_bands_pred_array']
    #         path = Path(str(energyband_data['path']))

    #         title = f"Energy bands of structure {n_atoms}_ATOMS/{structure}.<br>Used model {model_dir.parts[-1]}. Using SIESTA overlap matrix."
    #         filepath = savedir_struct / f"{n_atoms}atm_{structure}_energybands.html"
    #         # x_axis = [k_path]*energy_bands_pred.shape[1]
    #         n_series = energy_bands_pred.shape[0]
    #         titles_pred = [f"Predicted band {i}" for i in range(n_series)]
    #         titles_true = [f"True band {i}" for i in range(n_series)]
    #         plot_energy_bands(
    #             list(range(len(k_path))),
    #             energy_bands_true,
    #             energy_bands_pred,
    #             xlabel = "k_path index",
    #             ylabel = "Energy (eV)",
    #             title = title,
    #             titles_pred=titles_pred,
    #             titles_true=titles_true,
    #             filepath = filepath
    #         )
    #     print("Finished plotting energybands!")


    if plot_bands_and_dos:
        bands_paths = list(savedir.glob('*bands.npz'))
        dos_paths = list(savedir.glob('*dos.npz'))
        print(bands_paths, dos_paths)
        # For each file
        for k, bands_path in tqdm(enumerate(bands_paths)):
            bands_path = Path(bands_path)
            dos_path = Path(dos_paths[k])
            n_atoms = bands_path.parts[-1][0]
            structure = bands_path.stem.split("_")[1]
            savedir_struct = savedir / f"stats_{n_atoms}_ATOMS_{structure}"

            # Read it
            bands_data = np.load(bands_path)
            dos_data = np.load(dos_path)  # Assuming dos_paths has at least one file

            # Plot it
            k_len = bands_data['k_len']
            k_idx = bands_data['k_idx']
            k_label = bands_data['k_label']
            bands_true = bands_data['bands_true']
            bands_pred = bands_data['bands_pred']

            energies = dos_data['energies']
            dos_true = dos_data['dos_true']
            dos_pred = dos_data['dos_pred']

            path = Path(str(bands_data['path']))

            filepath = savedir_struct / f"{n_atoms}atm_{structure}_bands.html"
            fig_bands = plot_bands(k_len, bands_true, k_idx, k_label, predicted_bands=bands_pred, filepath=filepath)
            filepath = savedir_struct / f"{n_atoms}atm_{structure}_dos.html"
            fig_dos = plot_dos(energies, dos_true, predicted_dos=dos_pred, filepath=filepath)
            filepath = savedir_struct / f"{n_atoms}atm_{structure}_bandsdos.html"
            combine_band_and_dos(fig_bands, fig_dos, filepath=filepath)


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




if __name__ == "__main__":
    main()