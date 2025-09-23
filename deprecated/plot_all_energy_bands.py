# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
import sys
from pathlib import Path
import warnings

from tools.debug import create_sparse_matrix
from tools.scripts_utils import generate_g2m_dataset_from_paths, get_model_dataset, init_mace_g2m_model
# Add the root directory to Python path
root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
sys.path.append(str(root_dir))

import random
import sisl
import torch
import torch.optim as optim
import numpy as np
import scipy

from e3nn import o3
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from mace.modules import MACE, RealAgnosticResidualInteractionBlock
from graph2mat.bindings.e3nn import E3nnGraph2Mat
from graph2mat.models import MatrixMACE
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat import (
    # PointBasis,
    BasisTableWithEdges,
    BasisConfiguration,
    MatrixDataProcessor,
)

from graph2mat4abn.tools.plot import plot_columns_of_2darray, plot_error_matrices_big, plot_predictions_vs_truths
from graph2mat4abn.tools import load_config, flatten
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, get_orbital_indices_and_shifts_from_sile, get_kwargs, load_model, reconstruct_tim_from_coo, reduced_coord
from graph2mat4abn.tools.import_utils import get_object_from_module



def main():

    debug_mode = False
    # *********************************** #
    # * VARIABLES TO CHANGE BY THE USER * #
    # *********************************** #
    model_dir = Path("results/h_crystalls_1") # Results directory
    filename = "train_best_model.tar" # Model name (or relative path to the results directory)
    compute_calculations = False # Save or Load calculations.

    # *********************************** #

    # Hide some warnings
    warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
    warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")
    if debug_mode:
        print("**************************************************")
        print("*                                                *")
        print("*              DEBUG MODE ACTIVATED              *")
        print("*                                                *")
        print("**************************************************")

    savedir = model_dir / "results"
    savedir.mkdir(exist_ok=True, parents=True)
    calculations_path = savedir / "calculations_energybands.joblib"

    # Load the config of the model
    config = load_config(model_dir / "config.yaml")
    device = torch.device("cpu")

    # Load the same dataset used to train/validate the model (paths)
    train_paths, val_paths = get_model_dataset(model_dir, verbose=True)
    
    
    if compute_calculations:

        # Generate the G2M basis
        paths = train_paths + val_paths
        basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
        table = BasisTableWithEdges(basis)

        if not debug_mode:
            # Init the model, optimizer and others
            print("Initializing model...")
            model, optimizer, lr_scheduler, loss_fn = init_mace_g2m_model(config, table)
            
            # Load the model
            print("Loading model...")
            model_path = model_dir / filename
            initial_lr = float(config["optimizer"].get("initial_lr", None))

            model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, model_path, lr_scheduler=lr_scheduler, initial_lr=initial_lr, device=device)
            print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")
        else:
            model = create_sparse_matrix

        # Generate the G2M dataset (and splits)
        # if debug_mode:
        #     train_paths = [train_paths[i] for i in range(2)]
        #     val_paths = train_paths
        #     paths = train_paths
        train_dataset, val_dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, train_paths, val_paths, device=device, verbose=True) if not debug_mode else ([i for i in range(len(train_paths))], [i for i in range(len(val_paths))], None)
        splits = [train_dataset, val_dataset]











    dataloaders = [train_dataloader, val_dataloader]
    for dataloader_id, dataloader in enumerate(dataloaders):
        dataloader_type = "training" if dataloader_id == 0 else "validation"
        print(f"Plotting {dataloader_type} dataset...")

        #Count the number of plots already done for each structure type
        n_plotted = np.zeros([2, len(n_atoms_list)], dtype=np.int16)
        n_plotted[0] += n_atoms_list

        # For each split
        for j, data in enumerate(dataloader):
            print(f"====== Plotting matrix {j} ======")
            n_atoms = data.num_nodes
            col_idx = np.where(n_plotted[0] == n_atoms)[0][0]

            # Continue or break if already plotted the required number of plots
            if all(n_plotted[1][i] == n_plots_each[n_plotted[0][i]] for i in range(n_plotted.shape[1])):
                break
            if n_plotted[1][col_idx] == n_plots_each[n_atoms]:
                continue 
            
            # Generate prediction
            model_predictions = model(data=data)
            loss, _ = loss_fn(
                nodes_pred=model_predictions["node_labels"],
                nodes_ref=data.point_labels,
                edges_pred=model_predictions["edge_labels"],
                edges_ref=data.edge_labels,
            )

            # Predicted matrix
            h_pred = processor.matrix_from_data(
                data,
                predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]},
            )[0]

            # True matrix
            true_matrix = processor.matrix_from_data(
                data,
            )[0]

            # === Plot Hamiltonians ===
            filepath_html = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}_hamiltonian.html")
            filepath_png = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}_hamiltonian.png")
            if check_existing_plots:
                if filepath_html.is_file():
                    print(f"File {filepath_html} already exists. Continueing.")
                elif filepath_png.is_file():
                    print(f"File {filepath_png} already exists. Continueing.")
                else:
                    print("Plotting hamiltonian...")
                    title = f"Results of sample {j} of {dataloader_type} dataset (seed {config["dataset"]["seed"]}). There are {n_atoms} in the unit cell."
                    predicted_matrix_text = f"Saved training loss at epoch {checkpoint["epoch"]}:     {checkpoint["train_loss"]:.2f} eV²·100\nMSE evaluation:     {loss.item():.2f} eV²·100"
                    if n_atoms <= 32:
                        plot_error_matrices_big(
                            true_matrix.todense(), h_pred.todense(),
                            matrix_label="Hamiltonian",
                            figure_title=title,
                            predicted_matrix_text=predicted_matrix_text,
                            filepath = filepath_html
                        )
                    else:
                        plot_error_matrices_big(
                            true_matrix.todense(), h_pred.todense(),
                            matrix_label="Hamiltonian",
                            figure_title=title,
                            predicted_matrix_text=predicted_matrix_text,
                            filepath = filepath_png
                        )


            # ========= Plot energy bands =========
            print("Plotting energy bands...")
            filepath_html = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}_energybands.html")
            if filepath_html.is_file():
                print(f"File {filepath_html} already exists. Continueing.")
            else:
                file = sisl.get_sile(data.metadata["path"][0] / "aiida.fdf")
                geometry = file.read_geometry()
                cell = geometry.cell
                # orb_i, orb_j, isc = get_orbital_indices_and_shifts_from_sile(file)

                # Define a path in k-space
                kxs = np.linspace(0,1, num=40)
                kys = np.linspace(0,1, num=40)
                kzs = np.linspace(0,1, num=40) # * Change the resolution here
                k_dir_x = geometry.rcell[:,0]
                k_dir_y = geometry.rcell[:,1]
                k_dir_z = geometry.rcell[:,2]
                k_path_x=np.array([kx*k_dir_x for kx in kxs])
                k_path_y=np.array([ky*k_dir_y for ky in kys])
                k_path_z=np.array([kz*k_dir_z for kz in kzs])
                k_path=np.concatenate([k_path_x, k_path_y, k_path_z])

                k_path = np.array([[0, 0, 0], [0, 0, 1]]) if debug_mode else k_path



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
                
                # Plot energy bands
                energy_bands_pred_plot = np.stack(energy_bands_pred, axis=0)
                energy_bands_true_plot = np.stack(energy_bands_true, axis=0)

                title = f"Energy bands of sample {j} of {dataloader_type} dataset (seed {config["dataset"]["seed"]}). There are {n_atoms} in the unit cell. Using SIESTA overlap matrix."
                x_axis = [k_path]*energy_bands_pred_plot.shape[0]
                num_cols = energy_bands_pred_plot[:,0:n_bands].shape[1]
                titles_pred = [f"Predicted band {i}" for i in range(num_cols)]
                titles_true = [f"True band {i}" for i in range(num_cols)]
                plot_columns_of_2darray(
                    array_pred = energy_bands_pred_plot[:,0:n_bands],
                    array_true = energy_bands_true_plot[:,0:n_bands],
                    x = x_axis,
                    xlabel = "k", 
                    ylabel = "Energy (eV)", 
                    title = title,
                    titles_pred=titles_pred,
                    filepath = filepath_html
                )


            # === Plot eigenvalues ===
            filepath_html = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}eigenvalues.html")
            if filepath_html.is_file():
                print(f"File {filepath_html} already exists. Continueing.")
            else:
                titles_series = [f"[{"{:.2f}".format(k_point[0]) if k_point[0] != 0 else 0}, {"{:.2f}".format(k_point[1]) if k_point[1] != 0 else 0}, {"{:.2f}".format(k_point[2]) if k_point[2] != 0 else 0}]" for k_point in k_path]
                plot_predictions_vs_truths(
                    predictions=energy_bands_pred_plot,
                    truths=energy_bands_true_plot,
                    series_names=titles_series,
                    title='Eigenvalues comparison (eV)',
                    xaxis_title='True energy',
                    yaxis_title='Predicted energy',
                    legend_title='k points',
                    show_diagonal=True,
                    show_points_by_default=True,
                    filepath=filepath_html
                )

            # Count the number of plots done for each structure type
            n_plotted[1][col_idx] += 1



if __name__ == "__main__":
    main()