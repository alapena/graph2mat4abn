# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
import sys
from pathlib import Path

from tools.plot import plot_error_matrices_big
# Add the root directory to Python path
root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
sys.path.append(str(root_dir))

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

from graph2mat import (
    BasisTableWithEdges,
)


def main():
    debug_mode = False
    # *********************************** #
    # * VARIABLES TO CHANGE BY THE USER * #
    # *********************************** #
    paths = [
        # "./dataset/SHARE_OUTPUTS_2_ATOMS/c924-ac64-4837-a960-ff786d6c6836",
        # "./dataset/SHARE_OUTPUTS_8_ATOMS/bca3-f473-4c5e-8407-cbdc2d7c68a1",
        # "./dataset/SHARE_OUTPUTS_64_ATOMS/dcd8-ab99-4e8b-81ba-401f6739412e",

        # "./dataset/SHARE_OUTPUTS_8_ATOMS/59bc-dacc-4405-b7b4-195e66bd1481",
        # "dataset/SHARE_OUTPUTS_8_ATOMS/b5f1-87ae-481c-a4b2-058274105e7f",
        # "dataset/SHARE_OUTPUTS_2_ATOMS/5bc4-c7a6-4074-be4f-f983d5dbdbcb",
        "dataset/SHARE_OUTPUTS_8_ATOMS/b4a9-9312-4f83-b021-6e4195bd9d88",
        "dataset/SHARE_OUTPUTS_8_ATOMS/551a-934e-4d9f-8d56-f8ae1d61bb30",
        "dataset/SHARE_OUTPUTS_8_ATOMS/f3fd-f009-46fc-ad94-fc4ebf116390",
        "dataset/SHARE_OUTPUTS_8_ATOMS/cc7d-9eab-4f1a-a5dd-4fe7bf41d6bb",

    ]
    model_dir = Path("results/h_crystalls_1") # Results directory
    filename = "train_best_model.tar" # Model name (or relative path to the results directory)

    # compute_matrices_calculations = True # Save or Load calculations.
    # compute_eigenvalues_calculations = True
    # plot_nnzvalues_onsites_hops = True
    # plot_eigenvalues_and_energybands = True
    # plot_energybands = True

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

    # Load the config of the model
    config = load_config(model_dir / "config.yaml")
    device = torch.device("cpu")

    # Results directory
    paths = [Path(path) for path in paths]

    # Generate the G2M basis
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)

    # Init the model, optimizer and others
    print("Initializing model...")
    model, optimizer, lr_scheduler, loss_fn = init_mace_g2m_model(config, table)
    
    # Load the model
    print("Loading model...")
    model_path = model_dir / filename
    initial_lr = float(config["optimizer"].get("initial_lr", None))

    model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, model_path, lr_scheduler=lr_scheduler, initial_lr=initial_lr, device=device)
    print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")

    dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, paths, device=device, verbose=True)

    print("Starting loop.")
    for i, data in enumerate(dataset):

        path = Path(data.metadata["path"])
        n_atoms = path.parts[-2].split('_')[2]
        structure = path.parts[-1]

        # Create a different saving dir for each structure
        savedir = model_dir / "results" / f"stats_{n_atoms}_ATOMS_{structure}"
        print("Computing", savedir.parts[-1])
        savedir.mkdir(exist_ok=True, parents=True)

        # Generate prediction. Create dataloaders (needed to generate a prediction)
        data = DataLoader([data], 1)
        data = next(iter(data))

        model_predictions = model(data=data)

        # Reconstruct matrices
        h_pred = processor.matrix_from_data(data, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})[0]
        h_true = processor.matrix_from_data(data)[0]

        # 1. Plot structure
        print("Plotting structure...")
        file = sisl.get_sile(path / "aiida.fdf")
        fig = file.plot.geometry(axes="xyz")
        filepath = savedir / f"{n_atoms}atm_{structure}.png"
        fig.write_image(str(filepath))
        print("Saved structure plot at", filepath)

        # 2. Plot hamiltonian
        print("Plotting hamiltonian...")
        title = f"Hamiltonian of structure {n_atoms}_ATOMS/{structure}"
        filepath = savedir / f"{n_atoms}atm_{structure}_hamiltonian.html"
        plot_hamiltonian(
            h_true.todense(), h_pred.todense(),
            matrix_label="Hamiltonian",
            figure_title=title,
            predicted_matrix_text=None,
            filepath=filepath
        )
        filepath = savedir / f"{n_atoms}atm_{structure}_hamiltonian.png"
        plot_hamiltonian(
            h_true.todense(), h_pred.todense(),
            matrix_label="Hamiltonian",
            figure_title=title,
            predicted_matrix_text=None,
            filepath=filepath
        )
        print("Saved hamiltonian plot at", filepath)

    print(f"Finished! Results saved at {savedir}")




def plot_hamiltonian(true_matrix, predicted_matrix, matrix_label=None, figure_title=None, predicted_matrix_text=None, filepath=None, force_max_colorbar_abs_error=None):
    """Interactive Plotly visualization of error matrices."""

    # === Error matrices computation ===
    absolute_error_matrix = true_matrix - predicted_matrix

    threshold = 0.001
    mask = true_matrix >= threshold
    relative_error_matrix = np.where(mask, absolute_error_matrix / (true_matrix + threshold) * 100, 0)

    # === Colorbar limits ===
    vmin = np.min([np.min(true_matrix), np.min(predicted_matrix)])
    vmax = np.max([np.max(true_matrix), np.max(predicted_matrix)])
    lim_data = max(np.abs(vmin), np.abs(vmax))

    if force_max_colorbar_abs_error is None:
        lim_abs = np.max(np.abs(absolute_error_matrix))
    else:
        lim_abs = force_max_colorbar_abs_error

    lim_rel = 100.0  # %

    cbar_limits = [lim_data, lim_data, lim_abs, lim_rel]

    # === Titles ===
    if matrix_label is None:
        matrix_label = ''
    titles = [
        "True " + matrix_label,
        "Predicted " + matrix_label,
        "Absolute error (T-P)",
        f"Relative error (T-P)/(T) (masked where T is above {threshold})"
    ]
    cbar_titles = ["eV", "eV", "eV", "%"]

    # === Figure ===
    matrices = [true_matrix, predicted_matrix, absolute_error_matrix, relative_error_matrix]

    fig = make_subplots(
        rows=4, cols=1,
        vertical_spacing=0.1
    )

    for i, matrix in enumerate(matrices):
        row = i + 1
        col = 1

        heatmap = go.Heatmap(
            z=matrix,
            colorscale='RdBu',
            zmin=-cbar_limits[i],
            zmax=cbar_limits[i],
            
            colorbar=dict(title=cbar_titles[i], len=0.21, y=(0.92-0.275*i), ),
        )
        fig.add_trace(heatmap, row=row, col=col)

    # === Text annotations ===

    # Text under predicted matrix
    if predicted_matrix_text is not None:
        fig.add_annotation(
            text=predicted_matrix_text,
            xref='x2 domain', yref='y2 domain',
            x=1, y=-0.15,
            showarrow=False,
            font=dict(size=12),
            align='right'
        )

    # Absolute error stats
    abs = np.abs(absolute_error_matrix)
    mean = np.mean(abs[absolute_error_matrix != 0])
    std = np.std(abs[absolute_error_matrix != 0])

    max_absolute_error = np.max(absolute_error_matrix)
    min_absolute_error = np.min(absolute_error_matrix)
    max_abs = np.max(np.abs([max_absolute_error, min_absolute_error]))

    fig.add_annotation(
        text=f"mean_nnz(|T-P|) = {mean:.3f} eV, std_nnz(|T-P|) = {std:.3f} eV, |max| = {max_abs:.3f} eV",
        xref='x3 domain', yref='y3 domain',
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, weight=400),
        align='center'
    )

    # Relative error stats
    abs = np.abs(relative_error_matrix)
    mean = np.mean(abs[relative_error_matrix != 0])
    std = np.std(abs[relative_error_matrix != 0])

    max_relative_error = np.max(relative_error_matrix)
    min_relative_error = np.min(relative_error_matrix)
    max_abs = np.max(np.abs([max_relative_error, min_relative_error]))

    fig.add_annotation(
        text=f"mean_nnz(|T-P|) = {mean:.3f} %, std_nnz(|T-P|) = {std:.3f} %, |max| = {max_abs:.3f} %",
        xref='x4 domain', yref='y4 domain',
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, weight=400),
        align='center'
    )

    # === Layout of the whole figure ===
    fig.update_layout(
        height=1200,
        width=900,
        title_text=figure_title if figure_title else "Matrix Comparison and Errors",
        title_x=0.46,
        title_y=0.99,
        margin=dict(t=100, b=20, l=30),

        # Subplot titles
        xaxis1=dict(side="top", title_text=titles[0]), yaxis1=dict(autorange="reversed"),
        xaxis2=dict(side="top", title_text=titles[1]), yaxis2=dict(autorange="reversed"),
        xaxis3=dict(side="top", title_text=titles[2]), yaxis3=dict(autorange="reversed"),
        xaxis4=dict(side="top", title_text=titles[3]), yaxis4=dict(autorange="reversed"),
        font=dict(size=12),
    )

    # === Output ===
    if filepath:
        filepath = Path(filepath)
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath), height=1200, width=900,)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
        
    else:
        fig.show()

    del fig


if __name__ == "__main__":
    main()