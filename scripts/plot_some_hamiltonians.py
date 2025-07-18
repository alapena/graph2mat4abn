# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
import logging
import sys
from pathlib import Path
# Add the root directory to Python path
root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
sys.path.append(str(root_dir))

import numpy as np
import torch
from tools.import_utils import load_config
from tools.scripts_utils import generate_g2m_dataset_from_paths, get_model_dataset, init_mace_g2m_model
from tools.tools import get_basis_from_structures_paths, load_model
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
import sisl
from tools.debug import create_sparse_matrix
from tools.plot import plot_dataset_results, plot_error_matrices_big

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
        # "./dataset/SHARE_OUTPUTS_2_ATOMS/db65-6186-4188-b955-0dca4cd7daa1",
        # "./dataset/SHARE_OUTPUTS_2_ATOMS/4130-44f6-445e-8a26-4f6afcdd73ea",
        # "./dataset/SHARE_OUTPUTS_2_ATOMS/c185-8b61-445e-8068-be370e3617c6",
        # "./dataset/SHARE_OUTPUTS_8_ATOMS/bca3-f473-4c5e-8407-cbdc2d7c68a1",
        # "./dataset/SHARE_OUTPUTS_8_ATOMS/848a-19bd-414b-9ef2-40f39b1e2027",
        # "./dataset/SHARE_OUTPUTS_8_ATOMS/7651-acf9-490f-81d6-a8a9d1c2de76",
        "./dataset/SHARE_OUTPUTS_64_ATOMS/dcd8-ab99-4e8b-81ba-401f6739412e",
        "./dataset/SHARE_OUTPUTS_64_ATOMS/348a-4ea5-4c95-8d6b-7d32485b3156",
        "./dataset/SHARE_OUTPUTS_64_ATOMS/342f-eb3b-4268-820f-747c3936102b",
    ]
    split_type = "training"
    model_dir = Path("results/h_crystalls_1") # Results directory
    filename = "train_best_model.tar" # Model name (or relative path to the results directory)

    # *********************************** #

    # Hide some warnings
    warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
    warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")
    logging.getLogger("nodify.node").setLevel(logging.WARNING)


    # Load the config of the model
    config = load_config(model_dir / "config.yaml")
    device = torch.device("cpu")

    # Results directory
    paths = [Path(path) for path in paths]
    savedir = model_dir / "results"
    savedir.mkdir(exist_ok=True, parents=True)

    if debug_mode:
        print("**************************************************")
        print("*                                                *")
        print("*              DEBUG MODE ACTIVATED              *")
        print("*                                                *")
        print("**************************************************")

    # Generate the G2M basis
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
    dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, paths, device=device, verbose=True)

    # Iterate through all dataset and compute the magnitudes to plot.
    print("START OF THE LOOP...")
    for i, data in tqdm(enumerate(dataset)):
        if debug_mode and i==5:
            break
        # Generate prediction. Create dataloaders (needed to generate a prediction)
        data = DataLoader([data], 1)
        data = next(iter(data))

        # Generate prediction
        model_predictions = model(data=data) if not debug_mode else None
        loss, _ = loss_fn(
            nodes_pred=model_predictions["node_labels"],
            nodes_ref=data.point_labels,
            edges_pred=model_predictions["edge_labels"],
            edges_ref=data.edge_labels,
        ) #if not debug_mode else None, None

        # Reconstruct matrices
        h_pred = processor.matrix_from_data(data, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})[0] if not debug_mode else model(n_atoms=2, n_orbs=13, n_shifts=10, size=100)
        h_true = processor.matrix_from_data(data)[0] if not debug_mode else model(n_atoms=2, n_orbs=13, n_shifts=10, size=100)

        # 1. Plot structure
        print("Plotting structure...")
        path = Path(data.metadata["path"][0])
        file = sisl.get_sile(path / "aiida.fdf")
        geometry = file.read_geometry()
        n_atoms = len(geometry.atoms.Z)

        fig = file.plot.geometry(axes="xyz")
        filepath = savedir / f"{n_atoms}atm_{split_type}_struct_{path.parts[-1]}.png"
        fig.write_image(str(filepath))

        # 2. Plot hamiltonian
        print("Plotting hamiltonian...")
        title = f"Results of sample {i} of {split_type} dataset (seed {config["dataset"]["seed"]}). There are {n_atoms} in the unit cell."
        predicted_matrix_text = f"Saved training loss at epoch {checkpoint["epoch"]}:     {checkpoint["train_loss"]:.2f} eV²·100\nMSE evaluation:     {loss.item():.2f} eV²·100" if not debug_mode else None
        filepath = savedir / f"{n_atoms}atm_{split_type}_hamilt_{path.parts[-1]}.png"
        plot_error_matrices_big(
            h_true.todense(), h_pred.todense(),
            matrix_label="Hamiltonian",
            figure_title=title,
            predicted_matrix_text=predicted_matrix_text,
            filepath=filepath
        )
        print("Saved plot at", filepath)

        

    print(f"Results saved at {savedir}!")



if __name__ == "__main__":
    main()