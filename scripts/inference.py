# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
import sys
from pathlib import Path
import warnings

from tools.scripts_utils import init_mace_g2m_model
# Add the root directory to Python path
root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
sys.path.append(str(root_dir))


import random
import sisl
import torch
import torch.optim as optim
from e3nn import o3
import time

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from mace.modules import MACE, RealAgnosticResidualInteractionBlock
from torch_geometric.loader import DataLoader
from graph2mat.bindings.torch.data.dataset import InMemoryData
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.bindings.e3nn import E3nnGraph2Mat
from graph2mat.models import MatrixMACE
from graph2mat import (
    # PointBasis,
    BasisTableWithEdges,
    BasisConfiguration,
    MatrixDataProcessor,
)
from graph2mat4abn.tools import load_config, flatten
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, get_kwargs, load_model
from graph2mat4abn.tools.import_utils import get_object_from_module
from modules.trainer import Trainer
from graph2mat4abn.tools.plot import plot_error_matrices_big
# from graph2mat4abn.modules.models import MatrixMACE



def main():
    # *********************************** #
    # * VARIABLES TO CHANGE BY THE USER * #
    # *********************************** #
    model_dir = Path("results/h_crystalls_1") # Results directory
    filename = "train_best_model.tar" # Model name (or relative path to the results directory)
    path = Path('./dataset/SHARE_OUTPUTS_64_ATOMS/5dfc-7444-43ac-a2b0-db75f49a7bb9') # Path to desired structure

    # *********************************** #

    # Hide some warnings
    warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
    warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")

    time0 = time.time()
    # Load the config of the model
    config = load_config(model_dir / "config.yaml")
    device = torch.device("cpu")

    # Basis creation 
    basis = get_basis_from_structures_paths([path], verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)

    # Load the structure config
    file = sisl.get_sile(path / "aiida.fdf")
    file_h = sisl.get_sile(path / "aiida.HSX")
    geometry = file.read_geometry()

    # Load the true hamiltonian
    true_h = file_h.read_hamiltonian()
    embeddings_config = BasisConfiguration.from_matrix(
        matrix = true_h,
        geometry = geometry,
        labels = True,
        metadata={
            "device": device,
            "atom_types": torch.from_numpy(geometry.atoms.Z), # Unlike point_types, this is not rescaled.
        },
    )
    embeddings_config = [embeddings_config]

    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    dataset = TorchBasisMatrixDataset(embeddings_config, data_processor=processor)


    # Init the model, optimizer and others
    print("Initializing model...")
    model, optimizer, lr_scheduler, loss_fn = init_mace_g2m_model(config, table)
    
    # Load the model
    print("Loading model...")
    model_path = model_dir / filename
    initial_lr = float(config["optimizer"].get("initial_lr", None))

    model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, model_path, lr_scheduler=lr_scheduler, initial_lr=initial_lr, device=device)
    history = checkpoint["history"]
    print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")
    

    # === Inference ===
    dataloader = DataLoader(dataset, 1)
    model.eval()

    data = next(iter(dataloader))
    with torch.no_grad():
        # Model forward pass
        time1 = time.time()
        model_predictions = model(data=data)
        time2 = time.time()

        # Compute the loss
        loss, stats = loss_fn(
            nodes_pred=model_predictions["node_labels"],
            nodes_ref=data.point_labels,
            edges_pred=model_predictions["edge_labels"],
            edges_ref=data.edge_labels,
        )
        time3 = time.time()
        pred_matrix = processor.matrix_from_data(
            data,
            predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]},
        )[0].todense()
        time4 = time.time()

        # Save true matrix
        true_matrix = processor.matrix_from_data(
            data,
        )[0].todense()
        time5 = time.time()

    print("Time results of the inference:")
    print(f"Time to init the model: {time1 - time0:.2f}s")
    print(f"Time forward pass: {time2 - time1:.2f}s")
    print(f"Time reconstructing predicted matrix: {time4 - time3:.2f}s")
    print(f"Time reconstructing true matrix: {time5 - time4:.2f}s")

    # Plot
    results_directory = model_dir / "plots_inference"
    results_directory.mkdir(exist_ok=True)
    
    n_atoms = data.num_nodes
    title = f"Results (seed {config["dataset"]["seed"]}). There are {n_atoms} in the unit cell. Structure {path.parts[-1]}"
    predicted_matrix_text = f"Saved training loss at epoch {len(history["train_loss"])}:     {history["train_loss"][-1]:.2f} eV²·100<br>Saved validation loss at epoch {len(history["train_loss"])}:     {history["val_loss"][-1]:.2f} eV²·100<br>MSE evaluation:     {loss.item():.2f} eV²·100"
    plot_error_matrices_big(
        true_matrix, pred_matrix,
        matrix_label="Hamiltonian",
        figure_title=title,
        predicted_matrix_text=predicted_matrix_text,
        filepath = Path(results_directory / f"inference_{n_atoms}atoms_epoch{len(history["train_loss"])}_struct_{path.parts[-1]}.html")
    )
  

if __name__ == "__main__":
    main()