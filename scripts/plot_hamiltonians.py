# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
import sys
from pathlib import Path
# Add the root directory to Python path
root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
sys.path.append(str(root_dir))

import random
import sisl
import torch
import torch.optim as optim
import numpy as np

from e3nn import o3
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

from graph2mat4abn.tools.plot import plot_error_matrices
from graph2mat4abn.tools import load_config, flatten
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, get_kwargs, load_model
from graph2mat4abn.tools.import_utils import get_object_from_module

def main():
    # === Configuration load ===
    directory = Path("results/test2") # * Write here the directory where the model is stored
    filename = "val_best_model.tar"
    print(f"Loading model {directory / filename}...")


    config = load_config(directory / "config.yaml")

    # === List of paths to all structures ===
    parent_path = Path('./dataset')
    n_atoms_paths = list(parent_path.glob('*/'))
    paths = []
    for n_atoms_path in n_atoms_paths:
        structure_paths = list(n_atoms_path.glob('*/'))
        paths.append(structure_paths)
    paths = flatten(paths)
    
    random.seed(config["dataset"]["seed"])
    random.shuffle(paths)



    # == Basis creation === 
    basis = get_basis_from_structures_paths(paths, verbose=False, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)


    # === Model init ===
    env_config = config["environment_representation"]
    num_interactions = env_config["num_interactions"]
    hidden_irreps = o3.Irreps(env_config["hidden_irreps"])
    mace_descriptor = MACE(
        r_max=env_config["r_max"],
        num_bessel=env_config["num_bessel"],
        num_polynomial_cutoff=env_config["num_polynomial_cutoff"],
        max_ell=env_config["max_ell"],
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        num_interactions=num_interactions,
        num_elements=env_config["num_elements"],
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps(env_config["MLP_irreps"]),
        atomic_energies=torch.tensor(env_config["atomic_energies"]),
        avg_num_neighbors=env_config["avg_num_neighbors"],
        atomic_numbers=env_config["atomic_numbers"],
        correlation=env_config["correlation"],
        gate=get_object_from_module(env_config["gate"], "torch.nn.functional"),
    )

    model_config = config["model"]
    model = MatrixMACE(
        mace = mace_descriptor,
        readout_per_interaction=model_config.get("readout_per_interaction", False),
        graph2mat_cls = E3nnGraph2Mat,
        
        # Readout-specific arguments
        unique_basis = table,
        symmetric = True,

        # Preprocessing
        preprocessing_edges = get_object_from_module(
            model_config["preprocessing_edges"], 
            'graph2mat.bindings.e3nn.modules'
        ),
        preprocessing_edges_kwargs = get_kwargs(model_config["preprocessing_edges"], config),

        preprocessing_nodes = get_object_from_module(
            model_config["preprocessing_nodes"], 
            'graph2mat.bindings.e3nn.modules'
        ),
        preprocessing_nodes_kwargs = get_kwargs(model_config["preprocessing_nodes"], config),

        # Operations
        node_operation = get_object_from_module(
            model_config["node_operation"], 
            'graph2mat.bindings.e3nn.modules'
        ),
        node_operation_kwargs = get_kwargs(model_config["node_operation"], config),

        edge_operation = get_object_from_module(
            model_config["edge_operation"], 
            'graph2mat.bindings.e3nn.modules'
        ),
        edge_operation_kwargs = get_kwargs(model_config["edge_operation"], config),
    )
    # Optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"])
    )
    # Scheduler
    scheduler_config = config["scheduler"]
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(scheduler_config["t_0"]),
        T_mult=scheduler_config["t_multiplication"],
        eta_min=float(scheduler_config["eta_min"])
    )

    # === Dataset creation ===
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    embeddings_configs = []
    for i, path in enumerate(paths):
        if i==config["dataset"]["max_samples"]:
            break

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
                "atom_types": torch.from_numpy(geometry.atoms.Z), # Unlike point_types, this is not rescaled.
            },
        )

        embeddings_configs.append(embeddings_config)

    dataloader = TorchBasisMatrixDataset(embeddings_configs, data_processor=processor)

    n_atoms_list = [dataloader[i].num_nodes for i in range(len(dataloader))] if config["dataset"]["stratify"] == True else None
    train_dataset, val_dataset = train_test_split(
        dataloader, 
        train_size=config["dataset"]["train_split_ratio"],
        stratify=n_atoms_list,
        random_state=None # Dataset already shuffled (paths)
        )


    # === Model load ===
    model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, directory / filename, lr_scheduler=lr_scheduler)
    loss_fn = get_object_from_module(config["trainer"]["loss_function"], "graph2mat.core.data.metrics")



    # ====== Matrix plots ======
    results_directory = directory / "results"
    results_directory.mkdir(exist_ok=True)

    # Set the (max) number of each structure type that you want to plot
    n = 1
    n_plots_each = {
        2: n,
        3: n,
        8: n,
        32: n,
        64: n,
    }
    n_atoms_list = list(n_plots_each.keys())
    # Dataset plots
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, 1)
    val_dataloader = DataLoader(val_dataset, 1)

    dataloaders = [train_dataloader, val_dataloader]
    for dataloader_id, dataloader in enumerate(dataloaders):
        dataloader_type = "training" if dataloader_id == 0 else "validation"
        print(f"Plotting {dataloader_type} dataset...")

        n_plotted = np.zeros([2, len(n_atoms_list)], dtype=np.int16)
        n_plotted[0] += n_atoms_list
        for j, data in enumerate(dataloader):
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

            pred_matrix = processor.matrix_from_data(
                data,
                predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]},
            )[0].todense()

            # Save true matrix
            true_matrix = processor.matrix_from_data(
                data,
            )[0].todense()

            # Plot
            title = f"Results of sample {j} of {dataloader_type} dataset (seed {config["dataset"]["seed"]}). There are {n_atoms} in the unit cell."
            predicted_matrix_text = f"Saved training loss at epoch {checkpoint["epoch"]}:     {checkpoint["train_loss"]:.2f} eV²·100\nMSE evaluation:     {loss.item():.2f} eV²·100"
            plot_error_matrices(
                true_matrix, pred_matrix,
                matrix_label="Hamiltonian",
                figure_title=title,
                predicted_matrix_text=predicted_matrix_text,
                filepath = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}.html")
            )
            plot_error_matrices(
                true_matrix, pred_matrix,
                matrix_label="Hamiltonian",
                figure_title=title,
                predicted_matrix_text=predicted_matrix_text,
                filepath = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}.png")
            )

            print(f"Plotted {j} matrix")

            
            n_plotted[1][col_idx] += 1


if __name__ == "__main__":
    main()