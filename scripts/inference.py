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
from e3nn import o3

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
    # === Configuration load ===
    model_dir = Path('results/simple_blocks_blockmse_4') # * Set the directory where your model is saved
    model_filename = 'train_best_model.tar' # * Set the filename of your model


    config = load_config(model_dir / "config.yaml")
    device = torch.device("cpu") # torch.device(config["device"] if (torch.cuda.is_available() and config["device"]!="cpu") else 'cpu')

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
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)


    # === Data (graph) object creation ===
    i=0
    path = paths[i] # * Select the path to the structure you want to infere
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


    # === Enviroment descriptor initialization ===
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



    # === Glue between MACE and E3nnGraph2Mat init (Model init) ===
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



    # === Trainer initialization ===

    # Optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"])
    )

    # Scheduler
    scheduler_config = config["scheduler"]
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(scheduler_config["t_0"]),
        T_mult=scheduler_config["t_multiplication"],
        eta_min=float(scheduler_config["eta_min"])
    )

    # Loss function
    trainer_config = config["trainer"]
    loss_fn = get_object_from_module(trainer_config["loss_function"], "graph2mat.core.data.metrics")

    # Load saved model if required
    trained_model_path = config.get("results_dir", None)
    if trained_model_path is not None:
        trained_model_path = Path(trained_model_path) / model_filename
        model, checkpoint, optimizer, scheduler = load_model(model, optimizer, trained_model_path, lr_scheduler=scheduler, device=device)
        history = checkpoint["history"]
        print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")
    else:
        raise ValueError("Something went wrong, the trained_model_path is None. Please check your config.yaml file.")
    

    # === Inference ===
    dataloader = DataLoader(dataset, 1)
    model.eval()

    data = next(iter(dataloader))
    with torch.no_grad():
        # Model forward pass
        model_predictions = model(data=data)

        # Compute the loss
        loss, stats = loss_fn(
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
    results_directory = model_dir / "plots_inference"
    results_directory.mkdir(exist_ok=True)
    
    n_atoms = data.num_nodes
    title = f"Results of sample {i} (seed {config["dataset"]["seed"]}). There are {n_atoms} in the unit cell."
    predicted_matrix_text = f"Saved training loss at epoch {len(history["train_loss"])}:     {history["train_loss"][-1]:.2f} eV²·100<br>Saved validation loss at epoch {len(history["train_loss"])}:     {history["val_loss"][-1]:.2f} eV²·100<br>MSE evaluation:     {loss.item():.2f} eV²·100"
    plot_error_matrices_big(
        true_matrix, pred_matrix,
        matrix_label="Hamiltonian",
        figure_title=title,
        predicted_matrix_text=predicted_matrix_text,
        filepath = Path(results_directory / f"inference_{n_atoms}atoms_sample{i}_epoch{len(history["train_loss"])}.html")
    )
  

if __name__ == "__main__":
    main()