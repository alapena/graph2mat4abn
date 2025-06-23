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
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from mace.modules import MACE, RealAgnosticResidualInteractionBlock
from graph2mat.bindings.torch.data.dataset import InMemoryData
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.bindings.e3nn import E3nnGraph2Mat
from graph2mat import (
    # PointBasis,
    BasisTableWithEdges,
    BasisConfiguration,
    MatrixDataProcessor,
)
from graph2mat4abn.tools import load_config, flatten
from graph2mat4abn.tools.tools import get_basis_from_structures_paths
from graph2mat4abn.tools.import_utils import get_object_from_module
from graph2mat4abn.modules.enviroment_descriptor import EmbeddingBase, MACEDescriptor
from graph2mat4abn.modules.trainer import Trainer



def main():
    # === Configuration load ===
    config = load_config("./config.yaml")
    orbitals = config['orbitals']
    
    device = torch.device(config["device"] if (torch.cuda.is_available() and config["device"]!="cpu") 
    else 'cpu')



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
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=3)
    table = BasisTableWithEdges(basis)



    # === Dataset and dataloaders creation ===
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
                "device": device,
                "atom_types": torch.from_numpy(geometry.atoms.Z), # Unlike point_types, this is not rescaled.
            },
        )

        embeddings_configs.append(embeddings_config)

    dataset = TorchBasisMatrixDataset(embeddings_configs, data_processor=processor)
    # print(dataset[0])
    # a

    # Split dataset (also stratify)
    split = True if config["dataset"]["max_samples"] > 1 else False
    if split:
        n_atoms_list = [dataset[i].num_nodes for i in range(len(dataset))] if config["dataset"]["stratify"] == True else None
        train_dataset, val_dataset = train_test_split(
            dataset, 
            train_size=config["dataset"]["train_split_ratio"],
            stratify=n_atoms_list,
            random_state=None # Dataset already shuffled (paths)
            )
    else:
        train_dataset = dataset
        val_dataset = dataset
        print("There is just 1 sample in the dataset. Using it for both train and validation. Use this only for debugging.")
    
    # Keep all the dataset in memory
    train_dataset = InMemoryData(train_dataset)
    val_dataset = InMemoryData(val_dataset)



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



    # === Model initialization ===
    # Shape of inputs
    mace_out_irreps = hidden_irreps * (num_interactions - 1) + str(hidden_irreps[0])
    model_config = config["model"]

    # kwargs for each supported preprocessing/processing operation
    def get_preprocessing_nodes_kwargs(module: str) -> dict:
        if module == 'E3nnInteraction':
            kwargs = {
                'irreps': {
                    'node_feats_irreps': mace_out_irreps,
                    'edge_attrs_irreps': o3.Irreps.spherical_harmonics(1),
                    'edge_feats_irreps': o3.Irreps("8x0e"), 
                    # 'target_irreps ': mace_out_irreps,
                },
                'avg_num_neighbors': env_config["avg_num_neighbors"] # ? In principle it can be different from the enviroment decriptor, I think(?)
            }
        elif module == 'E3nnEdgeMessageBlock':
            kwargs = {
                'irreps': {
                    'node_feats_irreps': mace_out_irreps,
                    'edge_attrs_irreps': o3.Irreps.spherical_harmonics(1),
                    'edge_feats_irreps': o3.Irreps("8x0e"), 
                    'edge_hidden_irreps': mace_out_irreps,
                },
            }
        else:
            raise ValueError(f"Module {module} not supported yet. Write the kwargs in this function.")
        return kwargs

    # node_operation = get_object_from_module(model_config["node_operation"], "graph2mat4abn.modules.node_operations") if model_config["node_operation"] is not None else get_object_from_module('E3nnSimpleNodeBlock', "graph2mat.bindings.e3nn.modules.node_operations")
    model = E3nnGraph2Mat(
        unique_basis = table,
        irreps = dict(node_feats_irreps=mace_out_irreps),
        symmetric = True,
        # preprocessing_nodes = get_object_from_module(model_config["preprocessing_nodes"], 'graph2mat.bindings.e3nn.modules'),
        # preprocessing_nodes_kwargs = get_preprocessing_nodes_kwargs(model_config["preprocessing_nodes"]),
        preprocessing_edges = get_object_from_module(model_config["preprocessing_edges"], 'graph2mat.bindings.e3nn.modules'),
        preprocessing_edges_kwargs = get_preprocessing_nodes_kwargs(model_config["preprocessing_edges"])
        # node_operation=node_operation,
        # node_operation_kwargs={
        #     "irreps_in": mace_out_irreps,
        #     "config": model_config
        # }
        # node_operation_kwargs={
        #     "irreps": {"node_feats_irreps": o3.Irreps("0e")},
        #     # "config": model_config,
        # },
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
    
    # Trainer
    trainer = Trainer(
        environment_descriptor = mace_descriptor,
        model = model,
        config = config,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        loss_fn = loss_fn,
        optimizer = optimizer,
        device = device,
        lr_scheduler = scheduler,
        live_plot = trainer_config["live_plot"],
        live_plot_freq = trainer_config["live_plot_freq"],
        live_plot_matrix = trainer_config["live_plot_matrix"],
        live_plot_matrix_freq = trainer_config["live_plot_matrix_freq"],
        history = None,
        results_dir = config["results_dir"],
        checkpoint_freq = trainer_config["checkpoint_freq"],
        batch_size = trainer_config["batch_size"]
    )



    # === Start training ===
    print(f"\nTRAINING STARTS with {len(train_dataset)} train samples and {len(val_dataset)} validation samples.")
    print(f"Using device: {device}")
    trainer.train(num_epochs=trainer_config["num_epochs"])
    print("\nTraining completed successfully!")
  

if __name__ == "__main__":
    main()