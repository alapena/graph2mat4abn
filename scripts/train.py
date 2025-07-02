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
from graph2mat.bindings.torch.data.dataset import InMemoryData, RotatingPoolData
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
# from graph2mat4abn.modules.models import MatrixMACE



def main():
    # === Configuration load ===
    config = load_config("./config.yaml")
    trainer_config = config["trainer"]
    
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
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)



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
    model_config = config["model"]

    # === Glue between MACE and E3nnGraph2Mat init ===
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
    trained_model_path = config.get("trained_model_path", None)
    if trained_model_path is not None:
        model, checkpoint, optimizer, scheduler = load_model(model, optimizer, trained_model_path, lr_scheduler=scheduler, device=device)
        print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")
    

    # === Dataset creation ===
    print("Creating dataset...")
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    embeddings_configs = []
    for i, path in enumerate(paths):
        if i==config["dataset"]["max_samples"]:
            break

        # Load the structure config
        file = sisl.get_sile(path / "aiida.fdf")
        file_h = sisl.get_sile(path / "aiida.HSX")
        geometry = file.read_geometry()
        lattice_vectors = geometry.lattice

        matrix = trainer_config.get("matrix", "hamiltonian")

        if matrix == "hamiltonian":
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
        elif matrix == "tim":
            h_uc = file_h.read_hamiltonian()
            true_h = h_uc.Hk([0, 0, 0]).todense()

            embeddings_config = BasisConfiguration(
                point_types=geometry.atoms.Z,
                positions=geometry.xyz,
                basis=basis,
                cell=lattice_vectors.cell,
                pbc=(True, True, True),
                # pbc=(False, False, False),
                matrix = true_h
            )
        elif matrix == "overlap":
            true_h = file_h.read_overlap()

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

    # Split dataset (also stratify)
    if config["dataset"]["max_samples"] == None:
        split = True
    elif config["dataset"]["max_samples"] > 1:
        split = True
    else:
        split = False


    if split:
        n_atoms_list = [dataset[i].num_nodes for i in range(len(dataset))] if config["dataset"]["stratify"] == True else None
        train_dataset, val_dataset = train_test_split(
            dataset, 
            train_size=config["dataset"]["train_split_ratio"],
            stratify=n_atoms_list,
            random_state=None # Dataset already shuffled (paths)
            )
        print(f"Dataset splitted in {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    else:
        train_dataset = dataset
        val_dataset = dataset
        print("There is just 1 sample in the dataset. Using it for both train and validation. Use this only for debugging.")
    
    # Keep all the dataset in memory
    keep_in_memory = trainer_config.get("keep_in_memory", False)
    rotating_pool = trainer_config.get("rotating_pool", False)
    rotating_pool_size = trainer_config.get("rotating_pool_size", 50)
    if keep_in_memory:
        print("Keeping all the dataset in memory.")
        train_dataset = InMemoryData(train_dataset)
        val_dataset = InMemoryData(val_dataset)
    elif rotating_pool:
        print("Using rotating pool for the dataset.")
        train_dataset = RotatingPoolData(train_dataset, pool_size=rotating_pool_size)
        val_dataset = RotatingPoolData(val_dataset, pool_size=rotating_pool_size)

        
    # Trainer
    if matrix == "hamiltonian" or matrix == "overlap":
        from graph2mat4abn.modules.trainer_h import Trainer
        trainer = Trainer(
            # environment_descriptor = mace_descriptor,
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
            history = checkpoint["history"] if trained_model_path is not None else None,
            results_dir = config["results_dir"],
            checkpoint_freq = trainer_config["checkpoint_freq"],
            batch_size = trainer_config["batch_size"],
            processor=processor
        )
    elif matrix == "tim":
        from graph2mat4abn.modules.trainer_tim import Trainer
        trainer = Trainer(
            # environment_descriptor = mace_descriptor,
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
            history = checkpoint["history"] if trained_model_path is not None else None,
            results_dir = config["results_dir"],
            checkpoint_freq = trainer_config["checkpoint_freq"],
            batch_size = trainer_config["batch_size"],
            processor=processor
        )
    else:
        raise ValueError(f"Matrix type '{matrix}' not recognized.")



    # === Start training ===
    # TODO: This message shows pool_size instead of dataset length. 
    print(f"\nTRAINING STARTS with {len(train_dataset)} train samples and {len(val_dataset)} validation samples.")
    print(f"Using device: {device}")

    trainer.train(num_epochs=trainer_config["num_epochs"])
    
    print("\nTraining completed successfully!")
  

if __name__ == "__main__":
    main()