from tools.import_utils import get_object_from_module
import torch
import sisl
from e3nn import o3
from pathlib import Path
from tools.tools import get_basis_from_structures_paths, get_kwargs, get_scheduler_args_and_kwargs, read_structures_paths
from mace.modules import MACE, RealAgnosticResidualInteractionBlock
from tqdm import tqdm

from graph2mat.bindings.e3nn import E3nnGraph2Mat
from graph2mat.models import MatrixMACE
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.bindings.torch.data.dataset import InMemoryData, RotatingPoolData
from graph2mat import (
    BasisTableWithEdges,
    BasisConfiguration,
    MatrixDataProcessor,
)


def get_model_dataset(model_dir, verbose=False):
    dataset_dir = model_dir / "dataset"
    train_paths = [Path(path) for path in read_structures_paths(str(dataset_dir / f"train_dataset.txt"))]
    val_paths = [Path(path) for path in read_structures_paths(str(dataset_dir / f"val_dataset.txt"))]

    if verbose:
        print(f"Loaded {len(train_paths)} training paths and {len(val_paths)} validation paths.")

    return train_paths, val_paths



def generate_g2m_dataset_from_paths(config, basis, table, train_paths, val_paths, device="cpu", verbose=False):
    print("Generating dataset...") if verbose else None

    trainer_config = config["trainer"]
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)

    splits = [train_paths, val_paths]
    datasets = []
    for j, split in enumerate(splits):
        print(f"Generating split {j}...") if verbose else None
        embeddings_configs = []
        for i, path in tqdm(enumerate(split)):
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
                        "path": path
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
                        "path": path
                    },
                )

            embeddings_configs.append(embeddings_config)

        datasets.append(TorchBasisMatrixDataset(embeddings_configs, data_processor=processor))

    train_dataset = datasets[0]
    val_dataset = datasets[1]

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

    return train_dataset, val_dataset, processor



def init_mace_g2m_model(config, table):
    # === Enviroment descriptor initialization ===
    env_config = config["environment_representation"]

    num_interactions = env_config["num_interactions"]
    hidden_irreps = o3.Irreps(env_config["hidden_irreps"])
    
    # ! This operation is somehow time-consuming:
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



    # Optimizer
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        # weight_decay=float(optimizer_config["weight_decay"])
    )
    print(f"Using Optimizer {optimizer}")

    # Scheduler
    lr_scheduler_config = config["scheduler"]
    lr_scheduler = lr_scheduler_config.get("type", None)
    scheduler_args, scheduler_kwargs = get_scheduler_args_and_kwargs(config, verbose=True)
    if lr_scheduler is not None:
        lr_scheduler = get_object_from_module(lr_scheduler_config.get("type", None), "torch.optim.lr_scheduler")(optimizer, *(scheduler_args or ()), **scheduler_kwargs)

    # Loss function
    trainer_config = config["trainer"]
    loss_fn = get_object_from_module(trainer_config["loss_function"], "graph2mat.core.data.metrics")
    print(f"Using Loss function {loss_fn}")

    return model, optimizer, lr_scheduler, loss_fn