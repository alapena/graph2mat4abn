import torch
import sisl
from tqdm import tqdm
from e3nn import o3
from mace.modules import MACE, RealAgnosticResidualInteractionBlock
from pathlib import Path
from graph2mat.bindings.e3nn import E3nnGraph2Mat
from graph2mat.models import MatrixMACE
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.bindings.torch.data.dataset import InMemoryData, RotatingPoolData
from graph2mat import (
    BasisTableWithEdges,
    BasisConfiguration,
    MatrixDataProcessor,
)

from .import_utils import get_object_from_module
from .tools import get_basis_from_structures_paths, get_kwargs, get_scheduler_args_and_kwargs, read_structures_paths
from graph2mat4abn.modules.node_operations import HamGNNInspiredNodeBlock



def get_model_dataset(model_dir, verbose=False):
    dataset_dir = model_dir / "dataset"
    train_paths = [Path(path) for path in read_structures_paths(str(dataset_dir / f"train_dataset.txt"))]
    val_paths = [Path(path) for path in read_structures_paths(str(dataset_dir / f"val_dataset.txt"))]

    if verbose:
        print(f"Loaded {len(train_paths)} training paths and {len(val_paths)} validation paths.")

    return train_paths, val_paths



def generate_g2m_dataset_from_paths(config, basis, table, train_paths, val_paths=None, device="cpu", verbose=False):

    print("Generating dataset...") if verbose else None

    trainer_config = config["trainer"]
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)

    splits = [train_paths, val_paths] if val_paths is not None else [train_paths]
    # // splits = [train_paths, val_paths] if not extra_custom_validation else [train_paths, val_paths, val_paths_extra]
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
    val_dataset = datasets[1] if val_paths is not None else None

    # Keep all the dataset in memory
    keep_in_memory = trainer_config.get("keep_in_memory", False)
    rotating_pool = trainer_config.get("rotating_pool", False)
    rotating_pool_size = trainer_config.get("rotating_pool_size", 50)
    if keep_in_memory:
        print("Keeping all the dataset in memory.")
        train_dataset = InMemoryData(train_dataset)
        val_dataset = InMemoryData(val_dataset) if val_paths is not None else None
    elif rotating_pool:
        print("Using rotating pool for the dataset.")
        train_dataset = RotatingPoolData(train_dataset, pool_size=rotating_pool_size)
        val_dataset = RotatingPoolData(val_dataset, pool_size=rotating_pool_size) if val_paths is not None else None

    if val_paths is not None:
        return train_dataset, val_dataset, processor
    else:
        return train_dataset, processor



def init_mace_g2m_model(config, table):
    # **************** ENVIRONMENT DESCRIPTOR INIT **************** #

    env_config = config["environment_representation"]

    num_interactions = env_config["num_interactions"]
    hidden_irreps = o3.Irreps(env_config["hidden_irreps"])
    
    # / This operation is time-consuming:
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


    # **************** MODEL GRAPH2MAT+MACE INIT **************** #

    model_config = config["model"]

    model = MatrixMACE(
        mace = mace_descriptor,
        readout_per_interaction = model_config.get("readout_per_interaction", False),
        graph2mat_cls = E3nnGraph2Mat,

        # Readout-specific arguments:
        unique_basis = table,
        symmetric = True,

        preprocessing_edges = get_object_from_module(
            model_config["preprocessing_edges"], 
            'graph2mat.bindings.e3nn.modules'
        ),
        preprocessing_nodes = get_object_from_module(
            model_config["preprocessing_nodes"], 
            'graph2mat.bindings.e3nn.modules'
        ),
        node_operation = get_object_from_module(
            model_config["node_operation"], 
            'graph2mat.bindings.e3nn.modules'
        ),
        edge_operation = get_object_from_module(
            model_config["edge_operation"], 
            'graph2mat.bindings.e3nn.modules'
        ),

        preprocessing_edges_kwargs = get_kwargs(model_config["preprocessing_edges"], config),
        preprocessing_nodes_kwargs = get_kwargs(model_config["preprocessing_nodes"], config),
        node_operation_kwargs = get_kwargs(model_config["node_operation"], config),
        edge_operation_kwargs = get_kwargs(model_config["edge_operation"], config),
    )

    
    # **************** OPTIMIZER INIT **************** #

    optimizer_config = config["optimizer"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        # weight_decay=float(optimizer_config["weight_decay"])
    )

    print(f"Using Optimizer {optimizer}")


    # **************** SCHEDULER INIT **************** #

    scheduler_config = config["scheduler"]
    scheduler = scheduler_config.get("type", None)

    if scheduler_config.get("type", None) is not None:
        scheduler_args, scheduler_kwargs = get_scheduler_args_and_kwargs(config, verbose=True)

    if scheduler is not None:
        scheduler = get_object_from_module(scheduler_config.get("type", None), "torch.optim.lr_scheduler")(optimizer, *(scheduler_args or ()), **scheduler_kwargs)


    # **************** LOSS FUNCTION INIT **************** #

    trainer_config = config["trainer"]
    print("LOSS FN SELECTED: ", trainer_config["loss_function"])
    loss_fn = get_object_from_module(trainer_config["loss_function"], "graph2mat4abn.modules.loss_functions")

    print(f"Using Loss function {loss_fn}")

    return model, optimizer, scheduler, loss_fn



import numpy as np

def real_space_to_kspace(positions, b1, b2, b3):
    """
    Map real-space positions into reciprocal space (fractional and cartesian).
    Returns:
        k_frac: (N, 3) positions in fractional reciprocal coordinates
        k_cart: (N, 3) positions in cartesian k-space (nm^-1)
    """
    B = np.vstack([b1, b2, b3])  # reciprocal lattice vectors (3,3), Ang^-1
    # Fractional reciprocal coordinates
    k_frac = np.linalg.solve(B.T, positions.T).T  # shape (N,3)
    # Cartesian k-vectors
    k_cart = k_frac @ B
    return k_frac, k_cart