# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
import sys
from pathlib import Path
import warnings
# Add the root directory to Python path
root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
sys.path.append(str(root_dir))


import random
import sisl
import torch
import torch.optim as optim
from e3nn import o3

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, get_kwargs, get_scheduler_args_and_kwargs, load_model, read_structures_paths
from graph2mat4abn.tools.import_utils import get_object_from_module
from graph2mat4abn.modules.trainer import Trainer



def main():
    # === Configuration load ===
    config = load_config("./config_gpu1.yaml")
    debug_mode = config.get("debug_mode", False)
    trainer_config = config["trainer"]
    dataset_config = config["dataset"]
    
    device = torch.device(config["device"] if (torch.cuda.is_available() and config["device"]!="cpu") 
    else 'cpu')

    # Hide some warnings
    warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
    warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")



    # === List of paths to all structures ===
    # parent_path = Path('./dataset')

    # # Define which subdatasets to use
    # use_only_n_atoms = config["dataset"].get("use_only_n_atoms", None)

    # # # Define how many of each
    # # how_many_of_each = config["dataset"].get("how_many_of_each", None) 

    # # Filter the n_atoms_paths based on the use_only_n_atoms list
    # if use_only_n_atoms is not None:
    #     n_atoms_paths = [parent_path / f"SHARE_OUTPUTS_{n}" for n in use_only_n_atoms]
    # else:
    #     n_atoms_paths = list(parent_path.glob('*/'))

    # exclude_carbons = config["trainer"].get("exclude_carbons", False)

    # paths = []
    # for n_atoms_path in n_atoms_paths:
    #     structure_paths = list(n_atoms_path.glob('*/'))

    #     # In case you want to exclude carbon atoms, we need to use sisl.
    #     if exclude_carbons == True:
    #         structure_paths_nocarbon = structure_paths
    #         for structure_path in structure_paths:
    #             file = sisl.get_sile(structure_path / "aiida.fdf")
    #             geometry = file.read_geometry()
    #             zs = geometry.atoms.Z
    #             if 6 in zs:
    #                 # Exclude this structure
    #                 exclude = structure_path
    #                 structure_paths_nocarbon = [x for x in structure_paths_nocarbon if x != exclude]


    #         paths.append(structure_paths_nocarbon)

    #     else:
    #         paths.append(structure_paths)

    # paths = flatten(paths)

    # === List of paths to all desired structures ===
    extra_custom_validation = dataset_config.get("extra_custom_validation", False)
    exclude_carbons = dataset_config.get("exclude_carbons", True)
    use_only = dataset_config.get("use_only", None)
    custom_dataset = dataset_config.get("custom_dataset", False)
    # use_previous_dataset = dataset_config.get("use_previous_dataset", False)
    use_previous_dataset = True if config.get("trained_model_path", None) is not None else False

    true_dataset_folder = Path('./dataset')
    pointers_folder = Path('./dataset_nocarbon')

    # Use only determined subset of the dataset
    if use_only is not None:
        x_atoms_paths = [pointers_folder / f"SHARE_OUTPUTS_{n}" for n in use_only]
    else:
        x_atoms_paths = list(pointers_folder.glob('*/'))

    paths = []
    if not use_previous_dataset:
        # If you want no carbons
        if exclude_carbons:
        
            # Get all the structures
            filepath = "structures.txt"
            structures_paths = [[] for _ in x_atoms_paths]
            for i, x_atoms_path in enumerate(x_atoms_paths):
                structures = read_structures_paths(str(x_atoms_path / filepath))
                for structure in structures:
                    structures_paths[i].append(x_atoms_path.parts[-1] +"/"+ structure)

            # Now we join them with the true parent folder
            for structures in structures_paths:
                for structure in structures:
                    true_path = true_dataset_folder / structure
                    paths.append(true_path)
                
        # If you want carbons
        else:  
            # Join all structures in the paths variable
            for n_atoms_path in x_atoms_paths:
                structure_paths = list(n_atoms_path.glob('*/')) 
                for structure_path in structure_paths:
                    paths.append(structure_path)

        # Shuffle the dataset
        random.seed(config["dataset"]["seed"])
        random.shuffle(paths)

        # Get unique X_ATOMS
        unique_x_atoms = []
        for path in paths:
            x_atoms = path.parts[-2]
            if x_atoms not in unique_x_atoms:
                unique_x_atoms.append(x_atoms)

        # Split paths into training and validation datasets.
        if not custom_dataset:
            # // split = config["dataset"]["max_samples"] is None or config["dataset"]["max_samples"] > 1
            x_atoms_list = [int(Path(p.parts[1]).stem.split('_')[2]) for p in paths] if config["dataset"]["stratify"] == True else None
            train_paths, val_paths = train_test_split(
                paths, 
                train_size=config["dataset"]["train_split_ratio"],
                stratify=x_atoms_list,
                random_state=None # Dataset already shuffled (paths)
                )
            print(f"Dataset splitted in {len(train_paths)} training paths and {len(val_paths)} validation paths.")

            # Test the stratification.
            unique_x = [int(Path(p).stem.split('_')[2]) for p in unique_x_atoms]
            x_atoms_list_train = [int(Path(p.parts[1]).stem.split('_')[2]) for p in train_paths]
            x_atoms_list_val = [int(Path(p.parts[1]).stem.split('_')[2]) for p in val_paths]
            print(f"They are stratified: {[x_atoms_list_train.count(x) for x in unique_x]} versus {[x_atoms_list_val.count(x) for x in unique_x]}.")
            print(f"The proportions are \n{[x_atoms_list_train.count(x)/len(train_paths) for x in unique_x]} versus \n{[x_atoms_list_val.count(x)/len(val_paths) for x in unique_x]}.")

        # Custom training dataset
        else:
            pass
            # Train only on crystalls. This is, just 2, 8 atoms structures. At this point, the paths variable already has only these structures.
            # Stratify
            # x_atoms_list = [int(Path(p.parts[1]).stem.split('_')[2]) for p in paths] if config["dataset"]["stratify"] == True else None
            # train_paths, val_paths = train_test_split(
            #     paths, 
            #     train_size=config["dataset"]["train_split_ratio"],
            #     stratify=x_atoms_list,
            #     random_state=None # Dataset already shuffled (paths)
            #     )
            
            # Now, take out of the training dataset the structure of standard hBN


            # print(f"Dataset splitted in {len(train_paths)} training paths and {len(val_paths)} validation paths.")

            # # Test the stratification.
            # unique_x = [int(Path(p).stem.split('_')[2]) for p in unique_x_atoms]
            # x_atoms_list_train = [int(Path(p.parts[1]).stem.split('_')[2]) for p in train_paths]
            # x_atoms_list_val = [int(Path(p.parts[1]).stem.split('_')[2]) for p in val_paths]
            # print(f"They are stratified: {[x_atoms_list_train.count(x) for x in unique_x]} versus {[x_atoms_list_val.count(x) for x in unique_x]}.")
            # print(f"The proportions are \n{[x_atoms_list_train.count(x)/len(train_paths) for x in unique_x]} versus \n{[x_atoms_list_val.count(x)/len(val_paths) for x in unique_x]}.")

        # Set extra validation curve
            val_paths_extra = []
            if extra_custom_validation:
                use_only = ["64_ATOMS"]
                x_atoms_paths = [pointers_folder / f"SHARE_OUTPUTS_{n}" for n in use_only]
                filepath = "structures.txt"
                structures_paths = [[] for _ in x_atoms_paths]
                for i, x_atoms_path in enumerate(x_atoms_paths):
                    structures = read_structures_paths(str(x_atoms_path / filepath))
                    for structure in structures:
                        structures_paths[i].append(x_atoms_path.parts[-1] +"/"+ structure)

                # Now we join them with the true parent folder
                for structures in structures_paths:
                    for structure in structures:
                        true_path = true_dataset_folder / structure
                        val_paths_extra.append(true_path)

    # Use previous dataset
    else:
        if config.get("trained_model_path") is None:
            raise ValueError("There is no pretrained model.")
        
        previous_dataset_dir = Path(*Path(config.get("trained_model_path")).parts[:2]) / "dataset"
        train_paths = read_structures_paths(str(previous_dataset_dir / f"train_dataset.txt"))
        train_paths = [Path(path) for path in train_paths]
        val_paths = read_structures_paths(str(previous_dataset_dir / f"val_dataset.txt"))
        val_paths = [Path(path) for path in val_paths]
        paths = train_paths + val_paths

        val_paths_extra = read_structures_paths(str(previous_dataset_dir / f"val_dataset_extra.txt")) if extra_custom_validation else None
        val_paths_extra = [Path(path) for path in val_paths_extra] if extra_custom_validation else None

        if debug_mode:
            train_paths=[train_paths[0]]
            val_paths=[val_paths[0]]


    



    # == Basis creation === 
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)


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
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        # weight_decay=float(optimizer_config["weight_decay"])
    )
    print(f"Using Optimizer {optimizer}")

    # Scheduler
    scheduler_config = config["scheduler"]

    # len_train_dataloader = int(len(paths) * dataset_config.get("train_split_ratio"))
    scheduler_args, scheduler_kwargs = get_scheduler_args_and_kwargs(config, verbose=True)#, len_train_dataloader=len_train_dataloader)

    scheduler = scheduler_config.get("type", None)
    if scheduler is not None:
        scheduler = get_object_from_module(scheduler_config.get("type", None), "torch.optim.lr_scheduler")(optimizer, *(scheduler_args or ()), **scheduler_kwargs)
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=20,
    #     T_mult=2,
    #     eta_min=1e-25
    # )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, cooldown=0, min_lr=0, eps=0)


    # Loss function
    trainer_config = config["trainer"]
    loss_fn = get_object_from_module(trainer_config["loss_function"], "graph2mat.core.data.metrics")
    print(f"Using Loss function {loss_fn}")



    # === Trainer initialization ===

    # Load saved model if required
    trained_model_path = config.get("trained_model_path", None)
    if trained_model_path is not None:
        initial_lr = float(optimizer_config.get("initial_lr", None))
        model, checkpoint, optimizer, scheduler = load_model(model, optimizer, trained_model_path, lr_scheduler=scheduler, initial_lr=initial_lr, device=device)
        print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")
    else:
        checkpoint = None
    

    # === Dataset creation ===

    config["dataset"]["max_samples"] = 1 if debug_mode else None#config["dataset"]["max_samples"]

    print("Creating dataset...")
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    splits = [train_paths, val_paths] if not extra_custom_validation else [train_paths, val_paths, val_paths_extra]
    datasets = []
    for split in splits:
        embeddings_configs = []
        for i, path in enumerate(split):
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
    if extra_custom_validation:
        val_dataset_extra = datasets[2]

    # # Split dataset (also stratify)
    # split = config["dataset"]["max_samples"] is None or config["dataset"]["max_samples"] > 1
    # if split:
    #     n_atoms_list = [dataset[i].num_nodes for i in range(len(dataset))] if config["dataset"]["stratify"] == True else None
    #     train_dataset, val_dataset = train_test_split(
    #         dataset, 
    #         train_size=config["dataset"]["train_split_ratio"],
    #         stratify=n_atoms_list,
    #         random_state=None # Dataset already shuffled (paths)
    #         )
    #     print(f"Dataset splitted in {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    # else:
    #     train_dataset = dataset
    #     val_dataset = dataset
    #     print("There is just 1 sample in the dataset. Using it for both train and validation. Use this only for debugging.")
    
    # Keep all the dataset in memory
    keep_in_memory = trainer_config.get("keep_in_memory", False)
    rotating_pool = trainer_config.get("rotating_pool", False)
    rotating_pool_size = trainer_config.get("rotating_pool_size", 50)
    if keep_in_memory:
        print("Keeping all the dataset in memory.")
        train_dataset = InMemoryData(train_dataset)
        val_dataset = InMemoryData(val_dataset)
        val_dataset_extra = InMemoryData(val_dataset_extra) if extra_custom_validation else None
    elif rotating_pool:
        print("Using rotating pool for the dataset.")
        train_dataset = RotatingPoolData(train_dataset, pool_size=rotating_pool_size)
        val_dataset = RotatingPoolData(val_dataset, pool_size=rotating_pool_size)
        val_dataset_extra = RotatingPoolData(val_dataset_extra, pool_size=rotating_pool_size) if extra_custom_validation else None

        
    # Trainer
    trainer = Trainer(
        model = model,
        config = config,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        val_dataset_extra = val_dataset_extra,
        loss_fn = loss_fn,
        optimizer = optimizer,
        device = device,
        lr_scheduler = scheduler,
        live_plot = trainer_config["live_plot"],
        live_plot_freq = trainer_config["live_plot_freq"],
        live_plot_matrix = trainer_config["live_plot_matrix"],
        live_plot_matrix_freq = trainer_config["live_plot_matrix_freq"],
        results_dir = config["results_dir"],
        checkpoint_freq = trainer_config["checkpoint_freq"],
        batch_size = trainer_config["batch_size"],
        processor=processor,
        model_checkpoint = checkpoint,
    )



    # === Start training ===
    print(f"\nTRAINING STARTS with {len(train_dataset)} train samples and {len(val_dataset)} validation samples.")
    print(f"Using device: {device}")

    trainer.train(num_epochs=trainer_config["num_epochs"])
    
    print("\nTraining completed successfully!")
  

if __name__ == "__main__":
    main()