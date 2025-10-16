# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
import sys
from pathlib import Path
import warnings

from graph2mat4abn.modules.node_operations import HamGNNInspiredNodeBlock
# Add the root directory to Python path
# root_dir = Path(__file__).parent.parent  # Assuming train.py is in scripts/
# sys.path.append(str(root_dir))


import random
import sisl
import torch
import torch.optim as optim
from e3nn import o3
import graph2mat

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
from graph2mat4abn.tools.scripts_utils import generate_g2m_dataset_from_paths, init_mace_g2m_model



def main():
    # *********************************** #
    # * VARIABLES TO CHANGE BY THE USER * #
    # *********************************** #

    config = load_config("./config.yaml")


    # **************** CODE **************** #

    # Load configurations
    debug_mode = config.get("debug_mode", False)
    trainer_config = config["trainer"]
    dataset_config = config["dataset"]
    device = torch.device(config["device"] if (torch.cuda.is_available() and config["device"]!="cpu") 
    else 'cpu')

    # Hide some warnings
    warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
    warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")


    # **************** DATASET - PATHS **************** #

    # Load config variables
    extra_custom_validation = dataset_config.get("extra_custom_validation", False)
    exclude_carbons = dataset_config.get("exclude_carbons", True)
    use_only = dataset_config.get("use_only", None)
    custom_dataset = dataset_config.get("custom_dataset", False)
    use_previous_dataset = True if config.get("trained_model_path", None) is not None else False

    true_dataset_folder = Path('./dataset')
    pointers_folder = Path('./dataset_nocarbon')

    if use_only is not None:
        x_atoms_paths = [pointers_folder / f"SHARE_OUTPUTS_{n}" for n in use_only]
    else:
        x_atoms_paths = list(pointers_folder.glob('*/'))

    # Gather all structure paths
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


    if debug_mode:
        train_paths=train_paths[:2]
        val_paths=val_paths[:2]
        print("Debug mode: only 2 structures in train and validation.")

    # **************** BASIS GENERATION **************** #
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)
    

    # **************** MODEL - OPTIMIZER - LR_SCHEDULER - LOSS_FN INIT **************** #

    model, optimizer, scheduler, loss_fn = init_mace_g2m_model(config, table)


    # **************** DATASET GENERATION from paths **************** #

    train_dataset, val_dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, train_paths, val_paths=val_paths, device=device, verbose=True)


    # **************** LOAD PREVIOUS MODEL **************** #

    # Load model if specified
    optimizer_config = config["optimizer"]
    trained_model_path = config.get("trained_model_path", None)

    if trained_model_path is not None:

        initial_lr = float(optimizer_config.get("initial_lr", None))

        model, checkpoint, optimizer, _ = load_model(model, optimizer, trained_model_path, lr_scheduler=None, initial_lr=initial_lr, device=device)

        print(f"Loaded model in epoch {checkpoint['epoch']} with training loss {checkpoint['train_loss']} and validation loss {checkpoint['val_loss']}.")
    else:
        checkpoint = None
    
        
    # **************** TRAINER INIT **************** #

    trainer = Trainer(
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
        results_dir = config["results_dir"],
        checkpoint_freq = trainer_config["checkpoint_freq"],
        batch_size = trainer_config["batch_size"],
        processor=processor,
        model_checkpoint = checkpoint,
    )


    # **************** TRAINING LOOP **************** #

    print(f"\nTRAINING STARTS with {len(train_dataset)} train samples and {len(val_dataset)} validation samples.")
    print(f"Using device: {device}")

    trainer.train(num_epochs=trainer_config["num_epochs"])
    
    print("\nTraining completed successfully!")
  



if __name__ == "__main__":
    main()