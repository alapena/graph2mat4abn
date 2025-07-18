# === Simulate a proper Python package (temporal, I did not want to waste time on installing things) ===
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
from tools.plot import plot_dataset_results
from joblib import dump, load

from graph2mat import (
    BasisTableWithEdges,
)

def main():
    debug_mode = False
    # *********************************** #
    # * VARIABLES TO CHANGE BY THE USER * #
    # *********************************** #
    model_dir = Path("results/h_crystalls_1") # Results directory
    filename = "train_best_model.tar" # Model name (or relative path to the results directory)
    compute_calculations = True # Save or Load calculations.

    # *********************************** #

    # Hide some warnings
    warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
    warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")
    if debug_mode:
        print("**************************************************")
        print("*                                                *")
        print("*              DEBUG MODE ACTIVATED              *")
        print("*                                                *")
        print("**************************************************")

    savedir = model_dir / "results"
    savedir.mkdir(exist_ok=True, parents=True)
    calculations_path = savedir / "calculations_alldataset.joblib"
    
    # Define orbital labels (for now we will assume that all atoms have the same orbitals). Use the same order as appearance in the hamiltonian.
    orbitals = {
        0: "s1",
        1: "s2",
        2: "py1",
        3: "pz1",
        4: "px1",
        5: "py2",
        6: "pz2",
        7: "px2",
        8: "Pdxy",
        9: "Pdyz",
        10: "Pdz2",
        11: "Pdxz",
        12: "Pdx2-y2",
    }
    n_orbs = len(orbitals)

    # Load the config of the model
    config = load_config(model_dir / "config.yaml")
    device = torch.device("cpu")

    # Load the same dataset used to train/validate the model (paths)
    train_paths, val_paths = get_model_dataset(model_dir, verbose=True)
    
    if compute_calculations:

        # Generate the G2M basis
        paths = train_paths + val_paths
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
        train_dataset, val_dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, train_paths, val_paths, device=device, verbose=True) if not debug_mode else ([i for i in range(len(train_paths))], [i for i in range(len(val_paths))], None)
        splits = [train_dataset, val_dataset]
                
                
        # Iterate through all dataset and compute the magnitudes to plot.
        print("START OF THE LOOP...")
        true_sparse_matrices = [[] for _ in range(len(splits))] # [train, val]
        pred_sparse_matrices = [[] for _ in range(len(splits))] # [train, val]
        split_labels = [[] for _ in range(len(splits))] # [train, val]
        for k, dataset in enumerate(splits):
            print(f"Computing split {k}...")
            for i, data in tqdm(enumerate(dataset)):
                if debug_mode and i==5:
                    break
                # Generate prediction. Create dataloaders (needed to generate a prediction)
                data = DataLoader([data], 1)
                data = next(iter(data))

                # Generate prediction
                model_predictions = model(data=data) if not debug_mode else None

                # Reconstruct matrices (sparse format)
                h_pred = processor.matrix_from_data(data, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})[0].tocsr().tocoo() if not debug_mode else model(n_atoms=2, n_orbs=13, n_shifts=10, size=100).tocsr().tocoo()
                h_true = processor.matrix_from_data(data)[0].tocsr().tocoo() if not debug_mode else model(n_atoms=2, n_orbs=13, n_shifts=10, size=100).tocsr().tocoo()

                true_sparse_matrices[k].append(h_true)
                pred_sparse_matrices[k].append(h_pred)

                # Compute the labels
                # n_atoms = data.num_nodes if not debug_mode else h_pred.shape[0] // n_orbs
                # cols = h_pred.shape[1]

                # Orbitals incoming (cols) and shift indices
                # orb_in = np.empty([n_atoms*n_orbs,cols], dtype='<U10')
                # isc = np.empty([n_atoms*n_orbs,cols], dtype='<U10')

                path = Path(data.metadata["path"][0]) if not debug_mode else paths[i + k*len(splits[0])]
                geometry = sisl.get_sile(path / "aiida.fdf").read_geometry()
                # isc_list = [geometry.o2isc(io) for io in range(cols)]
                # for col in range(h_pred.shape[1]):
                #     orb_in[:,col] = orbitals[col % n_orbs]
                #     isc[:,col] = str(isc_list[col])

                # # Orbitals outcoming (rows)
                # orb_out = np.empty([n_atoms*n_orbs,cols], dtype='<U10')
                # for row in range(h_pred.shape[0]):
                #     orb_out[row,:] = orbitals[row % n_orbs]

                # # Join altogether
                # characters = [orb_in, " -> ", orb_out, " ", isc]
                # labels = np.char.add(characters[0], characters[1])
                # for j in range(1, len(characters)-1):
                #     labels = np.char.add(labels, characters[j+1])

                # for j, nnz_element in enumerate(h_pred.data):

                # Predicted labels
                for j in range(len(h_pred.data)):
                    row = h_pred.row[j]
                    col = h_pred.col[j]
                    orb_in = orbitals[col % n_orbs]
                    orb_out = orbitals[row % n_orbs]
                    isc = str(geometry.o2isc(col))

                    # Join altogether
                    label = ''.join([orb_in, " -> ", orb_out, " ", isc])

                    # Store the labels 
                    split_labels[k].append(label)
            

        print(f"Saving the results in {calculations_path}...")
        # Compute the other statistics
        train_data = (true_sparse_matrices[0], pred_sparse_matrices[0])  # (true, pred)
        val_data   = (true_sparse_matrices[1], pred_sparse_matrices[1])  # (true, pred)
        train_labels = split_labels[0]
        val_labels   = split_labels[1]

        # Save the results
        dump({
            'train_true':   train_data[0],
            'train_pred':   train_data[1],
            'val_true':     val_data[0],
            'val_pred':     val_data[1],
            'train_labels': train_labels,
            'val_labels':   val_labels
        }, calculations_path, compress=3)
        print("Calculations saved!")

    # Load the results
    try:
        data = load(calculations_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the saved calculations at {calculations_path}")

    # Reconstruct your tuples and labels
    train_data   = (data['train_true'],   data['train_pred'])
    val_data     = (data['val_true'],     data['val_pred'])
    train_labels = data['train_labels']
    val_labels   = data['val_labels']


    # Unpack for clarity
    train_true, train_pred = train_data
    val_true,   val_pred   = val_data

    # Means
    train_means = (
        np.array([m.mean() for m in train_true]),
        np.array([m.mean() for m in train_pred])
    )
    val_means = (
        np.array([m.mean() for m in val_true]),
        np.array([m.mean() for m in val_pred])
    )

    # Standard deviations (ddof=1)
    train_stds = (
        np.array([np.std(m.toarray(), ddof=0) for m in train_true]),
        np.array([np.std(m.toarray(), ddof=0) for m in train_pred])
    )
    val_stds = (
        np.array([np.std(m.toarray(), ddof=0) for m in val_true]),
        np.array([np.std(m.toarray(), ddof=0) for m in val_pred])
    )

    # Max absolute error
    maxae_train = np.array([
        np.max(np.abs(t - p))
        for t, p in zip(train_true, train_pred)
    ])
    maxae_val = np.array([
        np.max(np.abs(t - p))
        for t, p in zip(val_true, val_pred)
    ])

    maxaes = ([maxae_train, maxae_val])
    maxaes_labels = ([path.parts[-2][14:] +"/"+ path.parts[-1] for path in train_paths], [path.parts[-2][14:] +"/"+ path.parts[-1] for path in val_paths])

    print("Generating results...")
    
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # medium gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
        '#fdae61',  # sandy orange
        '#66c2a5',  # seafoam green
        '#fc8d62',  # coral
        '#a6d854',  # light lime
        '#ffd92f',  # sunflower
        '#e5c494',  # beige
        '#b3b3b3'   # soft gray
    ]
    filepath= savedir / "dataset_analysis.html"
    title = f"Dataset analysis. Used model {model_dir.parts[-1]}"
    plot_dataset_results(
        train_data=train_data, val_data=val_data,
        colors=colors, title=title,
        train_labels=train_labels, val_labels=val_labels,
        train_means=train_means, val_means=val_means,
        train_stds=train_stds, val_stds=val_stds,
        maxaes=maxaes, maxaes_labels=maxaes_labels,
        filepath=filepath
    )
    print(f"Results saved at {filepath}!")



if __name__ == "__main__":
    main()