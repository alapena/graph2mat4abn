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
from joblib import dump, load
import plotly.graph_objects as go

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
    compute_calculations = False # Save or Load calculations.

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

    savedir = model_dir / "results/alldataset"
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
                for k in range(len(h_pred.data)):
                    row = h_pred.row[k]
                    col = h_pred.col[k]
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
            'val_labels':   val_labels,
            'train_paths': train_paths,
            'val_paths': val_paths,
        }, calculations_path, compress=3)
        print("Calculations saved!")

        # Free the memory
        del true_sparse_matrices, pred_sparse_matrices, split_labels
        del train_data, val_data, train_labels, val_labels
        del train_dataset, val_dataset, processor, model, basis, table

    # Load the results
    print("Loading the results...")
    try:
        data = load(calculations_path)
        print("Results loaded!")
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

    n_train_samples = len(train_data[0])
    n_val_samples = len(val_data[0])

    # # Means
    # train_means = (
    #     np.array([m.mean() for m in train_true]),
    #     np.array([m.mean() for m in train_pred])
    # )
    # val_means = (
    #     np.array([m.mean() for m in val_true]),
    #     np.array([m.mean() for m in val_pred])
    # )

    # # Standard deviations (ddof=1)
    # train_stds = (
    #     np.array([np.std(m.toarray(), ddof=0) for m in train_true]),
    #     np.array([np.std(m.toarray(), ddof=0) for m in train_pred])
    # )
    # val_stds = (
    #     np.array([np.std(m.toarray(), ddof=0) for m in val_true]),
    #     np.array([np.std(m.toarray(), ddof=0) for m in val_pred])
    # )

    print("Generating results...")
    
    # filepath= savedir / "alldataset_matrix_elements.html"
    # title = f"Dataset analysis. Used model {model_dir.parts[-1]}"
    # plot_dataset_results(
    #     train_data=train_data, val_data=val_data,
    #     colors=colors, title=title,
    #     train_labels=train_labels, val_labels=val_labels,
    #     train_means=train_means, val_means=val_means,
    #     train_stds=train_stds, val_stds=val_stds,
    #     maxaes=maxaes, maxaes_labels=maxaes_labels,
    #     filepath=filepath
    # )

    # plot_pred_vs_true_matrix_elements(
    #     train_data=train_data, val_data=val_data,
    #     colors=colors, title=title,
    #     train_labels=train_labels, val_labels=val_labels,
    #     filepath=filepath
    # )
    # print(f"Results of matrix elements saved at {filepath}!")

    # filepath= savedir / "alldataset_mean.html"
    # title = f"Mean Dataset analysis. Used model {model_dir.parts[-1]}"
    # plot_pred_vs_true_mean(
    #     colors=colors, title=title,
    #     train_means=train_means, val_means=val_means,
    #     train_stds=train_stds, val_stds=val_stds,
    #     labels=maxaes_labels,
    #     n_train_samples=n_train_samples, n_val_samples=n_val_samples,
    #     filepath=filepath
    # )

    # Compute the labels of the scalar values plots.
    labels_train = [path.parts[-2][14:] +"/"+ path.parts[-1] for path in train_paths]
    labels_val = [path.parts[-2][14:] +"/"+ path.parts[-1] for path in val_paths]

    # Mean of the absolute error
    diff_train = [t - p for t, p in zip(train_true, train_pred)]
    diff_val = [t - p for t, p in zip(val_true, val_pred)]
    values_train = [np.mean(diff_train[i].data) for i in range(len(diff_train))]
    values_val = [np.mean(diff_val[i].data) for i in range(len(diff_val))]

    filepath= savedir / "alldataset_mean.html"
    title = f"Mean(T-P) of non-zero elements. Used model {model_dir.parts[-1]}"
    title_x="Mean(True-Pred) (eV)"
    plot_alldataset_struct_vs_scalar(
        title, title_x,
        values_train, labels_train, n_train_samples,
        values_val=values_val, labels_val=labels_val, n_val_samples=n_val_samples,
        filepath=filepath
    )
    print(f"Mean results saved at {filepath}")

    # Mean of the absolute value of the absolute error
    values_train = [np.mean(np.abs(diff_train[i].data)) for i in range(len(diff_train))]
    values_val = [np.mean(np.abs(diff_val[i].data)) for i in range(len(diff_val))]

    filepath= savedir / "alldataset_abs_mean.html"
    title = f"Mean(Abs(T-P)) of non-zero elements. Used model {model_dir.parts[-1]}"
    title_x="Mean(Abs(True-Pred)) (eV)"
    plot_alldataset_struct_vs_scalar(
        title, title_x,
        values_train, labels_train, n_train_samples,
        values_val=values_val, labels_val=labels_val, n_val_samples=n_val_samples,
        filepath=filepath
    )
    print(f"Mean(Abs) results saved at {filepath}")

    # Standard deviation of Absolute value of Absolute Error
    values_train = [np.std(np.abs(diff_train[i].data)) for i in range(len(diff_train))]
    values_val = [np.std(np.abs(diff_val[i].data)) for i in range(len(diff_val))]

    filepath= savedir / "alldataset_abs_std.html"
    title = f"Std(Abs(T-P)) of non-zero elements. Used model {model_dir.parts[-1]}"
    title_x="Std(Abs(True-Pred)) (eV)"
    plot_alldataset_struct_vs_scalar(
        title, title_x,
        values_train, labels_train, n_train_samples,
        values_val=values_val, labels_val=labels_val, n_val_samples=n_val_samples,
        filepath=filepath
    )
    print(f"Std results saved at {filepath}")

    # Maximum absolute Error
    values_train = np.array([np.max(np.abs(t - p)) for t, p in zip(train_true, train_pred)])
    values_val = np.array([np.max(np.abs(t - p)) for t, p in zip(val_true, val_pred)])

    filepath= savedir / "alldataset_maxae.html"
    title = f"Max Absolute Error Dataset analysis. Used model {model_dir.parts[-1]}"
    title_x="|True-Pred| (eV)"
    plot_alldataset_struct_vs_scalar(
        title, title_x,
        values_train, labels_train, n_train_samples,
        values_val=values_val, labels_val=labels_val, n_val_samples=n_val_samples,
        filepath=filepath
    )
    print(f"Maxae results saved at {filepath}")

    
    
    # Separate on-sites from hoppings:

    # Mean of only the on-sites
    splits = [train_data, val_data]
    split_paths = (train_paths, val_paths)


    splits_absmean_onsites = ([], [])
    splits_absmean_hops = ([], [])
    splits_absstd_onsites = ([], [])
    splits_absstd_hops = ([], [])
    # For each split (train, val)
    for k in range(len(splits)):
        if k != 0:
            break
        true_matrices = splits[k][0]
        pred_matrices = splits[k][1]
        n_matrices = len(true_matrices)

        # For each matrix
        onsite_values = []
        hopping_values = []
        for i in tqdm(range(n_matrices)):
            if i == 5:
                break

            diff_matrix = (true_matrices[i] - pred_matrices[i]).tocoo()

            # Compute labels for each nnz element
            path = Path("./") / split_paths[k][i]
            geometry = sisl.get_sile(path / "aiida.fdf").read_geometry()
            for j in range(len(diff_matrix.data)):
                row = diff_matrix.row[j]
                col = diff_matrix.col[j]
                orb_in = orbitals[col % n_orbs]
                orb_out = orbitals[row % n_orbs]
                isc = str(geometry.o2isc(col))

                # Join altogether
                label = ''.join([orb_in, " -> ", orb_out, " ", isc])

                # Store
                if isc == '[0 0 0]' and row == col:
                    onsite_values.append(diff_matrix.data[j])
                else:
                    hopping_values.append(diff_matrix.data[j])

            # Store the result for each matrix
            splits_absmean_onsites[k].append(np.mean(np.abs(onsite_values)))
            splits_absmean_hops[k].append(np.mean(np.abs(hopping_values)))
            splits_absstd_onsites[k].append(np.std(np.abs(onsite_values)))
            splits_absstd_hops[k].append(np.std(np.abs(hopping_values)))
            

    # Plot the results
    values_train_onsites = splits_absmean_onsites[0]
    values_val_onsites = splits_absmean_onsites[1]
    values_train_hops = splits_absmean_hops[0]
    values_val_hops = splits_absmean_hops[1]
    filepath= savedir / "alldataset_absmean.html"
    title = f"Mean(Abs(T-P)). Used model {model_dir.parts[-1]}"
    title_x="Mean(Abs(T-P)) (eV)"
    plot_alldataset_struct_vs_scalar_onsites_hoppings(
        title, title_x,
        values_train_onsite=values_train_onsites, values_train_hop=values_train_hops, labels_train=labels_train, n_train_samples=len(values_train_onsites),
        values_val_onsite=values_val_onsites, values_val_hop=values_val_hops, labels_val=labels_val, n_val_samples=len(values_val_onsites),
        filepath=None
    )
            

    # ! COSTY:
    # # Diagonal plot for each structure
    # split_paths = (train_paths, val_paths)

    # structures_train = [path.parts[-2][14:] +"/"+ path.parts[-1] for path in train_paths]
    # structures_val = [path.parts[-2][14:] +"/"+ path.parts[-1] for path in val_paths]
    # splits_structures = (structures_train, structures_val)

    # title_x="True matrix elements (eV)"
    # title_y="Predicted matrix elements (eV)"

    # splits = [train_data, val_data]
    # splits_str = ["train", "val"]

    # # For each split (train, val)
    # for j in range(len(splits)):
    #     print(f"Plotting split {j}...")
    #     true_matrices = splits[j][0]
    #     pred_matrices = splits[j][1]
    #     n_matrices = len(true_matrices)

    #     # For each matrix
    #     for i in tqdm(range(n_matrices)):

    #         # Compute labels
    #         path = Path("./") / split_paths[j][i]
    #         geometry = sisl.get_sile(path / "aiida.fdf").read_geometry()
    #         matrix_labels = []
    #         for k in range(len(true_matrices[i].data)):
    #             row = true_matrices[i].row[k]
    #             col = true_matrices[i].col[k]
    #             orb_in = orbitals[col % n_orbs]
    #             orb_out = orbitals[row % n_orbs]
    #             isc = str(geometry.o2isc(col))

    #             # Join altogether
    #             label = ''.join([orb_in, " -> ", orb_out, " ", isc])

    #             # Store the labels 
    #             matrix_labels.append(label)

    #         filepath= savedir / f"nnz_elements_{splits_str[j]}_{Path(splits_structures[j][i]).parts[-2]}_{Path(splits_structures[j][i]).parts[-1]}.html"
    #         title = f"Matrix elements of structure {splits_structures[j][i]}.<br>{splits_str[j]} dataset.<br>Used model {model_dir.parts[-1]}"
    #         true_values = true_matrices[i].data
    #         pred_values = pred_matrices[i].data
    #         plot_diagonal(
    #             true_values, pred_values, matrix_labels, # 1D array of elements.
    #             title=title, title_x=title_x, title_y=title_y, colors=None,
    #             filepath=filepath
    #         )
    # print(f"Matrix elements results saved at {savedir}")





    
def plot_diagonal(
    true_values, pred_values, labels, # 1D array of elements.
    title="Default title", title_x="X axis", title_y="Y axis", colors=None,
    filepath=None
):

    fig = go.Figure()
    
    # ====== TRAINING DATA ======
    traces = []
    # Training matrix elements
    trace = go.Scattergl(
        x=true_values,
        y=pred_values,
        mode='markers',
        marker=dict(
            # symbol='dash',
            size=5,
            # color=colors[i % len(colors)],
            line=dict(width=0)
        ),
        name=f'Matrix elements',
        text=labels,
        # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
        # legendgroup='training',
        # legendgrouptitle="Training samples",
        # showlegend=True
    )
    traces.append(trace)
        

    # Add identity line
    all_data = np.concatenate([true_values, pred_values])
    vmin, vmax = np.min(all_data), np.max(all_data)
    diagonal_trace = go.Scatter(
        x=[vmin, vmax],
        y=[vmin, vmax],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Ideal'
    )
    traces.append(diagonal_trace)


    # Create figure and update layout
    fig = go.Figure(data=traces)
    fig.update_layout(
        width=900,
        height=900,
        title=title,
        # xaxis_title=title_x,
        # yaxis_title=title_y,
        # legend_title='Legend',
        # hovermode='closest',
        # template='plotly_white',
        xaxis=dict(
            title=title_x,
            tickformat=".2f"
        ),
        yaxis=dict(
            title=title_y,
            tickformat=".2f"
        )
    )

    # Save to HTML if path is provided
    if filepath:
        f = open(filepath, "w")
        f.close()
        with open(filepath, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.close()
        
        # with open(f"{str(filepath)[:-4]}.json", "w") as f:
        #     f.write(fig.to_json())

    else:
        fig.show()

    return fig



def plot_alldataset_struct_vs_scalar(
    title, title_x,
    values_train, labels_train, n_train_samples,
    values_val=None, labels_val=None, n_val_samples=None,
    filepath=None
):
    # 1) Build a master list of unique labels in the order you want them
    all_labels = list(dict.fromkeys(
        labels_train + (labels_val if n_val_samples else [])
    ))

    # 2) Map each label to its integer position
    idx_map = {lab: i for i, lab in enumerate(all_labels)}

    # 3) Convert your y-arrays to numeric
    y_train = [idx_map[l] for l in labels_train]
    y_val   = [idx_map[l] for l in (labels_val if n_val_samples else [])]

    # 4) Build your traces using numeric y
    traces = [
        go.Scatter(
            x=values_train, y=y_train,
            mode='markers',
            marker=dict(symbol='x', size=6, color='blue'),
            name='Training',
            showlegend=True
        )
    ]

    # Dummy trace to show x axis on top
    traces.append(go.Scatter(
        x=[min(values_train), max(values_train)],
        y=[-1, -1],  # Place out of view
        xaxis='x2',
        mode='markers',
        marker=dict(opacity=0),
        visible=True,
        showlegend=False
    ))

    if n_val_samples:
        traces.append(
            go.Scatter(
                x=values_val, y=y_val,
                mode='markers',
                marker=dict(symbol='x', size=6, color='red'),
                name='Validation',
                showlegend=True
            )
        )

    fig = go.Figure(traces)

    # 5) Set the axis so categories sit exactly at integer ticks from 0…N-1
    fig.update_layout(
        width=800,
        height=(n_train_samples + (n_val_samples or 0)) * 15 + 100,
        title={
            'text': title,
            'y':1,
            'x':0.01,
            'xanchor': 'left',
            'yanchor': 'top',
            "pad": dict(t=10),
        },
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(all_labels))),
            ticktext=all_labels,
            range=[-0.5, len(all_labels) - 0.5],
            autorange=False
        ),
        xaxis=dict(
            title=title_x,
            title_standoff=0,
            side="bottom",
            showline=True,
            exponentformat='power',   # use 'e' for scientific, 'E' for capital E, or 'power' for 10^x
            showexponent='all',  
        ),
        xaxis2=dict(
            title=title_x,
            title_standoff=0,
            overlaying="x",   # share the same data range as xaxis
            side="top",       # draw it at the top
            showline=True,
            exponentformat='power',   # use 'e' for scientific, 'E' for capital E, or 'power' for 10^x
            showexponent='all',   
        ),
        margin=dict(l=40, r=20, t=65, b=45)
    )

    # 6) Save or show
    if filepath:
        f = open(filepath, "w")
        f.close()
        with open(filepath, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.close()
        
        # with open(f"{str(filepath)[:-4]}.json", "w") as f:
        #     f.write(fig.to_json())

    else:
        fig.show()

    return fig




def plot_alldataset_struct_vs_scalar_onsites_hoppings(
    title, title_x,
    values_train_onsite, values_train_hop, labels_train, n_train_samples,
    values_val_onsite=None, values_val_hop=None, labels_val=None, n_val_samples=None,
    filepath=None
):
    # 1) Build a master list of unique labels in the order you want them
    all_labels = list(dict.fromkeys(
        labels_train + (labels_val if n_val_samples else [])
    ))

    # 2) Map each label to its integer position
    idx_map = {lab: i for i, lab in enumerate(all_labels)}

    # 3) Convert your y-arrays to numeric
    y_train = [idx_map[l] for l in labels_train]
    y_val   = [idx_map[l] for l in (labels_val if n_val_samples else [])]

    # 4) Build your traces using numeric y
    traces = [
        go.Scatter(
            x=values_train_onsite, y=y_train,
            mode='markers',
            marker=dict(symbol='0', size=6, color='blue'),
            name='Training onsites',
            showlegend=True
        ),

        go.Scatter(
            x=values_train_hop, y=y_train,
            mode='markers',
            marker=dict(symbol='x', size=6, color='blue'),
            name='Training hoppings',
            showlegend=True
        ),
    ]

    # Dummy trace to show x axis on top
    traces.append(go.Scatter(
        x=[min(values_train_onsite), max(values_train_onsite)],
        y=[-1, -1],  # Place out of view
        xaxis='x2',
        mode='markers',
        marker=dict(opacity=0),
        visible=True,
        showlegend=False
    ))

    if n_val_samples:
        traces.append(
            go.Scatter(
                x=values_val_onsite, y=y_val,
                mode='markers',
                marker=dict(symbol='0', size=6, color='red'),
                name='Validation onsites',
                showlegend=True
            )
        )

        traces.append(
            go.Scatter(
                x=values_val_hop, y=y_val,
                mode='markers',
                marker=dict(symbol='x', size=6, color='red'),
                name='Validation hoppings',
                showlegend=True
            )
        )

    fig = go.Figure(data=traces)

    # 5) Set the axis so categories sit exactly at integer ticks from 0…N-1
    fig.update_layout(
        width=800,
        height=(n_train_samples + (n_val_samples or 0)) * 15 + 100,
        title={
            'text': title,
            'y':1,
            'x':0.01,
            'xanchor': 'left',
            'yanchor': 'top',
            "pad": dict(t=10),
        },
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(all_labels))),
            ticktext=all_labels,
            range=[-0.5, len(all_labels) - 0.5],
            autorange=False
        ),
        xaxis=dict(
            title=title_x,
            title_standoff=0,
            side="bottom",
            showline=True,
            exponentformat='power',   # use 'e' for scientific, 'E' for capital E, or 'power' for 10^x
            showexponent='all',  
        ),
        xaxis2=dict(
            title=title_x,
            title_standoff=0,
            overlaying="x",   # share the same data range as xaxis
            side="top",       # draw it at the top
            showline=True,
            exponentformat='power',   # use 'e' for scientific, 'E' for capital E, or 'power' for 10^x
            showexponent='all',   
        ),
        margin=dict(l=40, r=20, t=65, b=45)
    )

    # 6) Save or show
    if filepath:
        f = open(filepath, "w")
        f.close()
        with open(filepath, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.close()
        
        # with open(f"{str(filepath)[:-4]}.json", "w") as f:
        #     f.write(fig.to_json())

    else:
        fig.show()

    return fig







# def plot_pred_vs_true_mean(
#         colors, title,
#         train_means, val_means,
#         train_stds, val_stds,
#         labels,
#         n_train_samples, n_val_samples=None,
#         filepath=None
# ):

#     fig = go.Figure()

#     # ====== TRAINING DATA ======
#     mean_traces = []
#     std_traces = []
#     for i in range(n_train_samples):

#         # Training means
#         trace = go.Scatter(
#             x=[train_means[0][i]],
#             y=[train_means[1][i]],
#             mode='markers',
#             marker=dict(
#                 symbol='square',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=0)
#             ),
#             name=f'Mean training {i}',
#             # text=labels[0][i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='training_mean',
#             visible=True,
#             showlegend=True,
#         )
#         mean_traces.append(trace)

#         # Training std
#         trace = go.Scatter(
#             x=[train_stds[0][i]],
#             y=[train_stds[1][i]],
#             mode='markers',
#             marker=dict(
#                 symbol='triangle-up',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=0)
#             ),
#             name=f'Std training {i}',
#             text=labels[0][i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='training_std',
#             visible=False,
#             showlegend=True,
#         )
#         std_traces.append(trace)

#     # === Validation ===
#     for i in range(n_val_samples):

#         # Val means
#         trace = go.Scatter(
#             x=[val_means[0][i]],
#             y=[val_means[1][i]],
#             mode='markers',
#             marker=dict(
#                 symbol='square-open',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=1,)
#             ),
#             name=f'Mean val {i}',
#             text=labels[1][i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='val_mean',
#             visible=True,
#             showlegend=True,
#         )
#         mean_traces.append(trace)

#         # Validation std
#         trace = go.Scatter(
#             x=[val_stds[0][i]],
#             y=[val_stds[1][i]],
#             mode='markers',
#             marker=dict(
#                 symbol='triangle-up-open',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=0)
#             ),
#             name=f'Std validation {i}',
#             text=labels[1][i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='val_std',
#             visible=False,
#             showlegend=True,
#         )
#         std_traces.append(trace)
        


#     # Add identity line
#     all_x = np.concatenate([train_means[0], val_means[0],
#                             train_stds[0],  val_stds[0]])
#     vmin, vmax = all_x.min(), all_x.max()

#     # use this for the diagonal
#     diagonal_trace = go.Scatter(
#         x=[vmin, vmax], y=[vmin, vmax],
#         mode='lines', line=dict(color='black'),
#         name='Ideal', visible=True, showlegend=True
#     )

#     # Create figure and update layout
#     min_x_mean = np.min([train_means[0].min(), val_means[0].min()])
#     max_x_mean = np.max([train_means[0].max(), val_means[0].max()])
#     min_y_mean = np.min([train_means[1].min(), val_means[1].min()])
#     max_y_mean = np.max([train_means[1].max(), val_means[1].max()])

#     min_x_std = np.min([train_stds[0].min(), val_stds[0].min()])
#     max_x_std = np.max([train_stds[0].max(), val_stds[0].max()])
#     min_y_std = np.min([train_stds[1].min(), val_stds[1].min()])
#     max_y_std = np.max([train_stds[1].max(), val_stds[1].max()])

#     traces = mean_traces + std_traces + [diagonal_trace]
#     fig = go.Figure(data=traces)
#     fig.update_layout(
#         width=1000,
#         height=1000,
#         title=title,
#         # xaxis_title='True Values',
#         # yaxis_title='Predicted Values',
#         legend_title='Legend',
#         # hovermode='closest',
#         # template='plotly_white',
#         xaxis=dict(
#             title='True Values',
#             tickformat=".2f",
#             range = [min_x_mean, max_x_mean]
#         ),
#         yaxis=dict(
#             title='Predicted Values',
#             tickformat=".2f",
#             range = [min_y_mean, max_y_mean]
#         )
#     )

#     # Add dropdown
#     fig.update_layout(
#         updatemenus=[
#             dict(
#                 buttons=[
#                     dict(
#                         label="Mean",
#                         method="update",
#                         args=[
#                             {"visible": [True]*len(mean_traces) + [False]*len(std_traces) + [True]},
#                             {
#                                 "xaxis": {"range": [min_x_mean-0.0005*min_x_mean, max_x_mean+0.0005*max_x_mean]},
#                                 "yaxis": {"range": [min_y_mean-0.0005*min_y_mean, max_y_mean+0.0005*max_y_mean]},
#                             }

#                         ]
#                     ),
#                     dict(
#                         label="Std",
#                         method="update",
#                         args=[
#                             {"visible": [False]*len(mean_traces) + [True]*len(std_traces) + [True]},
#                             {
#                                 "xaxis": {"range": [min_x_std-0.0005*min_x_std, max_x_std+0.0005*max_x_std]},
#                                 "yaxis": {"range": [min_y_std-0.0005*min_y_std, max_y_std+0.0005*max_y_std]},
#                             }

#                         ]
#                     ),
#                 ],
#                 direction="down",
#                 pad={"r": 10, "t": 10},
#                 showactive=True,
#                 x=0.75,
#                 xanchor="left",
#                 y=1.1,
#                 yanchor="top"
#             ),
#         ]
#     )

#     # Save to HTML if path is provided
#     if filepath:
#         f = open(filepath, "w")
#         f.close()
#         with open(filepath, 'a') as f:
#             f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
#         f.close()
        
#         # with open(f"{str(filepath)[:-4]}.json", "w") as f:
#         #     f.write(fig.to_json())

    
#     return fig





# def plot_pred_vs_true_matrix_elements(
#         train_data, val_data,
#         colors, title,
#         train_labels, val_labels,
#         filepath
# ):
    
#     # Three figures. One for the matrix elements (HUGE), one for stats such as mean and std, and one for max abs difference (maxae) because it needs a very long fig. 

#     fig = go.Figure()
#     n_train_samples = len(train_data[0])
#     n_val_samples = len(val_data[0])

#     train_data = [[train_data[k][i].data for i in range(n_train_samples)] for k in range(len(train_data))]
#     val_data = [[val_data[k][i].data for i in range(n_val_samples)] for k in range(len(val_data))]
    

#     # ====== TRAINING DATA ======
#     matrix_traces = []
#     for i in range(n_train_samples):
#         # Training matrix elements
#         trace = go.Scattergl(
#             x=train_data[0][i],
#             y=train_data[1][i],
#             mode='markers',
#             marker=dict(
#                 # symbol='dash',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=0)
#             ),
#             name=f'Training sample {i}',
#             text=train_labels[i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='training',
#             # legendgrouptitle="Training samples",
#             showlegend=True
#         )
#         matrix_traces.append(trace)


#     # === Validation ===
#     for i in range(n_val_samples):
#         trace = go.Scattergl(
#             x=val_data[0][i],
#             y=val_data[1][i],
#             mode='markers',
#             marker=dict(
#                 symbol='circle-open',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=1,)
#             ),
#             name=f'Validation sample {i}',
#             text=val_labels[i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='validation',
#             # legendgrouptitle="Validation samples",
#             showlegend=True
#         )
#         matrix_traces.append(trace)
        


#     # Add identity line
#     train_flattened_data = ([np.min(train_data[0][i]) for i in range(n_train_samples)], [np.max(train_data[0][i]) for i in range(n_val_samples)]) # [train, val]
#     train_flattened_data = [train_flattened_data[k] for k in range(len(train_flattened_data))]
#     min, max = np.min(train_flattened_data[0]), np.max(train_flattened_data[1])
#     diagonal_trace = go.Scatter(
#         x=[min, max],
#         y=[min, max],
#         mode='lines',
#         line=dict(color='black', dash='solid'),
#         name='Ideal'
#     )


#     # Create figure and update layout
#     traces = matrix_traces + [diagonal_trace]
#     fig = go.Figure(data=traces)
#     fig.update_layout(
#         width=1000,
#         height=1000,
#         title=title,
#         # xaxis_title='True Values',
#         # yaxis_title='Predicted Values',
#         legend_title='Legend',
#         # hovermode='closest',
#         # template='plotly_white',
#         xaxis=dict(
#             title='True Values',
#             tickformat=".2f"
#         ),
#         yaxis=dict(
#             title='Predicted Values',
#             tickformat=".2f"
#         )
#     )

#     # Save to HTML if path is provided
#     if filepath:
#         f = open(filepath, "w")
#         f.close()
#         with open(filepath, 'a') as f:
#             f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
#         f.close()
        
#         # with open(f"{str(filepath)[:-4]}.json", "w") as f:
#         #     f.write(fig.to_json())

#     else:
#         fig.show()

#     return fig

if __name__ == "__main__":
    main()