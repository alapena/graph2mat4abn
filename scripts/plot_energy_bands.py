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
import scipy

from e3nn import o3
from tqdm import tqdm
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

from graph2mat4abn.tools.plot import plot_columns_of_2darray, plot_error_matrices_big, plot_predictions_vs_truths
from graph2mat4abn.tools import load_config, flatten
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, get_orbital_indices_and_shifts_from_sile, get_kwargs, load_model, reconstruct_tim_from_coo, reduced_coord
from graph2mat4abn.tools.import_utils import get_object_from_module



def main():

    debug_mode = False # Set all values to min so that the exec time is fastest
    check_existing_plots = True

    # * Write here the directory where the model is stored
    directory = Path("results/h_crystalls_1") 
    # * And the model name
    filename = "train_best_model.tar"
    # * Set the (max) number of each structure type that you want to plot
    n = 3
    # * Set the max number of energy bands you want in each plot
    n_bands = None


    n_plots_each = {
        2: n,
        3: n,
        8: n,
        32: n,
        64: n,
    }

    # === Configuration load ===
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
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



    # === Dataset creation ===
    config["dataset"]["max_samples"] = 1 if debug_mode else config["dataset"]["max_samples"]
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    embeddings_configs = []
    for i, path in enumerate(paths):

        # We need to keep track of the training/val splits, so we can't plot more than used for training (at least for training dataset)
        if i==config["dataset"]["max_samples"]:
            break
        
        print(f"Processing structure {i+1} of {len(paths)}...")

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
                "path": path,
            },
        )

        embeddings_configs.append(embeddings_config)

    dataset = TorchBasisMatrixDataset(embeddings_configs, data_processor=processor)

    # Split and stratify
    n_atoms_list = [dataset[i].num_nodes for i in range(len(dataset))] if config["dataset"]["stratify"] == True else None
    if not debug_mode:
        train_dataset, val_dataset = train_test_split(
            dataset, 
            train_size=config["dataset"]["train_split_ratio"],
            stratify=n_atoms_list,
            random_state=None # Dataset already shuffled (paths)
        )
    else:
        train_dataset = dataset
        val_dataset = dataset

    print("Initializing model...")

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

    # ! Careful now that we changed the scheduler for some models
    # Scheduler
    scheduler_config = config["scheduler"]
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(scheduler_config["t_0"]),
        T_mult=scheduler_config["t_multiplication"],
        eta_min=float(scheduler_config["eta_min"])
    )

    

    # === Model load ===
    model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, directory / filename, lr_scheduler=lr_scheduler, device=device)
    loss_fn = get_object_from_module(config["trainer"]["loss_function"], "graph2mat.core.data.metrics")



    # ====== Matrix plots ======
    results_directory = directory / "results"
    results_directory.mkdir(exist_ok=True)

    n_atoms_list = list(n_plots_each.keys())
    # Dataset plots
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, 1)
    val_dataloader = DataLoader(val_dataset, 1)

    dataloaders = [train_dataloader, val_dataloader]
    for dataloader_id, dataloader in enumerate(dataloaders):
        dataloader_type = "training" if dataloader_id == 0 else "validation"
        print(f"Plotting {dataloader_type} dataset...")

        #Count the number of plots already done for each structure type
        n_plotted = np.zeros([2, len(n_atoms_list)], dtype=np.int16)
        n_plotted[0] += n_atoms_list

        # For each split
        for j, data in enumerate(dataloader):
            print(f"====== Plotting matrix {j} ======")
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

            # Predicted matrix
            h_pred = processor.matrix_from_data(
                data,
                predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]},
            )[0]

            # True matrix
            true_matrix = processor.matrix_from_data(
                data,
            )[0]

            # === Plot Hamiltonians ===
            filepath_html = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}_hamiltonian.html")
            filepath_png = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}_hamiltonian.png")
            if check_existing_plots:
                if filepath_html.is_file():
                    print(f"File {filepath_html} already exists. Continueing.")
                elif filepath_png.is_file():
                    print(f"File {filepath_png} already exists. Continueing.")
                else:
                    print("Plotting hamiltonian...")
                    title = f"Results of sample {j} of {dataloader_type} dataset (seed {config["dataset"]["seed"]}). There are {n_atoms} in the unit cell."
                    predicted_matrix_text = f"Saved training loss at epoch {checkpoint["epoch"]}:     {checkpoint["train_loss"]:.2f} eV²·100\nMSE evaluation:     {loss.item():.2f} eV²·100"
                    if n_atoms <= 32:
                        plot_error_matrices_big(
                            true_matrix.todense(), h_pred.todense(),
                            matrix_label="Hamiltonian",
                            figure_title=title,
                            predicted_matrix_text=predicted_matrix_text,
                            filepath = filepath_html
                        )
                    else:
                        plot_error_matrices_big(
                            true_matrix.todense(), h_pred.todense(),
                            matrix_label="Hamiltonian",
                            figure_title=title,
                            predicted_matrix_text=predicted_matrix_text,
                            filepath = filepath_png
                        )


            # ========= Plot energy bands =========
            print("Plotting energy bands...")
            filepath_html = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}_energybands.html")
            if filepath_html.is_file():
                print(f"File {filepath_html} already exists. Continueing.")
            else:
                file = sisl.get_sile(data.metadata["path"][0] / "aiida.fdf")
                geometry = file.read_geometry()
                cell = geometry.cell
                # orb_i, orb_j, isc = get_orbital_indices_and_shifts_from_sile(file)

                # Define a path in k-space
                kxs = np.linspace(0,1, num=40)
                kys = np.linspace(0,1, num=40)
                kzs = np.linspace(0,1, num=40) # * Change the resolution here
                k_dir_x = geometry.rcell[:,0]
                k_dir_y = geometry.rcell[:,1]
                k_dir_z = geometry.rcell[:,2]
                k_path_x=np.array([kx*k_dir_x for kx in kxs])
                k_path_y=np.array([ky*k_dir_y for ky in kys])
                k_path_z=np.array([kz*k_dir_z for kz in kzs])
                k_path=np.concatenate([k_path_x, k_path_y, k_path_z])

                k_path = np.array([[0, 0, 0], [0, 0, 1]]) if debug_mode else k_path



                # TIM reconstruction
                h_uc = file.read_hamiltonian()
                s_uc = file.read_overlap()

                energy_bands_pred = []
                energy_bands_true = []
                for k_point in tqdm(k_path):
                    # Ground truth:
                    Hk_true = h_uc.Hk(reduced_coord(k_point, cell), gauge='cell').toarray()
                    Sk_true = s_uc.Sk(reduced_coord(k_point, cell), gauge='cell').toarray()

                    Ek_true = scipy.linalg.eigh(Hk_true, Sk_true, eigvals_only=True)
                    energy_bands_true.append(Ek_true)

                    # Prediction:
                    Hk_pred = reconstruct_tim_from_coo(k_point, h_pred.tocsr().tocoo(), geometry, cell)
                    # Sk = reconstruct_tim(k_point, s_pred, orb_i, orb_j, isc, cell)

                    Ek = scipy.linalg.eigh(Hk_pred, Sk_true, eigvals_only=True)

                    energy_bands_pred.append(Ek)
                
                # Plot energy bands
                energy_bands_pred_plot = np.stack(energy_bands_pred, axis=0)
                energy_bands_true_plot = np.stack(energy_bands_true, axis=0)

                title = f"Energy bands of sample {j} of {dataloader_type} dataset (seed {config["dataset"]["seed"]}). There are {n_atoms} in the unit cell. Using SIESTA overlap matrix."
                x_axis = [k_path]*energy_bands_pred_plot.shape[0]
                num_cols = energy_bands_pred_plot[:,0:n_bands].shape[1]
                titles_pred = [f"Predicted band {i}" for i in range(num_cols)]
                titles_true = [f"True band {i}" for i in range(num_cols)]
                plot_columns_of_2darray(
                    array_pred = energy_bands_pred_plot[:,0:n_bands],
                    array_true = energy_bands_true_plot[:,0:n_bands],
                    x = x_axis,
                    xlabel = "k", 
                    ylabel = "Energy (eV)", 
                    title = title,
                    titles_pred=titles_pred,
                    filepath = filepath_html
                )


            # === Plot eigenvalues ===
            filepath_html = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{checkpoint["epoch"]}eigenvalues.html")
            if filepath_html.is_file():
                print(f"File {filepath_html} already exists. Continueing.")
            else:
                titles_series = [f"[{"{:.2f}".format(k_point[0]) if k_point[0] != 0 else 0}, {"{:.2f}".format(k_point[1]) if k_point[1] != 0 else 0}, {"{:.2f}".format(k_point[2]) if k_point[2] != 0 else 0}]" for k_point in k_path]
                plot_predictions_vs_truths(
                    predictions=energy_bands_pred_plot,
                    truths=energy_bands_true_plot,
                    series_names=titles_series,
                    title='Eigenvalues comparison (eV)',
                    xaxis_title='True energy',
                    yaxis_title='Predicted energy',
                    legend_title='k points',
                    show_diagonal=True,
                    show_points_by_default=True,
                    filepath=filepath_html
                )

            # Count the number of plots done for each structure type
            n_plotted[1][col_idx] += 1



if __name__ == "__main__":
    main()