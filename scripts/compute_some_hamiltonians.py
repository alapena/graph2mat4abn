# import sys
# sys.path.insert(0, '/home/alapena/GitHub/graph2mat4abn')
# import os
# os.chdir('/home/ICN2/alapena/GitHub/graph2mat4abn') # Change to the root directory of the project

from graph2mat4abn.tools.import_utils import load_config, get_object_from_module
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, get_kwargs, load_model
from graph2mat4abn.tools.scripts_utils import get_model_dataset, init_mace_g2m_model
from graph2mat4abn.tools.script_plots import update_loss_plots, plot_grad_norms
from torch_geometric.data import DataLoader
from graph2mat4abn.tools.scripts_utils import generate_g2m_dataset_from_paths
from pathlib import Path
from mace.modules import MACE, RealAgnosticResidualInteractionBlock
from graph2mat.models import MatrixMACE
from graph2mat.bindings.e3nn import E3nnGraph2Mat
import torch
import warnings
from graph2mat import BasisTableWithEdges

warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")

from joblib import dump, load
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sisl

def main():
    paths = [
        Path('dataset/SHARE_OUTPUTS_8_ATOMS/d4f5-6b48-494f-b1de-c7e944c09f38'), #Hexagonal normal
        Path('dataset/SHARE_OUTPUTS_8_ATOMS/11ad-ba95-4a26-8f92-5267f5787553'), # Cubic
        Path('dataset/SHARE_OUTPUTS_2_ATOMS/fc1c-6ab6-4c0e-921e-99710e6fe41b'), # N-N
        Path('dataset/SHARE_OUTPUTS_2_ATOMS/7bbb-6d51-41eb-9de4-329298202ebf'), #B-B
        Path("dataset/SHARE_OUTPUTS_64_ATOMS/9410-b52a-4124-9c9c-210304f661a1"), #64
    ]
    # The current model:
    model_dir = Path("results/h_noc_2")
    filename = "train_best_model.tar"
    savedir = Path('results_dos')

    config = load_config(model_dir / "config.yaml")

    # Basis generation (needed to initialize the model)
    train_paths, val_paths = get_model_dataset(model_dir, verbose=False)
    paths = train_paths + val_paths
    basis = get_basis_from_structures_paths(paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)

    print("Initializing model...")
    model, optimizer, lr_scheduler, loss_fn = init_mace_g2m_model(config, table)

    # Load the model
    model_path = model_dir / filename
    model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, model_path, lr_scheduler=None, initial_lr=None, device='cpu')
    history = checkpoint["history"]
    print(f"Loaded model in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")

    for i, path in enumerate(paths):
        # === Inference ===
        dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, [path], verbose=False)
        dataloader = DataLoader(dataset, 1)
        model.eval()

        data = next(iter(dataloader))

        with torch.no_grad():
            model_predictions = model(data=data)

            h_pred = processor.matrix_from_data(data, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})[0]
            h_true = processor.matrix_from_data(data)[0]

        savedir.mkdir(exist_ok=True, parents=True)

        n_atoms = int(path.parts[-2].split('_')[2])
        structure = path.parts[-1]
        
        # Plot structure
        file = sisl.get_sile(path / "aiida.fdf")
        fig = file.plot.geometry(axes="xyz")
        filepath = savedir / f"{n_atoms}atm_{structure}.png"
        fig.write_image(str(filepath))
        filepath = savedir / f"{n_atoms}atm_{structure}.html"
        fig.write_html(str(filepath))
        print("Saved structure plot at", filepath)

        # Save hams
        filepath = savedir / f"{structure}_hamiltonian_pred.npz"
        np.savez(filepath, path=str(path), hamiltonian=h_pred)

        filepath = savedir / f"{structure}_hamiltonian_true.npz"
        np.savez(filepath, path=str(path), hamiltonian=h_true) 


if __name__ == "__main__":
    main()