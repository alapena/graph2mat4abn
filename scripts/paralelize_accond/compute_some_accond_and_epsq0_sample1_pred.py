import torch
import sisl
import tbplas as tb
import numpy as np
from pathlib import Path
from graph2mat import BasisTableWithEdges
from torch_geometric.data import DataLoader

from graph2mat4abn.modules.fermi import get_fermi_energy, read_orbital_count
from graph2mat4abn.tools.import_utils import load_config
from graph2mat4abn.tools.scripts_utils import generate_g2m_dataset_from_paths, get_model_dataset, init_mace_g2m_model
from graph2mat4abn.tools.tbplas_modules import get_tbplas_cell
from graph2mat4abn.tools.tools import get_basis_from_structures_paths, load_model

import warnings
warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")
warnings.filterwarnings("ignore", message=".*is not a known matrix type key.*")

def main():
    paths = [
        # hc8:
        # Path("dataset/SHARE_OUTPUTS_2_ATOMS/9b13-4a57-4de9-b863-1b35209370c4"), #train
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/4ed6-914e-4aa3-923a-53c873f0cc31"), #train
        # Path("dataset/SHARE_OUTPUTS_2_ATOMS/fc1c-6ab6-4c0e-921e-99710e6fe41b"), #val
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/7b57-1410-4da3-8535-5183ac1f2f61"), #val 
        # Path("dataset/SHARE_OUTPUTS_64_ATOMS/16eb-54f8-42cb-bdb1-7b16f24a650c"), #val

        #hnoc2
        # Path("dataset/SHARE_OUTPUTS_2_ATOMS/9b13-4a57-4de9-b863-1b35209370c4"), #train
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/e1df-2940-4ada-b9c0-d210a6bb2a19"), #train
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/4ed6-914e-4aa3-923a-53c873f0cc31"), #train
        # Path("dataset/SHARE_OUTPUTS_64_ATOMS/e46e-c792-412c-99ac-9e20794f7aad"), #train 
        # Path("dataset/SHARE_OUTPUTS_2_ATOMS/504a-71cd-4d25-a04a-b7fa45b92200"), #val
        Path("dataset/SHARE_OUTPUTS_8_ATOMS/f2b9-d7cc-4f42-9ccc-13dd371d22a5"), #val ---
        # Path("dataset/SHARE_OUTPUTS_8_ATOMS/72f5-effe-42c4-bc67-12314ba36f5e"), #val ---
        # Path("dataset/SHARE_OUTPUTS_64_ATOMS/16eb-54f8-42cb-bdb1-7b16f24a650c"), #val ---
    ]
    model_dir = Path("results/h_noc_2")
    filename = "val_best_model.tar"
    split = "val"
    true_or_pred_idx = 1
    delta = 0.04

    n_kpoints_accond = 300
    n_orbs_per_atom = 13
    n_B = 3 #num valence electrons of Boron for DOS calculation
    n_N = 5 #num valence electrons of Nitrogen for DOS calculation


    true_or_pred_str = 'true' if true_or_pred_idx == 0 else 'pred'
    savefolder = model_dir.parts[-1] + "_" + split
    savedir = Path("results_ac_cond") / savefolder
    savedir.mkdir(exist_ok=True, parents=True)
    print("Created savedir", savedir)

    config = load_config(model_dir / "config.yaml")
    print("Computing the AC Conductivity using TBPLaS of paths:", paths)

    # Basis generation (needed to initialize the model)
    train_paths, _ = get_model_dataset(model_dir, verbose=False)
    basis = get_basis_from_structures_paths(train_paths, verbose=True, num_unique_z=config["dataset"].get("num_unique_z", None))
    table = BasisTableWithEdges(basis)

    print("Initializing model...")
    model, optimizer, lr_scheduler, loss_fn = init_mace_g2m_model(config, table)

    # Load the model
    model_path = model_dir / filename
    model, checkpoint, optimizer, lr_scheduler = load_model(model, optimizer, model_path, lr_scheduler=None, initial_lr=None, device='cpu')
    history = checkpoint["history"]
    print(f"Loaded model {model_dir} in epoch {checkpoint["epoch"]} with training loss {checkpoint["train_loss"]} and validation loss {checkpoint["val_loss"]}.")

    for i, path in enumerate(paths):
        print("\nComputing ", path)

        n_atoms = int(path.parts[-2].split('_')[2])
        structure = path.parts[-1]

        # === Inference ===
        dataset, processor = generate_g2m_dataset_from_paths(config, basis, table, [path], verbose=False)
        dataloader = DataLoader(dataset, 1)
        model.eval()

        data = next(iter(dataloader))

        with torch.no_grad():
            model_predictions = model(data=data)

            h_pred = processor.matrix_from_data(data, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})[0]

        # Plot structure
        file = sisl.get_sile(path / "aiida.fdf")
        geometry = file.read_geometry()

        fig = file.plot.geometry(axes="xyz")
        filepath = savedir / f"{n_atoms}atm_{structure}.png"
        fig.write_image(str(filepath))
        filepath = savedir / f"{n_atoms}atm_{structure}.html"
        fig.write_html(str(filepath))
        print("Saved structure plot at", filepath)

        # Two iterations per sample: first for true, second for pred.
        # for true_or_pred_idx in range(2): # We comment this because the computations are so long that we prefer to compute the true and pred in paralel.
        if True:
            file = sisl.get_sile(path / "aiida.HSX")
            if true_or_pred_idx == 0:
                # True matrix
                hamiltonian = file.read_hamiltonian().tocsr().tocoo()
            elif true_or_pred_idx == 1:
                # Pred matrix
                hamiltonian = h_pred.tocsr().tocoo()
            else:
                raise ValueError("Something went wrong inside the loop.")
            
            # For now, we use the true overlap for both.
            overlap = file.read_overlap().tocsr().tocoo()

            # Build TBPLaS model
            ham_cell, overlap_cell = get_tbplas_cell(geometry, hamiltonian, overlap, units=tb.ANG)

            # Compute AC Conductivity

            # Load true DOS
            directory_dos = Path("results_dos") / savefolder
            struct_name = f"{n_atoms}atm_" + structure
       
            pattern = f"{split}_{struct_name}_dos_mesh*_{true_or_pred_str}.npz"
            matching_files = list(directory_dos.glob(pattern))
            if matching_files:
                filepath = matching_files[-1] # Load the last match (if any)
                print(f"Loading file {filepath}")
            else:
                raise ValueError(f"No matching file found in {directory_dos} with pattern {pattern}.")

            loaded_file = np.load(filepath)
            energies = loaded_file["energies"]
            dos_true = loaded_file["dos"]
            print("DOS data extracted from", filepath)

            n_orbs_supercell = read_orbital_count(path / "aiida.ORB_INDX")
            n_cells = n_orbs_supercell//n_orbs_per_atom//int(n_atoms)

            lind = tb.Lindhard(ham_cell, overlap=overlap_cell)
            e_min = 0
            e_fermi = get_fermi_energy(energies, dos_true, n_orbs_per_atom*int(n_atoms)*n_cells, (3*n_B + 5*n_N)*n_cells, ShowPlot=False)
            # e_max = np.min([np.max(energies), e_fermi+20])
            e_max = 18

            cfg = lind.config
            cfg.prefix       = "ac_cond"   # basename for saved results (optional)
            cfg.e_min        = e_min            
            cfg.e_max        = e_max
            cfg.dimension    = 3             
            cfg.num_spin     = 2              
            cfg.k_grid_size  = (n_kpoints_accond, n_kpoints_accond, n_kpoints_accond)    # k-mesh (increase for convergence)
            cfg.delta = delta

            # Optional (commonly used) physics knobs:
            # cfg.mu            = 0.0           # chemical potential (eV)
            # cfg.temperature   = 300      # temperature (K)
            # cfg.back_epsilon  = 1.0           # background dielectric constant
            cfg.ac_cond_component = 1      # choose σ_xx, σ_yy, etc.

            timer = tb.Timer()
            timer.tic("ac_cond")
            omegas, ac_cond = lind.calc_ac_cond()
            timer.toc("ac_cond")
            timer.report_total_time()

            if true_or_pred_idx == 0:
                filepath = savedir / f"{split}_{n_atoms}atm_{structure}_accond_true.npz"
            if true_or_pred_idx == 1:
                filepath = savedir / f"{split}_{n_atoms}atm_{structure}_accond_pred.npz"
            np.savez(filepath, path=str(path), omegas=omegas, ac_cond=ac_cond, time=timer._time_usage["ac_cond"])
            print(f"Saved AC Conductivity of structure {struct_name} at {filepath}")

            timer = tb.Timer()
            timer.tic("diel_demo")
            eps_q0 = lind.calc_epsilon_q0(omegas, ac_cond)
            timer.toc("diel_demo")

            if true_or_pred_idx == 0:
                filepath = savedir / f"{split}_{n_atoms}atm_{structure}_epsq0_true.npz"
            if true_or_pred_idx == 1:
                filepath = savedir / f"{split}_{n_atoms}atm_{structure}_epsq0_pred.npz"
            np.savez(filepath, path=str(path), eps_q0=eps_q0, time=timer._time_usage["diel_demo"])
            print(f"Saved dielectric function of structure {struct_name} at {filepath}")



if __name__=="__main__":
    main()