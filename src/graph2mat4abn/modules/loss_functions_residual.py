from graph2mat4abn.tools.tools import reconstruct_tim_from_coo
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import torch
import random
import sisl


def loss2_bands(config, batch, processor, model_predictions, device="cpu"):
      eigen_paths = [Path(str(p).replace("dataset", "dataset_eigen")) for p in batch.metadata["path"]]

      # --- subsample indices ---
      n_samples_batch= config["trainer"].get("n_samples_batch", 1)
      all_indices = list(range(len(eigen_paths)))
      sample_indices = random.sample(all_indices, min(n_samples_batch, len(all_indices)))


      loss2_losses = []
      hamiltonian_coo_batch_pred = processor.matrix_from_data(batch, predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]})
      for j in sample_indices:
            path = eigen_paths[j]

            bands = np.load(path/"bands.npz")["bands"]
            states = np.load(path/"states.npz")["states"]
            overlap_coo = load_npz(path/"overlap.npz")
            k = np.load(path/"k_path.npz")
            k_path, k_idx, k_label, k_len = k["k_path"], k["k_idx"], k["k_label"], k["k_len"]

            with torch.no_grad():
                  bands = torch.tensor(bands, dtype=torch.float32, device=device)
                  states = torch.tensor(states, dtype=torch.float32, device=device)

            path_dataset = Path(str(path).replace("dataset_eigen", "dataset"))
            geometry = sisl.get_sile(path_dataset / "aiida.fdf").read_geometry()
            cell = geometry.cell
            _, residual_norm_sq = loss_schrod_eq(bands, states, overlap_coo, k_path, geometry, cell, hamiltonian_coo_batch_pred[j].tocsr().tocoo(), device=device)
            loss2_losses.append(residual_norm_sq)

      loss2 = torch.stack(loss2_losses).mean()

      return loss2


def loss_schrod_eq(bands, states, overlap_coo, k_path, 
          geometry, cell, 
          hamiltonian_coo_pred,
          device="cpu"):
    
    overlap_tim = torch.tensor(reconstruct_tim_from_coo(k_path, overlap_coo, geometry, cell), dtype=torch.float32, device=device, requires_grad=True)
    hamiltonian_tim_pred = torch.tensor(reconstruct_tim_from_coo(k_path, hamiltonian_coo_pred, geometry, cell), dtype=torch.float32, device=device, requires_grad=True)

    true = bands[:, :, None] * (overlap_tim @ states)
    pred = hamiltonian_tim_pred @ states

    residual = pred - true
    residual_norm_sq = torch.sum(residual.conj() * residual).real

    return residual, residual_norm_sq/(bands.shape[1] * k_path.shape[0]) #Normalized by the nr of kpoints and bands.
