from graph2mat4abn.tools.tools import reconstruct_tim_from_coo
import torch


def bands(bands, states, overlap_coo, k_path, 
          geometry, cell, 
          hamiltonian_coo_pred,
          device="cpu"):
    
    overlap_tim = torch.tensor(reconstruct_tim_from_coo(k_path, overlap_coo, geometry, cell), dtype=torch.float32, device=device, requires_grad=True)
    hamiltonian_tim_pred = torch.tensor(reconstruct_tim_from_coo(k_path, hamiltonian_coo_pred, geometry, cell), dtype=torch.float32, device=device, requires_grad=True)

    true = bands[:, :, None] * (overlap_tim @ states)
    pred = hamiltonian_tim_pred @ states

    residual = pred - true
    residual_norm_sq = torch.sum(residual.conj() * residual).real

    return residual, residual_norm_sq
