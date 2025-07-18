import sisl
import torch
import numpy as np

from graph2mat import PointBasis
from tqdm import tqdm
from e3nn import o3
from pathlib import Path


def z_one_hot(z, orbitals, nr_bits):
    """
    Generate one-hot encodings from a list of single-value tensors.

    Args:
        z (list of torch.Tensor): A list of single-value tensors, e.g., [[2], [3], [4], [2], [2], ...].
        orbitals (dict): A dictionary mapping numbers to their corresponding values.
        nr_bits (int): The number of bits for one-hot encoding.

    Returns:
        torch.Tensor: A tensor containing the one-hot encodings.
    """

    # Extract values from the list of single-value tensors
    node_map={}
    k=0
    for key in orbitals.keys():
        node_map[key]=k
        k+=1

    indices = [tensor.item() for tensor in z]

    # Create an empty tensor for one-hot encoding
    one_hot = torch.zeros(len(indices), nr_bits)

    # Fill in the one-hot encoding based on the indices
    for i, idx in enumerate(indices):
        if idx in orbitals:  # Ensure the index exists in orbitals
            one_hot[i, int(node_map[idx])] = 1  # Set the corresponding bit to 1
        else:
            raise ValueError(f"Index {idx} not found in orbitals.")

    return one_hot

def flatten(xss):
    return [x for xs in xss for x in xs]



def get_basis_from_structures_paths(paths, verbose=False, num_unique_z=None):
    """_summary_

    Args:
        paths (list[str]): _description_
        verbose (bool, optional): _description_. Defaults to False.
        num_unique_z (int, optional): If known, input the total number of different atom types here for faster performance.

    Returns:
        list[PointBasis]: _description_
    """
    if verbose:
        print("="*60)
        print("Basis computation.")
        print(f"Number of structures to look on: {len(paths)}")
        print("Looking for unique atoms in each structure...")

    unique_atom_types = []
    unique_atom_types_path_idx = []

    # Look for all atom types in your list of structures
    iterator = tqdm(enumerate(paths)) if verbose else enumerate(paths)
    for i, path in iterator:
        path = Path(path)
        if num_unique_z is not None and len(unique_atom_types) == num_unique_z:
            print("Found enough basis points. Breaking the search...")
            break
        geometry = sisl.get_sile(path / "aiida.fdf").read_geometry()
        for z in geometry.atoms.Z:
            if z not in unique_atom_types:
                unique_atom_types.append(z)
                unique_atom_types_path_idx.append(i)
            if num_unique_z is not None and len(unique_atom_types) == num_unique_z:
                print("Found enough basis points. Breaking the search...")
                break
        # print("n_atoms= ", len(geometry.atoms.Z))

    if verbose:
        print(f"Found the following atomic numbers: {unique_atom_types}")
        print(f"Corresponding path indices: {unique_atom_types_path_idx}")

    # Build the basis
    basis = []
    unique_atom_types_basis = []
    for path_idx in unique_atom_types_path_idx:
        geometry = sisl.get_sile(paths[path_idx] / "aiida.fdf").read_geometry()
        for atom in geometry.atoms:
            if atom.Z not in unique_atom_types_basis:
                basis.append(PointBasis.from_sisl_atom(atom))
                unique_atom_types_basis.append(atom.Z)

    basis.sort(key=lambda x: x.type)
    unique_atom_types_basis.sort()

    if verbose:
        print(f"Basis with {len(basis)} elements built!")
        [print(f"\nBasis for atom {i}.\n\tAtom type: {basis[i].type}\n\tBasis: {basis[i].basis}\n\tBasis convention: {basis[i].basis_convention}\n\tR: {basis[i].R}") for i in range(len(basis))]

    return basis


def get_kwargs(module: str, config: dict) -> dict:
    env_config = config["environment_representation"]

    num_interactions = env_config["num_interactions"]
    hidden_irreps = o3.Irreps(env_config["hidden_irreps"])
    in_out_irreps = hidden_irreps * (num_interactions - 1) + str(hidden_irreps[0])

    if module == 'E3nnInteraction':
        kwargs = {
            'irreps': {
                'node_feats_irreps': in_out_irreps,
                'edge_attrs_irreps': o3.Irreps.spherical_harmonics(env_config.get("max_ell")),
                'edge_feats_irreps': o3.Irreps(f"{env_config.get("num_bessel")}x0e"), 
            },
        }

    elif module == 'E3nnEdgeMessageBlock':
        kwargs = {
            'irreps': {
                'node_feats_irreps': in_out_irreps,
                'edge_attrs_irreps': o3.Irreps.spherical_harmonics(env_config.get("max_ell")),
                'edge_feats_irreps': o3.Irreps(f"{env_config.get("num_bessel")}x0e"), 
                'edge_hidden_irreps': in_out_irreps,
            },
        }

    elif module == 'E3nnSimpleNodeBlock':
        kwargs = {}

    elif module == 'E3nnSimpleEdgeBlock':
        kwargs = {}

    else:
        raise ValueError(f"Module {module} not supported yet. Write the kwargs in this function.")
    
    return kwargs

def get_scheduler_args_and_kwargs(config, verbose=False):
    scheduler_config = config["scheduler"]

    type = scheduler_config.get("type", None)
    args = scheduler_config.get("args", None)
    kwargs = scheduler_config.get("kwargs", None)

    args = list(args.values()) if args is not None else None

    if verbose:
        print("LR Scheduler: ", type)
        print("Arguments: ", args)
        print("Keyword arguments: ", kwargs)

    return args, kwargs

    # if type is None:
    #     args = None
    #     kwargs = None

    # elif type == "CosineAnnealingWarmRestarts":
    #     scheduler_config = scheduler_config.get("CosineAnnealingWarmRestarts")
    #     args = None # TODO: Config this
    #     kwargs = {
    #         "t_0": scheduler_config.get("t_0"),
    #         "t_multiplication": scheduler_config.get("t_multiplication"),
    #         "eta_min": scheduler_config.get("eta_min"),
    #     }
    # elif type == "CyclicLR":
    #     scheduler_config = scheduler_config.get("CyclicLR")
    #     args = (
    #         float(scheduler_config.get("base_lr", None)),
    #         float(scheduler_config.get("max_lr", None)),
    #     )
    #     kwargs = {
    #         "step_size_up": scheduler_config.get("step_size_up", None),
    #         "cycle_momentum": scheduler_config.get("cycle_momentum", None),
    #         "mode": scheduler_config.get("mode", None),
    #     }
    # elif type == "OneCycleLR":
    #     scheduler_config = scheduler_config.get("OneCycleLR")
    #     len_train_dataloader = input_kwargs["len_train_dataloader"]
    #     epochs = config["trainer"].get("num_epochs")
    #     args = (
    #         float(scheduler_config.get("max_lr", None)),
    #     )
    #     kwargs = {
    #         # "total_steps": scheduler_config.get("epochs_cycle", None) * len_train_dataloader, # 
    #         # "epochs": epochs,                # Number of epochs
            
    #         # "pct_start": scheduler_config.get("pct_start", None),                # Fraction of steps spent increasing LR
    #         # "anneal_strategy": scheduler_config.get("anneal_strategy", None),        # Annealing strategy
    #         # "div_factor": float(scheduler_config.get("div_factor", None)),                # max_lr / initial_lr
    #         # "final_div_factor": float(scheduler_config.get("final_div_factor", None)),       # initial_lr / min_lr
    #     }
    # else:
    #     raise ValueError("Learning Rate scheduler not found.")

    return args, kwargs


def load_model(model, optimizer, path, lr_scheduler=None, initial_lr=None, device="cpu"):
    path = Path(path)

    checkpoint = torch.load(path, weights_only=True, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Set the initial lr of the optimizer if specified
    if initial_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr

    return model, checkpoint, optimizer, lr_scheduler


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_orbital_indices_and_shifts_from_sile(sile):
    geometry = sile.read_geometry()

    h_uc = sile.read_hamiltonian()

    # Convert to sparse format (coo)
    h_coo = h_uc.tocsr().tocoo()

    # Orbital indices and shifts extraction
    orb_i=-np.ones(h_coo.nnz, dtype=int) # We fill everything with minus ones to be able to detect iif something went wrong
    orb_j=-np.ones(h_coo.nnz, dtype=int) # Same
    isc=-np.ones((h_coo.nnz,3), dtype=int) # Same

    orb_i=h_coo.row
    for k in range(h_coo.nnz):
        orb_j[k]=geometry.osc2uc(h_coo.col[k])   # Unit cell orbital indices corresponding to the "supercell" indices
        isc[k]=geometry.o2isc(h_coo.col[k])      # Which unit cell / shift that orbital belongs to wrt. the (0,0,0) unit cell

    return orb_i, orb_j, isc


def reconstruct_tim_from_coo(k_point, M_coo, geometry, cell):
    no = M_coo.shape[0]  # Number of orbitals in the unit cell
    H_g = np.zeros((no, no), dtype=complex)  # Output matrix
    
    for k in range(M_coo.nnz):
        # Extract row index (already in unit cell basis)
        i_uc = M_coo.row[k]
        # Map supercell column index to unit cell orbital and shift
        j_uc = geometry.osc2uc(M_coo.col[k])
        isc_k = geometry.o2isc(M_coo.col[k])
        
        # Compute lattice vector for this shift
        R_uc = np.sum([cell[d] * isc_k[d] for d in range(3)], axis=0)
        Phase = np.exp(1j * k_point.dot(R_uc))
        
        # Accumulate value with phase
        H_g[i_uc, j_uc] += M_coo.data[k] * Phase
    
    return H_g


def reduced_coord(kpt, cell):
    """Convert a k-point to reduced coordinates."""
    return (cell.T)@kpt/(2*np.pi)


def write_structures_paths(structures_paths, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(structures_paths))

def read_structures_paths(filepath):
    try:
        with open(filepath, "r") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
