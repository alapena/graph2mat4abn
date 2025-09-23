import scipy.sparse as sp
import numpy as np

def create_sparse_matrix(n_atoms, n_orbs, n_shifts, size=3):
    # Calculate the dimensions
    rows = n_atoms * n_orbs
    cols = n_atoms * n_orbs * n_shifts
    
    # Create some random non-zero elements (for demonstration)
    # Here we add 3 random non-zero elements
    data = np.random.rand(size)
    row_indices = np.random.randint(0, rows, size=size)
    col_indices = np.random.randint(0, cols, size=size)
    
    # Create the sparse matrix in COO format
    sparse_mat = sp.coo_matrix((data, (row_indices, col_indices)), shape=(rows, cols))
    
    return sparse_mat